# --- derived from run_math.py by the shared-layer shim ---
import sys
import os
import argparse
import yaml
import json
import time
import torch
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile

# Which final-node prompt each dataset gets. math and amc take the authors'
# maths file; mbpp theirs; mmlu_pro and drop take files derived from the
# authors' mmlu.json, changing only the statement of how many options exist
# and what the answer looks like -- see shims/masrouter/install.py.
SHARED_FINAL_NODE = {
    'math': 'MAR/Roles/FinalNode/math.json',
    'amc': 'MAR/Roles/FinalNode/math.json',
    'mbpp': 'MAR/Roles/FinalNode/mbpp.json',
    'mmlu_pro': 'MAR/Roles/FinalNode/shared_mmlu_pro.json',
    'drop': 'MAR/Roles/FinalNode/shared_drop.json',
}
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from Datasets.shared_dataset import load_shared_dataset, shared_score, shared_task_labels
from MAR.Utils.log import configure_logging
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on MATH")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='stop each epoch after this many batches '
                             '(shim: for smoke runs; None = full epoch)')
    parser.add_argument("--shared_dataset", type=str, required=True,
                        choices=['math', 'amc', 'mbpp', 'drop', 'mmlu_pro'],
                        help="which shared benchmark to run")
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16,help="batch size")
    parser.add_argument('--epochs', type=int, default=5, help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='defaults to the author file matching --shared_dataset')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=100.0)
    parser.add_argument('--max_agent', type=int, default=6)
    args = parser.parse_args()
    if args.prompt_file is None:
        args.prompt_file = SHARED_FINAL_NODE[args.shared_dataset]
        print(f'shim: final-node prompt {args.prompt_file}', flush=True)
    return args


if __name__ == '__main__':
    args = parse_args()
    fix_random_seed(1234)
    train_dataset = load_shared_dataset(args.shared_dataset, split="train")
    test_dataset = load_shared_dataset(args.shared_dataset, split="test")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"{args.shared_dataset}_{current_time}.txt"
    configure_logging(log_name=log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=args.max_agent,device=device).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    logger.info("Start training...")
    num_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    if args.max_batches is not None:
        num_batches = min(num_batches, args.max_batches)
        logger.info(f'shim: capping each epoch at {num_batches} batch(es)')

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}",80*'-')
        total_solved, total_executed = (0, 0)
        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"{args.shared_dataset}_router_epoch{epoch}.pth", map_location=torch.device('cuda')))
            continue
        for i_batch in range(num_batches):
            logger.info(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            current_batch = dataloader(train_dataset,args.batch_size,i_batch)
            queries = [item['problem'] for item in current_batch]
            answers = [item['row'] for item in current_batch]
            task_labels = shared_task_labels(args.shared_dataset, current_batch)
            tasks_y = torch.tensor(task_labels).to(device)
            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs, vae_loss, agents_num = router.forward(queries, tasks, llms, reasonings, task_labels,prompt_file=args.prompt_file)

            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            agent_num_loss = 0
            utilities = []
            answers_loss = []
            is_solved_list = []
            for result, true_answer, log_prob, cost in zip(results, answers, log_probs, costs):
                is_solved = shared_score(args.shared_dataset, true_answer, result)
                predict_answer = result
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                utility = is_solved - cost * args.cost_rate
                utilities.append(utility)
                is_solved_list.append(is_solved)
                answer_loss:torch.Tensor = -log_prob * utility
                answers_loss.append(answer_loss)
                
            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            vae_loss = vae_loss.mean()
            is_solved_tensor = torch.tensor(is_solved_list, dtype=torch.float32, device=device).unsqueeze(1)  # shape: [N, 1]
            adjust_loss = ((1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num) + 0.25 * is_solved_tensor *  agents_num).mean()
            
            loss = task_loss + answer_loss + vae_loss*0.001 # + adjust_loss
            loss.backward()
            optimizer.step()
            
            accuracy = total_solved / total_executed
            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities:{utilities}")
        torch.save(router.state_dict(), f"{args.shared_dataset}_router_epoch{epoch}_new.pth")
    logger.info("Finish training...")
    logger.info("Start testing...")
    total_solved, total_executed = (0, 0)
    num_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size

    for i_batch in range(num_batches):
        logger.info(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        current_batch = dataloader(test_dataset,args.batch_size,i_batch)
        queries = [item['problem'] for item in current_batch]
        answers = [item['row'] for item in current_batch]
        task_labels = shared_task_labels(args.shared_dataset, current_batch)
        tasks_y = torch.tensor(task_labels).to(device)
        results, costs, log_probs, tasks_probs, vae_loss, agents_num  = router.forward(queries, tasks, llms, reasonings, task_labels,prompt_file=args.prompt_file)

        utilities = []
        for result, true_answer, log_prob, cost in zip(results, answers, log_probs, costs):
            is_solved = shared_score(args.shared_dataset, true_answer, result)
            predict_answer = result
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            utility = is_solved - cost * args.cost_rate
            utilities.append(utility)
            logger.debug(f"Predict: {predict_answer}")
            logger.debug(f"Truth: {true_answer}")
        
        accuracy = total_solved / total_executed
        logger.info(f"Batch time {time.time() - start_ts:.3f}")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"utilities:{utilities}")
    logger.info("Finish testing...")
