# --- derived from run_gsm8k.py by the shared-layer shim ---
import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.globals import Time
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.shared_dataset import shared_data_process, shared_score
import GDesigner.prompt.shared_prompt_sets  # noqa: F401  # registers the shared domains

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
    parser = argparse.ArgumentParser(description="GDesigner Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--train_items', type=int, default=0,
                        help='number of search items used for gradient updates')
    parser.add_argument('--search_items', type=int, default=0,
                        help='index where the held-out split starts')
    parser.add_argument('--eval_batch_size', type=int, default=0,
                        help='shim: batch size after the switch to evaluation, where no gradient is taken (0 = same as --batch_size)')
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=None,
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=None,
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default=None,
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if (args.agent_names is not None and args.agent_nums is not None
            and len(args.agent_names) != len(args.agent_nums)):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

_DOMAIN_DEFAULTS = {'math': (['MathSolver'], [4], 'FinalRefer'), 'code': (['CodeWriting'], [5], 'FinalWriteCode'), 'qa': (['AnalyzeAgent'], [5], 'FinalRefer')}
_DOMAIN_FAMILY = {'math': 'math', 'amc': 'math', 'mbpp': 'code', 'drop': 'qa', 'mmlu_pro': 'qa'}


async def main():
    args = parse_args()
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    # --- shared-layer shim (agent_wf_v2) --- smoke cap v1
    # Smoke mode: keep the first training batch plus a slice of REAL evaluation
    # items. The eval slice must come from past the train/eval boundary of the
    # train_then_eval file (the runner flips to evaluation after num_iterations
    # batches), or the "evaluation" records would carry train-split uids and the
    # collector would rightly refuse them. SHIM_SMOKE_EVAL_FROM carries that
    # boundary in items; both are set only by sweep.py --smoke.
    import os as _smoke_os
    _smoke_n = _smoke_os.getenv("SHIM_SMOKE_N")
    if _smoke_n:
        _smoke_n = int(_smoke_n)
        _head = args.batch_size * args.num_iterations
        _from = int(_smoke_os.getenv("SHIM_SMOKE_EVAL_FROM") or _head)
        _from = max(_from, _head)
        dataset = dataset[:_head] + dataset[_from:_from + _smoke_n]
    dataset = shared_data_process(dataset, args.domain)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/result/{args.domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
    # shared-layer shim: honour --result_file when given; the default name
    # collides between concurrent jobs of the same domain (1-second stamp).
    if args.result_file:
        result_file = Path(args.result_file)
        result_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
    
    # --- shared-layer shim: per-domain agent defaults ---
    _names, _nums, _decision = _DOMAIN_DEFAULTS[_DOMAIN_FAMILY[args.domain]]
    if args.agent_names is None:
        args.agent_names = _names
    if args.agent_nums is None:
        args.agent_nums = _nums
    if args.decision_method is None:
        args.decision_method = _decision
    print(f"[shim] domain={args.domain} agents={args.agent_names}x{args.agent_nums} decision={args.decision_method}")
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.gcn.train()
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=args.lr)   
    # --- shared-layer shim: explicit search/evaluation boundary ---
    # batch_size is a gradient hyperparameter and is left at the author's value for
    # the training batches. After the switch at --num_iterations the runner sets
    # optimized_spatial/temporal to False, and the optimiser step is guarded by
    # exactly that flag, so no gradient is taken over the remaining batches: there
    # the batch is purely an execution-concurrency knob.
    #
    # It has to be raised because these two repos dispatch only batch_size requests
    # at a time (every other method here runs 30-50 concurrently). Measured on
    # mmlu_pro: 48 questions in 72 minutes, i.e. 34 hours for one job's 1372
    # questions, and there are 16 such jobs. The evaluation split is what dominates
    # that -- 1120 of the 1372 -- and it is precisely the part with no gradient.
    _eval_bs = args.eval_batch_size or args.batch_size
    _train_items = args.train_items or args.num_iterations * args.batch_size
    _search_items = args.search_items or _train_items
    if not (0 < _train_items <= _search_items <= len(dataset)):
        raise ValueError("expected 0 < train_items <= search_items <= dataset size")
    _batch_plan = []
    _cursor = 0
    while _cursor < _train_items:
        _end = min(_cursor + args.batch_size, _train_items)
        _batch_plan.append((_cursor, _end))
        _cursor = _end
    if len(_batch_plan) != args.num_iterations:
        raise ValueError("num_iterations must equal ceil(train_items / batch_size)")
    # A control run may keep the author's smaller update budget. Skip the search
    # items it did not train on and begin inference at the real held-out boundary.
    _cursor = _search_items
    while _cursor < len(dataset):
        _end = min(_cursor + _eval_bs, len(dataset))
        if _end - _cursor < 1:
            break
        _batch_plan.append((_cursor, _end))
        _cursor = _end
    num_batches = len(_batch_plan)
    print(f"[shim] {args.num_iterations} training batch(es), {_train_items} item(s); "
          f"evaluation starts at item {_search_items} in "
          f"{num_batches - args.num_iterations} batch(es) of up to {_eval_bs}")
    total_solved, total_executed = (0, 0)
    
    for i_batch in range(num_batches):
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        
        _lo, _hi = _batch_plan[i_batch]
        
        current_batch = dataset[_lo:_hi]
        if current_batch is None:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            task = record["task"]
            step = record["step"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        
        for task, answer, log_prob, true_answer in zip(current_batch, raw_answers, log_probs, answers):
            _shared_score, predict_answer = shared_score(args.domain, task, answer[0])
            is_solved = _shared_score
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            updated_item = {
                "Question": task,
                "Answer": true_answer,
                "Step": step,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }
            data.append(updated_item)
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print("loss:", total_loss.item())
        
        if i_batch+1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.gcn.eval()
            print("Start Eval")
            
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())
