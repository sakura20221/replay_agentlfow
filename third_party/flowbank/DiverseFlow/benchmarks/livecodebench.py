import asyncio
import json
import os
import multiprocessing
import threading
import time
import base64
import zlib
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
import sys
sys.path.append("..")
sys.path.append("benchmarks")
# ensure lcb_runner is installed
from scripts.utils.lcb_runner import run_test

# Key functions copied from LiveCodeBench official evaluation
#sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use 'spawn' context so child processes do NOT inherit the parent's ~500MB+ dataset.
# With 'fork' (the default), every child copies the parent's full address space,
# causing OOM when multiple large-input problems run concurrently.
_spawn_ctx = multiprocessing.get_context("spawn")

def _pipe_worker(conn, sample, generation, debug, timeout, memory_limit_mb):
    """Worker that sends results back via Pipe instead of Manager.

    Runs in a spawned process (clean interpreter), so it only holds
    this single problem's data, not the full dataset.
    """
    try:
        # Set memory limit BEFORE executing user code
        import resource
        max_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except Exception:
        pass  # setrlimit may fail on some systems; spawn already helps

    try:
        from scripts.utils.lcb_runner import run_test as _run_test
        res, metadata = _run_test(sample, test=generation, debug=debug, timeout=timeout)
        conn.send((res, metadata))
    except MemoryError:
        conn.send(None)
    except Exception:
        conn.send(None)
    finally:
        conn.close()

def check_correctness(sample, generation, timeout, debug=True, memory_limit_mb=2048):
    parent_conn, child_conn = _spawn_ctx.Pipe(duplex=False)
    p = _spawn_ctx.Process(
        target=_pipe_worker,
        args=(child_conn, sample, generation, debug, timeout, memory_limit_mb),
    )
    p.start()
    child_conn.close()  # Close child end in parent process

    join_timeout = (timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    p.join(timeout=join_timeout)

    killed = False
    if p.is_alive():
        p.kill()
        p.join()
        killed = True

    result = None
    if parent_conn.poll():
        try:
            result = parent_conn.recv()
        except (EOFError, OSError):
            pass
    parent_conn.close()

    if result is None:
        in_outs = json.loads(sample["input_output"])
        default_res = [-1 for _ in range(len(in_outs["inputs"]))]
        reason = "Killed (timeout or memory limit)" if killed else "Crashed (likely memory limit)"
        default_metadata = {"error": reason, "error_code": -1, "error_message": reason}
        if debug:
            logger.warning(
                f"{reason}: {sample.get('question_id', 'unknown')}"
            )
        return default_res, default_metadata

    return result[0], result[1]

def evaluate_generations_by_problem(args):
    problem_generations, sample, debug, timeout = args
    res = []
    metadata = []
    for generation in problem_generations:
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(
                sample, generation, timeout=timeout, debug=debug
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            res.append(curr_res)
            metadata.append(curr_metadata)
    return res, metadata

class LiveCodeBench(BaseBenchmark):
    # LiveCodeBench test inputs can be 10-30MB each; generated (often naive)
    # code can use several GB per problem. Keep concurrency low to stay
    # within SLURM memory limits (128GB).
    DEFAULT_MAX_CONCURRENT_TASKS = 3

    def __init__(self, name: str, file_path: str, log_path: str, llm_config=None, timeout: int = 6):
        super().__init__(name, file_path, log_path, llm_config)
        self.timeout = timeout
        self.num_process_evaluate = min(16, os.cpu_count() or 4)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        exception_occurred = []
        stop_event = threading.Event()

        def target():
            try:
                return_value = func(*args)
                result.append(return_value)
            except Exception as e:
                exception_occurred.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")
        if exception_occurred:
            raise exception_occurred[0]
        return result[0] if result else None
    def parse_code(self, prediction):
        prediction = prediction.split("```python")[-1]
        prediction = prediction.split("```")[0]
        return prediction
    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        """Load data from JSONL and convert to LiveCodeBench evaluation format."""
        raw_data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                raw_data.append(json.loads(line))
        
        # Convert to evaluation format
        processed_data = []
        for item in raw_data:
            try:
                # Handle private test cases (only use private test cases for evaluation)
                try:
                    private_tests = json.loads(item["private_test_cases"])
                except:
                    private_tests = json.loads(
                        pickle.loads(
                            zlib.decompress(
                                base64.b64decode(item["private_test_cases"].encode("utf-8"))
                            )
                        )
                    )
                
                # Build evaluation sample
                fn_name = json.loads(item["metadata"]).get("func_name", None) if item["metadata"] else None
                processed_item = {
                    "question": item["question_content"],
                    "input_output": json.dumps({
                        "inputs": [t["input"] for t in private_tests],
                        "outputs": [t["output"] for t in private_tests],
                        "fn_name": fn_name
                    }),
                    "question_id": item['question_id'],
                    "canonical_solution": item.get("starter_code", ""),
                    "metadata": {
                        "func_name": fn_name,
                        "difficulty": item.get("difficulty", "unknown"),
                        "platform": item.get("platform", "unknown"),
                    }
                }
                processed_data.append(processed_item)   
            
            except Exception as e:
                logger.error(f"Error processing data: {str(e)}")
                continue
        
        if specific_indices is not None:
            return [processed_data[i] for i in specific_indices if i < len(processed_data)]
        return processed_data

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, agent: Callable, prompt: str, entry_point: str) -> Tuple[str, float]:
        return await asyncio.wait_for(agent(prompt, entry_point), timeout=120)

    async def evaluate_problem(self, problem: dict, agent: Callable, save_path: str = None) -> Tuple[str, str, str, float, Dict, float]:
        question = problem["question"]
        question_id = problem["question_id"]
        
        try:
            logger.info(
                f"Start evaluating LiveCodeBench problem: {question_id}"
            )
            
            # Generate code
            entry_point = problem["metadata"].get("func_name", "wrapped_function") if problem["metadata"] else "wrapped_function"
            logger.info(f"entry_point: {entry_point}")
            prediction, cost = await self._generate_output(
                agent, question, entry_point
            )
            logger.info(
                f"Finished code generation, task: {question_id}, cost: {cost}"
            )
            prediction = self.parse_code(prediction)
            # Use LiveCodeBench evaluation logic
            sample = {
                "question": question,
                "input_output": problem["input_output"],
                "question_id": question_id
            }
            # logger.info(f"Start evaluating sample {sample['input_output']}")
            
            # Evaluate in a multiprocessing environment.
            # Use thread pool (None) instead of ProcessPoolExecutor, because
            # check_correctness already spawns its own subprocess for isolation.
            # Using ProcessPoolExecutor here would double-fork, wasting memory.
            args = ([prediction], sample, False, self.timeout)
            loop = asyncio.get_running_loop()
            results, metadata = await loop.run_in_executor(
                None, evaluate_generations_by_problem, args
            )
            
            # Parse results
            logger.info(f"Test results: {results}")
            test_results = results[0]  # Take all test case results of the first (and only) generation
            test_metadata = metadata[0]
            passed = all(r == 1 for r in test_results)
            score = 1.0 if passed else 0.0
            
            # Build evaluation details
            evaluation_details = {
                "question_id": question_id,
                "test_results": test_results,
                "metadata": test_metadata,
                "execution_success": passed,
                "difficulty": problem.get("metadata", {}).get("difficulty", "unknown"),
                "platform": problem.get("metadata", {}).get("platform", "unknown")
            }
            
            # Build expected output for logging
            expected_output = {
                "question_id": question_id,
                "difficulty": evaluation_details["difficulty"],
                "platform": evaluation_details["platform"],
                "canonical_solution": problem.get("canonical_solution", "")
            }

            # Log failures
            if not passed:
                self.log_mismatch(
                    problem=question,
                    expected_output=json.dumps(expected_output),
                    prediction=prediction,
                    extracted_output=prediction,
                    extract_answer_code="N/A"
                )
                logger.warning(f"Task failed: {question_id}, score: {score}")
            else:
                logger.info(f"Task succeeded: {question_id}, score: {score}")

            result = (question, prediction, json.dumps(expected_output), score, evaluation_details, cost)
            
            # Save results
            if save_path:
                async with aiofiles.open(save_path, mode="a", encoding="utf-8") as file:
                    await file.write(json.dumps(result) + "\n")
            
            return result

        except asyncio.TimeoutError:
            logger.error(f"Code generation timeout: {question_id}")
            evaluation_details = {"question_id": question_id, "error": "Timeout"}
            return (question, "Timeout", "", 0.0, evaluation_details, 0.0)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Evaluation error: {question_id}, error: {e}")
            evaluation_details = {"question_id": question_id, "error": str(e)}
            return (
                question,
                f"Evaluation error: {str(e)}",
                "",
                0.0,
                evaluation_details,
                0.0,
            )

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        return 0.0, ""

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "evaluation_details", "cost"]

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = None):
        if max_concurrent_tasks is None:
            max_concurrent_tasks = self.DEFAULT_MAX_CONCURRENT_TASKS
        return await super().run_evaluation(agent, va_list, max_concurrent_tasks)

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = None):
        if max_concurrent_tasks is None:
            max_concurrent_tasks = self.DEFAULT_MAX_CONCURRENT_TASKS
        return await super().run_baseline(agent, max_concurrent_tasks)

    async def run_baseline_with_load_data(self, agent: Callable, past_data_path: str = None, max_concurrent_tasks: int = None):
        if max_concurrent_tasks is None:
            max_concurrent_tasks = self.DEFAULT_MAX_CONCURRENT_TASKS
        all_data = await self.load_data()

        if not past_data_path:
            past_data_path = os.path.join(self.log_path, f"{self.name}_results.jsonl")

        # Load past results
        past_results = {}
        if os.path.exists(past_data_path):
            async with aiofiles.open(past_data_path, mode="r", encoding="utf-8") as file:
                async for line in file:
                    try:
                        result = json.loads(line)
                        # Use question text as key
                        past_results[result[0]] = result
                    except:
                        continue

        # Filter new problems
        new_data = [p for p in all_data if p["question"] not in past_results]

        if not new_data:
            logger.info("All problems have been evaluated")
            return None, None, None

        logger.info(
            f"{len(new_data)} new problems to evaluate out of {len(all_data)} total"
        )

        # Evaluate new problems
        new_results = await self.evaluate_all_problems(
            new_data, agent, save_path=past_data_path, max_concurrent_tasks=max_concurrent_tasks
        )

        # Merge results
        all_results = list(past_results.values()) + new_results

        # Save final results
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results(all_results, columns)

        logger.info(f"{self.name} dataset average score: {average_score:.5f}")
        logger.info(f"Total cost: {total_cost:.5f}")
        logger.info(f"Average cost: {average_cost:.5f}")
        return average_score, average_cost, total_cost
