import os
import re
import json
from enum import Enum
from typing import Any, List, Tuple, Union


class CodeDataset(Enum):
    HUMAN_EVAL = "HumanEval"
    MBPP = "MBPP"
    LIVE_CODE_BENCH = "LiveCodeBench"
    CODEFORCES = "Codeforces"


def _resolve_dataset_file(fname: str) -> str:
    """Locate a public-test jsonl (e.g. 'mbpp_public_test.jsonl') robustly.

    The repo ships these at the top-level datasets/ dir. This file is
    DiverseFlow/scripts/utils/code.py, so the repo root is three dirs up;
    a few legacy/standalone layouts are also accepted. Returns the first
    existing candidate, else the top-level path (for a clear downstream error).
    """
    here = os.path.dirname(os.path.abspath(__file__))            # .../DiverseFlow/scripts/utils
    diverseflow_dir = os.path.dirname(os.path.dirname(here))      # .../DiverseFlow
    repo_root = os.path.dirname(diverseflow_dir)                  # .../<repo root>
    candidates = [
        os.path.join(repo_root, "datasets", fname),              # shared top-level
        os.path.join(diverseflow_dir, "datasets", fname),        # DiverseFlow/datasets
        os.path.join(diverseflow_dir, "data", "datasets", fname),
        os.path.join("data", "datasets", fname),                 # legacy cwd-relative
        os.path.join("datasets", fname),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def extract_test_cases_from_jsonl(entry_point: str, dataset: Union[CodeDataset, str, None] = None, problem: str = None):
    file_map = {
        "SHARED_MBPP": "shared_mbpp_public_test.jsonl",  # shared-layer shim
        CodeDataset.HUMAN_EVAL.value: "humaneval_public_test.jsonl",
        CodeDataset.MBPP.value: "mbpp_public_test.jsonl",
        CodeDataset.LIVE_CODE_BENCH.value: "livecodebench_public_test.jsonl",
        CodeDataset.CODEFORCES.value: "codeforces_public_test.jsonl",
    }
    hardcoded_cases_map = {
        CodeDataset.HUMAN_EVAL.value: {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        },
        CodeDataset.MBPP.value: {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        },
        CodeDataset.LIVE_CODE_BENCH.value: {},
        CodeDataset.CODEFORCES.value: {},
    }

    # Determine which datasets to search
    if dataset is not None:
        dataset_value = dataset.value if isinstance(dataset, CodeDataset) else dataset
        datasets_to_search = [dataset_value]
    else:
        # Auto-detect: search all datasets
        datasets_to_search = list(file_map.keys())

    for ds in datasets_to_search:
        hardcoded_cases = hardcoded_cases_map.get(ds, {})
        if problem is None and entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]

        fname = file_map.get(ds)
        if not fname:
            continue
        file_path = _resolve_dataset_file(fname)
        matches = []
        task_matches = []
        prompt_matches = []
        try:
            with open(file_path, "r") as file:
                for line in file:
                    data = json.loads(line)
                    if data.get("entry_point") != entry_point:
                        continue
                    task = str(data.get("task") or "")
                    prompt = str(data.get("prompt") or "")
                    if problem is not None and task and str(problem).startswith(task):
                        task_matches.append(data.get("test"))
                    if problem is not None and prompt and str(problem).startswith(prompt):
                        prompt_matches.append(data.get("test"))
                    matches.append(data.get("test"))
        except FileNotFoundError:
            continue
        if len(task_matches) == 1:
            return task_matches[0]
        if len(prompt_matches) == 1:
            return prompt_matches[0]
        if len(matches) == 1:
            return matches[0]

    return None


def extract_test_cases(docstring: str) -> List[Tuple[str, List[Any], Any]]:
    # Use regular expressions to match test cases, now capturing function names and any output
    pattern = r">>> (\w+)\((.*?)\)\n\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)

    test_cases = []
    for match in matches:
        func_name, input_str, expected_output = match

        # Process input
        input_list = []
        for item in input_str.split(","):
            item = item.strip()
            try:
                # Try to convert input to numeric type
                if "." in item:
                    input_list.append(float(item))
                else:
                    input_list.append(int(item))
            except ValueError:
                # If unable to convert to numeric, keep as string
                input_list.append(item.strip("'\""))

        # Process output
        try:
            # Try to convert output to numeric or boolean value
            if expected_output.lower() == "true":
                expected_output = True
            elif expected_output.lower() == "false":
                expected_output = False
            elif "." in expected_output:
                expected_output = float(expected_output)
            else:
                expected_output = int(expected_output)
        except ValueError:
            # If unable to convert, keep as string
            expected_output = expected_output.strip("'\"")

        test_cases.append([func_name, input_list, expected_output])

    return test_cases


def test_cases_2_test_functions(solution: str, test_cases: str):
    tester_function = f"""
{solution}

{test_cases}
"""
    return tester_function


def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function
