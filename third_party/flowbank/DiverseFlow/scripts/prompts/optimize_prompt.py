# =============================================================================
# Optimizer Prompt — Restructured with ScoreFlow-inspired ordering:
#   1. WORKFLOW_GOAL        (brief objective + dataset description)
#   2. generate_operator_usage()  (available operators)
#   3. WORKFLOW_CONTEXT     (previous best graph, prompt, experience, logs)
#   3.5 WORKFLOW_DIVERSITY_CONTEXT (Phase 2 only: coverage info + uncovered queries)
#   4. WORKFLOW_RULES       (all critical rules — placed LAST for emphasis)
# =============================================================================

# =============================================================================
# SECTION 1: Goal — What the optimizer LLM needs to do (placed FIRST)
# =============================================================================

WORKFLOW_GOAL = """You are building an LLM Graph (Python workflow) and corresponding Prompt to jointly solve {type} problems.
{dataset_description}
Referring to the given graph and prompt below, please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your single modification in XML tags in your reply.

When optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider Python's loops (for) and conditional statements (if-elif-else).
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process.
Introducing multiple operators at appropriate points can enhance performance. If some provided operators are not yet used, try incorporating them.
"""

# =============================================================================
# Dataset Descriptions — Help the optimizer LLM understand the task domain
# =============================================================================

DATASET_DESCRIPTIONS = {
    "MATH": """**Dataset: MATH** — A collection of competition mathematics problems from AMC, AIME, and other sources, spanning 7 categories: Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, and Precalculus. Difficulty levels range from 1 (easy) to 5 (competition-hard). Problems require multi-step mathematical reasoning, symbolic manipulation, and sometimes creative insight. Solutions must be expressed as exact values (fractions, radicals, etc.), not decimal approximations.""",

    "GSM8K": """**Dataset: GSM8K** — Grade school math word problems requiring multi-step arithmetic reasoning. Problems involve basic operations (addition, subtraction, multiplication, division) applied to real-world scenarios. Each problem requires 2-8 reasoning steps. The key challenge is correctly extracting quantities from natural language and chaining arithmetic operations without error.""",

    "AMC": """**Dataset: AMC** — Problems from American Mathematics Competitions (AMC 8, 10, 12). These are multiple-choice competition problems covering a wide range of mathematical topics including algebra, geometry, number theory, and combinatorics. Problems vary from straightforward to highly challenging, often requiring clever problem-solving strategies rather than brute-force computation.""",

    "HMMT": """**Dataset: HMMT** — Problems from the Harvard-MIT Mathematics Tournament, one of the most prestigious undergraduate math competitions. Problems are significantly harder than AMC, requiring deep mathematical insight, creative approaches, and strong proof-writing skills across algebra, combinatorics, geometry, and number theory.""",

    "HumanEval": """**Dataset: HumanEval** — A benchmark of hand-crafted Python programming challenges designed to evaluate code generation. Each problem includes a function signature, docstring with description and examples, and hidden test cases. Solutions must be syntactically correct Python functions that pass all test cases.""",

    "MBPP": """**Dataset: MBPP** — Mostly Basic Python Problems, a collection of crowd-sourced Python programming tasks. Problems range from simple string/list operations to moderate algorithmic challenges. Each includes a task description, function signature, and test cases.""",

    "HotpotQA": """**Dataset: HotpotQA** — A multi-hop question answering dataset requiring reasoning over multiple supporting documents. Questions are designed so that the answer cannot be found in a single paragraph — the model must combine information from different sources through multi-step reasoning.""",

    "DROP": """**Dataset: DROP** — Discrete Reasoning Over Paragraphs, a reading comprehension benchmark requiring numerical reasoning. Questions involve counting, sorting, arithmetic over numbers extracted from paragraphs, and compositional operations. Answers can be numbers, dates, or spans from the text.""",

    "LiveCodeBench": """**Dataset: LiveCodeBench** — A continuously updated benchmark of competitive programming problems from platforms like LeetCode, Codeforces, and AtCoder. Problems require algorithmic thinking, data structure knowledge, and efficient implementation.""",

    "Codeforces": """**Dataset: Codeforces** — Competitive programming problems from the Codeforces platform. Problems require advanced algorithmic knowledge including dynamic programming, graph algorithms, number theory, and computational geometry. Solutions must be efficient enough to pass within time limits.""",

    "MMLU_Pro": """**Dataset: MMLU_Pro** — A challenging multiple-choice benchmark covering 14 domains including math, physics, chemistry, biology, computer science, engineering, law, psychology, and more. Each question has up to 10 options (A-J). Problems require deep domain knowledge and multi-step reasoning. The workflow must output a single option letter as the final answer.""",
}


def get_dataset_description(dataset: str) -> str:
    """Get dataset description, returning empty string if not found."""
    return DATASET_DESCRIPTIONS.get(dataset, "")

# =============================================================================
# SECTION 2: Operator Usage — Available operators (placed after goal)
# The OPERATOR_USAGE dict and generate_operator_usage() are unchanged.
# =============================================================================

OPERATOR_USAGE = {
    "Custom": {
        "init": "self.custom = operator.Custom(self.llm)",
        "call": "solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)",
        "format": "custom(input: str, instruction: str) -> str",
        "note": "Returns the response string directly. The input and instruction are directly concatenated (instruction+input). Placeholders are not supported. You MUST define the prompt variable (e.g., `SOLVE_PROMPT = \"...\"`) in the `<prompt>` section for every `prompt_custom.XXXX` you use in graph.",
    },
    "ScEnsemble": {
        "init": "self.sc_ensemble = operator.ScEnsemble(self.llm)",
        "call": "best = await self.sc_ensemble(solutions=[sol1, sol2, sol3], problem=problem)",
        "format": "sc_ensemble(solutions: List[str], problem: str) -> str",
        "note": "Returns the best solution string directly. Selects the best from multiple candidates via voting.",
    },
    "AnswerGenerate": {
        "init": "self.answer_generate = operator.AnswerGenerate(self.llm)",
        "call": "answer = await self.answer_generate(input=problem)",
        "format": "answer_generate(input: str) -> str",
        "note": "Returns the answer string directly. Generates step-by-step reasoning internally.",
    },
    "CustomCodeGenerate": {
        "init": "self.custom_code_generate = operator.CustomCodeGenerate(self.llm)",
        "call": 'code = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="Analyze step by step and generate code.")',
        "format": "custom_code_generate(problem: str, entry_point: str, instruction: str) -> str",
        "note": "Returns the code string directly. The instruction should encourage step-by-step thinking.",
    },
    "Test": {
        "init": "self.test = operator.Test(self.llm, dataset=self.dataset)",
        "call": "test_result = await self.test(problem=problem, solution=solution, entry_point=entry_point)",
        "format": "test(problem: str, solution: str, entry_point: str) -> dict with keys 'result' (bool) and 'solution' (str)",
        "note": "Modify the input solution solution with public test cases. Always return test_result['solution'] (the improved solution). Use test_result['result'] (bool) for conditional retry if needed.",
    },
    "Programmer": {
        "init": "self.programmer = operator.Programmer(self.llm)",
        "call": 'result = await self.programmer(problem=problem, analysis="Step by step analysis")',
        "format": "programmer(problem: str, analysis: str = 'None') -> str",
        "note": "Writes and executes Python code, returns a string with the execution result.",
    },
    "QANumerical": {
        "init": "self.qa_numerical = operator.QANumerical(self.llm)",
        "call": 'result = await self.qa_numerical(problem=problem, analysis="Extract the relevant numbers/entities from the passage, then compute the answer")',
        "format": "qa_numerical(problem: str, analysis: str = 'None') -> str",
        "note": "Like Programmer but tailored for DROP-style passage+question inputs; writes and executes Python (the code may hard-code values read from the passage) and returns a string with the execution result.",
    },
}


def generate_operator_usage(operators):
    """Dynamically generate operator initialization and usage instructions based on the task's operator list."""
    usage = "\n## Operator Initialization and Usage (MUST follow exactly)\n\n"
    usage += "All operators MUST be initialized with `self.llm` in `__init__`. Example:\n```\n"
    for op in operators:
        if op in OPERATOR_USAGE:
            usage += f"        {OPERATOR_USAGE[op]['init']}\n"
    usage += "```\n\n"
    usage += "Operator call formats (MUST follow exactly):\n"
    for i, op in enumerate(operators):
        if op in OPERATOR_USAGE:
            info = OPERATOR_USAGE[op]
            usage += f"\n{i+1}. **{op}**\n"
            usage += f"   Format: `{info['format']}`\n"
            usage += f"   Example: `{info['call']}`\n"
            usage += f"   Note: {info['note']}\n"
    return usage


# =============================================================================
# SECTION 3: Context — Previous best graph, prompt, experience, logs
# =============================================================================

WORKFLOW_CONTEXT = """
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph.

<sample>
    <experience>{experience}</experience>
    <modification>(such as: add / delete / modify / ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>(only prompt_custom)
</sample>

Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}
"""

# =============================================================================
# SECTION 3.5: Diversity Context — Injected in Phase 2 only
# =============================================================================

WORKFLOW_DIVERSITY_CONTEXT = """
**SCORING**: Your workflow is evaluated by WEIGHTED accuracy — solving under-covered queries counts much more than solving well-covered ones. The score shown above reflects this weighted metric, not raw accuracy.

**Query weight distribution**:
- {n_weight_high} queries covered by 0 workflows (weight=1.00, highest impact on your score)
- {n_weight_mid} queries covered by 1-3 workflows (weight=0.25-0.50)
- {n_weight_low} queries covered by 4+ workflows (weight<0.20, low impact)

**High-weight examples** (solving these types has the most impact on your score):
{high_weight_examples}

**Existing strengths** (avoid duplicating):
{existing_strengths}

**The workflow is pure Python — use the full language freely.** Go beyond simple operator chains. Consider these design patterns:
- **Majority Vote**: Loop N calls + `collections.Counter` on extracted answers (more robust than ScEnsemble)
- **Dynamic Prompts**: Use `.format()` to inject prior outputs into prompts (define templates with `{{placeholders}}` in `<prompt>`)
- **Multi-Agent Debate**: Proposer → Critic → Judge, each seeing prior output
- **Conditional Routing**: Classify problem type → if/else → specialist prompt
- **Helper Methods**: `@staticmethod` for regex answer extraction, post-processing
- **Multi-Paradigm**: Combine forward reasoning + backward reasoning + code execution, then ensemble

You can import `re`, `collections`, `json`, `math`, etc. Think structurally different, not just prompt tweaks.

**Relaxed rules for diversity mode** (override the rules below where they conflict):
- Prompts MAY use `{{placeholders}}` filled via `.format()` in graph code (for dynamic prompt composition)
- Up to **10 lines** of code may be changed (structural changes need more room)
- You MAY define `@staticmethod` or helper methods on the Workflow class
"""

# =============================================================================
# SECTION 4: Rules & Constraints — All critical rules placed LAST for emphasis
# =============================================================================

WORKFLOW_RULES = """
Now, provide your optimization. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.

**OPERATOR RULES (MUST follow strictly):**
1. Do NOT create new operators. Only use the operators listed in the Operator Usage section above.
2. All operators MUST be initialized in `__init__` with `self.llm` as the first argument. Example: `self.custom = operator.Custom(self.llm)`. Never call `operator.XXX()` without `self.llm`.
3. Follow the exact call format for each operator as specified in the Operator Usage section.
4. Loop iteration MUST <= 5 to avoid timeout.
5. The graph complexity should not exceed 8 nodes.

**`__call__` SIGNATURE AND RETURN FORMAT (MUST follow exactly):**
{call_signature}

**PROMPT GENERATION RULES (CRITICAL):**
- Only generate prompts used by Custom operator via `prompt_custom.YOUR_PROMPT_NAME` in the graph.
- You MUST define every `prompt_custom.XXXX` variable you reference in graph in the `<prompt>` section. For example, if you use `prompt_custom.SOLVE_PROMPT` in graph, you must output `SOLVE_PROMPT = "..."` in `<prompt>`.
- Other operators (ScEnsemble, Programmer, AnswerGenerate, CustomCodeGenerate, Test) have built-in prompts — do NOT generate prompts for them.
- **The generated prompt MUST be plain text. Do NOT use placeholders like {{{{problem}}}}, {{{{entry_point}}}}, or {{{{input}}}}. The prompt will be directly concatenated with the input, NOT formatted with .format().**
- Remove any unused prompts from prompt_custom.

**MODIFICATION RULES:**
- **Only ONE detail point can be modified at a time**, and no more than 5 lines of code (including graph and prompt) may be changed per modification — extensive modifications are strictly prohibited to maintain project focus!
- When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
- **Under no circumstances should Graph output None for any field.**
- Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
- It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
"""

# Per-task-type __call__ signatures and return format constraints
CALL_SIGNATURES = {
    "math": """The `__call__` method signature MUST be:
```python
async def __call__(self, problem: str):
    # ... your workflow logic ...
    return solution, self.llm.usage_tracker.get_summary()["total_cost"]
```
- Input: `problem` (str) — the math problem text.
- Return: a **2-tuple** of `(solution_string, cost_float)`. Do NOT return only the solution without cost. Do NOT return a 3-tuple or any other format.""",

    "code": """The `__call__` method signature MUST be:
```python
async def __call__(self, problem: str, entry_point: str):
    # ... your workflow logic ...
    return solution, self.llm.usage_tracker.get_summary()["total_cost"]
```
- Input: `problem` (str) — the code problem description; `entry_point` (str) — the function name to implement.
- Return: a **2-tuple** of `(solution_string, cost_float)`. Do NOT return only the solution without cost. Do NOT return a 3-tuple or any other format.""",

    "qa": """The `__call__` method signature MUST be:
```python
async def __call__(self, problem: str):
    # ... your workflow logic ...
    return solution, self.llm.usage_tracker.get_summary()["total_cost"]
```
- Input: `problem` (str) — the question text.
- Return: a **2-tuple** of `(solution_string, cost_float)`. Do NOT return only the solution without cost. Do NOT return a 3-tuple or any other format.""",
}


# =============================================================================
# Error correction prompt for auto-fixing failed graphs
# =============================================================================

WORKFLOW_FIX_PROMPT = """The following workflow graph failed with an error. Please fix the code.

**Error message:**
```
{error_msg}
```

**Current graph code (the class body inside graph.py):**
```python
{graph_code}
```

**Current prompt code (prompt.py):**
```python
{prompt_code}
```

{operator_usage}

**`__call__` SIGNATURE AND RETURN FORMAT (MUST follow exactly):**
{call_signature}

**RULES:**
1. Do NOT create new operators. Only use the operators listed above.
2. All operators MUST be initialized in `__init__` with `self.llm` as the first argument.
3. Follow the exact call format for each operator.
4. Fix the error while preserving the original intent of the graph as much as possible.
5. If the error is caused by using a non-existent operator, replace it with one of the available operators.

Output your fix in the same XML format:
- <modification>Describe what you fixed (include original modification + fix)</modification>
- <graph>The fixed class body (everything starting from `class Workflow:`)</graph>
- <prompt>The prompt code (only prompt_custom variables used by Custom operator)</prompt>

**Under no circumstances should any field be empty.**
"""

# =============================================================================
# Template for writing graph.py files
# =============================================================================

WORKFLOW_TEMPLATE = """from typing import Literal
import {optimized_path}.{dataset}.workflows.template.operator as operator
import {optimized_path}.{dataset}.workflows.round_{round}.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

{graph}
"""
