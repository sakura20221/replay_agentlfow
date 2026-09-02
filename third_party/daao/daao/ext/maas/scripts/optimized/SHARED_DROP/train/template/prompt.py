IMPROVE_CODE_PROMPT = """
The previous solution failed some test cases. Please analyze the problem carefully and provide an improved solution that addresses all edge cases and requirements. Ensure your code is efficient and follows best practices.
"""

GENERATE_SOLUTION_PROMPT = """
Please solve the given mathematical problem step by step. Follow these guidelines:

1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
"""

REFINE_ANSWER_PROMPT = """
Given the mathematical problem and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the problem.
2. Explain the approach and any formulas or concepts used.
3. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer, enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your response should be comprehensive, mathematically rigorous, and easy to follow.
"""

SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
"""

MATH_SOLUTION_PROMPT = """
Please solve the given mathematical problem step by step. Follow these guidelines:

1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
"""

MATH_SOLVE_PROMPT = """
You are a highly skilled mathematician tasked with solving a math problem. Follow these steps carefully:

1. Read and understand the problem thoroughly.
2. Identify all key information, variables, and relationships.
3. Determine the appropriate mathematical concepts, formulas, or equations to use.
4. Solve the problem step-by-step, showing all your work clearly.
5. Double-check your calculations and reasoning at each step.
6. Provide a clear and concise final answer.
7. Verify your solution by plugging it back into the original problem or using an alternative method if possible.

Format your answer as follows:
- Use LaTeX notation for mathematical expressions where appropriate.
- Show each step of your solution process clearly.
- Clearly state your final answer at the end of your solution.
- Express numerical answers as precise values (avoid rounding unless specified).
- Ensure that your final answer is a single numerical value without any units or additional text.
- Do not include any explanatory text with your final answer, just the number itself.

For example, if the final answer is 42.5, your response should end with just:
42.5

Here's the problem to solve:

"""

DETAILED_SOLUTION_PROMPT = """
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
"""


# --- shared-layer shim (agent_wf_v2) --- prompt task adaptation v4
# See shims/maas_family/install.py for why this is appended rather than edited in.
# Every override below is a raw string: the mis-escaped LaTeX repaired just above
# is exactly the bug that non-raw prompt literals cause.
import os as _shim_ap_os
import sys as _shim_ap_sys


def _shim_ap_task() -> str:
    """Which dataset this process is running.

    Read from the environment first (sweep.py exports SHIM_DATASET per job) and
    from argv second, so a manual `optimize.py --dataset SHARED_DROP` outside the
    sweep is adapted too instead of silently falling back to the maths wording.
    MMLU-Pro is tested before MATH because "SHARED_MMLUPRO" contains neither, and
    DROP before MBPP only for readability -- the markers are disjoint.
    """
    marker = (_shim_ap_os.getenv("SHIM_DATASET", "") or " ".join(_shim_ap_sys.argv)).upper()
    for _needle, _name in (("MMLUPRO", "mmlu_pro"), ("MMLU_PRO", "mmlu_pro"),
                           ("DROP", "drop"), ("MBPP", "mbpp"), ("AMC", "amc")):
        if _needle in marker:
            return _name
    return "math"


_SHIM_AP_TASK = _shim_ap_task()

# (1) Repair, unconditional. Only these three tokens are rewritten, not every
# control character: 0x09 is also an ordinary tab and 0x0a an ordinary newline in
# prompt text, so a blanket rule would corrupt legitimate whitespace. The
# installer's --check asserts that nothing else remains.
for _shim_ap_name in [_n for _n in list(globals()) if _n.endswith("_PROMPT")]:
    _shim_ap_text = globals()[_shim_ap_name]
    if isinstance(_shim_ap_text, str):
        globals()[_shim_ap_name] = (_shim_ap_text
                                    .replace("\x08oxed", "\\boxed")
                                    .replace("\x0crac", "\\frac")
                                    .replace("\x09imes", "\\times"))
# Deleted, not just left to fall out of use: _shim_ap_text still holds the
# *pre-repair* text of the last constant, and it would sit in the module namespace
# where anything walking globals() -- including this project's own scanners -- reads
# it as a live prompt. Measured: 46 phantom hits per repo before this line existed.
del _shim_ap_name, _shim_ap_text

# (2) Adaptation, per dataset.
if _SHIM_AP_TASK == "drop":
    # Gated on the task alone. Gating it on "GENERATE_SOLUTION_PROMPT" in globals()
    # -- as an earlier revision did -- skipped this whole block inside op_prompt.py,
    # which is where GENERATE_COT_PROMPT and PYTHON_CODE_VERIFIER_PROMPT live: the
    # live transcripts then showed 53% of DAAO's DROP prompts still carrying the
    # GSM8K worked examples and "based on a given mathematical problem". The names
    # this defines in the module that does not use them are unread and harmless;
    # skipping the ones that ARE used was not.
    GENERATE_SOLUTION_PROMPT = r"""
Please answer the given reading comprehension question about the passage step by step. Follow these guidelines:

1. State the question clearly.
2. Outline the approach and identify the parts of the passage that bear on it.
3. Provide the detailed derivation, quoting the figures, dates or names the passage gives.
4. Explain each step of your reasoning.
5. Present the final answer on a last line of the form 'Answer: <answer>', where <answer> is the concise answer (a span, number, date, or list as appropriate).
6. Quote the passage exactly when the question asks for a textual span.

Your solution should be thorough, faithful to the passage, and easy to understand.
"""
    MATH_SOLUTION_PROMPT = GENERATE_SOLUTION_PROMPT
    REFINE_ANSWER_PROMPT = r"""
Given the reading comprehension question, its passage and the output from the code execution, please provide a well-formatted and detailed answer. Follow these guidelines:

1. Begin with a clear statement of the question.
2. Explain the approach and which parts of the passage were used.
3. Show the step-by-step derivation, quoting the figures the passage gives.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer on a last line of the form 'Answer: <answer>', where <answer> is the concise answer (a span, number, date, or list as appropriate).
6. Quote the passage exactly when the question asks for a textual span.

Your response should be comprehensive, faithful to the passage, and easy to follow.
"""
    SOLUTION_PROMPT = r"""
Provide a comprehensive, step-by-step answer to the given reading comprehension question. Your response should include:

1. A clear restatement of the question.
2. An explanation of what the passage says on the point at issue.
3. A detailed, logical progression of steps leading to the answer.
4. Clear explanations for each step, including the reasoning behind it.
5. All figures and dates quoted exactly as the passage gives them.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked on a last line of the form 'Answer: <answer>', where <answer> is the concise answer (a span, number, date, or list as appropriate).
8. A brief explanation of the significance of the result, if relevant.

Ensure your answer is rigorous, easy to follow, and faithful to the passage.
"""
    DETAILED_SOLUTION_PROMPT = SOLUTION_PROMPT
    MATH_SOLVE_PROMPT = r"""
You are a highly skilled reading comprehension analyst tasked with answering a question about a passage. Follow these steps carefully:

1. Read and understand the passage and the question thoroughly.
2. Identify all key figures, dates, names and relationships the passage states.
3. Determine what the question asks for: a span to quote, a count, or an arithmetic result over stated figures.
4. Work the answer out step-by-step, showing all your work clearly.
5. Double-check your reading and any arithmetic at each step.
6. Provide a clear and concise final answer.
7. Verify your answer against the passage and any arithmetic against the stated figures.

Format your answer as follows:
- Quote textual spans exactly as the passage writes them.
- Show each step of your reasoning clearly.
- Clearly state your final answer at the end of your solution.
- Express numerical answers as precise values (avoid rounding unless specified).
- Ensure that your final answer is concise: a span, number, date, or list as appropriate.
- Do not include any explanatory text on the final answer line.

For example, if the final answer is 57, your response should end with just:
Answer: 57

Here's the question to answer:

"""
    if "PYTHON_CODE_VERIFIER_PROMPT" in globals():
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "based on a given mathematical problem and output the answer",
            "based on a given reading comprehension question and its passage, and output the answer")
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "Implement the calculation steps described in the problem.",
            "Implement the counting or arithmetic steps over the figures the passage states.")
    if "GENERATE_COT_PROMPT" in globals():
        GENERATE_COT_PROMPT = r"""
Reading Comprehension Reasoning Instruction
{instruction}

Current Problem: {input}

Demonstration Examples (DROP style):

1. Passage: "The city council approved 14 permits in March and 9 permits in April."
   Question: How many permits were approved in total over the two months?
   Analysis:
   Locate both figures in the passage: 14 in March, 9 in April
   The question asks for the total, so add them: 14 + 9 = 23
   Answer: 23

2. Passage: "Ferrer won the 2007 final, and Nadal won in 2008 and 2009."
   Question: Who won the final the year before Nadal's first title?
   Analysis:
   Nadal's first title is 2008, so the year before is 2007
   The passage names the 2007 winner: Ferrer
   Copy the span exactly as written
   Answer: Ferrer

Solution Protocol:
1. Parse the passage and the question carefully
2. Identify the spans and figures that bear on the question
3. Perform the stepwise derivation over those figures
4. Verify intermediate results against the passage
5. Present the final answer on a last line of the form 'Answer: <answer>'

Step-by-Step Analysis:
"""

if _SHIM_AP_TASK == "mmlu_pro":
    # Same reasoning as the DROP block above.
    GENERATE_SOLUTION_PROMPT = r"""
Please solve the given multiple-choice question step by step. Follow these guidelines:

1. State the question clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide the detailed reasoning, using LaTeX notation for any mathematical expressions.
4. Explain each step of your reasoning, including why the other options are wrong.
5. Present the final answer on a last line of the form 'Answer: (X)', where X is a single option letter.
6. Choose exactly one of the listed options.

Your solution should be thorough, well reasoned, and easy to understand.
"""
    MATH_SOLUTION_PROMPT = GENERATE_SOLUTION_PROMPT
    REFINE_ANSWER_PROMPT = r"""
Given the multiple-choice question and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the question.
2. Explain the approach and any formulas or concepts used.
3. Show the step-by-step reasoning, using LaTeX notation for any mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer on a last line of the form 'Answer: (X)', where X is a single option letter.
6. Choose exactly one of the listed options.

Your response should be comprehensive, rigorous, and easy to follow.
"""
    SOLUTION_PROMPT = r"""
Provide a comprehensive, step-by-step solution to the given multiple-choice question. Your response should include:

1. A clear restatement of the question.
2. An explanation of the concepts and principles involved.
3. A detailed, logical progression of steps leading to the answer.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked on a last line of the form 'Answer: (X)', where X is a single option letter.
8. A brief explanation of why the remaining options are wrong, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
"""
    DETAILED_SOLUTION_PROMPT = SOLUTION_PROMPT
    MATH_SOLVE_PROMPT = r"""
You are a highly skilled expert tasked with answering a multiple-choice question. Follow these steps carefully:

1. Read and understand the question and every option thoroughly.
2. Identify all key information, variables, and relationships.
3. Determine the appropriate concepts, formulas, or equations to use.
4. Work the question out step-by-step, showing all your work clearly.
5. Double-check your reasoning and calculations at each step.
6. Provide a clear and concise final answer.
7. Verify your answer by checking the remaining options can be ruled out.

Format your answer as follows:
- Use LaTeX notation for mathematical expressions where appropriate.
- Show each step of your solution process clearly.
- Clearly state your final answer at the end of your solution.
- Give exactly one option letter, chosen from the options listed with the question.
- Ensure that your final answer is a single option letter without any units or additional text.
- Do not include any explanatory text with your final answer, just the letter itself.

For example, if the correct option is C, your response should end with just:
Answer: (C)

Here's the question to answer:

"""
    if "PYTHON_CODE_VERIFIER_PROMPT" in globals():
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "based on a given mathematical problem and output the answer",
            "based on a given multiple-choice question and output the answer")
        PYTHON_CODE_VERIFIER_PROMPT = PYTHON_CODE_VERIFIER_PROMPT.replace(
            "Implement the calculation steps described in the problem.",
            "Implement the calculation steps needed to decide between the options.")
    if "GENERATE_COT_PROMPT" in globals():
        GENERATE_COT_PROMPT = r"""
Multiple-Choice Reasoning Instruction
{instruction}

Current Problem: {input}

Demonstration Examples (MMLU-Pro style):

1. Question: Which unit measures electric current?
   Options: (A) volt (B) ampere (C) ohm (D) watt
   Analysis:
   Current is charge per unit time, whose SI unit is the ampere
   Volt measures potential, ohm resistance, watt power, so all three are ruled out
   Answer: (B)

2. Question: A body accelerates from rest at 3 m/s^2 for 4 s. What is its final speed?
   Options: (A) 7 m/s (B) 10 m/s (C) 12 m/s (D) 24 m/s
   Analysis:
   From rest, $v = at$
   Substitute values: $v = 3 \times 4 = 12$ m/s
   Match against the options: 12 m/s is option C
   Answer: (C)

Solution Protocol:
1. Parse the question and every option carefully
2. Identify the relevant concepts
3. Perform the stepwise derivation
4. Rule out the remaining options
5. Present the final answer on a last line of the form 'Answer: (X)'

Step-by-Step Analysis:
"""

if _SHIM_AP_TASK == "mbpp":
    # The MBPP cell imports the authors' HumanEval template, so both constants
    # name the wrong benchmark to the model. Nothing else in this template is
    # HumanEval-specific: the operator-level prompts are generic.
    for _shim_ap_name in ("IMPROVE_CODE_PROMPT", "GENERATE_CODE_PROMPT"):
        if _shim_ap_name in globals() and isinstance(globals()[_shim_ap_name], str):
            globals()[_shim_ap_name] = (globals()[_shim_ap_name]
                                        .replace("HumanEval benchmark", "MBPP benchmark")
                                        .replace("HumanEval dataset", "MBPP dataset"))
