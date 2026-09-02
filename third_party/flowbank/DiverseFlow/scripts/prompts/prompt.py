# -*- coding: utf-8 -*-
# @Desc    : prompts of operators

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

FORMAT_PROMPT = """
For the question described as {problem_description},
please extract a short and concise answer contains only one word/few words from the following solution: {solution}.
Make sure there are no additional comments or explanations in your response.
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

QA_NUMERICAL_PROMPT = """
You are given a reading comprehension question that may benefit from programmatic computation.
Your task is to write Python code that helps answer the question by extracting relevant
information from the passage and performing any necessary operations (counting, arithmetic,
sorting, set operations, date differences, comparisons, etc.).

Passage and question:
{problem}

Other analysis (may or may not be provided):
{analysis}
{feedback}

Your code should:
1. Define a function named `solve` that takes NO arguments and returns the final answer.
2. Hard-code the relevant values extracted from the passage as variables inside the function.
3. Use Python to perform the operations needed (e.g., sum, count, max, min, sort, date diff, set ops).
4. The return value should be in the exact format the question expects:
   - A number (int or float) if the question asks for a count, amount, year, percentage, difference, etc.
   - A string (text span) if the question asks "who", "which", "what" — return the extracted span.
   - A date string (e.g., "December 15, 1945") if the question asks for a date.
   - If the question is essentially not computational (e.g., "what is the main idea"), you may
     still return a string answer directly from the passage — treat the code as a bookkeeping tool.

Important constraints:
- Do NOT use external libraries beyond the standard library. Prefer pure Python.
- Do NOT attempt to re-read the passage at runtime — extract values at code-writing time.
- Keep the code short and focused; avoid unnecessary complexity.
- The return value will be consumed by a downstream LLM to produce the final answer string,
  so clear numeric/textual output is best.

Example (numerical):
    def solve():
        # Extracted from passage: GDP in 2020 = 500, in 2025 = 750
        gdp_2020 = 500
        gdp_2025 = 750
        growth_pct = (gdp_2025 - gdp_2020) / gdp_2020 * 100
        return growth_pct  # 50.0

Example (span):
    def solve():
        # Extracted from passage: Alice scored 20, Bob scored 25, Carol scored 15
        scores = {{"Alice": 20, "Bob": 25, "Carol": 15}}
        return max(scores, key=scores.get)  # "Bob"

Now write the code.
"""


PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
"""

MD_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the solution that is more capable of solving the problem compared to other solutions, as this is crucial for problem-solving.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

REVIEW_PROMPT = """
Given a problem and a thoughtful solution, your task is to using critical thinking (questioning) to review the solution's correctness and provide a review result in boolean format.

problem: {problem}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, please return False and give a feedback for the error. Otherwise, please return True and give a explanation for the correctness.
"""

REVISE_PROMPT = """
Given a problem and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question and ensure the final code solution is wrapped with ```python```.

problem: {problem}
solution: {solution}
feedback: {feedback}

Ensure the output code is self-contained, and without any additional text or test cases.
"""
