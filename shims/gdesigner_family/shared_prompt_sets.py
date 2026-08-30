"""Register a prompt-set domain for each shared dataset.

The repo ships three domains -- gsm8k, humaneval, mmlu -- and the bake-off needs
five. Each new domain subclasses whichever shipped set matches its task type.
Prompting is part of the method under test, so everything that is strategy --
roles, role descriptions, the connection graph, reasoning instructions, word
limits, decision structure -- is inherited rather than replaced.

MATH, AMC and MBPP override nothing at all: their inherited constraints are role
descriptions and a generic "function signature and docstring" instruction, which
describe the task correctly.

DROP and MMLU-Pro do override, in one respect only: the `mmlu` domain states an
option count, and that statement is false for both of them. See the comment above
those two classes for what was measured before changing it, and for the exact
list of what was left alone.

DROP is also the one genuine domain gap: the repo has no reading-comprehension
set. It inherits `mmlu`, whose "Knowlegable Expert" roles are question-answering
oriented, rather than `gsm8k`, whose roles are mathematical analysts -- feeding a
passage-comprehension task to a maths persona would handicap the method for a
reason unrelated to topology design. This substitution is declared, not hidden.
"""

from __future__ import annotations

from GDesigner.prompt.gsm8k_prompt_set import GSM8KPromptSet
from GDesigner.prompt.humaneval_prompt_set import HumanEvalPromptSet
from GDesigner.prompt.mmlu_prompt_set import ROLE_DESCRIPTION, MMLUPromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@PromptSetRegistry.register("math")
class MathSharedPromptSet(GSM8KPromptSet):
    """MATH-500: maths word problems, same shape as GSM8K."""


@PromptSetRegistry.register("amc")
class AMCSharedPromptSet(GSM8KPromptSet):
    """AMC: competition maths, same shape as GSM8K."""


@PromptSetRegistry.register("mbpp")
class MBPPSharedPromptSet(HumanEvalPromptSet):
    """MBPP: Python function synthesis, same shape as HumanEval."""


# The `mmlu` domain hardcodes four options in four separate places -- get_constraint,
# get_analyze_constraint, get_decision_constraint and get_adversarial_answer_prompt --
# because MMLU is a four-way task. Inheriting it unchanged told the model, on every
# call, that "I will also give you 4 answers enumerated as A, B, C and D. Only one
# answer out of the offered 4 is correct", on a ten-way dataset and on a dataset with
# no options at all.
#
# Measured before deciding what to do about it (letter_space.py, over the recorded
# transcripts): 5,488 MMLU-Pro and 5,024 DROP calls carried that claim, and it did
# NOT bind -- 52.5% of the final answers were E through J against 53.9% of gold
# answers there, and DROP decisions came back as real spans ("Answer: Corey Dillon"),
# not letters. The model resolves the contradiction in favour of the task text, which
# shared/bench.py appends. So the expected score effect of this fix is small.
#
# It is still made, for two reasons: a prompt that asserts something false about the
# dataset is not a protocol anyone should report, and "the model happened to ignore
# it" is a property of this executor, not a guarantee. Only the four option-counting
# statements move. Role lists, role descriptions, connection graph, word limits, the
# first-line rule, the refer-to-other-agents clause and the decision role are
# inherited untouched -- including get_analyze_constraint's operator precedence,
# which is reproduced exactly rather than corrected.


@PromptSetRegistry.register("drop")
class DROPSharedPromptSet(MMLUPromptSet):
    """DROP: passage comprehension; closest shipped domain is the QA one.

    Its answer is a span from the passage, so the letter-picking constraints are
    restated in terms of a span. postprocess_answer is overridden for the same
    reason: the inherited one keeps only the first character, which is right for a
    letter and destroys a span.
    """

    @staticmethod
    def get_constraint():
        return """
            I will ask you a question about a passage.
            I will also give you the passage the question refers to.
            Exactly one span of the passage answers the question.
            You must find the span that answers the question.
            Your response must end with a line of the form 'Answer: <answer>',
            where <answer> is the shortest exact span from the passage.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The last line of your reply must contain only the answer line(for example : Answer: 57)
        """

    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""+ """
I will ask you a question about a passage, and give you the passage it refers to.
Exactly one span of the passage answers the question.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The last line of your reply must contain only the answer line(for example : Answer: 57)
"""

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question about a passage.
        I will also give you the passage the question refers to.
        Exactly one span of the passage answers the question.
        You must find the span that answers the question.
        Your response must end with a line of the form 'Answer: <answer>',
        where <answer> is the shortest exact span from the passage.
        I will give you some other people's answers and analysis.
        Your reply must only contain that answer line and cannot have any other characters.
        For example, your reply can be Answer: 57.
        """

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The last line of your reply must contain only the answer line(for example : Answer: 57)
                """

    def postprocess_answer(self, answer):
        """Keep the whole reply.

        The inherited implementation ends with `answer = answer[0]`, taking the
        first character -- correct when the answer is a letter, and destructive on a
        span, where "Corey Dillon" would become "C". The list-flattening and
        type-checking above it are kept.
        """
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        return answer


@PromptSetRegistry.register("mmlu_pro")
class MMLUProSharedPromptSet(MMLUPromptSet):
    """MMLU-Pro: multiple choice with up to ten options instead of four.

    The option count is left unstated rather than changed from four to ten: items
    in this split carry differing numbers of options, so naming any single count
    would be false on some of them.
    """

    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you the answer options, enumerated as A, B, C and so on.
            Only one answer out of the offered options is correct.
            You must choose the correct answer to the question.
            Your response must be one of the option letters offered with the question,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or J)
        """

    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""+ """
I will ask you a question and the answer options, enumerated as A, B, C and so on.
Only one answer out of the offered options is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or J)
"""

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you the answer options, enumerated as A, B, C and so on.
        Only one answer out of the offered options is correct.
        You must choose the correct answer to the question.
        Your response must be one of the option letters offered with the question,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or J)
                """


SHARED_DOMAINS = ("math", "amc", "mbpp", "drop", "mmlu_pro")
