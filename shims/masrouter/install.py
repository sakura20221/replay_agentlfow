#!/usr/bin/env python3
"""Install the shared-layer shim into MasRouter.

`Experiments/run_shared.py` is derived from the repo's own `run_math.py` by
targeted substitution, not rewritten: that file holds the router's training
objective (a task-classification cross-entropy, a VAE term, and
`answer_loss = -log_prob * utility` with `utility = is_solved - cost * rate`),
and reimplementing it would change the method under measurement.

Five things move: the dataset loader, the task label (MasRouter classifies each
query into Math / Commonsense / Code and trains that classifier, so a hardcoded
label would teach it that every query is maths), the answers carried through the
batch, and the two grading sites -- one in the training loop and one in the test
loop. Missing the second would leave the reported numbers graded by MATH's
checker regardless of dataset, which is exactly the kind of failure that looks
like a working run.

    python shims/masrouter/install.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPO = ROOT / "third_party" / "masrouter"
DATA = ROOT / "shared" / "data"

MARKER = "# --- derived from run_math.py by the shared-layer shim ---"
DATASETS = ("math", "amc", "mbpp", "drop", "mmlu_pro")

problems: list[str] = []


def report(ok: bool, message: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {message}")
    if not ok:
        problems.append(message)


def substitute(text: str, pattern: str, replacement: str, label: str, expected: int = 1) -> str:
    updated, count = re.subn(pattern, replacement, text)
    if count != expected:
        report(False, f"anchor {label}: matched {count}x, expected {expected}x")
        return text
    return updated


def derive_runner() -> None:
    source = REPO / "Experiments" / "run_math.py"
    if not source.exists():
        report(False, "run_math.py missing")
        return
    text = source.read_text(encoding="utf-8")

    # 1. imports.
    text = substitute(
        text,
        r"from Datasets\.math_dataset import load_math_dataset\s*,\s*MATH_is_correct\s*,\s*MATH_get_predict",
        "from Datasets.shared_dataset import load_shared_dataset, shared_score, shared_task_labels",
        "import line",
    )

    # 2. a flag naming which shared benchmark to run.
    text = substitute(
        text,
        r'(parser\.add_argument\("--result_file", type=str, default=None\))',
        r'\1\n    parser.add_argument("--shared_dataset", type=str, required=True,\n'
        f'                        choices={list(DATASETS)!r},\n'
        '                        help="which shared benchmark to run")',
        "--shared_dataset flag",
    )

    # 3. dataset loading.
    text = substitute(
        text,
        r'train_dataset = load_math_dataset\("Datasets/MATH",split="train"\)\s*\n\s*'
        r'test_dataset = load_math_dataset\("Datasets/MATH",split="test"\)',
        'train_dataset = load_shared_dataset(args.shared_dataset, split="train")\n'
        '    test_dataset = load_shared_dataset(args.shared_dataset, split="test")',
        "dataset loading",
    )

    # 4. answers carry the full record so grading can reach the test harness /
    #    answer spans, and the task label follows the dataset instead of being
    #    pinned to Math. Both appear twice: training loop and test loop.
    text = substitute(text, r"answers = \[item\['solution'\] for item in current_batch\]",
                      "answers = [item['row'] for item in current_batch]",
                      "answers list", expected=2)
    text = substitute(text, r"task_labels = \[0 for _ in current_batch\]",
                      "task_labels = shared_task_labels(args.shared_dataset, current_batch)",
                      "task labels", expected=2)

    # 5. grading, in both loops.
    text = substitute(
        text,
        r"predict_answer = MATH_get_predict\(result\)\n(\s*)is_solved = MATH_is_correct\(predict_answer,true_answer\)",
        r"is_solved = shared_score(args.shared_dataset, true_answer, result)\n\1predict_answer = result",
        "grading", expected=2,
    )

    # 6. log file name follows the dataset.
    text = re.sub(r'log_file = f"MATH_\{current_time\}\.txt"',
                  'log_file = f"{args.shared_dataset}_{current_time}.txt"', text, count=1)

    # 6b. Keep the last partial batch in both training and evaluation. The
    # author's int(N / batch_size) silently drops every remainder; slicing in
    # dataloader already handles a smaller final batch correctly.
    text = substitute(
        text,
        r"num_batches = int\(len\((train_dataset|test_dataset)\)/args\.batch_size\)",
        r"num_batches = (len(\1) + args.batch_size - 1) // args.batch_size",
        "complete batch coverage", expected=2,
    )

    # 7. the final decision node's prompt follows the dataset.
    #
    # This is the graded call: FinalRefer produces the answer that gets scored. The
    # authors switch the file per dataset in their own runners -- run_math.py uses
    # math.json, run_mmlu.py mmlu.json, run_mbpp.py mbpp.json -- and this runner was
    # derived from run_math.py, so it inherited math.json for all five datasets.
    # Every MBPP answer was therefore requested as "the answer is \boxed{...}
    # without any units" instead of a Python code block, and every MMLU-Pro and DROP
    # answer as a boxed maths expression.
    text = substitute(
        text,
        r"parser\.add_argument\('--prompt_file', type=str,\s*default='MAR/Roles/FinalNode/math\.json'\)",
        "parser.add_argument('--prompt_file', type=str, default=None,\n"
        "                        help='defaults to the author file matching --shared_dataset')",
        "prompt_file default",
    )
    text = substitute(
        text,
        r"(args = parser\.parse_args\(\))",
        r"\1\n"
        "    if args.prompt_file is None:\n"
        "        args.prompt_file = SHARED_FINAL_NODE[args.shared_dataset]\n"
        "        print(f'shim: final-node prompt {args.prompt_file}', flush=True)",
        "final-node selection",
    )
    text = substitute(
        text,
        r"(from MAR\.Prompts\.tasks_profile import tasks_profile)",
        r"\1\n"
        "\n# Which final-node prompt each dataset gets. math and amc take the authors'\n"
        "# maths file; mbpp theirs; mmlu_pro and drop take files derived from the\n"
        "# authors' mmlu.json, changing only the statement of how many options exist\n"
        "# and what the answer looks like -- see shims/masrouter/install.py.\n"
        "SHARED_FINAL_NODE = {\n"
        "    'math': 'MAR/Roles/FinalNode/math.json',\n"
        "    'amc': 'MAR/Roles/FinalNode/math.json',\n"
        "    'mbpp': 'MAR/Roles/FinalNode/mbpp.json',\n"
        "    'mmlu_pro': 'MAR/Roles/FinalNode/shared_mmlu_pro.json',\n"
        "    'drop': 'MAR/Roles/FinalNode/shared_drop.json',\n"
        "}",
        "final-node table",
    )

    target = REPO / "Experiments" / "run_shared.py"
    target.write_text(MARKER + "\n" + text, encoding="utf-8")
    report(target.exists(), f"runner -> {target.relative_to(ROOT)}")


# Derived from the authors' MAR/Roles/FinalNode/mmlu.json. The system line, the
# decision framing, the "\boxed{}" output format and the one-sentence last-line
# rule are theirs verbatim; only the option count changes, because MMLU-Pro items
# carry up to ten options and the count is left unstated rather than restated as a
# different fixed number (items differ in how many they offer).
FINAL_NODE_MMLU_PRO = {
    "system": "You are the top decision-maker and are good at analyzing and "
              "summarizing other people's opinions, finding errors and giving final answers.",
    "user": "\nOnly one answer out of the offered options is correct.\n"
            "You must choose the correct answer to the question.\n"
            "Your response must be one of the option letters offered with the question, "
            "corresponding to the correct answer.\n"
            "I will give you some other people's answers and analysis.\n"
            "The last line of the reply should contain only one sentence"
            "(the answer is \\boxed{X}.) and nothing else.\n"
            "For example, The answer is the answer is \\boxed{A}.",
}

# Same provenance, restated for a task whose answer is a span rather than a letter.
# The sweep runs MasRouter on DROP, so this prevents the final node from using a
# maths or multiple-choice answer contract on a passage span.
FINAL_NODE_DROP = {
    "system": "You are the top decision-maker and are good at analyzing and "
              "summarizing other people's opinions, finding errors and giving final answers.",
    "user": "\nThe answer may be a passage span, number, date, or short list.\n"
            "You must give the concise answer supported by the passage.\n"
            "I will give you some other people's answers and analysis.\n"
            "The last line of the reply must contain only one line of the form "
            "Answer: <answer> and nothing else.\n"
            "For example, Answer: 57",
}


# DROP's role pool, derived from the authors' Commonsense pool.
#
# MasRouter classifies every query into one of three task types it ships and then
# loads that type's roles from MAR/Roles/<Type>/. DROP is routed to Commonsense --
# the same type the authors' own run_mmlu.py uses for MMLU -- because that pool is
# the question-answering one. Adding a fourth type was rejected: the task
# classifier scores a query against every type description, so a fourth entry would
# change the routing distribution on MATH and MBPP too, contaminating cells that
# were correct.
#
# Three of the seven role descriptions state an answer shape that is wrong for a
# span task, and only those three phrases are rewritten. The other four roles, the
# MessageAggregation, OutputFormat, PostProcess and PostDescription fields, the
# number of roles and their names are copied unchanged -- so the collaboration
# structure MasRouter learns over is the authors'.
DROP_ROLE_EDITS = [
    ("Please analyze step by step and choose the correct answer.",
     "Please analyze step by step and give the concise answer supported by the passage."),
    ("You will be given a complex math problem .",
     "You will be given a reading comprehension question about a passage ."),
    ("You will be given a complex math problem.",
     "You will be given a reading comprehension question about a passage."),
]

MMLU_PRO_ROLE_EDITS = [
    ("You will be given a complex math problem .",
     "You will be given a multiple-choice question ."),
    ("You will be given a complex math problem.",
     "You will be given a multiple-choice question."),
]

# Appended to encoder_roles: prefer a dataset-specific pool when one exists.
ROLE_VARIANT_MARKER = "# --- shared-layer shim (agent_wf_v2) --- dataset role pool v1"
ROLE_VARIANT_BLOCK = '''
        {marker}
        # A dataset may need one of its role descriptions worded differently
        # without changing which task type it routes to. Roles/<Type>_<dataset>/,
        # when present, replaces <Type>'s pool for that dataset only -- the type
        # list the classifier scores against is untouched, so no other dataset's
        # routing moves. See shims/masrouter/install.py.
        _shim_ds = os.getenv("SHIM_DATASET", "")
        if _shim_ds:
            for _shim_task in list(task_role_database):
                _shim_variant = f"{{_shim_task}}_{{_shim_ds}}"
                if task_role_database.get(_shim_variant):
                    task_role_database[_shim_task] = task_role_database[_shim_variant]
                    task_role_emb[_shim_task] = task_role_emb[_shim_variant]
                    logger.info(f"shim: {{_shim_task}} roles taken from {{_shim_variant}}")
'''


def write_dataset_role_pools() -> None:
    """Derive dataset-worded role pools from the authors' Commonsense pool."""
    base = REPO / "MAR" / "Roles"
    source = base / "Commonsense"
    if not source.exists():
        report(False, "MAR/Roles/Commonsense missing")
        return
    source_paths = sorted(source.glob("*.json"))
    for dataset, edits in (("drop", DROP_ROLE_EDITS),
                           ("mmlu_pro", MMLU_PRO_ROLE_EDITS)):
        target = base / f"Commonsense_{dataset}"
        target.mkdir(parents=True, exist_ok=True)
        edited = copied = 0
        for path in source_paths:
            profile = json.loads(path.read_text(encoding="utf-8"))
            changed = False
            for field in ("Description", "PostDescription"):
                value = profile.get(field)
                if not isinstance(value, str):
                    continue
                for before, after in edits:
                    if before in value:
                        value = value.replace(before, after)
                        changed = True
                profile[field] = value
            (target / path.name).write_text(
                json.dumps(profile, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8")
            edited += 1 if changed else 0
            copied += 1
        report(copied == len(source_paths) and edited > 0,
               f"{dataset} role pool derived "
               f"({copied} roles copied, {edited} reworded)")


def patch_role_variants() -> None:
    """Let encoder_roles pick up the dataset-specific pool."""
    path = REPO / "MAR" / "MasRouter" / "mas_router.py"
    text = path.read_text(encoding="utf-8")
    if ROLE_VARIANT_MARKER in text:
        report(True, "role-variant lookup already installed")
        return
    anchor = "        logger.info('Role embeddings loaded.')"
    if anchor not in text:
        report(False, "encoder_roles anchor not found")
        return
    block = ROLE_VARIANT_BLOCK.format(marker=ROLE_VARIANT_MARKER)
    text = text.replace(anchor, block + anchor, 1)
    path.write_text(text, encoding="utf-8")
    report(ROLE_VARIANT_MARKER in path.read_text(encoding="utf-8"),
           "role-variant lookup installed in encoder_roles")



REGISTRY_MARKER = "# --- shared-layer shim (agent_wf_v2) --- dataset role profile v1"

# Inserted at the top of RoleRegistry.get_role_profile. Written as a plain string
# and joined with the marker by concatenation, NOT by str.format: the block
# contains f-string braces of the target's own ({self.domain}, {self.role}), and
# .format() reads those as replacement fields and raises KeyError('self.domain') --
# which is why the first attempt at this patch silently did nothing.
REGISTRY_BLOCK = '''        # The agent's profile is read from disk here, by task-type name, so a
        # dataset-specific pool has to be preferred at THIS point. Patching
        # encoder_roles alone only changed which role the router picks, not the
        # text that role is then given: measured in the live transcripts, 15.6% of
        # DROP prompts still said "choose the correct answer" and the adapted
        # wording appeared zero times. See shims/masrouter/install.py.
        import os as _shim_rr_os

        _shim_rr_ds = _shim_rr_os.getenv("SHIM_DATASET", "")
        if _shim_rr_ds:
            _shim_rr_path = f"MAR/Roles/{self.domain}_{_shim_rr_ds}/{self.role}.json"
            if _shim_rr_os.path.exists(_shim_rr_path):
                return json.load(open(_shim_rr_path, encoding="utf-8"))
'''


def patch_role_registry() -> None:
    """Prefer Roles/<Type>_<dataset>/ where the profile is actually loaded."""
    path = REPO / "MAR" / "Roles" / "role_registry.py"
    text = path.read_text(encoding="utf-8")
    if REGISTRY_MARKER in text:
        report(True, "role profile lookup already dataset-aware")
        return
    signature = "    def get_role_profile(self):\n"
    if signature not in text:
        report(False, "get_role_profile signature not found")
        return
    block = "        " + REGISTRY_MARKER + "\n" + REGISTRY_BLOCK
    text = text.replace(signature, signature + block, 1)
    path.write_text(text, encoding="utf-8")
    verify_syntax(path)
    report(REGISTRY_MARKER in path.read_text(encoding="utf-8"),
           "role profile lookup made dataset-aware")


def verify_syntax(path: Path) -> None:
    """A patch that produces a SyntaxError fails at import time, far from here."""
    import ast

    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        report(False, f"{path.name} does not parse after patching: {exc}")


def write_final_node_prompts() -> None:
    """Add the two derived final-node prompts next to the authors' own."""
    base = REPO / "MAR" / "Roles" / "FinalNode"
    if not base.exists():
        report(False, "MAR/Roles/FinalNode missing")
        return
    written = []
    for name, payload in (("shared_mmlu_pro.json", FINAL_NODE_MMLU_PRO),
                          ("shared_drop.json", FINAL_NODE_DROP)):
        path = base / name
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        written.append(name)
    # The authors' own files must still be there and unmodified: math.json and
    # mbpp.json are used as-is, so a missing one is a silent fallback waiting to
    # happen.
    missing = [n for n in ("math.json", "mbpp.json", "mmlu.json")
               if not (base / n).exists()]
    report(not missing, f"final-node prompts ({', '.join(written)} written; "
                        f"authors' files present: {not missing})")


def install_files() -> None:
    shutil.copyfile(HERE / "shared_dataset.py", REPO / "Datasets" / "shared_dataset.py")
    report((REPO / "Datasets" / "shared_dataset.py").exists(), "Datasets/shared_dataset.py")


def link_data() -> None:
    target_dir = REPO / "Datasets" / "shared"
    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = ([f"{name}.jsonl" for name in DATASETS]
              + [f"{name}_search.jsonl" for name in DATASETS])
    for filename in sorted(set(wanted)):
        source = DATA / filename
        if not source.exists():
            report(False, f"missing shared split {filename}")
            continue
        target = target_dir / filename
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copyfile(source, target)
    report(len(list(target_dir.glob("*.jsonl"))) >= 9,
           f"data wired ({len(list(target_dir.glob('*.jsonl')))} files)")


# MasRouter's headline contribution is a joint search over task type,
# collaboration pattern, agent count, role and *LLM*. The shipped llm_profile
# lists five cloud models with per-model prices and published accuracies, and the
# experiment proxy serves exactly one model -- so left alone, the router would
# learn a distribution over five names that are all Qwen3-8B while reading
# descriptions asserting they differ. Every request would succeed and the numbers
# would look plausible.
#
# Collapsing the pool to one truthful entry makes the LLM head a one-way choice:
# its log-prob is exactly 0 and contributes no gradient, while the task,
# collaboration, count and role heads keep learning. That is a real loss of one
# of MasRouter's four axes and must be stated in the results, but note the loss
# here is task_loss + answer_loss + 0.001*vae_loss with no cost term, so the LLM
# head could only ever have traded accuracy for price -- removing it does not
# handicap MasRouter on the accuracy metric this comparison ranks by.
#
# The description below states only what is verifiable about the served model. No
# benchmark accuracies are asserted, because inventing them is exactly the
# failure this replaces.
LLM_PROFILE = '''llm_profile = [
                {'Name': 'qwen3-8b',
                 'Description': 'Qwen3-8B, an 8-billion-parameter hybrid-reasoning\
                    model from the Qwen3 family, served locally by vLLM behind the\
                    experiment proxy. Thinking mode is disabled and temperature is\
                    fixed at 0 for every call, so replies are direct answers.\
                    Running locally, the model has no per-token monetary price.'},
                ]
'''


def patch_llm_profile() -> None:
    path = REPO / "MAR" / "LLM" / "llm_profile.py"
    text = path.read_text(encoding="utf-8")
    if "served locally by vLLM" in text:
        report(True, "llm_profile already collapsed to the served model")
        return
    names = re.findall(r"'Name': '([^']+)'", text)
    path.write_text(LLM_PROFILE, encoding="utf-8")
    new_names = re.findall(r"'Name': '([^']+)'", path.read_text(encoding="utf-8"))
    report(new_names == ["qwen3-8b"],
           f"llm_profile: {len(names)} cloud models -> {new_names} "
           f"(replaced {', '.join(names)})")


def patch_price_table() -> None:
    """Register the served model so token accounting is not silently zeroed.

    cost_count() returns (0, 0, 0) for a model name absent from MODEL_PRICE, and
    it returns *before* incrementing PromptTokens/CompletionTokens -- so an
    unregistered name zeroes the token counters too, not just the price. A
    zero-price entry keeps the counters live and reports the honest monetary cost
    of local inference. Cost columns are therefore not comparable with the
    published cloud-API numbers, by construction.
    """
    path = REPO / "MAR" / "LLM" / "price.py"
    text = path.read_text(encoding="utf-8")
    if '"qwen3-8b"' in text:
        report(True, "price table already has the served model")
        return
    anchor = "MODEL_PRICE = {\n"
    if anchor not in text:
        report(False, "MODEL_PRICE anchor missing")
        return
    text = text.replace(
        anchor,
        anchor + '    # shared-layer shim: local inference has no per-token price,\n'
                 '    # but the entry must exist or cost_count() zeroes the token counters.\n'
                 '    "qwen3-8b": {\n        "input": 0.0,\n        "output": 0.0\n    },\n',
        1,
    )
    path.write_text(text, encoding="utf-8")
    report('"qwen3-8b"' in path.read_text(encoding="utf-8"), "price table: qwen3-8b registered")


def patch_max_batches() -> None:
    """Add --max_batches so a closed loop can be verified without a full epoch.

    The train loop runs len(train_dataset)/batch_size batches with no way to stop
    early, which is fine for the real sweep but means the smallest possible
    smoke test is a whole epoch over the shared search split. The default is None,
    so the sweep behaviour is unchanged.
    """
    path = REPO / "Experiments" / "run_shared.py"
    text = path.read_text(encoding="utf-8")
    if "--max_batches" in text:
        report(True, "runner already has --max_batches")
        return
    anchor = '    parser.add_argument("--result_file", type=str, default=None)\n'
    if anchor not in text:
        report(False, "runner result_file anchor missing")
        return
    text = text.replace(
        anchor,
        anchor + "    parser.add_argument('--max_batches', type=int, default=None,\n"
                 "                        help='stop each epoch after this many batches '\n"
                 "                             '(shim: for smoke runs; None = full epoch)')\n",
        1,
    )
    anchors = (
        "    num_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size\n",
        "    num_batches = int(len(train_dataset)/args.batch_size)\n",
    )
    anchor_line = next((line for line in anchors if line in text), None)
    if anchor_line is None:
        report(False, "runner num_batches anchor missing")
        return
    text = text.replace(
        anchor_line,
        anchor_line + "    if args.max_batches is not None:\n"
                      "        num_batches = min(num_batches, args.max_batches)\n"
                      "        logger.info(f'shim: capping each epoch at {num_batches} batch(es)')\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    report("--max_batches" in path.read_text(encoding="utf-8"), "runner gained --max_batches")


def patch_sentence_encoder() -> None:
    """Return embeddings autograd can use.

    sentence-transformers >= 3 runs encode() inside torch.inference_mode(), and
    an inference tensor cannot be saved for backward. MasRouter feeds these
    embeddings straight into trainable Linear layers in every one of its heads,
    so on this version the first batch dies with "Inference tensors cannot be
    saved for backward" -- the repo was written against sentence-transformers 2.x,
    which used no_grad(). Cloning outside the inference context yields an ordinary
    tensor, which is the workaround PyTorch's own error message names. The encoder
    stays frozen either way, so no gradient reaches it and the method is unchanged.
    """
    path = REPO / "MAR" / "LLM" / "llm_embedding.py"
    text = path.read_text(encoding="utf-8")
    if "inference tensor" in text.lower():
        report(True, "sentence encoder already returns a normal tensor")
        return
    old = ("        embeddings = self.model.encode(sentence,convert_to_tensor=True,device=self.device)\n"
           "        return embeddings")
    if old not in text:
        report(False, "sentence encoder forward not found")
        return
    new = ("        embeddings = self.model.encode(sentence,convert_to_tensor=True,device=self.device)\n"
           "        # --- shared-layer shim ---\n"
           "        # sentence-transformers >= 3 encodes under torch.inference_mode(), and an\n"
           "        # inference tensor cannot be saved for backward. These embeddings feed the\n"
           "        # router's trainable heads, so cloning here (outside the inference context)\n"
           "        # is required to make them ordinary tensors. The encoder remains frozen.\n"
           "        return embeddings.clone()")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    report("embeddings.clone()" in path.read_text(encoding="utf-8"),
           "sentence encoder returns a clone (autograd-safe)")


def write_env() -> None:
    """Point MasRouter at the proxy.

    llm_registry.get() sends every non-DeepSeek model name to ALLChat, and
    ALLChat builds an OpenAI/AsyncOpenAI client from os.environ["URL"] and
    os.environ["KEY"] -- so URL must be the API *base* ending in /v1, not a
    complete endpoint path. BASE_URL/API_KEY are set as well because the sibling
    GPTChat class reads those instead; it is not selected here, but leaving it
    unconfigured would make any future registry change fail the same way.

    Without these, the OpenAI client raises "Missing credentials" inside
    tenacity's @retry(wait_random_exponential(max=100), stop_after_attempt(10)),
    which turns a one-line configuration error into a multi-minute stall with
    nothing written to the log and no request ever reaching the proxy.
    """
    base = "http://127.0.0.1:18080/train/masrouter/v1"
    path = REPO / ".env"
    path.write_text(
        f"URL = '{base}'\n"
        "KEY = 'local'\n"
        f"BASE_URL = '{base}/chat/completions'\n"
        "API_KEY = 'local'\n",
        encoding="utf-8",
    )
    report("URL = " in path.read_text(encoding="utf-8"), f"env -> {path.relative_to(ROOT)}")


def patch_model_suffix() -> None:
    """Send the model name the proxy actually serves.

    This is on the GPTChat path, which llm_registry does not select for our model
    names, so it changes nothing about the current runs -- it is applied because
    the pattern is a live trap: GPTChat requests `model + '-y'`, a tag for the
    authors' internal gateway. The
    proxy refuses names it does not serve rather than answering with a different
    model, so the suffix has to go -- and dropping it is preferable to allowlisting
    'qwen3-8b-y', since cost_count() is already keyed on the unsuffixed name and a
    mismatch between the two is what makes token accounting silently vanish.
    """
    path = REPO / "MAR" / "LLM" / "gpt_chat.py"
    text = path.read_text(encoding="utf-8")
    if "shim: no gateway suffix" in text:
        report(True, "model suffix already dropped")
        return
    # The two call sites are spelled differently (model+'-y' and model + '-y'),
    # so match on a whitespace-tolerant pattern rather than one literal.
    pattern = re.compile(r'"name":\s*model\s*\+\s*\'-y\',')
    count = len(pattern.findall(text))
    if count == 0:
        report(False, "model suffix pattern not found")
        return
    text = pattern.sub('"name": model,  # shim: no gateway suffix', text)
    path.write_text(text, encoding="utf-8")
    report("shim: no gateway suffix" in path.read_text(encoding="utf-8"),
           f"model suffix '-y' dropped ({count} call site(s))")


def patch_checkpoint_name() -> None:
    """Give each dataset its own checkpoint file.

    run_shared.py was derived from run_math.py and inherited its literal
    `math_router_epoch{epoch}_new.pth`. The name carries no dataset, so the five
    datasets of one sweep -- which run concurrently from the same working
    directory -- all write and read the same file and silently clobber each other.
    """
    path = REPO / "Experiments" / "run_shared.py"
    text = path.read_text(encoding="utf-8")
    if "args.shared_dataset}_router" in text:
        report(True, "checkpoint name already per-dataset")
        return
    old = 'f"math_router_epoch{epoch}_new.pth"'
    if old not in text:
        report(False, "checkpoint name pattern not found")
        return
    text = text.replace(old, 'f"{args.shared_dataset}_router_epoch{epoch}_new.pth"')
    # The resume path reads the same family of files; keep the two consistent so a
    # future --start_epoch cannot load another dataset's router.
    text = text.replace('f"math_router_epoch{epoch}.pth"',
                        'f"{args.shared_dataset}_router_epoch{epoch}.pth"')
    path.write_text(text, encoding="utf-8")
    report("args.shared_dataset}_router" in path.read_text(encoding="utf-8"),
           "checkpoint name is per-dataset")


CONCURRENT_QUERIES = """        # --- shared-layer shim: run a batch's queries in parallel threads ---
        # forward() executed the batch one query at a time. Every other method here
        # dispatches its batch concurrently, so the sequential loop did not make the
        # comparison fairer -- it made MasRouter ~20x slower in wall clock for the
        # same computation. At the measured ~2 min/query the search alone (252 items
        # x 5 epochs) would have needed ~42h per dataset, and the 1120-item MMLU-Pro
        # test loop another ~37h, which would have excluded the method in practice.
        #
        # Threads rather than the repo's Graph.arun: arun exists but is dead code --
        # calling it raises "a coroutine was expected, got None" because the agent
        # classes implement only synchronous execution. The synchronous run() blocks
        # on HTTP, so threads give real concurrency.
        #
        # Safe because the query loop is pure inference:
        #   * the gradient-bearing log_probs (llm + role + collab) are computed by
        #     the router *before* this loop and are untouched here;
        #   * each query builds its own Graph, and Graph defaults optimized_spatial
        #     and optimized_temporal to False -- MasRouter never passes them -- so
        #     its spatial_logits carry requires_grad=False;
        #   * run()'s own log_probs are discarded: only [0][0], the answer, is kept.
        #
        # The Cost/PromptTokens singletons are incremented non-atomically and so
        # become unattributable per query. That is acceptable only because local
        # inference is priced at 0.0 in MAR/LLM/price.py, making every per-query
        # cost 0 and utility reduce to is_solved; token totals for budget accounting
        # are taken from the proxy, which counts per request. Revisit if a non-zero
        # price is ever configured.
        import concurrent.futures as _futures

        def _run_one(_item):
            query, task, llms, collab, roles = _item
            kwargs = get_kwargs(collab['Name'], len(llms))
            llm_names = [llm['Name'] for llm in llms]
            role_names = [role['Name'] for role in roles]
            logger.info(f'Query: {query}')
            logger.info(f'Task: {task["Name"]}')
            logger.info(f'LLMs: {llm_names}')
            logger.info(f'Reasoning: {collab["Name"]}')
            logger.info(f'Roles: {role_names}')
            logger.info('-----------------------------------')
            g = Graph(domain=task['Name'], llm_names=llm_names, agent_names=role_names,
                      decision_method="FinalRefer", prompt_file=prompt_file,
                      reasoning_name=collab["Name"], **kwargs)
            return g.run(inputs={"query": query}, num_rounds=kwargs["num_rounds"])[0][0]

        _items = list(zip(queries, selected_tasks, selected_llms,
                          selected_collabs, selected_roles))
        with _futures.ThreadPoolExecutor(max_workers=max(len(_items), 1)) as _pool:
            final_result = list(_pool.map(_run_one, _items))
        costs = [0.0 for _ in _items]
"""


def patch_concurrent_queries() -> None:
    path = REPO / "MAR" / "MasRouter" / "mas_router.py"
    text = path.read_text(encoding="utf-8")
    # Tested against the first line of the block this function actually writes.
    # The earlier check looked for "run a batch's queries concurrently", which the
    # block does not contain -- it says "in parallel threads" -- so an already
    # patched file reported FAIL on every re-run, and the failure was cosmetic
    # noise sitting next to real ones.
    if CONCURRENT_QUERIES.splitlines()[0].strip() in text:
        # An older version stored each thread-local Graph on self.g. Concurrent
        # workers then raced to overwrite it even though no code ever read it.
        # Remove that stale assignment when upgrading an already-patched tree.
        upgraded = text.replace("            self.g = g\n", "")
        if upgraded != text:
            path.write_text(upgraded, encoding="utf-8")
        report(True, "batch already dispatched concurrently")
        return
    start = text.find("        final_result = []\n        costs = []\n")
    end = text.find("        return final_result, costs, log_probs")
    if start == -1 or end == -1 or end < start:
        report(False, "forward() query loop not found")
        return
    text = text[:start] + CONCURRENT_QUERIES + "\n" + text[end:]
    path.write_text(text, encoding="utf-8")
    final = path.read_text(encoding="utf-8")
    ok = "ThreadPoolExecutor" in final and final.count("g.run(inputs=") == 1
    report(ok, "batch queries dispatched concurrently (thread pool)")


def check() -> None:
    print("\n[masrouter] verification")
    runner = REPO / "Experiments" / "run_shared.py"
    report(runner.exists(), "run_shared.py")
    # The graded call's prompt must follow the dataset. Asserted on the generated
    # runner rather than on the installer, because the runner is what runs.
    if runner.exists():
        runner_text = runner.read_text(encoding="utf-8")
        report("SHARED_FINAL_NODE[args.shared_dataset]" in runner_text
               and "'mbpp': 'MAR/Roles/FinalNode/mbpp.json'" in runner_text,
               "final-node prompt selected per dataset")
        base = REPO / "MAR" / "Roles" / "FinalNode"
        derived = [n for n in ("shared_mmlu_pro.json", "shared_drop.json")
                   if not (base / n).exists()]
        report(not derived, f"derived final-node prompts present"
                            + ("" if not derived else f" (missing: {derived})"))
        roles = REPO / "MAR" / "Roles"
        author_pool = sorted((roles / "Commonsense").glob("*.json"))
        for dataset in ("drop", "mmlu_pro"):
            pool = sorted((roles / f"Commonsense_{dataset}").glob("*.json"))
            report(len(pool) == len(author_pool) and pool,
                   f"{dataset} role pool has the authors' role set "
                   f"({len(pool)}/{len(author_pool)} roles)")
            stale = [p.name for p in pool
                     if "complex math problem" in p.read_text(encoding="utf-8")
                     or (dataset == "drop" and "choose the correct answer" in
                         p.read_text(encoding="utf-8"))]
            report(not stale, f"{dataset} roles carry no wrong-task wording"
                              + ("" if not stale else f" (still: {stale})"))
        report(ROLE_VARIANT_MARKER in (REPO / "MAR" / "MasRouter" / "mas_router.py")
               .read_text(encoding="utf-8"),
               "encoder_roles prefers the dataset pool")
        report(REGISTRY_MARKER in (REPO / "MAR" / "Roles" / "role_registry.py")
               .read_text(encoding="utf-8"),
               "role profile lookup prefers the dataset pool (the layer that "
               "actually reaches the model)")
        # The authors' maths file must still ask for \\boxed, i.e. it was read and
        # reused rather than edited into a generic one.
        math_file = base / "math.json"
        if math_file.exists():
            report("boxed" in math_file.read_text(encoding="utf-8"),
                   "authors' math.json unmodified")
        drop_final = base / "shared_drop.json"
        if drop_final.exists():
            drop_text = drop_final.read_text(encoding="utf-8")
            report("Answer: <answer>" in drop_text and "\\\\boxed" not in drop_text,
                   "DROP final node uses the shared span-answer contract")
    if runner.exists():
        text = runner.read_text(encoding="utf-8")
        report("MATH_get_predict" not in text, "no leftover MATH_get_predict")
        report(text.count("shared_score(args.shared_dataset") == 2, "grading replaced in both loops")
        report(text.count("shared_task_labels(") == 2, "task labels replaced in both loops")
        report(text.count("+ args.batch_size - 1) // args.batch_size") == 2,
               "training and evaluation keep their final partial batch")
        report("--max_batches" in text and "min(num_batches, args.max_batches)" in text,
               "smoke runs can cap training batches")
    router_text = (REPO / "MAR" / "MasRouter" / "mas_router.py").read_text(encoding="utf-8")
    report("ThreadPoolExecutor" in router_text and "self.g = g" not in router_text,
           "parallel Graph objects are thread-local")
    report((REPO / "Datasets" / "shared_dataset.py").exists(), "shared_dataset.py")
    for name in DATASETS:
        path = REPO / "Datasets" / "shared" / f"{name}.jsonl"
        report(path.exists(), f"data {name}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not REPO.exists():
        report(False, f"missing {REPO}")
    elif args.check:
        check()
    else:
        print("[masrouter] installing")
        install_files()
        write_final_node_prompts()
        write_dataset_role_pools()
        patch_role_variants()
        patch_role_registry()
        patch_llm_profile()
        patch_price_table()
        derive_runner()
        # after derive_runner: it regenerates run_shared.py from run_math.py
        # and would otherwise discard this patch.
        patch_max_batches()
        patch_sentence_encoder()
        patch_checkpoint_name()
        patch_concurrent_queries()
        patch_model_suffix()
        write_env()
        link_data()

    print("\n" + "=" * 60)
    if problems:
        print(f"MasRouter shim: {len(problems)} problem(s)")
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print("MasRouter shim OK")


if __name__ == "__main__":
    main()
