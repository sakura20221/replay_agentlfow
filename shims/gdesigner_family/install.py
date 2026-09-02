#!/usr/bin/env python3
"""Install the shared-layer shim into G-Designer and CARD.

The runner is *derived* from each repo's own `run_gsm8k.py` by targeted
substitution rather than rewritten. That loop contains the topology policy
gradient (`utility = is_solved`, `single_loss = -log_prob * utility`, then the
Adam step on `graph.gcn`), and reimplementing it would risk changing the method
being measured. Only three things move: where the data comes from, which prompt
domain is selected, and which scorer grades the answer. Every anchor is checked,
so a repo update that shifts the code fails loudly instead of silently producing
a runner that trains nothing.

CARD is a fork of G-Designer but its run_gsm8k.py is not identical, so each repo
is derived from its own copy.

    python shims/gdesigner_family/install.py [--check]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "shared" / "data"

MARKER = "# --- derived from run_gsm8k.py by the shared-layer shim ---"

# repo dir -> inner package name
REPOS = {
    "gdesigner": ("third_party/gdesigner", "GDesigner"),
    "card": ("third_party/card", "CARD"),
}

DATASETS = ("math", "amc", "mbpp", "drop", "mmlu_pro")

problems: list[str] = []


def report(ok: bool, message: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {message}")
    if not ok:
        problems.append(message)


def substitute(text: str, pattern: str, replacement: str, label: str) -> str:
    """Regex substitution with a loud failure when the anchor is gone.

    Patterns rather than literals because CARD is a black-formatted fork of
    G-Designer: same code, different whitespace (`import a, b` vs `import a,b`,
    `Graph(\\n    domain=` vs `Graph(domain=`, `x == y` vs `x==y`).
    """
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        report(False, f"anchor missing: {label}")
        return text
    return updated


def derive_runner(repo: Path, package: str) -> None:
    source = repo / "experiments" / "run_gsm8k.py"
    if not source.exists():
        report(False, f"{source.name} missing in {repo.name}")
        return
    text = source.read_text(encoding="utf-8")

    # `--domain` already exists in both repos and its own help text says it is
    # "the same as dataset name", so it doubles as the shared-split selector and
    # no new flag is needed -- a smaller diff against the authors' code.

    # 1. imports: shared data + scorer, and register the extra prompt domains.
    text = substitute(
        text,
        r"from datasets\.gsm8k_dataset import\s+gsm_data_process\s*,\s*gsm_get_predict",
        "from datasets.shared_dataset import shared_data_process, shared_score\n"
        f"import {package}.prompt.shared_prompt_sets  # noqa: F401  # registers the shared domains",
        "import line",
    )

    # 2. data conversion, keyed on the domain.
    text = substitute(text, r"dataset = gsm_data_process\(dataset\)",
                      "dataset = shared_data_process(dataset, args.domain)",
                      "data process call")

    # 3. results directory per dataset.
    text = substitute(text, r'(result_dir = Path\(f"\{[A-Za-z_]+\})/result/gsm8k("\))',
                      r'\1/result/{args.domain}\2', "result dir")

    # 4. the domain must follow the flag rather than stay pinned to gsm8k.
    text = substitute(text, r'domain\s*=\s*"gsm8k"', "domain=args.domain", "Graph(domain=...)")

    # 5. grading: one scorer for every method, and partial credit survives
    #    (DROP is F1, so is_solved is fractional rather than boolean).
    text = substitute(
        text,
        r"predict_answer = gsm_get_predict\(answer\[0\]\)\n(\s*)is_solved = float\(predict_answer\)\s*==\s*float\(true_answer\)",
        r"_shared_score, predict_answer = shared_score(args.domain, task, answer[0])\n\1is_solved = _shared_score",
        "scoring lines",
    )

    target = repo / "experiments" / "run_shared.py"
    target.write_text(MARKER + "\n" + text, encoding="utf-8")
    report(target.exists(), f"runner -> {target.relative_to(ROOT)}")



# Both repos read BASE_URL / API_KEY from a .env via python-dotenv and POST the
# full URL directly, so it must include the path. G-Designer sends the legacy
# {name, inputs:{msg}} shape and CARD sends standard OpenAI JSON; the proxy
# detects which by the body, so the same URL works for both. Written by the
# installer rather than by hand: a hand-made .env is not reproducible, and its
# absence makes the repo fail with an unhelpful connection error.
ENV_TEMPLATE = "BASE_URL = 'http://127.0.0.1:18080/train/{label}/v1/chat/completions'\nAPI_KEY = 'local'\n"

# CARD alone routes through a provider switch: llm_registry reads SERVER from the
# environment and looks the requested model up in MODEL_NAME_MAP[SERVER]. An unset
# SERVER makes that .get() return None and the lookup raises AttributeError before
# any request is made, so the value is part of the wiring, not a preference.
EXTRA_ENV = {"card": "SERVER = 'local'\n"}


def write_env(repo: Path, label: str) -> None:
    path = repo / ".env"
    path.write_text(ENV_TEMPLATE.format(label=label) + EXTRA_ENV.get(label, ""), encoding="utf-8")
    report("18080" in path.read_text(encoding="utf-8"), f"env -> {path.relative_to(ROOT)}")


# CARD's registry translates the CLI's --llm_name through MODEL_NAME_MAP[SERVER]
# before constructing the client, and an unknown name maps to None -- which is
# then sent as the request's "model" field. Rather than borrow the "qwen" entry
# (whose values are DashScope's cloud names) a "local" provider is added that
# maps to the name vLLM actually serves.
#
# Note an upstream bug left alone here: get() does `if MY_SERVER == "together":
# ... ` followed by a separate `if/else` whose else-branch unconditionally
# overwrites the choice with GPTChat, so the TogetherChat path is dead for
# SERVER=together. We never take it, and changing it would alter author logic.
LOCAL_PROVIDER = """        "local": {
            None: "Qwen/Qwen3-8B",
            "": "Qwen/Qwen3-8B",
            "qwen3-8b": "Qwen/Qwen3-8B",
        },
"""


def patch_model_name_map(repo: Path) -> None:
    path = repo / "CARD" / "llm" / "llm_registry.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if '"local"' in text:
        report(True, "llm_registry already has the local provider")
        return
    anchor = "    MODEL_NAME_MAP = {\n"
    if anchor not in text:
        report(False, "llm_registry MODEL_NAME_MAP anchor missing")
        return
    text = text.replace(anchor, anchor + LOCAL_PROVIDER, 1)
    path.write_text(text, encoding="utf-8")
    report('"local"' in path.read_text(encoding="utf-8"), "llm_registry: local provider added")


def patch_optional_together(repo: Path) -> None:
    """Make the Together client import optional.

    CARD/llm/__init__.py imports TogetherChat unconditionally, which imports the
    `together` SDK, so `import CARD` fails with ModuleNotFoundError on a host that
    has no Together account. Installing the SDK to satisfy an import for a hosted
    API we never call would add ~90MB to a filesystem at 99% capacity; the guard
    keeps the symbol absent instead, and any real use of it would still raise.
    """
    path = repo / "CARD" / "llm" / "__init__.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "shared-layer shim" in text:
        report(True, "together import already optional")
        return
    old = "from CARD.llm.together_chat import TogetherChat"
    if old not in text:
        report(False, "together import line not found")
        return
    text = text.replace(
        old,
        "# --- shared-layer shim: the Together SDK is not installed and never used ---\n"
        "try:\n"
        f"    {old}\n"
        "except ModuleNotFoundError:  # pragma: no cover - hosted provider unused\n"
        "    TogetherChat = None",
        1,
    )
    path.write_text(text, encoding="utf-8")
    report("ModuleNotFoundError" in path.read_text(encoding="utf-8"), "together import made optional")


# CARD's method is a search over *combinations*: node_kwargs is a dict mapping a
# combination name to a list of per-node configs (llm_name, role, external tool),
# and Graph caches a feature vector per combination for the GCN to score. The
# authors ship both heterogeneous pools (qwen-7b+qwen-14b+qwen-72b.json) and
# single-model configs (qwen-72B.json, gpt-4o.json), so a one-backbone run is an
# author-supported configuration rather than a degenerate one -- but note that
# under a single backbone CARD's model-selection axis collapses and only the role
# and topology axes remain active. That is a property of the protocol, not a bug,
# and it belongs in the results discussion.
#
# The role lists below are copied verbatim from the authors' own configs for the
# matching domain (math/qwen-72B.json, humaneval/qwen-72B.json,
# mmlu_node_config.json) so nothing about the method is invented here; only
# llm_name is rewritten to the locally served model.
NODE_CONFIG_SOURCE = {
    "math": ("math", "qwen-72B.json"),
    "code": ("humaneval", "qwen-72B.json"),
    "qa": (None, "mmlu_node_config.json"),
}

DOMAIN_FAMILY = {"math": "math", "amc": "math", "mbpp": "code",
                 "drop": "qa", "mmlu_pro": "qa"}


def write_node_configs(repo: Path) -> None:
    config_root = repo / "CARD" / "config"
    out_dir = config_root / "shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, (subdir, name) in NODE_CONFIG_SOURCE.items():
        source = (config_root / subdir / name) if subdir else (config_root / name)
        if not source.exists():
            report(False, f"node config source missing: {source.name}")
            continue
        groups = json.loads(source.read_text(encoding="utf-8"))
        # Keep a single combination: with one backbone the author's multi-group
        # files differ only by llm_name, so they would collapse into duplicates
        # and inflate the feature cache with identical entries.
        first = next(iter(groups.values()))
        adapted = {"combination_1": [dict(node, llm_name="qwen3-8b") for node in first]}
        target = out_dir / f"{family}.json"
        target.write_text(json.dumps(adapted, indent=4, ensure_ascii=False), encoding="utf-8")
        roles = [n.get("role") for n in adapted["combination_1"]]
        report(target.exists(), f"node config {family}.json ({len(roles)} nodes: {', '.join(roles)})")


RUNNER_NODE_PATCH = """    # --- shared-layer shim: supply the combination dict CARD requires ---
    # Graph.__init__ defaults node_kwargs to [{} ...] when it is None, but the
    # very next statement builds all_node_config_groups from the *parameter*
    # rather than the attribute, so the default is discarded and
    # init_with_node_config(None) raises TypeError in init_nodes(). Every mode
    # except DirectAnswer therefore needs an explicit dict from the caller --
    # which is what the authors' own run_mmlu.py passes and run_gsm8k.py does
    # not. get_kwargs() also returns a node_kwargs key, which would collide with
    # the explicit argument, so it is removed here.
    kwargs.pop("node_kwargs", None)
    if args.node_config_file is None:
        args.node_config_file = str(
            Path(CARD_ROOT) / "CARD" / "config" / "shared"
            / f"{_DOMAIN_FAMILY[args.domain]}.json"
        )
    with open(args.node_config_file, encoding="utf-8") as _fh:
        node_config = json.load(_fh)
    for _combo, _group in node_config.items():
        assert len(_group) == len(agent_names), (
            f"{_combo} has {len(_group)} nodes but --agent_nums gives "
            f"{len(agent_names)}; CARD requires them to match"
        )
    print(f"[shim] node config: {args.node_config_file} "
          f"({len(node_config)} combination(s), {len(agent_names)} nodes)")
"""


def patch_runner_node_config(repo: Path) -> None:
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        report(False, "run_shared.py missing, cannot wire node config")
        return
    text = path.read_text(encoding="utf-8")

    # Added before the early return so a file patched by an earlier revision
    # still gains the table: the injected block references _DOMAIN_FAMILY, so
    # testing for the bare name matches the reference and never the definition.
    if "_DOMAIN_FAMILY = " not in text:
        text = text.replace("async def main(",
                            "_DOMAIN_FAMILY = " + repr(DOMAIN_FAMILY) + "\n\n\nasync def main(", 1)
        for line in ("from pathlib import Path", "import json"):
            if line not in text:
                text = line + "\n" + text
        path.write_text(text, encoding="utf-8")

    # CARD_ROOT is the repo root, so the config path needs the package dir too;
    # applied unconditionally because it is a literal swap and must also reach
    # files written by an earlier revision.
    if 'Path(CARD_ROOT) / "config" / "shared"' in text:
        text = text.replace('Path(CARD_ROOT) / "config" / "shared"',
                            'Path(CARD_ROOT) / "CARD" / "config" / "shared"')
        path.write_text(text, encoding="utf-8")

    if "supply the combination dict" in text:
        report("_DOMAIN_FAMILY = " in text, "runner already wires the node config")
        return

    anchor = "    kwargs = get_kwargs(args.mode, len(agent_names))\n"
    if anchor not in text:
        report(False, "runner get_kwargs anchor missing")
        return
    text = text.replace(anchor, anchor + RUNNER_NODE_PATCH, 1)

    text = text.replace(
        "        optimized_temporal=args.optimized_temporal,\n        **kwargs,",
        "        optimized_temporal=args.optimized_temporal,\n"
        "        node_kwargs=node_config,\n"
        "        allow_random_combination=False,\n"
        "        **kwargs,",
        1,
    )
    text = text.replace(
        '    parser.add_argument("--optimized_spatial", action="store_true")',
        '    parser.add_argument("--node_config_file", type=str, default=None,\n'
        '                        help="CARD combination dict; defaults to the shared "\n'
        '                             "single-backbone config for this domain")\n'
        '    parser.add_argument("--optimized_spatial", action="store_true")',
        1,
    )
    path.write_text(text, encoding="utf-8")
    final = path.read_text(encoding="utf-8")
    report("node_kwargs=node_config" in final and "--node_config_file" in final,
           "runner wires the node config")


# The authors use a different agent class, count and decision head per domain --
# run_gsm8k: MathSolver / FinalRefer, run_humaneval: CodeWriting / FinalWriteCode,
# run_mmlu: AnalyzeAgent / FinalRefer -- and run_shared.py was derived from the
# gsm8k one, so it carried gsm8k's defaults into every domain. AnalyzeAgent is not
# a drop-in either: it calls prompt_set.get_analyze_constraint(), which only the
# MMLU prompt sets define, so using it on math raises AttributeError.
#
# Node count is 5 for all three families because that is what the authors' own
# node-config files contain, and Graph asserts the two match.
# Both repos pick a different agent class, count and decision head per domain, and
# both run_shared.py files were derived from the gsm8k runner -- so they carried
# gsm8k's defaults into every domain. That is not a preference but a hard
# incompatibility: MathSolver and CodeWriting call
# prompt_set.get_constraint(role) while AnalyzeAgent calls
# get_analyze_constraint(role), and the MMLU prompt sets define get_constraint()
# with *no* parameters. Running MathSolver on the qa domains therefore dies with
# "MMLUPromptSet.get_constraint() takes 0 positional arguments but 1 was given".
#
# Values are each repo's own author defaults, read from its run_gsm8k / run_mmlu /
# run_humaneval: G-Designer uses 4 math agents, CARD 5 (its node config ships 5).
AUTHOR_DEFAULTS = {
    "gdesigner": {"math": (["MathSolver"], [4], "FinalRefer"),
                  "code": (["CodeWriting"], [5], "FinalWriteCode"),
                  "qa": (["AnalyzeAgent"], [5], "FinalRefer")},
    "card": {"math": (["MathSolver"], [5], "FinalRefer"),
             "code": (["CodeWriting"], [5], "FinalWriteCode"),
             "qa": (["AnalyzeAgent"], [5], "FinalRefer")},
}


def patch_runner_domain_defaults(repo: Path, label: str) -> None:
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")

    # parse_args() cross-checks the two lists before main() fills in the per-domain
    # defaults, so with default=None it raises on len(None). The check is kept for
    # explicitly supplied lists; the defaults below are consistent by construction.
    guard_old = "    if len(args.agent_names) != len(args.agent_nums):"
    guard_new = ("    if (args.agent_names is not None and args.agent_nums is not None\n"
                 "            and len(args.agent_names) != len(args.agent_nums)):")
    if guard_old in text:
        text = text.replace(guard_old, guard_new, 1)
        path.write_text(text, encoding="utf-8")

    if "_DOMAIN_DEFAULTS" in text:
        report("args.agent_names is not None" in text,
               f"{label}: runner already resolves per-domain defaults")
        return

    # Quote style differs between the two repos (CARD is black-formatted), so the
    # defaults are nulled by regex rather than literal match.
    for option in ("agent_names", "agent_nums", "decision_method"):
        # Quote style differs between the two repos (G-Designer uses single
        # quotes, CARD is black-formatted with double), so match either.
        pattern = re.compile(r'(["\']--' + option + r'["\'],.{0,300}?)default\s*=\s*[^,\n]+', re.S)
        text, count = pattern.subn(r"\1default=None", text, count=1)
        if count == 0:
            report(False, f"{label}: could not null the default for --{option}")
            return

    table = AUTHOR_DEFAULTS[label]
    block = ["    # --- shared-layer shim: per-domain agent defaults ---",
             "    _names, _nums, _decision = _DOMAIN_DEFAULTS[_DOMAIN_FAMILY[args.domain]]",
             "    if args.agent_names is None:",
             "        args.agent_names = _names",
             "    if args.agent_nums is None:",
             "        args.agent_nums = _nums",
             "    if args.decision_method is None:",
             "        args.decision_method = _decision",
             '    print(f"[shim] domain={args.domain} '
             'agents={args.agent_names}x{args.agent_nums} decision={args.decision_method}")',
             ""]
    # No trailing newline in the anchor: G-Designer builds agent_names as a
    # one-line comprehension while CARD's is wrapped across lines.
    anchor = "    agent_names = ["
    if anchor not in text:
        report(False, f"{label}: runner agent_names anchor missing")
        return
    text = text.replace(anchor, "\n".join(block) + anchor, 1)

    header = f"_DOMAIN_DEFAULTS = {table!r}\n"
    if "_DOMAIN_FAMILY = " not in text:
        header += "_DOMAIN_FAMILY = " + repr(DOMAIN_FAMILY) + "\n"
    text = text.replace("async def main(", header + "\n\nasync def main(", 1)
    path.write_text(text, encoding="utf-8")
    final = path.read_text(encoding="utf-8")
    report("_DOMAIN_DEFAULTS" in final and "_DOMAIN_FAMILY" in final,
           f"{label}: runner resolves per-domain agent defaults")


KWARG_FILTER_BLOCK = """    # --- shared-layer shim: keep only kwargs the agent class accepts ---
    # init_nodes() forwards every key of a node config straight into the agent
    # constructor, but the authors' configs carry external_tool / _type / _source,
    # which only AnalyzeAgent and CodeWriting declare -- MathSolver raises
    # TypeError on them. Filtering here leaves the author configs verbatim and
    # keeps the role assignment, which is the part the method actually searches
    # over. Nothing is dropped for the tool-using classes.
    import inspect as _inspect

    from CARD.agents.agent_registry import AgentRegistry as _AgentRegistry

    _accepted = set()
    for _agent_name in set(agent_names):
        if _agent_name in _AgentRegistry.registry:
            _accepted |= set(
                _inspect.signature(
                    _AgentRegistry.registry.get_class(_agent_name).__init__
                ).parameters
            )
    _dropped = sorted({k for _g in node_config.values() for _n in _g for k in _n} - _accepted)
    if _dropped:
        print(f"[shim] node kwargs not accepted by {sorted(set(agent_names))}: {_dropped}")
    node_config = {
        _name: [{k: v for k, v in _node.items() if k in _accepted} for _node in _group]
        for _name, _group in node_config.items()
    }
"""


def patch_runner_agent_kwargs(repo: Path) -> None:
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "keep only kwargs the agent class accepts" in text:
        report(True, "runner already filters node kwargs")
        return
    anchor = '{len(agent_names)} nodes)")\n'
    if anchor not in text:
        report(False, "runner node-config print anchor missing")
        return
    text = text.replace(anchor, anchor + KWARG_FILTER_BLOCK, 1)
    path.write_text(text, encoding="utf-8")
    report("keep only kwargs" in path.read_text(encoding="utf-8"), "runner filters node kwargs")


def patch_runner_fixed_group(repo: Path) -> None:
    """Name the combination each arun() should execute.

    Graph.arun picks a combination at random when allow_random_combination is set
    and otherwise asserts that the caller named one, so with a deterministic
    single-combination config the name has to be passed explicitly. It is read
    from the loaded config rather than hardcoded, so a multi-combination config
    still selects its own first group instead of a stale literal.
    """
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")

    # Inserted before the early return: the call-site rewrite below references
    # _fixed_group, so a file patched by an earlier revision can already have the
    # reference while still missing the definition.
    anchor = "    print(f\"[shim] node config: "
    if "_fixed_group = next" not in text:
        if anchor not in text:
            report(False, "runner node-config print anchor missing")
            return
        text = text.replace(anchor, "    _fixed_group = next(iter(node_config))\n" + anchor, 1)
        path.write_text(text, encoding="utf-8")

    if "fixed_group=_fixed_group" in text:
        report("_fixed_group = next" in text, "runner already names the combination")
        return
    old = "realized_graph.arun(input_dict, args.num_rounds)"
    if old not in text:
        report(False, "runner arun call not found")
        return
    count = text.count(old)
    text = text.replace(old, "realized_graph.arun(\n"
                             "                    input_dict, args.num_rounds, fixed_group=_fixed_group\n"
                             "                )")
    path.write_text(text, encoding="utf-8")
    report("fixed_group=_fixed_group" in path.read_text(encoding="utf-8"),
           f"runner names the combination ({count} arun call site(s))")


def patch_result_file_arg(repo: Path, label: str) -> None:
    """Make --result_file actually take effect.

    Both runners declare the flag and then overwrite it unconditionally with
    `result_dir / f"{domain}_{llm_name}_{current_time}.json"`. The timestamp has
    one-second resolution, and the sweep launches the main and author-default
    variants of a method from one thread pool -- observed 22:59:46 and 22:59:47,
    i.e. one second apart by luck. Two jobs landing in the same second would append
    their per-item records to a single file, and that file is what the held-out
    accuracy is recomputed from.
    """
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "args.result_file:" in text:
        report(True, f"{label}: --result_file already honoured")
        return
    old = '    result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"'
    if old not in text:
        report(False, f"{label}: result_file assignment not found")
        return
    text = text.replace(
        old,
        "    # shared-layer shim: honour --result_file when given; the default name\n"
        "    # collides between concurrent jobs of the same domain (1-second stamp).\n"
        "    if args.result_file:\n"
        "        result_file = Path(args.result_file)\n"
        "        result_file.parent.mkdir(parents=True, exist_ok=True)\n"
        "    else:\n"
        + old.replace("    result_file", "        result_file"),
        1,
    )
    if "from pathlib import Path" not in text:
        text = "from pathlib import Path\n" + text
    path.write_text(text, encoding="utf-8")
    report("args.result_file:" in path.read_text(encoding="utf-8"),
           f"{label}: --result_file honoured")


BATCH_PLAN = """    # --- shared-layer shim: explicit search/evaluation boundary ---
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
"""


def patch_eval_batch_size(repo: Path, label: str) -> None:
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "_batch_plan" in text:
        report(True, f"{label}: evaluation batch already planned")
        return

    import re as _re

    num_batches = _re.search(r"^\s*num_batches = int\(len\(dataset\) ?/ ?args\.batch_size\)\s*$",
                             text, _re.M)
    current = _re.search(r"^(\s*)current_batch = dataloader\(dataset, ?args\.batch_size, ?i_batch\)\s*$",
                         text, _re.M)
    if not num_batches or not current:
        report(False, f"{label}: batch construction not found")
        return

    text = text[:num_batches.start()] + BATCH_PLAN.rstrip("\n") + text[num_batches.end():]
    current = _re.search(r"^(\s*)current_batch = dataloader\(dataset, ?args\.batch_size, ?i_batch\)\s*$",
                         text, _re.M)
    indent = current.group(1)
    text = (text[:current.start()]
            + f"{indent}_lo, _hi = _batch_plan[i_batch]\n"
            + f"{indent}current_batch = dataset[_lo:_hi]"
            + text[current.end():])

    anchor = "    parser.add_argument(\"--result_file\", type=str, default=None)"
    if anchor not in text:
        anchor = _re.search(r"^\s*parser\.add_argument\(['\"]--batch_size['\"].*$", text, _re.M)
        anchor = anchor.group(0) if anchor else None
    if anchor and "--eval_batch_size" not in text:
        text = text.replace(
            anchor,
            anchor + "\n    parser.add_argument('--eval_batch_size', type=int, default=0,\n"
                     "                        help='shim: batch size after the switch to "
                     "evaluation, where no gradient is taken (0 = same as --batch_size)')",
            1,
        )
    if anchor and "--train_items" not in text:
        text = text.replace(
            anchor,
            anchor + "\n    parser.add_argument('--train_items', type=int, default=0,\n"
                     "                        help='number of search items used for gradient updates')\n"
                     "    parser.add_argument('--search_items', type=int, default=0,\n"
                     "                        help='index where the held-out split starts')",
            1,
        )
    path.write_text(text, encoding="utf-8")
    final = path.read_text(encoding="utf-8")
    report("_batch_plan" in final and "--eval_batch_size" in final
           and "--train_items" in final and "--search_items" in final,
           f"{label}: train budget and held-out boundary planned independently")


def patch_dangling_import(repo: Path, package: str) -> None:
    """Neutralise `from <pkg>.llm import VisualLLMRegistry`.

    That name exists nowhere in the G-Designer repo -- the authors removed the
    class but left the import, so `readers.py` cannot be imported at all, which
    takes the whole runner down. It is used only by the image and video readers,
    which text benchmarks never reach (and which would fail regardless, the class
    being absent). CARD's own authors hit this and commented the same line out,
    so the fix here matches theirs.
    """
    path = repo / package / "tools" / "reader" / "readers.py"
    if not path.exists():
        report(False, f"{path.name} missing")
        return
    text = path.read_text(encoding="utf-8")
    target = f"from {package}.llm import VisualLLMRegistry"
    matching = [line for line in text.splitlines() if target in line]
    if not matching:
        report(True, "dangling VisualLLMRegistry import already absent")
        return

    canonical = f"# {target}  # shim: name does not exist in this repo (image/video readers only)"
    lines = text.splitlines(keepends=True)
    changed = False
    for index, line in enumerate(lines):
        if target not in line:
            continue
        stripped = line.strip()
        # CARD already comments this import upstream. Preserve the author's line
        # byte-for-byte; no local patch is required there.
        if stripped == f"# {target}":
            report(True, "dangling VisualLLMRegistry import already commented upstream")
            return
        ending = "\n" if line.endswith("\n") else ""
        indent = line[: len(line) - len(line.lstrip())]
        lines[index] = indent + canonical + ending
        changed = lines[index] != line
        break
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    report(canonical in path.read_text(encoding="utf-8"), "dangling import commented out")


LOG_MODULE = '''"""Added by the shared-layer shim.

G-Designer imports `{package}.utils.log` from readers.py and
tools/coding/executor_factory.py but never ships the module, so the package
cannot be imported at all. CARD, its fork, restores exactly this file; the
content below matches CARD's.
"""
import logging

logger = logging.getLogger("{package}")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)
'''


def ensure_log_module(repo: Path, package: str) -> None:
    path = repo / package / "utils" / "log.py"
    if path.exists():
        report(True, f"{package}/utils/log.py already present")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LOG_MODULE.format(package=package), encoding="utf-8")
    report(path.exists(), f"{package}/utils/log.py created (missing upstream)")


def patch_token_counter(repo: Path, package: str) -> None:
    """Stop `cal_token` from raising on a model tiktoken has never heard of.

    `cost_count` already routes non-OpenAI models to an "other" branch with zero
    price and zero tokens, but it calls `cal_token` first, and
    `tiktoken.encoding_for_model("Qwen/Qwen3-8B")` raises KeyError before the
    branching is reached. The exception surfaces inside the repo's tenacity retry
    as an opaque `RetryError[... raised KeyError]` after every LLM call has
    already succeeded -- 73 requests returned 200 at the proxy while the run
    still died.

    A fixed fallback encoder keeps the real model name in the logs instead of
    mislabelling the run as an OpenAI model just to satisfy the price table. Token
    and cost accounting for this experiment come from the proxy's per-namespace
    records, not from here.
    """
    path = repo / package / "llm" / "price.py"
    if not path.exists():
        report(False, f"{package}/llm/price.py missing")
        return
    text = path.read_text(encoding="utf-8")
    if "shim: fallback encoder" in text:
        report(True, "cal_token already patched")
        return
    anchor = "    encoder = tiktoken.encoding_for_model(model)"
    if anchor not in text:
        report(False, "cal_token anchor missing")
        return
    text = text.replace(
        anchor,
        "    try:  # shim: fallback encoder for locally served models\n"
        "        encoder = tiktoken.encoding_for_model(model)\n"
        "    except KeyError:\n"
        "        encoder = tiktoken.get_encoding(\"cl100k_base\")",
        1,
    )
    path.write_text(text, encoding="utf-8")
    report("shim: fallback encoder" in path.read_text(encoding="utf-8"), "cal_token made tolerant")


def install_files(repo: Path, package: str) -> None:
    shutil.copyfile(HERE / "shared_dataset.py", repo / "datasets" / "shared_dataset.py")
    report((repo / "datasets" / "shared_dataset.py").exists(), "datasets/shared_dataset.py")

    prompt_text = (HERE / "shared_prompt_sets.py").read_text(encoding="utf-8")
    if package != "GDesigner":
        prompt_text = prompt_text.replace("GDesigner.prompt", f"{package}.prompt")
    (repo / package / "prompt" / "shared_prompt_sets.py").write_text(prompt_text, encoding="utf-8")
    report((repo / package / "prompt" / "shared_prompt_sets.py").exists(),
           f"{package}/prompt/shared_prompt_sets.py")


def link_data(repo: Path) -> None:
    target_dir = repo / "datasets" / "shared"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        # _train_then_eval is the file these two repos actually run on: their loop
        # trains for --num_iterations batches and then evaluates the remainder of
        # the same file, so search and evaluation data have to arrive concatenated.
        # It was generated into shared/data/ but never linked here, and the jobs
        # died with FileNotFoundError before issuing a single request.
        for suffix, source_name in (("", f"{name}.jsonl"),
                                    ("_search", f"{name}_search.jsonl"),
                                    ("_train_then_eval", f"{name}_train_then_eval.jsonl")):
            source = DATA / source_name
            if not source.exists():
                report(False, f"missing shared split {source_name}")
                continue
            target = target_dir / f"{name}{suffix}.jsonl"
            if target.is_symlink() or target.exists():
                target.unlink()
            try:
                target.symlink_to(os.path.relpath(source, start=target.parent))
            except OSError:
                shutil.copyfile(source, target)
    count = len(list(target_dir.glob("*.jsonl")))
    report(count >= 2 * len(DATASETS), f"data wired ({count} files)")


EXEC_TIMEOUT_MARKER = "# --- shared-layer shim (agent_wf_v2) --- exec timeout v1"


def patch_exec_timeout(repo: Path, package: str) -> None:
    """Bound execute_code_get_return, the one raw exec of model-written code.

    The family's own function_with_timeout (thread + join(timeout)) already
    bounds get_output and PyExecutor.execute; this path ran exec() bare. Only
    the math domain reaches it (MathSolver._async_execute), which is why drop
    and mmlu_pro never hung. On the official L5 split a generated
    compute_n_plus_k blocked forever with zero CPU and took the whole job with
    it (runs_v5/gdesigner_authordefault/math 2026-08-24, stack in the job's
    stacks.txt). A thread guard is the right shape here precisely because the
    observed failure is a blocked wait: the timeout leaks a parked thread and
    nothing else. The 30 s budget matches the exec guard the other families
    got (shims/exec_guard_patch.py), keeping the policy identical per method.
    """
    path = repo / package / "tools" / "coding" / "python_executor.py"
    if not path.exists():
        report(False, f"exec timeout: {path} missing")
        return
    text = path.read_text(encoding="utf-8")
    if EXEC_TIMEOUT_MARKER in text:
        report(True, "exec timeout already applied")
        return
    old = "    try:\n        exec(code, {}, local_vars)\n"
    new = ("    try:\n"
           "        " + EXEC_TIMEOUT_MARKER + "\n"
           "        function_with_timeout(exec, (code, {}, local_vars), 30)\n")
    if old not in text:
        report(False, "exec timeout: anchor not found in python_executor.py")
        return
    text = text.replace(old, new, 1)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        report(False, f"exec timeout: patched file does not parse: {exc}")
        return
    path.write_text(text, encoding="utf-8")
    report(True, "exec timeout on execute_code_get_return (30 s)")


def patch_daemon_guard(repo: Path, package: str) -> None:
    # v2 (2026-08-24): the guard thread must be a daemon. function_with_timeout
    # leaks its PropagatingThread when the guarded code never returns, and a
    # non-daemon leaked thread keeps the interpreter alive after main() ends --
    # gdesigner/mbpp and card_authordefault/mbpp finished all 756 records and
    # then hung at exit for 75-96 minutes until killed by hand.
    utils = repo / package / "tools" / "coding" / "executor_utils.py"
    utext = utils.read_text(encoding="utf-8")
    if "thread.daemon = True" in utext:
        report(True, "guard thread already daemonised")
        return
    anchor = "    thread = PropagatingThread(target=wrapper)\n    thread.start()"
    if anchor not in utext:
        report(False, "daemonise: anchor not found in executor_utils.py")
        return
    utext = utext.replace(anchor, "    thread = PropagatingThread(target=wrapper)\n"
                                  "    thread.daemon = True\n    thread.start()", 1)
    try:
        ast.parse(utext)
    except SyntaxError as exc:
        report(False, f"daemonise: patched file does not parse: {exc}")
        return
    utils.write_text(utext, encoding="utf-8")
    report(True, "guard thread daemonised (leaked threads no longer block exit)")


def check(repo: Path, package: str, label: str) -> None:
    print(f"\n[{label}] verification")
    report((repo / "experiments" / "run_shared.py").exists(), "run_shared.py")
    report((repo / ".env").exists(), ".env")
    report((repo / "datasets" / "shared_dataset.py").exists(), "shared_dataset.py")
    report((repo / package / "prompt" / "shared_prompt_sets.py").exists(), "shared_prompt_sets.py")
    executor_text = (repo / package / "tools" / "coding" / "python_executor.py").read_text(encoding="utf-8")
    report(EXEC_TIMEOUT_MARKER in executor_text
           and "function_with_timeout(exec, (code, {}, local_vars), 30)" in executor_text,
           "execute_code_get_return bounded (30 s)")
    for name in DATASETS:
        path = repo / "datasets" / "shared" / f"{name}.jsonl"
        portable = not path.is_symlink() or not Path(os.readlink(path)).is_absolute()
        report(path.exists() and path.stat().st_size > 0 and portable,
               f"data {name}.jsonl")
    links = list((repo / "datasets" / "shared").glob("*.jsonl"))
    report(all(not path.is_symlink() or
               not Path(os.readlink(path)).is_absolute() for path in links),
           "shared-data links are relative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    for label, (rel, package) in REPOS.items():
        repo = ROOT / rel
        if not repo.exists():
            report(False, f"{label}: missing {rel}")
            continue
        if args.check:
            check(repo, package, label)
            continue
        print(f"\n[{label}] installing")
        install_files(repo, package)
        patch_dangling_import(repo, package)
        ensure_log_module(repo, package)
        patch_token_counter(repo, package)
        patch_exec_timeout(repo, package)
        patch_daemon_guard(repo, package)
        derive_runner(repo, package)
        link_data(repo)
        write_env(repo, label)
        patch_runner_domain_defaults(repo, label)
        patch_result_file_arg(repo, label)
        patch_eval_batch_size(repo, label)
        patch_smoke_cap(repo)
        if label == "card":
            patch_model_name_map(repo)
            patch_optional_together(repo)
            write_node_configs(repo)
            patch_runner_node_config(repo)
            patch_runner_agent_kwargs(repo)
            patch_runner_fixed_group(repo)

    print("\n" + "=" * 60)
    if problems:
        print(f"G-Designer-family shim: {len(problems)} problem(s)")
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print("G-Designer-family shim OK")



SMOKE_CAP_MARKER = "# --- shared-layer shim (agent_wf_v2) --- smoke cap v1"
SMOKE_CAP_BLOCK = """    {marker}
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
"""


def patch_smoke_cap(repo: Path) -> None:
    path = repo / "experiments" / "run_shared.py"
    if not path.exists():
        report(False, f"missing {path}")
        return
    text = path.read_text(encoding="utf-8")
    if SMOKE_CAP_MARKER in text:
        report(True, "smoke cap already installed")
        return
    anchor = "    dataset = JSONLReader.parse_file(args.dataset_json)\n"
    if anchor not in text:
        report(False, "smoke cap anchor (JSONLReader line) not found")
        return
    text = text.replace(anchor, anchor + SMOKE_CAP_BLOCK.format(marker=SMOKE_CAP_MARKER), 1)
    path.write_text(text, encoding="utf-8")
    report(SMOKE_CAP_MARKER in path.read_text(encoding="utf-8"), "smoke cap installed")


if __name__ == "__main__":
    main()
