"""Generate workflow descriptions by LLM-summarizing each workflow's source.

The QueryMatching selector represents every workflow as a graph node whose
feature vector is the embedding of a free-text *description* of that workflow
(see ``build_selector_data.py``: descriptions -> text-embedding-3-small ->
workflow_embeddings.pkl). Without descriptions the builder falls back to the
bare label string, so the workflow node features carry no semantics.

This script closes that gap the way the paper's shipped examples were made:
for each selected workflow it reads the generated ``graph.py`` + ``prompt.py``
(and the optimizer's per-round ``experience.json`` "modification" note as extra
context) and asks an LLM to summarize the workflow's architecture, strategy,
rough LLM-call/cost profile, and what query types it suits — one prose paragraph
in the style of ``data/example/<bench>/workflow_descriptions.json``.

The output is a ``{label: {"feature": description}}`` JSON that feeds directly
into ``build_selector_data.py --descriptions``.

Model / endpoint
----------------
Any OpenAI-compatible chat endpoint. The base_url / api_key / temperature are
resolved with precedence: explicit CLI flag > --config model entry > environment
/ default. The simplest way to point at a local vLLM Qwen — reusing the SAME
config.yaml DiverseFlow uses, WITHOUT touching the OpenAI creds the embedding step
needs — is ``--config``:

    python data_processing/describe_workflows.py --dataset MBPP \
        --model "Qwen/Qwen3-8B" --config DiverseFlow/config/config.yaml \
        --workflow Flow_5 DiverseFlow/workspace/MBPP/workflows/round_5 \
        --out runs/mbpp_descriptions.json

Equivalently, pass the endpoint explicitly:
``--model "Qwen/Qwen3-8B" --base-url http://localhost:8002/v1 --api-key EMPTY``.
With neither, it falls back to $OPENAI_BASE_URL / $OPENAI_API_KEY and gpt-4o-mini.

The request-level temperature is sent in the call, so it overrides any server
default (default 0.2 unless set via --config/--temperature). Any
``<think>...</think>`` reasoning block (Qwen3 and other thinking models emit one
before the answer) is stripped, so only the description paragraph is kept.

Usage
-----
    # one --workflow LABEL ROUND_DIR per selected workflow (same labels you
    # will pass to build_selector_data / aggregate_round_scores)
    python data_processing/describe_workflows.py --dataset MBPP \
        --workflow Flow_5 DiverseFlow/workspace/MBPP/workflows/round_5 \
        --workflow Flow_6 DiverseFlow/workspace/MBPP/workflows/round_6 \
        --out runs/mbpp_descriptions.json

    # then feed it in:
    python data_processing/build_selector_data.py \
        --train-scores runs/mbpp_train/sources.json --train-queries runs/mbpp_train/queries.json \
        --test-scores  runs/mbpp_test/sources.json  --test-queries  runs/mbpp_test/queries.json \
        --descriptions runs/mbpp_descriptions.json \
        --task-id MBPP --embedding-backend openai --out-dir data/mbpp_full

Use ``--dry-run`` to assemble and inspect the prompts without calling the API
(writes the prompt itself as each "feature", so the JSON shape can be checked
offline).
"""
import argparse
import json
import os
import re

import yaml

# Bound the source we send per workflow (generated graphs are small; this is a
# safety cap for unusually large ones).
_MAX_SRC_CHARS = 12000

SYSTEM_PROMPT = (
    "You are documenting agentic LLM workflows for a research codebase. "
    "Given a workflow's Python source, write ONE dense paragraph (roughly "
    "110-180 words) describing it, matching this style:\n"
    "  - the architecture: which operators it uses and the control flow "
    "(sequence, branching, loops, ensembling, refinement);\n"
    "  - the problem-solving strategy and what makes it distinctive;\n"
    "  - its rough LLM-call / cost profile (e.g. fixed vs variable, worst-case "
    "number of calls);\n"
    "  - what query types it suits or struggles with.\n"
    "Write plain prose only: no markdown, no headings, no bullet points, no "
    "code. Describe what the workflow DOES, not the Python syntax. Output only "
    "the paragraph."
)


def read_workflow_files(round_dir):
    """Return (graph_src, prompt_src, modification) for a round directory."""
    def _read(name):
        p = os.path.join(round_dir, name)
        if not os.path.exists(p):
            return ""
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    graph_src = _read("graph.py")
    prompt_src = _read("prompt.py")
    modification = ""
    exp_path = os.path.join(round_dir, "experience.json")
    if os.path.exists(exp_path):
        try:
            with open(exp_path, "r", encoding="utf-8") as f:
                modification = (json.load(f) or {}).get("modification", "") or ""
        except (json.JSONDecodeError, OSError):
            modification = ""
    return graph_src, prompt_src, modification


def build_summary_prompt(label, dataset, graph_src, prompt_src, modification):
    """Assemble the user prompt sent to the chat model for one workflow."""
    graph_src = (graph_src or "")[:_MAX_SRC_CHARS]
    prompt_src = (prompt_src or "")[:_MAX_SRC_CHARS]
    parts = [f"Workflow label: {label}"]
    if dataset:
        parts.append(f"Task / dataset: {dataset}")
    if modification:
        parts.append(f"Optimizer note (change vs its parent): {modification}")
    parts.append("\n--- graph.py ---\n" + graph_src)
    if prompt_src.strip():
        parts.append("\n--- prompt.py (custom prompts) ---\n" + prompt_src)
    parts.append("\nWrite the one-paragraph description now.")
    return "\n".join(parts)


def _strip_think(text):
    """Drop a leading chain-of-thought block (Qwen3 etc. emit <think>...</think>
    before the answer). No-op for models that don't think."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def resolve_endpoint(args, ap):
    """Resolve (base_url, api_key, temperature) with precedence:
    explicit CLI flag > --config model entry > environment / default.

    --config points at a DiverseFlow-style YAML with a top-level ``models:`` map
    (the same config.yaml DiverseFlow uses); the entry for ``--model`` supplies
    base_url / api_key / temperature, so the writer is configured in one place.
    """
    cfg_model = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        models = cfg.get("models", {}) or {}
        if args.model not in models:
            ap.error(f"--model {args.model!r} not found in {args.config} "
                     f"(models: {', '.join(models) or 'none'})")
        cfg_model = models[args.model] or {}

    base_url = args.base_url or cfg_model.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or cfg_model.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    if args.temperature is not None:
        temperature = args.temperature
    elif "temperature" in cfg_model:
        temperature = cfg_model["temperature"]
    else:
        temperature = 0.2
    return base_url, api_key, temperature


def summarize_chat(prompt, model, base_url, api_key, temperature):
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return _strip_think(resp.choices[0].message.content.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", nargs=2, action="append", required=True,
                    metavar=("LABEL", "ROUND_DIR"),
                    help="workflow label and its generated round directory (repeatable)")
    ap.add_argument("--out", required=True, help="output descriptions JSON")
    ap.add_argument("--dataset", default="", help="task/dataset name, for context")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="chat model (OpenAI-compatible); default gpt-4o-mini. "
                         "For a local vLLM server use e.g. 'Qwen/Qwen3-8B'.")
    ap.add_argument("--config",
                    help="DiverseFlow-style YAML with a 'models:' map; the entry for "
                         "--model supplies base_url/api_key/temperature "
                         "(e.g. DiverseFlow/config/config.yaml).")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL. Overrides --config; "
                         "falls back to $OPENAI_BASE_URL. Local vLLM Qwen: http://localhost:8002/v1")
    ap.add_argument("--api-key", default=None,
                    help="API key. Overrides --config; falls back to $OPENAI_API_KEY, "
                         "else 'EMPTY' for local servers.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature (sent in the request, so it overrides any "
                         "server default). Overrides --config; default 0.2.")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble prompts but do NOT call the API; write the "
                         "prompt as each feature (for offline inspection)")
    args = ap.parse_args()

    base_url, api_key, temperature = resolve_endpoint(args, ap)
    if not args.dry_run:
        print(f"writer: model={args.model}  base_url={base_url or 'default OpenAI'}  "
              f"temperature={temperature}")

    descriptions = {}
    for label, round_dir in args.workflow:
        if not os.path.isdir(round_dir):
            raise FileNotFoundError(f"{label}: round dir not found: {round_dir}")
        graph_src, prompt_src, modification = read_workflow_files(round_dir)
        if not graph_src.strip():
            raise FileNotFoundError(f"{label}: no graph.py found in {round_dir}")
        prompt = build_summary_prompt(label, args.dataset, graph_src, prompt_src, modification)
        if args.dry_run:
            feature = "[DRY-RUN PROMPT]\n" + prompt
        else:
            feature = summarize_chat(prompt, args.model, base_url, api_key, temperature)
        descriptions[label] = {"feature": feature}
        n_words = len(feature.split())
        print(f"{label}: {n_words} words" + ("  (dry-run)" if args.dry_run else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(descriptions)} descriptions -> {args.out}")
    print("  next: build_selector_data.py ... --descriptions " + args.out)


if __name__ == "__main__":
    main()
