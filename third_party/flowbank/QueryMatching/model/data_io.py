"""Selector-data loader: read either a compact ``.npz`` or a verbose ``.csv``.

The selector (``multi_task_graph_selector.py``) historically called
``pd.read_csv`` on a ``selector_data.csv`` whose embedding columns are JSON text.
This module returns the *same* DataFrame from either:

* ``selector_data.npz``  — the compact lossless artifact shipped in this repo
  (embeddings stored once as float64; see ``scripts/make_slim_data.py``), or
* ``selector_data.csv``  — the verbose format produced by
  ``data_processing/build_selector_data.py`` when you run the full pipeline.

Downstream code (``prepare_data_for_GNN``, ``form_data``, the GNN, ``test``)
is untouched: it sees an identical DataFrame, so results are bit-for-bit equal
whether the data came from ``.npz`` or ``.csv``. For the ``.npz`` path the
embedding cells are returned as already-parsed nested lists ``[[...]]``; the
selector's ``_parse_embedding`` passes non-strings through unchanged.
"""
import os

import numpy as np
import pandas as pd


# Candidate filenames, in priority order. A freshly pipeline-generated CSV wins
# over a shipped npz so re-running the pipeline takes effect without renaming.
_CANDIDATES = ("selector_data.csv", "selector_data.npz")


def resolve_data_path(path):
    """Resolve ``path`` (a file or directory) to an existing data file.

    If ``path`` exists, use it. Otherwise look in its directory (or in ``path``
    itself if it is a directory) for the first existing candidate filename.
    """
    if os.path.isfile(path):
        return path
    search_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
    for name in _CANDIDATES:
        cand = os.path.join(search_dir, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"No selector data found at '{path}'. Looked for {_CANDIDATES} in {search_dir}/"
    )


def _df_from_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    num_llms = int(z["num_llms"])
    num_query = int(z["num_query"])
    query_emb = z["query_emb"]                 # (num_query, dim) float64
    task_emb = z["task_emb"]                    # (dim,) or (num_query, dim) float64
    num_rows = num_query * num_llms

    # Rebuild the per-row embedding cells as nested lists [[...]], reusing one
    # object per query/task so memory stays modest. _parse_embedding returns
    # these unchanged and the selector takes cell[0] to get the 1536-vector.
    q_cells_by_query = [[query_emb[q]] for q in range(num_query)]
    if task_emb.ndim == 1:
        t_cell = [task_emb]
        task_cells = [t_cell] * num_rows
    else:
        t_cells_by_query = [[task_emb[q]] for q in range(num_query)]
        task_cells = [t_cells_by_query[i // num_llms] for i in range(num_rows)]
    query_cells = [q_cells_by_query[i // num_llms] for i in range(num_rows)]

    # query text may be stored per-row (make_slim_data) or per-query
    # (build_selector_data); expand the per-query case to per-row.
    if "query_text" in z:
        qt = z["query_text"].astype(str)
        query_col = ([qt[i // num_llms] for i in range(num_rows)]
                     if len(qt) == num_query else list(qt))
    else:
        query_col = [""] * num_rows

    data = {
        "task_id": z["task_id"].astype(str) if "task_id" in z else np.array([""] * num_rows),
        "query": query_col,
        "query_embedding": query_cells,
        "task_description_embedding": task_cells,
        "effect": z["effect"].astype(float),
        "cost": z["cost"].astype(float),
        "raw_cost": z["raw_cost"].astype(float) if "raw_cost" in z else z["cost"].astype(float),
        "llm": z["llm"].astype(str) if "llm" in z else np.array([""] * num_rows),
        "split": z["split"].astype(str) if "split" in z else np.array([""] * num_rows),
    }
    if "metric" in z:
        data["metric"] = z["metric"].astype(str)
    return pd.DataFrame(data)


def load_selector_dataframe(path):
    """Load selector data from ``.npz`` or ``.csv`` into a DataFrame."""
    resolved = resolve_data_path(path)
    if resolved.endswith(".npz"):
        return _df_from_npz(resolved)
    return pd.read_csv(resolved)
