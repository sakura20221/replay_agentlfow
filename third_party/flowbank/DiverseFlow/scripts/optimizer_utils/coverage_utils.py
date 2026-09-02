"""Coverage matrix utilities for diversity-aware workflow optimization.

Builds and maintains a per-query coverage matrix across all workflows in the pool.
Core capability: compute Weighted Score for all rounds using coverage-based weights.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from scripts.logs import logger


class CoverageUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        # {round_number: {query_idx: score}}
        self.coverage_matrix: Dict[int, Dict[int, float]] = {}
        # Cached validation queries (loaded once)
        self._validation_queries: Optional[List[dict]] = None

    # ------------------------------------------------------------------
    # Coverage Matrix
    # ------------------------------------------------------------------

    def build_coverage_matrix(self) -> Dict[int, Dict[int, float]]:
        """Scan all round_*/*.jsonl files to build per-query coverage matrix.

        Returns:
            {round_number: {query_idx: score}} where query_idx is the line index
            in the .jsonl file (consistent across rounds since they evaluate the
            same validation set).
        """
        workflows_dir = os.path.join(self.root_path, "workflows")
        self.coverage_matrix = {}

        for round_dir in sorted(os.listdir(workflows_dir)):
            if not round_dir.startswith("round_") or round_dir == "round_0":
                continue
            round_path = os.path.join(workflows_dir, round_dir)
            if not os.path.isdir(round_path):
                continue

            try:
                round_num = int(round_dir.split("_")[1])
            except (IndexError, ValueError):
                continue

            # Find the validation .jsonl file (not test_results)
            jsonl_files = [
                f for f in os.listdir(round_path)
                if f.endswith(".jsonl") and not f.startswith("test_")
            ]
            if not jsonl_files:
                continue

            # Use the latest result file (highest score or most recent)
            jsonl_files.sort(reverse=True)
            jsonl_path = os.path.join(round_path, jsonl_files[0])

            scores = {}
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        scores[idx] = float(record.get("score", 0.0))
            except Exception as e:
                logger.warning(f"Failed to read {jsonl_path}: {e}")
                continue

            if scores:
                self.coverage_matrix[round_num] = scores

        logger.info(
            f"Coverage matrix built: {len(self.coverage_matrix)} rounds, "
            f"{self._get_n_queries()} queries"
        )
        return self.coverage_matrix

    def _get_n_queries(self) -> int:
        """Get total number of queries from coverage matrix."""
        if not self.coverage_matrix:
            return 0
        return max(len(scores) for scores in self.coverage_matrix.values())

    # ------------------------------------------------------------------
    # Weighted Score (core mechanism)
    # ------------------------------------------------------------------

    def compute_all_weighted_scores(self) -> Dict[int, float]:
        """Recompute weighted scores for ALL rounds using current coverage matrix.

        weight(q) = 1 / (n_solvers(q) + 1)
        weighted_score(r) = Sigma_q score(r,q) * w(q) / Sigma_q w(q)

        This ensures all rounds are scored on the same scale with the latest weights.
        Must be called after build_coverage_matrix().

        Returns:
            {round_num: weighted_score}
        """
        if not self.coverage_matrix:
            return {}

        n_queries = self._get_n_queries()
        if n_queries == 0:
            return {}

        # Step 1: compute n_solvers(q) for each query
        n_solvers = defaultdict(int)
        for round_scores in self.coverage_matrix.values():
            for q_idx, score in round_scores.items():
                if score > 0:
                    n_solvers[q_idx] += 1

        # Step 2: compute weight(q) = 1 / (n_solvers(q) + 1)
        weights = {}
        for q_idx in range(n_queries):
            weights[q_idx] = 1.0 / (n_solvers[q_idx] + 1)

        total_weight = sum(weights.values())
        if total_weight == 0:
            return {r: 0.0 for r in self.coverage_matrix}

        # Step 3: compute weighted score for each round
        weighted_scores = {}
        for round_num, round_scores in self.coverage_matrix.items():
            ws = sum(
                round_scores.get(q_idx, 0.0) * weights[q_idx]
                for q_idx in range(n_queries)
            )
            weighted_scores[round_num] = ws / total_weight

        return weighted_scores

    def recompute_weighted_scores(self) -> Dict[int, float]:
        """Convenience: rebuild coverage matrix and compute all weighted scores."""
        self.build_coverage_matrix()
        return self.compute_all_weighted_scores()

    # ------------------------------------------------------------------
    # Weight Distribution (for prompt injection)
    # ------------------------------------------------------------------

    def get_weight_distribution(self) -> dict:
        """Compute weight distribution statistics for prompt injection.

        Returns:
            {
                "n_weight_high": count of queries with 0 solvers (weight=1.0),
                "n_weight_mid": count of queries with 1-3 solvers (weight=0.25-0.50),
                "n_weight_low": count of queries with 4+ solvers (weight<0.20),
                "n_total": total queries,
            }
        """
        if not self.coverage_matrix:
            return {"n_weight_high": 0, "n_weight_mid": 0, "n_weight_low": 0, "n_total": 0}

        n_queries = self._get_n_queries()
        n_solvers = defaultdict(int)
        for round_scores in self.coverage_matrix.values():
            for q_idx, score in round_scores.items():
                if score > 0:
                    n_solvers[q_idx] += 1

        n_high = 0  # 0 solvers: weight = 1.0
        n_mid = 0   # 1-3 solvers: weight = 0.25-0.50
        n_low = 0   # 4+ solvers: weight < 0.20
        for q_idx in range(n_queries):
            ns = n_solvers[q_idx]
            if ns == 0:
                n_high += 1
            elif ns <= 3:
                n_mid += 1
            else:
                n_low += 1

        return {
            "n_weight_high": n_high,
            "n_weight_mid": n_mid,
            "n_weight_low": n_low,
            "n_total": n_queries,
        }

    def sample_high_weight_queries(self, dataset: str, n: int = 5) -> List[str]:
        """Sample n high-weight query texts for prompt injection.

        Prioritizes completely uncovered queries (weight=1.0), then fills with
        low-coverage queries (n_solvers=1-2). This ensures useful examples even
        when uncovered queries are exhausted late in optimization.

        Args:
            dataset: Dataset name (e.g., "MATH", "GSM8K").
            n: Number of queries to sample.

        Returns:
            List of query text strings with coverage annotation.
        """
        if not self.coverage_matrix:
            return []

        n_queries = self._get_n_queries()
        n_solvers = defaultdict(int)
        for round_scores in self.coverage_matrix.values():
            for q_idx, score in round_scores.items():
                if score > 0:
                    n_solvers[q_idx] += 1

        # Sort queries by n_solvers ascending (lowest coverage first)
        sorted_queries = sorted(range(n_queries), key=lambda q: n_solvers[q])

        # Take extra candidates in case some can't be loaded
        candidate_indices = sorted_queries[:n * 2]

        queries = self.load_validation_queries(dataset)
        if not queries:
            return []

        # Shuffle candidates to avoid always showing the same ones
        random.shuffle(candidate_indices)
        sampled = []
        for idx in candidate_indices:
            if len(sampled) >= n:
                break
            if idx < len(queries):
                q = queries[idx]
                text = q.get("problem") or q.get("inputs") or q.get("question", "")
                if text:
                    ns = n_solvers[idx]
                    prefix = "[uncovered]" if ns == 0 else f"[{ns} solver{'s' if ns > 1 else ''}]"
                    if len(text) > 500:
                        text = text[:500] + "..."
                    sampled.append(f"{prefix} {text}")

        return sampled

    # ------------------------------------------------------------------
    # Existing Workflow Strengths (for prompt injection)
    # ------------------------------------------------------------------

    def get_existing_workflow_strengths(self, weighted_scores: Dict[int, float] = None,
                                        top_k: int = 5) -> str:
        """Summarize existing workflows' strengths using weighted scores."""
        if not self.coverage_matrix:
            return "No existing workflows yet."

        lines = []
        for round_num in sorted(self.coverage_matrix.keys()):
            scores = self.coverage_matrix[round_num]
            accuracy = sum(1 for s in scores.values() if s > 0) / len(scores) if scores else 0
            ws = weighted_scores.get(round_num, accuracy) if weighted_scores else accuracy
            lines.append(
                f"- round_{round_num}: accuracy={accuracy:.1%}, weighted_score={ws:.4f}"
            )

        # Sort by weighted_score descending
        if weighted_scores:
            lines.sort(
                key=lambda l: -float(l.split("weighted_score=")[1])
            )
        return "\n".join(lines[:top_k])

    # ------------------------------------------------------------------
    # Coverage Stats
    # ------------------------------------------------------------------

    def get_coverage_stats(self) -> dict:
        """Compute coverage statistics across all workflows."""
        if not self.coverage_matrix:
            return {"n_workflows": 0, "n_total": 0, "n_covered": 0, "n_uncovered": 0,
                    "coverage_rate": 0.0}

        n_queries = self._get_n_queries()
        n_covered = 0
        for q_idx in range(n_queries):
            if any(rs.get(q_idx, 0.0) > 0 for rs in self.coverage_matrix.values()):
                n_covered += 1

        return {
            "n_workflows": len(self.coverage_matrix),
            "n_total": n_queries,
            "n_covered": n_covered,
            "n_uncovered": n_queries - n_covered,
            "coverage_rate": n_covered / n_queries if n_queries > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Validation Data Loading
    # ------------------------------------------------------------------

    def load_validation_queries(self, dataset: str) -> List[dict]:
        """Load validation dataset queries for example sampling."""
        if self._validation_queries is not None:
            return self._validation_queries

        # Resolve the validation split robustly: the repo ships splits at the
        # top-level datasets/ dir. This file is
        # DiverseFlow/scripts/optimizer_utils/coverage_utils.py, so the repo root
        # is three dirs up; legacy/cwd-relative layouts are also accepted.
        here = os.path.dirname(os.path.abspath(__file__))            # .../DiverseFlow/scripts/optimizer_utils
        diverseflow_dir = os.path.dirname(os.path.dirname(here))      # .../DiverseFlow
        repo_root = os.path.dirname(diverseflow_dir)                  # .../<repo root>
        data_dir = os.path.join(os.path.dirname(os.path.dirname(self.root_path)), "data", "datasets")
        candidates = []
        for root in (os.path.join(repo_root, "datasets"),
                     os.path.join(diverseflow_dir, "datasets"),
                     data_dir, "datasets"):
            candidates.append(os.path.join(root, f"{dataset.lower()}_validate.jsonl"))
            candidates.append(os.path.join(root, f"{dataset}_validate.jsonl"))

        for path in candidates:
            if os.path.exists(path):
                queries = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            queries.append(json.loads(line))
                self._validation_queries = queries
                return queries

        logger.warning(f"Validation data not found for {dataset} in {data_dir}")
        self._validation_queries = []
        return []

    def read_round_scores(self, round_num: int) -> Dict[int, float]:
        """Read per-query scores for a specific round from .jsonl file."""
        workflows_dir = os.path.join(self.root_path, "workflows")
        round_path = os.path.join(workflows_dir, f"round_{round_num}")

        if not os.path.isdir(round_path):
            return {}

        jsonl_files = [
            f for f in os.listdir(round_path)
            if f.endswith(".jsonl") and not f.startswith("test_")
        ]
        if not jsonl_files:
            return {}

        jsonl_files.sort(reverse=True)
        jsonl_path = os.path.join(round_path, jsonl_files[0])

        scores = {}
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    scores[idx] = float(record.get("score", 0.0))
        except Exception as e:
            logger.warning(f"Failed to read scores for round {round_num}: {e}")

        return scores
