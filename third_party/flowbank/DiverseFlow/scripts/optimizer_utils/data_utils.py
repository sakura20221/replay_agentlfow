import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []

    def load_results(self, path: str) -> list:
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    def get_top_rounds(self, sample: int, path=None, mode="Graph", use_weighted: bool = False):
        self._load_scores(path, mode, use_weighted=use_weighted)
        unique_rounds = set()
        unique_top_scores = []

        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items):
        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round, path=None, mode: str = "Graph"):
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, "workflows", f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        if not os.path.exists(log_dir):
            return ""
        logger.info(log_dir)
        data = read_json_file(log_dir, encoding="utf-8")

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        max_total_chars = 40000
        drop_fields = {"extract_answer_code"}
        max_output_chars = 8000

        log = ""
        for sample in random_samples:
            cleaned = {}
            for k, v in sample.items():
                if k in drop_fields:
                    continue
                v_str = str(v)
                if k == "model_output" and len(v_str) > max_output_chars:
                    head_len = max_output_chars // 2
                    tail_len = max_output_chars // 2
                    v_str = v_str[:head_len] + "\n... [middle truncated] ...\n" + v_str[-tail_len:]
                cleaned[k] = v_str
            entry = json.dumps(cleaned, indent=4, ensure_ascii=False) + "\n\n"
            if len(log) + len(entry) > max_total_chars:
                break
            log += entry

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float) -> dict:
        now = datetime.datetime.now()
        return {"round": round, "score": score, "weighted_score": None,
                "avg_cost": avg_cost, "total_cost": total_cost, "time": now}

    def save_results(self, json_file_path: str, data: list):
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def update_all_weighted_scores(self, weighted_scores: dict):
        """Update weighted_score field for ALL entries in results.json.

        Called after each evaluation to keep all rounds on the same weight scale.
        """
        results_path = os.path.join(self.root_path, "workflows", "results.json")
        if not os.path.exists(results_path):
            return

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            r = entry.get("round")
            entry["weighted_score"] = weighted_scores.get(r)

        write_json_file(results_path, data, encoding="utf-8", indent=4)
        logger.info(f"Updated weighted_score for {len(data)} entries in results.json")

    def _load_scores(self, path=None, mode="Graph", use_weighted: bool = False):
        """Load scores from results.json for parent selection ranking.

        Args:
            use_weighted: If True (Phase 2), rank by weighted_score instead of raw score.
        """
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = read_json_file(result_file, encoding="utf-8")
        if not data:
            return self.top_scores
        df = pd.DataFrame(data)
        if "round" not in df.columns:
            return self.top_scores

        # Choose which score field to use for ranking
        score_field = "score"
        if use_weighted and "weighted_score" in df.columns:
            # Check if weighted_score is available and non-null
            if df["weighted_score"].notna().any():
                score_field = "weighted_score"
                # Fill NaN with 0 for rounds that don't have weighted_score yet
                df["weighted_score"] = df["weighted_score"].fillna(0.0)

        scores_per_round = df.groupby("round")[score_field].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores
