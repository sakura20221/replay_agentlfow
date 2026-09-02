import json
import os
from collections import defaultdict

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file

class ExperienceUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def load_experience(self, path=None, mode: str = "Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        experience_data = defaultdict(lambda: {
            "score": None, "weighted_score": None,
            "success": {}, "failure": {},
        })

        for round_dir in os.listdir(rounds_dir):
            if os.path.isdir(os.path.join(rounds_dir, round_dir)) and round_dir.startswith("round_"):
                round_path = os.path.join(rounds_dir, round_dir)
                try:
                    round_number = int(round_dir.split("_")[1])
                    json_file_path = os.path.join(round_path, "experience.json")
                    if os.path.exists(json_file_path):
                        data = read_json_file(json_file_path, encoding="utf-8")
                        father_node = data["father node"]

                        # Parent scores (first time we see this parent)
                        # Compatible with both old AFlow format (before/after)
                        # and new DiverseFlow format (score_before/score_after)
                        # Use explicit None check: 0.0 is a valid score, `or` would skip it
                        def _get(d, new_key, old_key):
                            v = d.get(new_key)
                            return v if v is not None else d.get(old_key)

                        if experience_data[father_node]["score"] is None:
                            experience_data[father_node]["score"] = _get(data, "score_before", "before")
                        if experience_data[father_node]["weighted_score"] is None:
                            experience_data[father_node]["weighted_score"] = data.get("weighted_score_before")

                        entry = {
                            "modification": data["modification"],
                            "score": _get(data, "score_after", "after"),
                            "weighted_score": data.get("weighted_score_after"),
                        }

                        if data["succeed"]:
                            experience_data[father_node]["success"][round_number] = entry
                        else:
                            experience_data[father_node]["failure"][round_number] = entry
                except Exception as e:
                    logger.info(f"Error processing {round_dir}: {str(e)}")

        experience_data = dict(experience_data)

        output_path = os.path.join(rounds_dir, "processed_experience.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(experience_data, outfile, indent=4, ensure_ascii=False)

        logger.info(f"Processed experience data saved to {output_path}")
        return experience_data

    def format_experience(self, processed_experience, sample_round, use_weighted: bool = False):
        """Format experience for prompt injection.

        Args:
            use_weighted: If True (Phase 2), display weighted scores instead of raw scores.
        """
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            score_label = "Weighted Score" if use_weighted else "Score"
            if use_weighted and experience_data.get("weighted_score") is not None:
                parent_score = experience_data["weighted_score"]
            else:
                parent_score = experience_data["score"]
            experience = f"Original {score_label}: {parent_score}\n"
            experience += "These are some conclusions drawn from experience:\n\n"

            for key, value in experience_data["failure"].items():
                if use_weighted and value.get("weighted_score") is not None:
                    display_score = value["weighted_score"]
                else:
                    display_score = value["score"]
                experience += f"-Absolutely prohibit {value['modification']} ({score_label}: {display_score})\n"
            for key, value in experience_data["success"].items():
                experience += f"-Absolutely prohibit {value['modification']} \n"

            experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."
        else:
            experience = f"No experience data found for round {sample_round}."
        return experience

    def check_modification(self, processed_experience, modification, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            for key, value in experience_data["failure"].items():
                if value["modification"] == modification:
                    return False
            for key, value in experience_data["success"].items():
                if value["modification"] == modification:
                    return False
            return True
        else:
            return True

    def get_past_modifications(self, processed_experience, sample_round):
        """Get all past modification descriptions for a given parent round."""
        modifications = []
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            for value in experience_data["failure"].values():
                modifications.append(value["modification"])
            for value in experience_data["success"].values():
                modifications.append(value["modification"])
        return modifications

    def create_experience_data(self, sample, modification):
        """Create initial experience data. Scores filled in by update_experience()."""
        return {
            "father node": sample["round"],
            "modification": modification,
            "score_before": None,
            "score_after": None,
            "weighted_score_before": None,
            "weighted_score_after": None,
            "succeed": None,
        }

    def update_experience(self, directory, experience, avg_score,
                          parent_raw_score: float = None,
                          weighted_scores: dict = None,
                          parent_round: int = None, child_round: int = None,
                          use_weighted_for_success: bool = False):
        """Update experience with both raw and weighted scores.

        Always records both score types. Success/fail criterion depends on phase:
        - Phase 1 (use_weighted_for_success=False): raw accuracy comparison
        - Phase 2 (use_weighted_for_success=True): weighted score comparison

        Args:
            directory: Round directory path.
            experience: Experience dict to update.
            avg_score: Raw accuracy score on validation set.
            parent_raw_score: Parent's raw accuracy score.
            weighted_scores: Dict {round: weighted_score} from global recomputation.
            parent_round: Parent's round number (for weighted_scores lookup).
            child_round: Child's round number (for weighted_scores lookup).
            use_weighted_for_success: If True, use weighted scores for success/fail.
        """
        # Always record raw scores
        experience["score_before"] = parent_raw_score
        experience["score_after"] = avg_score

        # Record weighted scores (available for both phases after global recomputation)
        parent_ws = None
        child_ws = None
        if weighted_scores:
            parent_ws = weighted_scores.get(parent_round)
            child_ws = weighted_scores.get(child_round)
        experience["weighted_score_before"] = parent_ws
        experience["weighted_score_after"] = child_ws

        # Success/fail criterion
        if use_weighted_for_success and parent_ws is not None and child_ws is not None:
            experience["succeed"] = bool(child_ws > parent_ws)
            logger.info(
                f"[Phase 2] weighted: {child_ws:.4f} vs {parent_ws:.4f} → "
                f"{'success' if experience['succeed'] else 'failure'} "
                f"(raw: {avg_score:.4f} vs {parent_raw_score:.4f})"
            )
        else:
            experience["succeed"] = bool(avg_score > (parent_raw_score or 0))

        write_json_file(os.path.join(directory, "experience.json"), experience, encoding="utf-8", indent=4)
