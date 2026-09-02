import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from scripts.logs import logger


class VisualizationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    # ------------------------------------------------------------------
    # 1. MCT Tree from processed_experience.json
    # ------------------------------------------------------------------
    def plot_mct_tree(self, output_dir: Optional[str] = None):
        """Visualize the Monte-Carlo Tree from processed_experience.json."""
        exp_path = os.path.join(self.root_path, "workflows", "processed_experience.json")
        if not os.path.exists(exp_path):
            logger.warning(f"processed_experience.json not found at {exp_path}, skipping MCT tree plot.")
            return

        with open(exp_path, "r", encoding="utf-8") as f:
            experience = json.load(f)

        G = nx.DiGraph()

        # Build nodes and edges
        for parent_round_str, data in experience.items():
            parent_round = int(parent_round_str)
            parent_score = data["score"]
            if parent_round not in G:
                G.add_node(parent_round, score=parent_score, node_type="normal")

            for child_round_str, child_data in data["success"].items():
                child_round = int(child_round_str)
                child_score = child_data["score"]
                mod = child_data["modification"]
                G.add_node(child_round, score=child_score, node_type="success")
                G.add_edge(parent_round, child_round, label=self._shorten(mod), edge_type="success")

            for child_round_str, child_data in data["failure"].items():
                child_round = int(child_round_str)
                child_score = child_data["score"]
                mod = child_data["modification"]
                G.add_node(child_round, score=child_score, node_type="failure")
                G.add_edge(parent_round, child_round, label=self._shorten(mod), edge_type="failure")

        if len(G.nodes) == 0:
            logger.warning("No nodes to plot in MCT tree.")
            return

        # Layout
        pos = self._tree_layout(G)

        fig, ax = plt.subplots(1, 1, figsize=(max(14, len(G.nodes) * 1.2), max(8, len(G.nodes) * 0.5)))

        # Draw edges
        success_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "success"]
        failure_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "failure"]

        nx.draw_networkx_edges(G, pos, edgelist=success_edges, edge_color="#2ecc71", width=2.0,
                               arrows=True, arrowsize=15, ax=ax, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edges(G, pos, edgelist=failure_edges, edge_color="#e74c3c", width=1.5,
                               arrows=True, arrowsize=15, ax=ax, style="dashed", connectionstyle="arc3,rad=0.1")

        # Node colors based on score
        scores = [G.nodes[n].get("score", 0) or 0 for n in G.nodes]
        min_score = min(s for s in scores if s > 0) if any(s > 0 for s in scores) else 0
        max_score = max(scores) if scores else 1

        node_colors = []
        for n in G.nodes:
            s = G.nodes[n].get("score", 0)
            if s == 0:
                node_colors.append("#bdc3c7")  # grey for zero/error
            else:
                # Normalize to [0.3, 1.0] for color intensity
                intensity = 0.3 + 0.7 * (s - min_score) / (max_score - min_score + 1e-9)
                node_colors.append(plt.cm.Blues(intensity))

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors,
                               edgecolors="black", linewidths=1.5, ax=ax)

        # Labels: "R{round}\n{score:.2f}"
        labels = {n: f"R{n}\n{G.nodes[n].get('score', 0):.3f}" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight="bold", ax=ax)

        # Edge labels (shortened modification)
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5,
                                     font_color="#555555", ax=ax)

        # Legend
        success_patch = mpatches.Patch(color="#2ecc71", label="Success (score improved)")
        failure_patch = mpatches.Patch(color="#e74c3c", label="Failure (score not improved)")
        ax.legend(handles=[success_patch, failure_patch], loc="upper left", fontsize=9)

        ax.set_title("MCT Tree — Optimization History", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        save_dir = output_dir or os.path.join(self.root_path, "workflows")
        save_path = os.path.join(save_dir, "mct_tree.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"MCT tree saved to {save_path}")

    # ------------------------------------------------------------------
    # 2. Performance & Cost from results.json
    # ------------------------------------------------------------------
    def plot_performance_cost(self, output_dir: Optional[str] = None):
        """Visualize per-round performance and cost from results.json.

        If weighted_score is available in results.json, plots an additional
        'Weighted Score (final)' line showing coverage-weighted performance.
        """
        results_path = os.path.join(self.root_path, "workflows", "results.json")
        if not os.path.exists(results_path):
            logger.warning(f"results.json not found at {results_path}, skipping performance plot.")
            return

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            logger.warning("results.json is empty, skipping performance plot.")
            return

        rounds = [r["round"] for r in results]
        scores = [r["score"] for r in results]
        avg_costs = [r["avg_cost"] for r in results]

        # Check if weighted_score is available
        weighted_scores = [r.get("weighted_score") for r in results]
        has_weighted = any(ws is not None for ws in weighted_scores)

        # Compute running best score
        best_scores = []
        current_best = 0
        for s in scores:
            current_best = max(current_best, s)
            best_scores.append(current_best)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # --- Top: Score ---
        ax1.plot(rounds, scores, "o-", color="#3498db", linewidth=1.5, markersize=5, label="Raw Accuracy", alpha=0.7)
        ax1.plot(rounds, best_scores, "s--", color="#e74c3c", linewidth=2, markersize=4, label="Best Accuracy So Far")

        # Weighted score line (if available)
        if has_weighted:
            ws_valid = [(r, ws) for r, ws in zip(rounds, weighted_scores) if ws is not None]
            if ws_valid:
                ws_rounds, ws_scores = zip(*ws_valid)
                ax1.plot(ws_rounds, ws_scores, "^-", color="#9b59b6", linewidth=1.5, markersize=5,
                         label="Weighted Score (final)", alpha=0.8)
                # Annotate best weighted score
                best_ws_idx = ws_scores.index(max(ws_scores))
                ax1.annotate(
                    f"Best WS: {ws_scores[best_ws_idx]:.3f}\n(Round {ws_rounds[best_ws_idx]})",
                    xy=(ws_rounds[best_ws_idx], ws_scores[best_ws_idx]),
                    xytext=(ws_rounds[best_ws_idx] + 0.5, ws_scores[best_ws_idx] + 0.05),
                    arrowprops=dict(arrowstyle="->", color="#9b59b6"),
                    fontsize=8, color="#9b59b6",
                )

        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_title("Optimization Performance per Round", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Annotate best raw score
        best_idx = scores.index(max(scores))
        ax1.annotate(
            f"Best: {scores[best_idx]:.3f}\n(Round {rounds[best_idx]})",
            xy=(rounds[best_idx], scores[best_idx]),
            xytext=(rounds[best_idx] + 0.5, scores[best_idx] - 0.08),
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
            fontsize=8, color="#e74c3c",
        )

        # --- Bottom: Cost ---
        ax2.bar(rounds, avg_costs, color="#2ecc71", alpha=0.7, label="Avg Cost per Problem")
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Avg Cost ($)", fontsize=12)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_title("Cost per Round", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xticks(rounds)

        plt.tight_layout()

        save_dir = output_dir or os.path.join(self.root_path, "workflows")
        save_path = os.path.join(save_dir, "performance_cost.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Performance & cost chart saved to {save_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _shorten(text: str, max_len: int = 30) -> str:
        """Shorten text for edge labels."""
        text = text.strip()
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    @staticmethod
    def _tree_layout(G: nx.DiGraph):
        """Compute a top-down tree layout. Roots are nodes with no incoming edges."""
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            roots = [min(G.nodes)]

        pos = {}
        visited = set()

        def bfs_levels(root):
            """BFS from root, returns dict: level -> list of nodes."""
            levels = {}
            queue = [(root, 0)]
            seen = {root}
            while queue:
                node, level = queue.pop(0)
                levels.setdefault(level, []).append(node)
                for child in sorted(G.successors(node)):
                    if child not in seen:
                        seen.add(child)
                        queue.append((child, level + 1))
            return levels, seen

        x_offset = 0
        for root in sorted(roots):
            if root in visited:
                continue
            levels, seen = bfs_levels(root)
            visited |= seen

            max_width = max(len(nodes) for nodes in levels.values())
            for level, nodes in levels.items():
                for i, node in enumerate(nodes):
                    x = x_offset + (i - len(nodes) / 2) * 2.5
                    y = -level * 2.0
                    pos[node] = (x, y)
            x_offset += max_width * 2.5 + 2

        # Place any remaining unvisited nodes
        for node in G.nodes:
            if node not in pos:
                pos[node] = (x_offset, 0)
                x_offset += 2

        return pos
