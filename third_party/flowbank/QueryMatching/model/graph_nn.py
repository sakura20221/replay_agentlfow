import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim*2)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self,task_id, query_features, llm_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_two_features=torch.cat([aligned_task_features,aligned_query_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_features = torch.cat([aligned_two_features, aligned_llm_features], 0)
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features * 2)
        self.bn2 = nn.BatchNorm1d(hidden_features * 2)

    def forward(self, task_id, query_features, llm_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):
        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]
        edge_weight_mask=F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1,self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features))
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        raw_scores = (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1)
        return F.sigmoid(raw_scores)

class form_data:

    def __init__(self,device):
        self.device = device

    def formulation(self,task_id,query_feature,llm_feature,org_node,des_node,edge_feature,label,edge_mask,combined_edge,train_mask,valide_mask,test_mask):

        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        task_id=torch.tensor(task_id, dtype=torch.float).to(self.device)
        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]
        des_node=[(i+1 + org_node[-1]) for i in des_node]
        edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(self.device)
        edge_weight = torch.tensor(edge_feature, dtype=torch.float).reshape(-1,1).to(self.device)
        combined_edge=torch.tensor(combined_edge, dtype=torch.float).reshape(-1,2).to(self.device)
        combined_edge=torch.cat((edge_weight, combined_edge), dim=-1)
        data = Data(task_id=task_id,query_features=query_features, llm_features=llm_features, edge_index=edge_index,
                        edge_attr=edge_weight,query_indices=query_indices, llm_indices=llm_indices,label=torch.tensor(label, dtype=torch.float).to(self.device),
                        edge_mask=edge_mask,combined_edge=combined_edge,
                    train_mask=train_mask,valide_mask=valide_mask,test_mask=test_mask)

        return data


class GNN_prediction:
    def __init__(self, query_feature_dim, llm_feature_dim,hidden_features_size,in_edges_size,wandb,config,device):
        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim, llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,in_edges=in_edges_size).to(device)
        self.wandb = wandb
        self.config = config
        self.device = device
        self.optimizer =AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        # The selector is trained with a per-edge BCE loss against the (cost-weighted)
        # effect label; the model outputs sigmoid probabilities.
        self.criterion = torch.nn.BCELoss()

    def train_validate(self,data,data_validate,data_for_test,test_task_ids=None):
        self.test_task_ids = test_task_ids
        self.save_path= self.config['model_path']
        self.num_edges = len(data.edge_attr)
        self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)

        # Check if val set exists
        has_val = self.valide_mask.any()
        # Both val_selector_effect and test_result are higher-better, so use -1 baseline either way.
        best_metric = -1

        # Setup training log
        log_path = self.config.get('train_log_path', 'model_path/training_log.csv')
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        log_file = open(log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['epoch', 'train_loss', 'train_result', 'train_golden',
                             'val_loss', 'val_accuracy', 'val_f1', 'val_result', 'val_golden',
                             'test_loss', 'test_result', 'test_golden'])

        for epoch in range(self.config['train_epoch']):
            self.model.train()
            loss_mean=0
            mask_train = data.edge_mask
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone()
                mask = mask.bool()
                # Randomly hide a fraction of train edges each step (edge-level masking).
                random_mask = torch.rand(mask.size()) < self.config['train_mask_rate']
                random_mask = random_mask.to(torch.bool)
                mask = torch.where(mask & random_mask, torch.tensor(False, dtype=torch.bool), mask)
                mask = mask.bool()
                edge_can_see = torch.logical_and(~mask, self.train_mask)
                self.optimizer.zero_grad()
                predicted_edges= self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                                            edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean+=loss
            loss_mean=loss_mean/self.config['batch_size']
            loss_mean.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # Training set accuracy: use all train edges for message passing,
                # predict on train edges, compute avg effect of selector's picks
                train_edge_can_see = self.train_mask.clone()
                predicted_edges_train = self.model(task_id=data.task_id, query_features=data.query_features,
                                                    llm_features=data.llm_features, edge_index=data.edge_index,
                                                    edge_mask=self.train_mask, edge_can_see=train_edge_can_see,
                                                    edge_weight=data.combined_edge)
                pred_train = predicted_edges_train.reshape(-1, self.config['llm_num'])
                train_max_idx = torch.argmax(pred_train, 1)
                value_train = data.edge_attr[self.train_mask].reshape(-1, self.config['llm_num'])
                train_label_idx = torch.argmax(value_train, 1)
                train_row_indices = torch.arange(len(value_train))
                train_result = value_train[train_row_indices, train_max_idx].mean()
                train_golden = value_train[train_row_indices, train_label_idx].mean()

                # Validate metrics (if val set exists)
                validate_accuracy = 0.0
                f1 = 0.0
                loss_validate_val = 0.0
                val_result = 0.0    # avg effect of selector's pick on val (analog of test_result)
                val_golden = 0.0    # oracle on val (analog of test_golden)
                if has_val:
                    mask_validate = torch.tensor(data_validate.edge_mask, dtype=torch.bool)
                    edge_can_see = self.train_mask
                    predicted_edges_validate = self.model(task_id=data_validate.task_id,query_features=data_validate.query_features,
                                                                                llm_features=data_validate.llm_features,
                                                                                edge_index=data_validate.edge_index,
                                                                                edge_mask=mask_validate,edge_can_see=edge_can_see, edge_weight=data_validate.combined_edge)
                    observe_edge= predicted_edges_validate.reshape(-1, self.config['llm_num'])
                    observe_idx = torch.argmax(observe_edge, 1)
                    value_validate=data_validate.edge_attr[mask_validate].reshape(-1, self.config['llm_num'])
                    label_idx = torch.argmax(value_validate, 1)
                    correct = (observe_idx == label_idx).sum().item()
                    total = label_idx.size(0)
                    validate_accuracy = correct / total
                    observe_idx_ = observe_idx.cpu().numpy()
                    label_idx_ = label_idx.cpu().numpy()
                    f1 = f1_score(label_idx_, observe_idx_, average='macro')
                    loss_validate_val = self.criterion(predicted_edges_validate.reshape(-1), data_validate.label[mask_validate].reshape(-1))
                    # Selector-effect style val metrics: mean of actual effect of selector's pick
                    val_row_indices = torch.arange(len(value_validate))
                    val_result = value_validate[val_row_indices, observe_idx].mean().item()
                    val_golden = value_validate[val_row_indices, label_idx].mean().item()

                # Test
                test_result,test_loss=self.test(data_for_test,self.config['model_path'])

                # Model selection: if val set exists, use val_result (selector-effect on val,
                # higher=better, same metric as test_result). Otherwise fall back to test_result
                # (potential data leakage but no alternative without val set).
                if has_val:
                    select_metric = val_result
                else:
                    select_metric = test_result.item() if hasattr(test_result, 'item') else test_result
                if select_metric > best_metric:
                    best_metric = select_metric
                    torch.save(self.model.state_dict(), self.save_path)

                self.wandb.log({"train_loss":loss_mean, "train_result": train_result, "train_golden": train_golden,
                                "validate_loss": loss_validate_val,"test_loss":test_loss,
                                "validate_accuracy": validate_accuracy,"validate_f1": f1,
                                "val_result": val_result, "val_golden": val_golden,
                                "test_result": test_result})

                # Log to CSV
                log_writer.writerow([
                    epoch,
                    f"{loss_mean.item():.6f}",
                    f"{train_result.item():.4f}",
                    f"{train_golden.item():.4f}",
                    f"{loss_validate_val.item() if has_val else 0.0:.6f}",
                    f"{validate_accuracy:.4f}",
                    f"{f1:.4f}",
                    f"{val_result:.4f}",
                    f"{val_golden:.4f}",
                    f"{test_loss.item():.6f}",
                    f"{test_result.item():.4f}",
                    f"{self._last_test_golden.item():.4f}"
                ])
                log_file.flush()

        # Close training log
        log_file.close()
        print(f"Training log saved to {log_path}")

        # Final test with best model, save per-query predictions
        self.model.load_state_dict(torch.load(self.save_path))
        self.test(data_for_test, self.save_path, save_predictions=True)

        # Generate visualizations
        fig_dir = os.path.dirname(log_path) or '.'
        self._plot_training_curves(log_path, fig_dir)
        pred_path = self.config.get('test_predictions_path', 'model_path/test_predictions.json')
        if os.path.exists(pred_path):
            self._plot_test_predictions(pred_path, fig_dir)

    def _plot_training_curves(self, log_path, fig_dir):
        """Plot training curves from the CSV log."""
        epochs, train_loss, train_result, train_golden = [], [], [], []
        val_loss, val_acc, val_f1, val_result, val_golden = [], [], [], [], []
        test_loss, test_result, test_golden = [], [], []
        with open(log_path, 'rb') as fb:
            clean = fb.read().replace(b'\x00', b'')
        import io
        reader = csv.DictReader(io.StringIO(clean.decode('utf-8')))
        for row in reader:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss']))
                train_result.append(float(row.get('train_result', 0)))
                train_golden.append(float(row.get('train_golden', 0)))
                val_loss.append(float(row['val_loss']))
                val_acc.append(float(row['val_accuracy']))
                val_f1.append(float(row['val_f1']))
                # Selector-style val metrics (may be absent in legacy logs)
                val_result.append(float(row.get('val_result', 0)))
                val_golden.append(float(row.get('val_golden', 0)))
                test_loss.append(float(row['test_loss']))
                test_result.append(float(row['test_result']))
                test_golden.append(float(row['test_golden']))

        # Detect whether val set was present this run (val_result non-zero at least once)
        has_val_data = any(v > 0 for v in val_result)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, train_loss, label='Train Loss', alpha=0.8)
        if has_val_data:
            ax.plot(epochs, val_loss, label='Val Loss', alpha=0.8)
        ax.plot(epochs, test_loss, label='Test Loss', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Selector Performance — train / val / test (if val available)
        ax = axes[0, 1]
        ax.plot(epochs, train_result, label='Train Selector', alpha=0.8, color='tab:blue')
        ax.plot(epochs, train_golden, label='Train Oracle', alpha=0.8, color='tab:blue', linestyle='--')
        if has_val_data:
            ax.plot(epochs, val_result, label='Val Selector', alpha=0.8, color='tab:green')
            ax.plot(epochs, val_golden, label='Val Oracle', alpha=0.8, color='tab:green', linestyle='--')
        ax.plot(epochs, test_result, label='Test Selector', alpha=0.8, color='tab:orange')
        ax.plot(epochs, test_golden, label='Test Oracle', alpha=0.8, color='tab:orange', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg Score')
        title = 'Selector Performance (Train / Val / Test)' if has_val_data else 'Selector Performance (Train vs Test)'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation metrics
        ax = axes[1, 0]
        if has_val_data:
            ax.plot(epochs, val_acc, label='Val Accuracy (argmax match)', alpha=0.8)
            ax.plot(epochs, val_f1, label='Val Macro F1', alpha=0.8)
            ax.set_title('Validation Classification Metrics')
        else:
            ax.text(0.5, 0.5, 'No validation set', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title('Validation Metrics (no val set)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Performance gap
        ax = axes[1, 1]
        gap = [g - r for g, r in zip(test_golden, test_result)]
        ax.plot(epochs, gap, label='Oracle - Selector', color='red', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gap')
        ax.set_title('Performance Gap (Oracle - Selector)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(fig_dir, 'training_curves.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved training curves to {path}")

    def _plot_test_predictions(self, pred_path, fig_dir):
        """Plot test prediction analysis with per-dataset breakdown."""
        with open(pred_path) as f:
            preds = json.load(f)

        per_query = preds['per_query']
        llm_num = len(per_query[0]['all_scores'])
        per_dataset = preds.get('per_dataset', {})
        has_task_ids = 'task_id' in per_query[0]

        # Get unique datasets
        datasets = sorted(set(q['task_id'] for q in per_query)) if has_task_ids else ['All']
        n_datasets = len(datasets)

        has_cost_data = 'all_costs' in per_query[0]
        n_cols = 4 if has_cost_data else 3

        # rows: row0=overall, row1..n=per-dataset
        n_rows = 1 + n_datasets
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        def plot_row(ax_row, queries, title_prefix):
            predicted_wfs = [q['predicted_workflow'] for q in queries]
            oracle_wfs = [q['oracle_workflow'] for q in queries]
            predicted_scores = [q['predicted_score'] for q in queries]
            oracle_scores = [q['oracle_score'] for q in queries]

            # Mean combined score per workflow
            ax = ax_row[0]
            wf_labels = [f'W{i+1}' for i in range(llm_num)]
            # Oracle: average combined score for each workflow across all queries
            all_scores_matrix = np.array([q['all_scores'] for q in queries])  # (n_queries, llm_num)
            oracle_mean_scores = all_scores_matrix.mean(axis=0)
            # Selector: selection frequency per workflow
            n_queries = len(queries)
            selector_freq = np.array([predicted_wfs.count(i) / n_queries for i in range(llm_num)])
            x = np.arange(llm_num)
            w = 0.35
            ax.bar(x - w/2, oracle_mean_scores, w, label='Per-WF Mean Score', alpha=0.8, color='tab:orange')
            ax.bar(x + w/2, selector_freq, w, label='Selector Select Freq', alpha=0.8, color='tab:blue')
            ax.set_xticks(x)
            ax.set_xticklabels(wf_labels)
            ax.set_ylabel('Score / Frequency')
            r_score = np.mean(predicted_scores)
            o_score = np.mean(oracle_scores)
            ax.set_title(f'{title_prefix} WF Quality (n={len(queries)}, R={r_score:.4f}, O={o_score:.4f})')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')

            # Score gap histogram
            ax = ax_row[1]
            diffs = [o - p for o, p in zip(oracle_scores, predicted_scores)]
            ax.hist(diffs, bins=20, alpha=0.8, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Oracle - Selector Score')
            ax.set_ylabel('Count')
            ax.set_title(f'{title_prefix} Gap (mean={np.mean(diffs):.4f})')
            ax.grid(True, alpha=0.3, axis='y')

            # Confusion matrix
            ax = ax_row[2]
            # Confusion matrix: distribute tied oracle workflows evenly
            conf = np.zeros((llm_num, llm_num), dtype=float)
            for q in queries:
                p = q['predicted_workflow']
                scores = q['all_scores']
                max_s = max(scores)
                tied = [i for i, s in enumerate(scores) if abs(s - max_s) < 1e-6]
                for t in tied:
                    conf[t][p] += 1.0 / len(tied)
            im = ax.imshow(conf, cmap='Blues')
            ax.set_xticks(range(llm_num))
            ax.set_xticklabels(wf_labels)
            ax.set_yticks(range(llm_num))
            ax.set_yticklabels(wf_labels)
            ax.set_xlabel('Selector Predicted')
            ax.set_ylabel('Oracle')
            ax.set_title(f'{title_prefix} Confusion')
            for i in range(llm_num):
                for j in range(llm_num):
                    if conf[i][j] > 0.1:
                        label_text = f'{conf[i][j]:.0f}' if conf[i][j] == int(conf[i][j]) else f'{conf[i][j]:.1f}'
                        ax.text(j, i, label_text, ha='center', va='center',
                                color='white' if conf[i][j] > conf.max()/2 else 'black', fontsize=8)
            plt.colorbar(im, ax=ax)

            # Effect-Cost scatter (col 3) — only if cost data available
            if has_cost_data and len(ax_row) > 3:
                ax = ax_row[3]
                pred_effects = [q['predicted_effect_raw'] for q in queries]
                pred_costs = [q['predicted_cost'] for q in queries]

                # Per-workflow averages as reference points
                wf_effects = np.mean([q['all_effects_raw'] for q in queries], axis=0)
                wf_costs = np.mean([q['all_costs'] for q in queries], axis=0)

                # Scatter: each query's selector selection
                ax.scatter(pred_costs, pred_effects, alpha=0.15, s=8, c='blue', label='Selector choices')
                # Per-workflow average markers
                for wi in range(llm_num):
                    ax.scatter(wf_costs[wi], wf_effects[wi], s=120, marker='*', zorder=5,
                              edgecolors='black', linewidths=0.5)
                    ax.annotate(f'W{wi+1}', (wf_costs[wi], wf_effects[wi]),
                               textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')

                # Selector average
                r_eff = np.mean(pred_effects)
                r_cost = np.mean(pred_costs)
                ax.scatter(r_cost, r_eff, s=150, marker='D', c='red', zorder=6,
                          edgecolors='black', linewidths=1, label=f'Selector avg ({r_eff:.3f}, {r_cost:.3f})')

                ax.set_xlabel('Cost')
                ax.set_ylabel('Effect (raw)')
                ax.set_title(f'{title_prefix} Effect vs Cost')
                ax.legend(fontsize=7, loc='lower right')
                ax.grid(True, alpha=0.3)

        # Overall row
        plot_row(axes[0], per_query, 'Overall')

        # Per-dataset rows
        if has_task_ids:
            for di, ds in enumerate(datasets):
                ds_queries = [q for q in per_query if q['task_id'] == ds]
                plot_row(axes[1 + di], ds_queries, ds)

        plt.tight_layout()
        path = os.path.join(fig_dir, 'test_predictions.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved test prediction analysis to {path}")

    def test(self,data,model_path,save_predictions=False):
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                             edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, 1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, 1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        self._last_test_golden = result_golden
        print("result_predict:", result, "result_golden:",result_golden)

        # Per-dataset breakdown
        if self.test_task_ids is not None and len(self.test_task_ids) == len(value_test):
            task_ids_arr = np.array(self.test_task_ids)
            for task_name in np.unique(task_ids_arr):
                mask_task = task_ids_arr == task_name
                task_predict = value_test[mask_task][torch.arange(mask_task.sum()), max_idx[torch.tensor(mask_task)]].mean()
                task_golden = value_test[mask_task][torch.arange(mask_task.sum()), label_idx[torch.tensor(mask_task)]].mean()
                print(f"  {task_name}: predict={task_predict:.4f}, oracle={task_golden:.4f}")

        if save_predictions:
            pred_path = self.config.get('test_predictions_path', 'model_path/test_predictions.json')
            os.makedirs(os.path.dirname(pred_path) or '.', exist_ok=True)

            # Extract cost and raw effect from combined_edge
            # combined_edge layout: [weighted_score, raw_cost, raw_effect]
            combined_test = data.combined_edge[mask].reshape(-1, self.config['llm_num'], 3)
            cost_test = combined_test[:, :, 1]   # (N_test, llm_num)
            effect_raw_test = combined_test[:, :, 2]  # original effect before cost_weight weighting

            # Raw cost (un-normalized) for fair comparison with other methods
            has_raw_cost = hasattr(data, 'raw_cost') and data.raw_cost is not None
            if has_raw_cost:
                raw_cost_test = data.raw_cost[mask].reshape(-1, self.config['llm_num']).to(cost_test.device)
            else:
                raw_cost_test = cost_test  # fallback to normalized cost

            # Per-workflow means on test set
            llm_num = self.config['llm_num']
            test_wf_combined_means = value_test.mean(dim=0)           # cost_weight-weighted
            test_wf_effect_means = effect_raw_test.mean(dim=0)        # raw effect
            test_wf_cost_means = cost_test.mean(dim=0)                # cost

            # Per-workflow means on training set
            train_mask_bool = torch.tensor(data.train_mask, dtype=torch.bool)
            value_train = data.edge_attr[train_mask_bool].reshape(-1, llm_num)
            combined_train = data.combined_edge[train_mask_bool].reshape(-1, llm_num, 3)
            train_wf_combined_means = value_train.mean(dim=0)
            train_wf_effect_means = combined_train[:, :, 2].mean(dim=0)
            train_wf_cost_means = combined_train[:, :, 1].mean(dim=0)

            test_wf_raw_cost_means = raw_cost_test.mean(dim=0)

            def make_baseline(wf_idx, label):
                """Build a baseline dict for a given workflow index."""
                return {
                    f"{label}_wf": int(wf_idx),
                    f"{label}_combined": test_wf_combined_means[wf_idx].item(),
                    f"{label}_effect": test_wf_effect_means[wf_idx].item(),
                    f"{label}_cost": test_wf_cost_means[wf_idx].item(),
                    f"{label}_raw_cost": test_wf_raw_cost_means[wf_idx].item(),
                }

            # 6 baselines
            # 1. Best combined (test) — highest cost_weight-weighted score on test
            best_combined_test_idx = torch.argmax(test_wf_combined_means).item()
            # 2. Best combined (train→test) — highest on train, evaluated on test
            best_combined_train_idx = torch.argmax(train_wf_combined_means).item()
            # 3. Best effect (test) — highest raw effect on test
            best_effect_test_idx = torch.argmax(test_wf_effect_means).item()
            # 4. Best effect (train→test) — highest raw effect on train, evaluated on test
            best_effect_train_idx = torch.argmax(train_wf_effect_means).item()
            # 5. Lowest cost (test) — lowest cost on test
            best_cost_test_idx = torch.argmin(test_wf_cost_means).item()
            # 6. Lowest cost (train→test) — lowest cost on train, evaluated on test
            best_cost_train_idx = torch.argmin(train_wf_cost_means).item()

            baselines = {
                **make_baseline(best_combined_test_idx, "best_combined_test"),
                **make_baseline(best_combined_train_idx, "best_combined_train"),
                **make_baseline(best_effect_test_idx, "best_effect_test"),
                **make_baseline(best_effect_train_idx, "best_effect_train"),
                **make_baseline(best_cost_test_idx, "best_cost_test"),
                **make_baseline(best_cost_train_idx, "best_cost_train"),
            }

            # Selector & Oracle metrics
            selector_combined = result.item()
            oracle_combined = result_golden.item()
            selector_effect = effect_raw_test[row_indices, max_idx].mean().item()
            oracle_effect = effect_raw_test[row_indices, label_idx].mean().item()
            selector_cost = cost_test[row_indices, max_idx].mean().item()
            oracle_cost = cost_test[row_indices, label_idx].mean().item()
            selector_raw_cost = raw_cost_test[row_indices, max_idx].mean().item()
            oracle_raw_cost = raw_cost_test[row_indices, label_idx].mean().item()

            # Keep backward-compatible field names
            best_test_wf_idx = best_combined_test_idx
            best_test_wf_score = test_wf_combined_means[best_test_wf_idx].item()
            best_train_wf_idx = best_combined_train_idx
            best_train_wf_test_score = test_wf_combined_means[best_train_wf_idx].item()

            # Per-dataset summary
            per_dataset = {}
            if self.test_task_ids is not None and len(self.test_task_ids) == len(value_test):
                task_ids_arr = np.array(self.test_task_ids)
                for task_name in np.unique(task_ids_arr):
                    mask_task = task_ids_arr == task_name
                    indices_task = torch.tensor(mask_task)
                    t_predict = value_test[mask_task][torch.arange(mask_task.sum()), max_idx[indices_task]].mean()
                    t_golden = value_test[mask_task][torch.arange(mask_task.sum()), label_idx[indices_task]].mean()
                    t_test_wf_means = value_test[mask_task].mean(dim=0)
                    per_dataset[task_name] = {
                        "result_predict": t_predict.item(),
                        "result_golden": t_golden.item(),
                        "num_queries": int(mask_task.sum()),
                        "best_performed_workflow": best_test_wf_idx,
                        "best_performed_workflow_score": t_test_wf_means[best_test_wf_idx].item(),
                        "best_performed_training_workflow": best_train_wf_idx,
                        "best_performed_training_workflow_score": t_test_wf_means[best_train_wf_idx].item(),
                    }
                    # Add per-dataset cost/effect breakdown for baselines
                    t_cost_means = cost_test[torch.tensor(mask_task)].mean(dim=0)
                    t_effect_means = effect_raw_test[torch.tensor(mask_task)].mean(dim=0)
                    t_raw_cost_means = raw_cost_test[torch.tensor(mask_task)].mean(dim=0)
                    per_dataset[task_name]["per_wf_effect"] = t_effect_means.cpu().tolist()
                    per_dataset[task_name]["per_wf_cost"] = t_cost_means.cpu().tolist()
                    per_dataset[task_name]["per_wf_raw_cost"] = t_raw_cost_means.cpu().tolist()

            predictions = {
                "summary": {
                    "result_predict": selector_combined,
                    "result_golden": oracle_combined,
                    "test_loss": loss_test.item(),
                    "num_queries": len(value_test),
                    # Backward-compatible fields
                    "best_performed_workflow": best_test_wf_idx,
                    "best_performed_workflow_score": best_test_wf_score,
                    "best_performed_training_workflow": best_train_wf_idx,
                    "best_performed_training_workflow_score": best_train_wf_test_score,
                    # Selector & Oracle detailed metrics
                    "selector": {"combined": selector_combined, "effect": selector_effect, "cost": selector_cost, "raw_cost": selector_raw_cost},
                    "oracle": {"combined": oracle_combined, "effect": oracle_effect, "cost": oracle_cost, "raw_cost": oracle_raw_cost},
                    # 6 baselines (each has: _wf, _combined, _effect, _cost)
                    **baselines,
                },
                "per_dataset": per_dataset,
                "per_query": []
            }
            # Convert logits to probabilities for saving
            edge_probs = edge_predict  # model already outputs sigmoid probabilities
            for i in range(len(value_test)):
                scores = value_test[i].cpu().tolist()
                probs = edge_probs[i].cpu().tolist()
                # Find all tied-best workflows (oracle)
                max_score = max(scores)
                oracle_workflows = [j for j, s in enumerate(scores) if abs(s - max_score) < 1e-6]
                costs_i = cost_test[i].cpu().tolist()
                effects_raw_i = effect_raw_test[i].cpu().tolist()
                raw_costs_i = raw_cost_test[i].cpu().tolist()
                entry = {
                    "query_idx": i,
                    "predicted_workflow": max_idx[i].item(),
                    "oracle_workflow": oracle_workflows,  # list of all tied-best WFs
                    "predicted_score": scores[max_idx[i].item()],
                    "oracle_score": max_score,
                    "all_scores": scores,
                    "all_probs": probs,
                    "all_costs": costs_i,
                    "all_raw_costs": raw_costs_i,
                    "all_effects_raw": effects_raw_i,
                    "predicted_cost": costs_i[max_idx[i].item()],
                    "predicted_raw_cost": raw_costs_i[max_idx[i].item()],
                    "predicted_effect_raw": effects_raw_i[max_idx[i].item()],
                }
                if self.test_task_ids is not None and i < len(self.test_task_ids):
                    entry["task_id"] = self.test_task_ids[i]
                predictions["per_query"].append(entry)
            with open(pred_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved test predictions to {pred_path}")

        return result,loss_test