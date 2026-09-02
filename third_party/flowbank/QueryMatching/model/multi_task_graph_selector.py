import random
import numpy as np
import torch
from graph_nn import  form_data,GNN_prediction
from utils import savejson,loadjson,savepkl,loadpkl
from data_io import load_selector_dataframe
import pandas as pd
import json
import re
import yaml
device = "cuda" if torch.cuda.is_available() else "cpu"

class graph_selector_prediction:
    def __init__(self, selector_data_path,llm_path,llm_embedding_path,config,wandb,
                 eval_only=False, save_predictions=False):
        self.config = config
        self.wandb = wandb
        # Loads either a compact selector_data.npz or a verbose selector_data.csv;
        # see QueryMatching/model/data_io.py. The downstream DataFrame is identical.
        self.data_df = load_selector_dataframe(selector_data_path)
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.num_llms=len(self.llm_names)
        self.num_query=int(len(self.data_df)/self.num_llms)
        self.num_task=config['num_task']
        self.set_seed(self.config['seed'])
        self.llm_description_embedding=loadpkl(llm_embedding_path)
        self.prepare_data_for_GNN()
        self.split_data()
        self.form_data = form_data(device)
        self.query_dim = self.query_embedding_list.shape[1]
        self.llm_dim = self.llm_description_embedding.shape[1]
        self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                                    hidden_features_size=self.config['embedding_dim'], in_edges_size=self.config['edge_dim'],wandb=self.wandb,config=self.config,device=device)
        if eval_only:
            print("GNN selector initialized for inference (eval_only).")
            self.result_predict, self.result_golden = self.eval_only_predict(
                save_predictions=save_predictions)
        else:
            print("GNN training successfully initialized.")
            self.train_GNN()


    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def split_data(self):
        if 'split' in self.data_df.columns:
            self._split_data_predefined()
        else:
            self._split_data_positional()

    def _split_data_positional(self):
        self.query_per_task=int(self.num_query/self.num_task)
        split_ratio = self.config['split_ratio']

        # Calculate the size of each set for each task
        train_size = int(self.query_per_task * split_ratio[0])
        val_size = int(self.query_per_task * split_ratio[1])
        test_size = int(self.query_per_task * split_ratio[2])

        # Generate indices
        train_idx = []
        validate_idx = []
        test_idx = []

        for task_id in range(self.num_task):
            # Starting index for each task
            start_idx = task_id * self.query_per_task * self.num_llms

            # Add training set indices
            train_idx.extend(range(start_idx, start_idx + train_size* self.num_llms))

            # Add validation set indices
            validate_idx.extend(range(start_idx + train_size* self.num_llms,
                                      start_idx + train_size* self.num_llms + val_size* self.num_llms))

            # Add test set indices
            test_idx.extend(range(start_idx + train_size* self.num_llms + val_size* self.num_llms,
                                  start_idx + train_size* self.num_llms + val_size* self.num_llms + test_size* self.num_llms))

        self._finalize_split(train_idx, validate_idx, test_idx)

    def _split_data_predefined(self):
        """Use 'split' column from CSV to determine train/val/test indices."""
        split_col = self.data_df['split'].values
        train_idx = [i for i, s in enumerate(split_col) if s == 'train']
        validate_idx = [i for i, s in enumerate(split_col) if s == 'val']
        test_idx = [i for i, s in enumerate(split_col) if s == 'test']

        print(f"Predefined split: train={len(train_idx)}, val={len(validate_idx)}, test={len(test_idx)}")
        self._finalize_split(train_idx, validate_idx, test_idx)

    def _finalize_split(self, train_idx, validate_idx, test_idx):
        # Preserve raw cost/effect for reporting in test()
        self.combined_edge=np.concatenate((self.cost_list.reshape(-1,1),self.effect_list.reshape(-1,1)),axis=1)

        # Min-Max normalize effect and cost to [0, 1] before weighting
        eff_min, eff_max = self.effect_list.min(), self.effect_list.max()
        cost_min, cost_max = self.cost_list.min(), self.cost_list.max()

        if eff_max > eff_min:
            effect_norm = (self.effect_list - eff_min) / (eff_max - eff_min)
        else:
            effect_norm = np.zeros_like(self.effect_list)

        if cost_max > cost_min:
            cost_norm = (self.cost_list - cost_min) / (cost_max - cost_min)
        else:
            cost_norm = np.zeros_like(self.cost_list)

        # cost_weight: 0.0 = pure performance, 0.5 = balanced, 1.0 = pure cost
        cost_weight = self.config.get('cost_weight', 0.0)
        effect_weight = 1.0 - cost_weight
        self.effect_list = effect_weight * effect_norm - cost_weight * cost_norm

        # BCE: use the (cost-weighted) effect as an independent per-edge label,
        # clamped to [0, 1]. This is the only supported loss.
        self.label = np.clip(self.effect_list, 0.0, 1.0).reshape(-1, 1)
        self.edge_org_id=[num for num in range(self.num_query) for _ in range(self.num_llms)]
        self.edge_des_id=list(range(self.edge_org_id[0],self.edge_org_id[0]+self.num_llms))*self.num_query

        self.mask_train =torch.zeros(len(self.edge_org_id))
        self.mask_train[train_idx]=1

        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[validate_idx] = 1

        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[test_idx] = 1


    @staticmethod
    def _parse_embedding(s):
        """Parse embedding string to list. Handles both JSON and numpy formats."""
        # Already-parsed cell (e.g. nested list/array from the .npz loader): pass through.
        if not isinstance(s, str):
            return s
        s = s.strip()
        # Try direct JSON parse first (for JSON-formatted embeddings)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Fallback: numpy space-separated format like "[[ 0.123  0.456 ]]"
        s = re.sub(r'\s+', ', ', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s = s.replace("[[,", "[[")
            return json.loads(s)

    def prepare_data_for_GNN(self):
        unique_index_list=list(range(0, len(self.data_df), self.num_llms))
        query_embedding_list_raw=self.data_df['query_embedding'].tolist()
        task_embedding_list_raw = self.data_df['task_description_embedding'].tolist()
        self.query_embedding_list= []
        self.task_embedding_list= []
        for inter in query_embedding_list_raw:
            inter = self._parse_embedding(inter)
            self.query_embedding_list.append(inter[0])

        for inter in task_embedding_list_raw:
            inter = self._parse_embedding(inter)
            self.task_embedding_list.append(inter[0])
        self.query_embedding_list=np.array(self.query_embedding_list)[unique_index_list]
        self.task_embedding_list = np.array(self.task_embedding_list)[unique_index_list]
        self.effect_list=np.array(self.data_df['effect'].tolist())
        self.cost_list=np.array(self.data_df['cost'].tolist())
        if 'raw_cost' in self.data_df.columns:
            self.raw_cost_list = np.array(self.data_df['raw_cost'].tolist())
        else:
            self.raw_cost_list = self.cost_list.copy()



    def _build_gnn_data(self):
        """Build the train/validate/test graph Data objects and per-query task ids.

        Shared by both training (train_GNN) and inference (eval_only_predict);
        returns the list of per-query test task ids. Identical to the original
        train_GNN data-construction so results are unchanged.
        """
        self.data_for_GNN_train = self.form_data.formulation(task_id=self.task_embedding_list,
                                                             query_feature=self.query_embedding_list,
                                                             llm_feature=self.llm_description_embedding,
                                                             org_node=self.edge_org_id,
                                                             des_node=self.edge_des_id,
                                                             edge_feature=self.effect_list, edge_mask=self.mask_train,
                                                             label=self.label, combined_edge=self.combined_edge,
                                                             train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                             test_mask=self.mask_test)
        self.data_for_GNN_validate = self.form_data.formulation(task_id=self.task_embedding_list,
                                                                query_feature=self.query_embedding_list,
                                                                llm_feature=self.llm_description_embedding,
                                                                org_node=self.edge_org_id,
                                                                des_node=self.edge_des_id,
                                                                edge_feature=self.effect_list,
                                                                edge_mask=self.mask_validate, label=self.label,
                                                                combined_edge=self.combined_edge,
                                                                train_mask=self.mask_train,
                                                                valide_mask=self.mask_validate,
                                                                test_mask=self.mask_test)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.effect_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test)
        # Attach raw_cost for reporting in test predictions
        import torch as _torch
        raw_cost_tensor = _torch.tensor(self.raw_cost_list, dtype=_torch.float)
        self.data_for_GNN_train.raw_cost = raw_cost_tensor
        self.data_for_GNN_validate.raw_cost = raw_cost_tensor
        self.data_for_test.raw_cost = raw_cost_tensor

        # Build per-query task labels for per-dataset reporting
        # Each query has num_llms rows in the DataFrame; take every num_llms-th row
        unique_idx = list(range(0, len(self.data_df), self.num_llms))
        self.query_task_ids = self.data_df['task_id'].values[unique_idx]
        # Extract test query task_ids using the test mask
        test_edge_indices = [i for i, v in enumerate(self.mask_test) if v == 1]
        test_query_indices = sorted(set(i // self.num_llms for i in test_edge_indices))
        test_task_ids = [self.query_task_ids[qi] for qi in test_query_indices]
        self._test_task_ids = test_task_ids
        return test_task_ids

    def train_GNN(self):
        test_task_ids = self._build_gnn_data()
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_GNN_validate,
                                        data_for_test=self.data_for_test, test_task_ids=test_task_ids)

    def eval_only_predict(self, save_predictions=False):
        """Load a trained checkpoint and recompute test metrics WITHOUT training.

        Reuses the exact graph construction and GNN_prediction.test() path used
        during training, so the returned result_predict matches the value stored
        in the run's test_predictions.json. The masks and test_task_ids that
        test() reads are normally set inside train_validate(); we set them here.
        """
        test_task_ids = self._build_gnn_data()
        gp = self.GNN_predict
        gp.test_task_ids = test_task_ids
        gp.save_path = self.config['model_path']
        gp.num_edges = len(self.data_for_test.edge_attr)
        gp.train_mask = torch.tensor(self.data_for_test.train_mask, dtype=torch.bool)
        gp.valide_mask = torch.tensor(self.data_for_test.valide_mask, dtype=torch.bool)
        gp.test_mask = torch.tensor(self.data_for_test.test_mask, dtype=torch.bool)
        state = torch.load(self.config['model_path'], map_location=device)
        gp.model.load_state_dict(state)
        gp.model.eval()
        result, _loss = gp.test(self.data_for_test, self.config['model_path'],
                                save_predictions=save_predictions)
        result_predict = result.item() if hasattr(result, 'item') else float(result)
        golden = gp._last_test_golden
        result_golden = golden.item() if hasattr(golden, 'item') else float(golden)
        return result_predict, result_golden

    def test_GNN(self):
        predicted_result = self.GNN_predict.test(data=self.data_for_test,model_path=self.config['model_path'])




if __name__ == "__main__":
    import wandb
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)
    wandb.init(project="graph_selector")
    graph_selector_prediction(selector_data_path=config['saved_selector_data_path'],llm_path=config['llm_description_path'],
                            llm_embedding_path=config['llm_embedding_path'],config=config,wandb=wandb)