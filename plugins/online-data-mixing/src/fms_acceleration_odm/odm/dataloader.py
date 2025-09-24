# dataloader + RL agent
from datasets import DatasetDict
from torch.utils.data import IterableDataset
from typing import Optional, List
import math
import random
from logging import getLogger
from torch.utils.data import DataLoader
from .reward import compute_reward, Reward
import torch
import os
import json

logger = getLogger(__name__)


class OnlineData(IterableDataset):
    def __init__(
            self,
            dataset_dict: DatasetDict,
            collators_dict: dict,
            eval_dataset_dict: DatasetDict,
            eval_collators_dict: dict,
            sampling_weights: Optional[List[float]]=None,
            gamma: float = 0.1,
            eta: float = 0.3,
            sampling_interval: int = 1, # sample data category every 1 sample,
            eval_batch_size: int = 5,
            output_dir="odm"
        ):
        """
        Mixes datasets with sampling ratios learnt using Multi Armed Bandit (MAB) and rewards defined.

        Args:
            - dataset_dict: DatasetDict - Expects a `dataset_dict` with keys as category names
                and values as corresponding HF datasets. As long as the above is maintained, the OnlineData should work OOB.
            - sampling_weights: Optional[List[float]] - Sampling weights to start with. If left None,
                sampling weights for each category would be n_i/total where n_i = total number samples in category i.
            - max_iter: int - If negative, sample till infinity, otherwise sample until `max_iter`
            - gamma: float - MAB variable
            - eta: float - MAB variable
        """
        logger.info(f"Using gamma: {gamma} and eta: {eta}")

        self.gamma = gamma
        self.eta = eta
        self.sampling_interval = sampling_interval
        self.collators_dict = collators_dict
        self.eval_collators_dict = eval_collators_dict
        self.eval_dataset_dict = eval_dataset_dict
        self.eval_dataset_dict_dl = {}
        for k, _ in dataset_dict.items():
            dataset_dict[k] = iter(DataLoader(dataset_dict[k], 1, shuffle=False, num_workers=1, collate_fn=collators_dict[k]))
        self.eval_batch_size = eval_batch_size
        self.dataset_dict = dataset_dict
        self.eval_dataset_dict = eval_dataset_dict
        self.category_list = sorted(dataset_dict.keys())
        self.id2cat = {i: c for i, c in enumerate(self.category_list)}
        self.cat2id = {c: i for i, c in enumerate(self.category_list)}
        self.total_categories = len(self.category_list)
        if sampling_weights is None:
            sampling_weights = [1]*self.total_categories

        self.sampling_weights = torch.tensor(sampling_weights, dtype=torch.float64)
        self.sampling_ratio = []
        self._update_sampling_ratio(self.sampling_weights)
        self.curr_idx = [0] * self.total_categories
        self.produced = 0
        self.arm_idx = 0
        self.reward_type = Reward.ENTROPY
        self.output_dir = output_dir
        self.K = self.total_categories
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.log_file_path = os.path.join(self.output_dir, "odm.jsonl")
        self.log = {"samples_produced_so_far": 0, 
                    "sampling_interval": self.sampling_interval,
                    "total_categories": self.total_categories, 
                    "current_sampling_weights": self.sampling_weights.tolist(), 
                    "current_sampling_ratio": self.sampling_ratio,
                    "arm_dix": self.arm_idx,
                    "category_level_counts_so_far": self.curr_idx,
                    "rewards": [0]*self.total_categories,
                    "count": 0,
                    "action": "",
                    }

    def log_to_file(self):
        with open(self.log_file_path, "a") as f:
            f.write(json.dumps(self.log) + "\n")

    def __iter__(self):
        self.produced = 0
        return self

    def __next__(self):
        if self.produced % self.sampling_interval == 0:            
            self.arm_idx = random.choices(
                range(self.total_categories),
                weights=self.sampling_ratio,
                k=1
            )[0]

        sample = next(self.dataset_dict[self.id2cat[self.arm_idx]])
        self.curr_idx[self.arm_idx] += 1
        self.produced += 1
        sample = {
            "input_ids": sample["input_ids"][0],
            "attention_mask": sample["attention_mask"][0],
            "labels": sample["labels"][0]
        }
        self.log["arm_dix"] = self.arm_idx
        self.log["samples_produced_so_far"] = self.produced
        self.log["category_level_counts_so_far"] = self.curr_idx
        self.log["action"] = "sample"
        self.log_to_file()
        return sample

    def _reset_eval_dataloaders(self):
        self.eval_dataset_dict_dl = {}
        for k, _ in self.eval_dataset_dict.items():
            # this can be improved with persistent workers and caching dataloaders and resetting them when needed.
            self.eval_dataset_dict_dl[k] = iter(DataLoader(self.eval_dataset_dict[k], self.eval_batch_size, shuffle=False, num_workers=1, collate_fn=self.eval_collators_dict[k]))

    def _update_sampling_ratio(self, weights):
        w = weights
        w_sum = w.sum()
        K = len(w)

        base = (1.0 - self.gamma) * (w / w_sum)
        expl = self.gamma / K
        self.sampling_ratio = (base + expl).tolist()
        return self.sampling_ratio

    def update_weights(self, count, rewards):
        """
        batch_categories  : list of categories of the samples in the batch
        rewards: list[float] (same length) -- reward in [0,1]
        """

        for arm in range(self.K):
            avg_r = rewards[arm] / count[arm]     # empirical reward
            est_r = avg_r / self.sampling_ratio[arm]
            self.sampling_weights[arm] *= math.exp(self.eta * est_r / self.K)
        return self._update_sampling_ratio(self.sampling_weights)

    def get_weights(self):
        return self.sampling_weights.copy()

    def get_sampling_ratio(self): 
        return self.sampling_ratio.copy()
    
    def update_sampling_weights(self, model, accelerator, metrics):
        rewards = [0] * self.total_categories
        count = [0] * self.total_categories
        eval_dataset_dict = {}
        self._reset_eval_dataloaders()
        for c in range(self.total_categories):
            eval_dataset_dict[self.id2cat[c]] = accelerator.prepare(self.eval_dataset_dict_dl[self.id2cat[c]])
        for c in range(self.total_categories):
            for batch in eval_dataset_dict[self.id2cat[c]]:
                cc, rc = compute_reward(model=model, batch={k: v.to(accelerator.device) for k, v in batch.items()}, vocab_size=32000, reward_type=self.reward_type, train_loop_metrics=metrics)
                rewards[c] += rc
                count[c] += cc
        rewards = torch.tensor(rewards, device=accelerator.device)
        count = torch.tensor(count, device=accelerator.device)
        rewards = accelerator.reduce(rewards, reduction="sum")
        count = accelerator.reduce(count, reduction="sum")
        if accelerator.is_main_process:
            logger.info(f"new rewards {rewards}")
            logger.info(f"new counts {count}")
            self.update_weights(rewards, count)
        self.log["current_sampling_weights"] = self.sampling_weights.tolist()
        self.log["current_sampling_ratio"] = self.sampling_ratio
        self.log["rewards"] = rewards.tolist()
        self.log["count"] = count.tolist()
        self.log["action"] = "update"
        self.log_to_file()
