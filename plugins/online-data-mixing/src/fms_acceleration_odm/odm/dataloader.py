# dataloader + RL agent
from datasets import IterableDataset, DatasetDict
from typing import Optional, List
import math
import numpy as np
import random
from logging import getLogger
from torch.utils.data import DataLoader
logger = getLogger(__name__)

class OnlineData(IterableDataset):
    def __init__(
            self,
            dataset_dict: DatasetDict,
            collators_dict: dict,
            sampling_weights: Optional[List[float]]=None,
            gamma: float = 0.1,
            eta: float = 0.3,
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
        self.collators_dict = collators_dict
        for k, _ in dataset_dict.items():
            dataset_dict[k] = iter(DataLoader(dataset_dict[k], 1, shuffle=False, num_workers=1, collate_fn=collators_dict[k]))
        self.dataset_dict = dataset_dict
        self.category_list = sorted(dataset_dict.keys())
        self.id2cat = {i: c for i, c in enumerate(self.category_list)}
        self.total_categories = len(self.category_list)
        logger.info(f"Dataset categories: {self.category_list}")
        if sampling_weights is None:
            sampling_weights = [1]*self.total_categories

        self.sampling_weights = np.array(sampling_weights, dtype=np.float64)
        self.sampling_ratio = []
        self.update_sampling_ratio(self.sampling_weights)
        self.curr_idx = [0] * self.total_categories
        self.produced = 0

    def __iter__(self):
        self.produced = 0
        return self

    def __next__(self):
        arm_idx = random.choices(
            range(self.total_categories),
            weights=self.sampling_ratio,
            k=1
        )[0]

        sample = next(self.dataset_dict[self.id2cat[arm_idx]])
        self.curr_idx[arm_idx] += 1
        self.produced += 1
        return sample[0]

    def update_weights(self, batch_categories, rewards):
        """
        batch_categories  : list of categories of the samples in the batch
        rewards: list[float] (same length) -- reward in [0,1]
        """
        cat_sum, cat_cnt = {}, {}

        for c, r in zip(batch_categories, rewards):
            cat_sum[c] = cat_sum.get(c, 0.0) + float(r)
            cat_cnt[c] = cat_cnt.get(c, 0) + 1

        for arm in range(self.K):
            if arm in cat_sum:                          # arm was sampled
                avg_r = cat_sum[arm] / cat_cnt[arm]     # empirical reward
                est_r = avg_r / self.sampling_ratio[arm]
            else:                                       # arm not sampled
                est_r = 0.0

            self.sampling_weights[arm] *= math.exp(self.eta * est_r / self.K)

        return self.sampling_weights

    def update_sampling_ratio(self, new_weights):
        new_weights = np.asarray(new_weights, dtype=np.float64)
        assert new_weights.shape == self.sampling_weights.shape
        self.sampling_weights[:] = new_weights

        w = self.sampling_weights
        w_sum = w.sum()
        K = len(w)

        base = (1.0 - self.gamma) * (w / w_sum)
        expl = self.gamma / K
        self.sampling_ratio = (base + expl).tolist()

        return self.sampling_ratio

    def get_weights(self):
        return self.sampling_weights.copy()

    def get_sampling_ratio(self): 
        return self.sampling_ratio.copy()