# samples entropy

import torch
import torch.nn.functional as F

from typing import Dict, Tuple, Any
from transformers import PreTrainedModel
from enum import StrEnum, auto


class Reward(StrEnum):
    ENTROPY = auto()
    ENTROPY3_VARENT1 = auto()
    ENTROPY_LAST_TOKEN = auto()
    TRAIN_LOSS = auto()
    VALIDATION_LOSS = auto()


def compute_reward(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    vocab_size: int,
    reward_type: Reward,
    train_loop_metrics 
) -> Tuple[Any, torch.Tensor]:
    """
    Compute rewards based on the provided reward_type. You should be extending this function for new rewards.
    
    Supported rewards:
        Entropy related rewards:
        Calculates the entropy and variance of entropy of every sequence in the batch.
        For every sequence,
            1. The token level metrics are computed
            2. The metrics are averaged per sequence after applying the attention mask
        Train loss reward:
        Validation loss reward:

    Args:
        model (PreTrainedModel): HF Model object
        batch (torch.Tensor): The batch is assumed to be a tensor which is ready to be processed by the model. (batch_size x seq_len)
        vocab_size (int): Maximum vocab size of the model
        reward_type (Reward): Type of the reward
        train_loop_metrics (): Metrics from the training loop such as training loss, grad norm etc

    Returns:
        torch.Tensor
    """
    if reward_type.startswith(Reward.ENTROPY):
        with torch.inference_mode():
            outputs = model(**batch)
            shift_logits = outputs.logits[:, :-1, :]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            probs = torch.exp(log_probs)

            entropy = -torch.sum(probs * log_probs, dim=-1)
            sum_p_log_sq = torch.sum(probs * (log_probs**2), dim=-1)
            varentropy = sum_p_log_sq - (entropy**2)

            entropy_last_token = entropy[:, -1]

            mask = batch["attention_mask"][:, 1:]

            entropy = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
            varentropy = (varentropy * mask).sum(dim=-1) / mask.sum(dim=-1)

        max_entropy = torch.log(
            torch.tensor(vocab_size, dtype=entropy.dtype, device=entropy.device)
        )

        entropy = (entropy / max_entropy).clamp(0.0, 1.0)
        varentropy = (varentropy / max_entropy**2).clamp(0.0, 1.0)
        entropy_last_token = (entropy_last_token / max_entropy).clamp(0.0, 1.0)
        if reward_type == Reward.ENTROPY:
            return entropy.sum().item()
        elif reward_type == Reward.ENTROPY3_VARENT1:
            return 0.75 * entropy.sum().item() + 0.25 * varentropy.sum().item()
        elif reward_type == Reward.ENTROPY_LAST_TOKEN:
            return entropy_last_token.sum().item()
    elif reward_type == Reward.TRAIN_LOSS:
        pass
    elif reward_type == Reward.VALIDATION_LOSS:
        pass
    else:
        raise TypeError(f"Reward {reward_type} not supported")
