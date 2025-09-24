# samples entropy

# Standard
from enum import StrEnum, auto
from typing import Dict

# Third Party
from transformers import PreTrainedModel
import torch
import torch.nn.functional as F


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
    train_loop_metrics=None,
) -> float:
    """
    Compute rewards based on the provided reward_type.
    You should be extending this function for new rewards.

    Supported rewards:

        Entropy related rewards: ENTROPY, ENTROPY3_VARENT1 & ENTROPY_LAST_TOKEN
        Calculates the entropy and variance of entropy of every sequence in the batch.
        For every sequence,
            1. The token level metrics are computed
            2. The metrics are averaged per sequence after applying the attention mask

        Train loss reward: TRAIN_LOSS
        Validation loss reward: VALIDATION_LOSS

    Args:
        model (PreTrainedModel): HF Model object
        batch (torch.Tensor): Batch of samples (input_ids, labels, attention_mask)
        vocab_size (int): Maximum vocab size of the model used by ENTROPY rewards
        reward_type (Reward): Type of the reward
        train_loop_metrics: Metrics from the training loop such as training loss,
        grad norm etc.
    Returns:
        float
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
        if reward_type == Reward.ENTROPY3_VARENT1:
            return (0.75 * entropy.sum().item() + 0.25 * varentropy.sum().item(),)
        if reward_type == Reward.ENTROPY_LAST_TOKEN:
            return entropy_last_token.sum().item()
    elif reward_type == Reward.TRAIN_LOSS:
        return 0
    elif reward_type == Reward.VALIDATION_LOSS:
        return 0
    raise TypeError(f"Reward {reward_type} not supported")
