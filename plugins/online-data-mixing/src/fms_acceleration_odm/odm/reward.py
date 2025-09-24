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
    GRADNORM = auto()


TRAIN_LOSS = {"buffer": None}

EVAL_LOSS = {"buffer": None}

GRADNORM = {"buffer": None}


def compute_reward(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    vocab_size: int,
    reward_type: Reward,
    train_loss_history=None,
    eval_loss_history=None,
    gradnorm_history=None,
    last_sampled_category=None,
    total_categories=None,
    current_category=None,
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
        Categories giving higher train loss reward are chosen.
        Validation loss reward: VALIDATION_LOSS
        Grad norm reward: GRADNORM

    Args:
        model (PreTrainedModel): HF Model object
        batch (torch.Tensor): Batch of samples (input_ids, labels, attention_mask)
        vocab_size (int): Maximum vocab size of the model used by ENTROPY rewards
        reward_type (Reward): Type of the reward
        train_loss_history: list of dicts each holding information on the training loss
        eval_loss_history: list of dicts each holding information on the eval loss
        gradnorm_history: list of dicts each holding information on the grad_norm
        last_sampled_category: index of the last sampled category
        total_categories: total number of categories
        current_category: currently being reward computed category
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
    if reward_type == Reward.TRAIN_LOSS:
        if not TRAIN_LOSS["buffer"]:
            TRAIN_LOSS["buffer"] = [1e-100] * total_categories
        TRAIN_LOSS["buffer"][last_sampled_category] = train_loss_history[-1]["loss"]
        return TRAIN_LOSS["buffer"][current_category]
    if reward_type == Reward.VALIDATION_LOSS:
        if not EVAL_LOSS["buffer"]:
            EVAL_LOSS["buffer"] = [1e-100] * total_categories
        EVAL_LOSS["buffer"][last_sampled_category] = eval_loss_history[-1]["loss"]
        return EVAL_LOSS["buffer"][current_category]
    if reward_type == Reward.GRADNORM:
        if not GRADNORM["buffer"]:
            GRADNORM["buffer"] = [1e-100] * total_categories
        GRADNORM["buffer"][last_sampled_category] = 1 / (
            gradnorm_history[-1]["grad_norm"] + 0.0001
        )
        return GRADNORM["buffer"][current_category]
    raise TypeError(f"Reward {reward_type} not supported")
