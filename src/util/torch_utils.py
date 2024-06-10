import gc
from typing import Optional, List

import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PretrainedConfig


def get_device(device_ids: Optional[List[int]] = None):
    if not torch.cuda.is_available():
        return "cpu"

    if device_ids is None or len(device_ids) == 0:
        return "cuda"

    return f"cuda:{device_ids[0]}"


def mean_pool_output_with_input_mask(
        sequence_output: Optional[torch.LongTensor] = None,
        input_mask: Optional[torch.FloatTensor] = None,
):
    masked_sequence_output = sequence_output * input_mask
    sum_embeddings_a = torch.sum(masked_sequence_output, 1)
    sum_mask = input_mask.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings_a / sum_mask


def max_pool_output_with_input_mask(
        sequence_output: Optional[torch.LongTensor] = None,
        input_mask: Optional[torch.FloatTensor] = None,
):
    sequence_output[input_mask == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(sequence_output, 1)[0]


def determine_bos_token_id(
        config,
        input_ids: Optional[torch.LongTensor] = None,
):
    if hasattr(config, "bos_token_id") and config.bos_token_id is not None:
        return config.bos_token_id

    if input_ids is None:
        raise ValueError("Unable to determine bos token. BOS token not set in config and no input ids passed.")

    return input_ids[0, 0].item()


def mean_pool_sequence_outputs_for_cross_encoder(
        bos_token_id: int,
        sequence_output: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
):
    batch_size, seq_len = input_ids.shape
    first_seq_end_idx = (input_ids[:, 1:] == bos_token_id).long().argmax(1)\
        .repeat_interleave(seq_len).view(-1, seq_len)
    indices = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device) \
        .repeat(1, batch_size).view(batch_size, -1)

    token_type_ids = (indices > first_seq_end_idx).long()

    # [CLS] [Tokens A] [SEP] [CLS] [Tokens B] [SEP]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
    token_type_ids_expanded = token_type_ids.unsqueeze(-1).expand(sequence_output.size()).float()

    input_mask_expanded_a = input_mask_expanded * (1 - token_type_ids_expanded)
    pooled_output_a = mean_pool_output_with_input_mask(sequence_output, input_mask_expanded_a)
    input_mask_expanded_b = input_mask_expanded * token_type_ids_expanded
    pooled_output_b = mean_pool_output_with_input_mask(sequence_output, input_mask_expanded_b)
    return pooled_output_a, pooled_output_b


def get_pairwise_sep_token_indices(
        for_first_sequence,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None
):
    batch_size, seq_len = attention_mask.shape
    sep_idx = torch.arange(0, seq_len, dtype=torch.long, device=token_type_ids.device) \
        .repeat(1, batch_size).view(batch_size, -1)

    sep_idx = sep_idx
    if for_first_sequence:
        input_mask = (1 - token_type_ids) * attention_mask
    else:
        input_mask = token_type_ids * attention_mask
    sep_idx[input_mask == 0] = 0
    sep_idx = sep_idx.max(dim=1).values
    return sep_idx


def pool_sequence_output(
        sequence_output: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        include_cls: bool = True,
        include_sep: bool = True,
        use_mean_pooling: bool = True
):
    attention_mask_filtered = attention_mask.clone()
    if not include_cls:
        attention_mask_filtered[:, 0] = 0
    if not include_sep:
        batch_size, seq_len = attention_mask.shape
        all_idx = torch.arange(0, seq_len, dtype=torch.long, device=attention_mask.device) \
            .repeat(1, batch_size).view(batch_size, -1)
        sep_idx = all_idx.clone()
        sep_idx[attention_mask == 0] = 0
        sep_idx = sep_idx.max(dim=1).values
        sep_idx = sep_idx.unsqueeze(1).repeat(1, seq_len)
        attention_mask_filtered = attention_mask_filtered & (sep_idx != all_idx)

    input_mask_expanded = attention_mask_filtered.unsqueeze(-1).expand(sequence_output.size()).float()
    if use_mean_pooling:
        return mean_pool_output_with_input_mask(sequence_output, input_mask_expanded)
    else:
        return max_pool_output_with_input_mask(sequence_output, input_mask_expanded)


def get_sequence_classification_loss(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        config: PretrainedConfig,
        num_labels: int
) -> torch.FloatTensor:

    problem_type = None if config is None else config.problem_type
    if problem_type is None:
        if num_labels == 1:
            problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            problem_type = "single_label_classification"
        else:
            problem_type = "multi_label_classification"

    loss = None
    if problem_type == "regression":
        loss_fct = MSELoss()
        if num_labels == 1:
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
            loss = loss_fct(logits, labels)
    elif problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    elif problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

    return loss


def clear_memory():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
