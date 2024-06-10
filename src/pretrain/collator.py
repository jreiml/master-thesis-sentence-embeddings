import random
from typing import Dict, List, Union

import nltk
import numpy as np
import torch
from nltk import word_tokenize, TreebankWordDetokenizer
from sentence_splitter import SentenceSplitter
from torch import Tensor
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding

from data.constants import TEXT_COL, TEXT_A_COL, TEXT_B_COL


def prepare_for_sop(
        tokenizer, text_col, max_length, features, return_special_tokens_mask=False, return_list=False
) -> Union[BatchEncoding, List[Dict[str, Tensor]]]:
    splitter = SentenceSplitter("en")
    texts = [[], []]
    labels = []

    for feature in features:
        text = feature[text_col]
        split_text = splitter.split(text)
        if len(split_text) == 1:
            split_text = word_tokenize(split_text[0])

        # Split down the middle, or decide randomly for uneven amount of sentences
        split_index = len(split_text) // 2
        if len(split_text) % 2 == 1:
            split_index += random.randint(0, 1)

        text_a = ' '.join(split_text[:split_index])
        text_b = ' '.join(split_text[split_index:])

        swap_label = random.randint(0, 1)
        if swap_label == 0:
            text_tmp = text_a
            text_a = text_b
            text_b = text_tmp
        texts[0].append(text_a)
        texts[1].append(text_b)
        labels.append(swap_label)

    tokenized = tokenizer(
        *texts,
        padding=True,
        max_length=max_length,
        truncation=TruncationStrategy.LONGEST_FIRST,
        return_tensors="pt",
        return_special_tokens_mask=return_special_tokens_mask
    )
    tokenized["next_sentence_label"] = torch.tensor(labels, dtype=torch.long)

    if not return_list:
        return tokenized

    new_features = []
    for key, feature in tokenized.items():
        new_feature = {}
        for row in feature:
            new_feature[key] = row

        new_features.append(new_feature)

    return new_features


def get_special_tokens_mask(tokenizer, input_ids, special_tokens_mask=None):
    if special_tokens_mask is not None:
        return special_tokens_mask.bool()

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
    ]
    return torch.tensor(special_tokens_mask, dtype=torch.bool)


def get_mlm_eval_labels(tokenizer, input_ids, special_tokens_mask=None):
    special_tokens_mask = get_special_tokens_mask(tokenizer, input_ids, special_tokens_mask)
    labels = input_ids.clone()
    labels[special_tokens_mask] = -100
    return labels, special_tokens_mask


def apply_masking(tokenizer, mlm_probability, input_ids, special_tokens_mask):
    # Use uniform masking without random/same word prediction (https://arxiv.org/pdf/2202.08005.pdf)
    target = torch.bernoulli(torch.full(input_ids.shape, mlm_probability, device=input_ids.device)).bool()
    special_tokens_mask = get_special_tokens_mask(tokenizer, input_ids, special_tokens_mask)
    target = target & ~special_tokens_mask

    labels = input_ids.clone()
    labels[~target] = -100
    input_ids[target] = tokenizer.mask_token_id
    return labels, special_tokens_mask


def mlm_eval_collator(tokenizer, max_length, text_col=TEXT_COL):
    def inner(features):
        texts = [feature[text_col] for feature in features]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        labels, _ = get_mlm_eval_labels(tokenizer, tokenized["input_ids"], tokenized["special_tokens_mask"])
        tokenized["labels"] = labels
        del tokenized["special_tokens_mask"]
        return tokenized

    return inner


def mlm_train_collator(tokenizer, mlm_probability, max_length, text_col=TEXT_COL):
    def inner(features):
        texts = [feature[text_col] for feature in features]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        labels, _ = apply_masking(tokenizer, mlm_probability, tokenized["input_ids"], tokenized["special_tokens_mask"])
        tokenized["labels"] = labels
        del tokenized["special_tokens_mask"]
        return tokenized

    return inner


def mlm_pair_eval_collator(tokenizer, max_length):
    def inner(features):
        texts_a = [feature[TEXT_A_COL] for feature in features]
        texts_b = [feature[TEXT_B_COL] for feature in features]
        tokenized = tokenizer(
            texts_a,
            texts_b,
            max_length=max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        labels, _ = get_mlm_eval_labels(tokenizer, tokenized["input_ids"], tokenized["special_tokens_mask"])
        tokenized["labels"] = labels
        del tokenized["special_tokens_mask"]
        return tokenized

    return inner


def mlm_pair_train_collator(tokenizer, mlm_probability, max_length):
    def inner(features):
        texts_a = [feature[TEXT_A_COL] for feature in features]
        texts_b = [feature[TEXT_B_COL] for feature in features]
        tokenized = tokenizer(
            texts_a,
            texts_b,
            max_length=max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        labels, _ = apply_masking(tokenizer, mlm_probability, tokenized["input_ids"], tokenized["special_tokens_mask"])
        tokenized["labels"] = labels
        del tokenized["special_tokens_mask"]
        return tokenized

    return inner


def mlm_sop_train_collator(tokenizer, mlm_probability, text_col, max_length):
    def inner(features):
        tokenized = prepare_for_sop(tokenizer, text_col, max_length, features, return_special_tokens_mask=True)
        labels, _ = apply_masking(tokenizer, mlm_probability, tokenized["input_ids"], tokenized["special_tokens_mask"])
        tokenized["labels"] = labels
        del tokenized["special_tokens_mask"]
        return tokenized

    return inner


def mlm_sop_eval_collator(tokenizer):
    padding_collator = DataCollatorWithPadding(tokenizer)

    def inner(features):
        tokenized = padding_collator(features)
        labels, _ = get_mlm_eval_labels(tokenizer, tokenized["input_ids"], tokenized["special_tokens_mask"])
        tokenized["labels"] = labels
        del tokenized["special_tokens_mask"]
        return tokenized

    return inner


def delete(text, del_ratio=0.6, delete_at_least_one=True):
    words = nltk.word_tokenize(text)
    n = len(words)
    if n <= 1:
        return text

    keep_or_not = np.random.rand(n) > del_ratio
    total_keep = sum(keep_or_not)
    if total_keep == 0:
        keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
    if total_keep == n and n > 1 and delete_at_least_one:
        keep_or_not[np.random.choice(n)] = False  # guarantee that at least one word is deleted

    words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
    return words_processed


def msd_delete_collator(tokenizer, max_length, del_ratio, text_col=TEXT_COL,
                        delete_at_least_one=True, enforce_delete_subset=False):
    def inner(features):
        texts = [feature[text_col] for feature in features]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        texts_noised = tokenizer.batch_decode(tokenized.input_ids, skip_special_tokens=True)
        if del_ratio > 0.0:
            # bug in experiments, keeping it for reproducibility
            if not enforce_delete_subset:
                texts_noised = texts
            texts_noised = [delete(
                text, del_ratio=del_ratio, delete_at_least_one=delete_at_least_one
            ) for text in texts_noised]
        tokenized_noised = tokenizer(
            texts_noised,
            max_length=max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
            padding=True,
            return_tensors="pt",
        )

        return {
            "noised_input_ids": tokenized_noised["input_ids"],
            "noised_attention_mask": tokenized_noised["attention_mask"],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "special_tokens_mask": tokenized["special_tokens_mask"]
        }

    return inner
