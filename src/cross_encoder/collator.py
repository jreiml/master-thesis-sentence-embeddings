from typing import List

import torch
from sentence_transformers import InputExample
from transformers.tokenization_utils_base import TruncationStrategy


def text_only_input_example_collator(tokenizer, max_length):
    def inner(features: List[List[str]]):
        sentence_count = len(features[0])
        assert sentence_count == 2, "Only pairs are supported!"

        text_pairs = [[], []]

        for feature in features:
            text_a = feature[0].strip()
            text_b = feature[1].strip()
            text_pairs[0].append(text_a)
            text_pairs[1].append(text_b)

        return tokenizer(
            *text_pairs,
            padding=True,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors="pt",
            max_length=max_length
        )

    return inner


def input_example_collator(tokenizer, max_length):
    def inner(features: List[InputExample]):
        sentence_count = len(features[0].texts)
        assert sentence_count == 2, "Only pairs are supported!"

        text_pairs = [[], []]
        labels = []

        for feature in features:
            text_a = feature.texts[0].strip()
            text_b = feature.texts[1].strip()
            text_pairs[0].append(text_a)
            text_pairs[1].append(text_b)

            labels.append(feature.label)

        tokenized = tokenizer(
            *text_pairs,
            padding=True,
            truncation=TruncationStrategy.LONGEST_FIRST,
            return_tensors="pt",
            max_length=max_length
        )

        labels = torch.tensor(labels, dtype=torch.float)
        return {
            **tokenized,
            "labels": labels
        }

    return inner
