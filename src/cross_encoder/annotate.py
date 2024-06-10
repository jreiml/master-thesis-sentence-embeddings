import logging
import os
import pickle
from typing import List, Optional, Dict, Tuple, Any, Callable

import torch
from accelerate import Accelerator
from datasets import Dataset
from sentence_transformers import CrossEncoder, InputExample
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.pair_tokenizer import get_tokenizer_for_cross_bi_encoder
from util.args import DataArguments
from cross_encoder.collator import text_only_input_example_collator
from data.constants import TEXT_A_COL, TEXT_B_COL, PROCESSED_LABEL_A_COL, PROCESSED_LABEL_B_COL
from data.dataset_loader import load_pair_dataset_from_args
from data.multitask_pair_dataset import MultitaskInputExample, MultitaskPairDataset, MultitaskPairDatasetDict
from data.pair_dataset import PairDataset, PairDatasetDict
from model import AutoModelForEmbeddingSimilarityCrossBiEncoder


def get_cross_encoder_predictions(model: Any, collate_fn: Callable, sentence_pairs: List[List[str]]) -> List[float]:
    activation_fct = nn.Identity()
    model = model
    if isinstance(model, CrossEncoder):
        activation_fct = model.default_activation_function
        model = model.model

    score_targets = []
    dataloader = DataLoader(sentence_pairs, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model.eval()
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    iterator = tqdm(dataloader, desc="Cross-Encoder targets")

    with torch.no_grad():
        for features in iterator:
            outputs = model(**features, return_dict=True)
            score_output = accelerator.gather(outputs.logits)
            score_output = activation_fct(score_output)

            score_target = score_output.cpu().view(-1)
            score_targets.extend(score_target)

    return score_targets


def get_sentence_pair_targets_for_dataset(model: Any, collate_fn: Callable, dataset: Dataset) -> List[float]:
    text_a = dataset[TEXT_A_COL]
    text_b = dataset[TEXT_B_COL]
    text_pairs = list(zip(text_a, text_b))
    return get_cross_encoder_predictions(model, collate_fn, text_pairs)


def build_pair_dataset_split(
        split: PairDataset, model: Any, collate_fn: Callable
) -> Optional[PairDataset]:
    if split is None:
        return None

    text_pairs = [example.texts for example in split.input_examples]
    logging.debug(f"Collecting predictions for silver dataset ...")
    scores = get_cross_encoder_predictions(model, collate_fn, text_pairs)
    input_examples = [InputExample(texts=text_pair, label=label) for text_pair, label in zip(text_pairs, scores)]
    return PairDataset(input_examples)


def generate_silver_dataset(dataset: PairDatasetDict,
                            model_name: str,
                            max_length: int,
                            silver_dataset_cache_path: Optional[str] = None) -> PairDatasetDict:
    if silver_dataset_cache_path is not None and os.path.exists(silver_dataset_cache_path):
        logging.info(f"Loading cached silver dataset from {silver_dataset_cache_path} ...")
        with open(silver_dataset_cache_path, 'rb') as f:
            return pickle.load(f)

    cross_encoder = AutoModelForEmbeddingSimilarityCrossBiEncoder.from_pretrained(model_name)
    tokenizer = get_tokenizer_for_cross_bi_encoder(model_name)
    collate_fn = text_only_input_example_collator(tokenizer, max_length)

    logging.info(f"Generating silver dataset ...")
    silver_dataset = PairDatasetDict(
        train=build_pair_dataset_split(dataset.train, cross_encoder, collate_fn),
        validation=build_pair_dataset_split(dataset.validation, cross_encoder, collate_fn),
        test=dataset.test
    )

    if silver_dataset_cache_path is not None:
        logging.debug(f"Saving silver dataset to {silver_dataset_cache_path} ...")
        dirname = os.path.dirname(silver_dataset_cache_path)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        with open(silver_dataset_cache_path, 'wb') as f:
            pickle.dump(silver_dataset, f)

    return silver_dataset


def build_multitask_pair_dataset_split(
        split: Dataset,
        domain_label_pair_distribution: Dict[Tuple[Any, Any], float],
        model: Any,
        collate_fn: Callable,
) -> MultitaskPairDataset:
    text_pairs = [[example[TEXT_A_COL], example[TEXT_B_COL]] for example in split]
    # if has labels
    if len(domain_label_pair_distribution) > 0:
        labels = [[example[PROCESSED_LABEL_A_COL], example[PROCESSED_LABEL_B_COL]] for example in split]
    else:
        labels = [None] * len(text_pairs)

    logging.debug(f"Collecting predictions for silver dataset ...")
    scores = get_cross_encoder_predictions(model, collate_fn, text_pairs)
    input_examples = [MultitaskInputExample(
        texts=text_pair, similarity_label=similarity_label, domain_labels=domain_labels
    ) for text_pair, similarity_label, domain_labels in zip(text_pairs, scores, labels)]
    return MultitaskPairDataset(input_examples, domain_label_pair_distribution)


def generate_multitask_silver_dataset(
        data_args: DataArguments,
        model_name: str,
        max_length: int
) -> MultitaskPairDatasetDict:
    if data_args.silver_dataset_cache_path is not None and os.path.exists(data_args.silver_dataset_cache_path):
        logging.info(f"Loading cached silver dataset from {data_args.silver_dataset_cache_path} ...")
        with open(data_args.silver_dataset_cache_path, 'rb') as f:
            return pickle.load(f)

    cross_encoder = AutoModelForEmbeddingSimilarityCrossBiEncoder.from_pretrained(model_name)
    tokenizer = get_tokenizer_for_cross_bi_encoder(model_name)
    collate_fn = text_only_input_example_collator(tokenizer, max_length)
    pair_dataset, label_pair_distribution = load_pair_dataset_from_args(data_args)

    logging.info(f"Generating silver dataset ...")
    train = build_multitask_pair_dataset_split(
        pair_dataset["train"], label_pair_distribution, cross_encoder, collate_fn)
    validation = None
    if "validation" in pair_dataset:
        validation = build_multitask_pair_dataset_split(
            pair_dataset["validation"], label_pair_distribution, cross_encoder, collate_fn)
    test = None
    if "test" in pair_dataset:
        test = build_multitask_pair_dataset_split(
            pair_dataset["test"], label_pair_distribution, cross_encoder, collate_fn)

    silver_dataset = MultitaskPairDatasetDict(
        train=train,
        validation=validation,
        test=test
    )

    if data_args.silver_dataset_cache_path is not None:
        logging.debug(f"Saving silver dataset to {data_args.silver_dataset_cache_path} ...")
        dirname = os.path.dirname(data_args.silver_dataset_cache_path)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        with open(data_args.silver_dataset_cache_path, 'wb') as f:
            pickle.dump(silver_dataset, f)

    return silver_dataset
