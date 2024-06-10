import logging
import math
import os
from collections import Counter
import random
from typing import Tuple, Dict, Any

from datasets import concatenate_datasets, load_dataset, DatasetDict, load_from_disk, Dataset
from transformers import set_seed

from util.args import DataArguments, PairGenerationStrategy
from data.constants import PROCESSED_LABEL_COL, TEXT_COL
from data.data_preprocessing import normalize_labels, \
    apply_on_dataset_dict, split_documents_to_sentence_grams, truncate_labels_to_length, apply_tokenizer_formatting, \
    filter_duplicate_rows
from data.pair_generator import get_label_to_indices, get_label_pair_distribution, SentencePairGenerator
from data.pair_set import PairSet

logger = logging.getLogger(__name__)


def load_raw_dataset(args: DataArguments) -> DatasetDict:
    if args.raw_dataset_cache_path is not None:
        dataset_file = os.path.join(args.raw_dataset_cache_path, "dataset_dict.json")
        if os.path.exists(dataset_file):
            logger.info(f"Loading cached dataset from {args.raw_dataset_cache_path} ...")
            return load_from_disk(args.raw_dataset_cache_path)

    logger.info(f"Loading dataset from {args.dataset_path} ...")
    set_seed(args.data_generation_seed)
    if args.is_hugging_face_dataset:
        dataset = load_dataset(args.dataset_path)
        if isinstance(dataset, DatasetDict):
            dataset = concatenate_datasets(list(dataset.values()))
    elif args.is_text_dataset:
        dataset = load_dataset('text', data_files=args.dataset_path, split="train")
    elif args.is_csv_dataset:
        dataset = load_dataset('csv', data_files=args.dataset_path, split="train", delimiter=args.csv_delimiter)
    else:
        dataset = load_from_disk(args.dataset_path)

    if args.text_col != TEXT_COL:
        dataset = dataset.rename_column(args.text_col, TEXT_COL)
    if args.use_data_percentage is not None and 0 < args.use_data_percentage < 1:
        logger.debug(f"Only using {args.use_data_percentage:.1%} of the data for debugging ...")
        num_shards = math.ceil(1 / args.use_data_percentage)
        dataset = dataset.shard(num_shards, 1)

    dataset = dataset.filter(lambda ex: ex[TEXT_COL] is not None)
    if args.strip_whitespaces:
        logger.debug("Stripping whitespaces...")
        dataset = dataset.map(lambda ex: {TEXT_COL: ex[TEXT_COL].strip()})
    if args.do_lowercase:
        logger.debug("Using lowercase...")
        dataset = dataset.map(lambda ex: {TEXT_COL: ex[TEXT_COL].lower()})
    if args.split_into_sentence_grams is not None:
        logger.debug("Splitting into sentence grams...")
        dataset = split_documents_to_sentence_grams(dataset, TEXT_COL, args.split_into_sentence_grams)

    if args.tokenizer_name is not None:
        dataset = apply_tokenizer_formatting(dataset, TEXT_COL, args.tokenizer_name, args.max_length, False)

    if args.raw_filter_fn is not None:
        logger.debug(f"Applying custom filter function ...")
        dataset = dataset.filter(args.raw_filter_fn)
    if args.raw_map_fn is not None:
        logger.debug(f"Applying custom map function ...")
        dataset = dataset.filter(args.raw_map_fn)

    logger.debug(f"Filtering empty rows ...")
    dataset = dataset.filter(lambda ex: not ex[TEXT_COL].isspace() and len(ex[TEXT_COL]) > 0)
    if args.label_col is not None:
        dataset = dataset.map(lambda ex: {PROCESSED_LABEL_COL: ex[args.label_col]})

    if args.filter_duplicates:
        logger.debug(f"Filtering duplicate rows ...")
        dataset = filter_duplicate_rows(dataset)

    if args.dataset_size_limit is not None and args.dataset_size_limit > 0:
        logger.debug(f"Limiting dataset size to {args.dataset_size_limit}")
        dataset = dataset.shuffle(seed=args.data_generation_seed)
        dataset = Dataset.from_dict(dataset[:args.dataset_size_limit])

    if args.label_col is not None:
        logger.debug(f"Normalizing labels for {args.label_col} ...")
        dataset = normalize_labels(dataset, PROCESSED_LABEL_COL)

    eval_test_percent = args.validation_percent + args.test_percent
    if eval_test_percent <= 0:
        dataset = DatasetDict({
            "train": dataset
        })
    else:
        dataset = dataset.train_test_split(eval_test_percent)
        if args.validation_percent > 0 and args.test_percent > 0:
            test_percent = args.test_percent / eval_test_percent
            test_dataset = dataset["test"].train_test_split(test_percent)
            dataset["validation"] = test_dataset["train"]
            dataset["test"] = test_dataset["test"]
        elif args.validation_percent > 0:
            dataset["validation"] = dataset["test"]
            del dataset["test"]

    if args.raw_dataset_cache_path is not None:
        logger.debug(f"Saving raw dataset to {args.raw_dataset_cache_path} ...")
        dataset.save_to_disk(args.raw_dataset_cache_path)

    return dataset


def check_filter_minority_class(args: DataArguments, dataset: DatasetDict) -> DatasetDict:
    minimum_required_samples_per_class = args.minimum_required_samples_per_class
    if minimum_required_samples_per_class is None or args.label_col is None:
        return dataset

    labels = [label for split in dataset.values() for label in split[PROCESSED_LABEL_COL]]
    label_count = Counter(labels)
    keep_labels = set([label for label, count in label_count.items()
                       if count >= minimum_required_samples_per_class])

    main_dataset = dataset.filter(lambda ex: ex[PROCESSED_LABEL_COL] in keep_labels)
    if not args.new_label_for_filtered_samples:
        main_dataset = normalize_labels(main_dataset, PROCESSED_LABEL_COL)
        return main_dataset

    minority_dataset = dataset.filter(lambda ex: ex[PROCESSED_LABEL_COL] not in keep_labels)
    new_label = max(keep_labels) + 1
    minority_dataset = minority_dataset.map(lambda _: {
        PROCESSED_LABEL_COL: new_label
    })

    for split in dataset.keys():
        dataset[split] = concatenate_datasets([main_dataset[split], minority_dataset[split]])
    dataset = normalize_labels(dataset, PROCESSED_LABEL_COL)
    return dataset


def load_processed_dataset(args: DataArguments) -> DatasetDict:
    dataset = load_raw_dataset(args)
    if args.truncate_hierarchical_label_to_length is None:
        return dataset

    if args.label_col is not None:
        logger.debug(f"Truncating hierarchical labels for {args.label_col} ...")
        dataset = truncate_labels_to_length(dataset, args.label_col, args.truncate_hierarchical_label_to_length)
        dataset = normalize_labels(dataset, args.label_col)
    return dataset


def get_label_pair_distribution_for_dataset_dict(dataset: DatasetDict) -> Dict[Tuple[Any, Any], float]:
    combined_dataset = concatenate_datasets(list(dataset.values()))
    label_distribution = get_label_pair_distribution(get_label_to_indices(combined_dataset, PROCESSED_LABEL_COL))
    return label_distribution


def load_pair_dataset_from_args(args: DataArguments) -> Tuple[DatasetDict, Dict[Any, float]]:
    dataset = load_processed_dataset(args)
    if args.pair_dataset_cache_path is not None:
        dataset_file = os.path.join(args.pair_dataset_cache_path, "dataset_dict.json")
        if os.path.exists(dataset_file):
            logger.info(f"Loading cached pair dataset from {args.pair_dataset_cache_path} ...")
            label_distribution = {} if args.label_col is None else get_label_pair_distribution_for_dataset_dict(dataset)
            return load_from_disk(args.pair_dataset_cache_path), label_distribution

    set_seed(args.data_generation_seed)
    if args.tokenizer_name is not None:
        dataset = apply_tokenizer_formatting(dataset, TEXT_COL, args.tokenizer_name, args.max_length, True)

    label_distribution = {}
    if args.label_col is not None and not any([
        isinstance(label, float) for d in dataset.values() for label in d[PROCESSED_LABEL_COL]
    ]):
        label_distribution = get_label_pair_distribution_for_dataset_dict(dataset)
    dataset = check_filter_minority_class(args, dataset)

    def gen_sent_pairs(_, split):
        skip_pairs = PairSet()
        pair_datasets = []
        sentences = random.sample(split[TEXT_COL], len(split[TEXT_COL]))
        labels = None if args.label_col is None else split[PROCESSED_LABEL_COL]
        batch_size = args.pair_generation_batch_size if args.pair_generation_batch_size > 0 else len(sentences)
        for batch_start in range(0, len(sentences), batch_size):
            batch_end = batch_start + batch_size
            sentences_batch = sentences[batch_start:batch_end]
            labels_batch = None if labels is None else labels[batch_start:batch_end]
            gen = SentencePairGenerator(sentences_batch, labels_batch)

            for top_k, strategy in zip(args.top_k_pairs, args.pair_generation_strategies):
                logger.debug(f"Generating top {top_k} pairs using strategy {strategy} ...")

                if strategy == PairGenerationStrategy.REVERSE_SEMANTIC_SEARCH:
                    pair_dataset = gen.generate_semantic_search(
                        top_k, least_similar=True, semantic_model_name=args.semantic_search_model, skip_pairs=skip_pairs)
                elif strategy == PairGenerationStrategy.SEMANTIC_SEARCH:
                    pair_dataset = gen.generate_semantic_search(
                        top_k, least_similar=False, semantic_model_name=args.semantic_search_model, skip_pairs=skip_pairs)
                elif strategy == PairGenerationStrategy.REVERSE_BM25:
                    pair_dataset = gen.generate_bm25(top_k, least_similar=True, skip_pairs=skip_pairs)
                elif strategy == PairGenerationStrategy.BM25:
                    pair_dataset = gen.generate_bm25(top_k, least_similar=False, skip_pairs=skip_pairs)
                elif strategy == PairGenerationStrategy.RANDOM:
                    pair_dataset = gen.generate_random(top_k, skip_pairs=skip_pairs)
                else:
                    raise ValueError(f"Invalid pairing strategy: {strategy}!")

                pair_datasets.append(pair_dataset)
        return concatenate_datasets(pair_datasets)

    dataset = apply_on_dataset_dict(dataset, gen_sent_pairs)

    if args.pair_dataset_cache_path is not None:
        logger.debug(f"Saving pair dataset to {args.pair_dataset_cache_path} ...")
        dataset.save_to_disk(args.pair_dataset_cache_path)

    return dataset, label_distribution


def load_processed_dataset_for_sop(args: DataArguments) -> DatasetDict:
    new_cache_path = None
    if args.raw_dataset_cache_path is not None:
        new_cache_path = os.path.join(args.raw_dataset_cache_path, 'sop')
    args = args._replace(
        raw_dataset_cache_path=new_cache_path,
        split_into_sentence_grams=2
    )
    return load_processed_dataset(args)
