from collections import defaultdict
from typing import Callable, Union, List, Dict, Any, Optional

from datasets import DatasetDict, concatenate_datasets, Dataset
from nltk import ngrams
from sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer

from data.constants import PROCESSED_LABEL_COL, TEXT_COL


def apply_on_dataset_dict(dataset_dict: DatasetDict, apply_fn: Callable[[str, Dataset], Dataset]) -> DatasetDict:
    new_dataset_dict = {}
    for key, split in dataset_dict.items():
        new_dataset_dict[key] = apply_fn(key, split)
    return DatasetDict(new_dataset_dict)


def get_unique_labels(dataset: Union[Dataset, DatasetDict], label_col: str) -> List[int]:
    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets([split for split in dataset.values()])
    return sorted(set(dataset[label_col]))


def split_documents_to_sentence_grams(
        dataset: Union[Dataset, DatasetDict],
        text_col: str,
        sentence_count: int
) -> Union[Dataset, DatasetDict]:
    splitter = SentenceSplitter("en")

    def add_sentence(mapped_batch, batch, batch_index, new_sentence):
        mapped_batch[text_col].append(new_sentence)

        for key in batch:
            if key == text_col:
                continue
            value = batch[key][batch_index]
            mapped_batch[key].append(value)

    def map_fn(batch):
        mapped_batch = defaultdict(list)

        for i, document in enumerate(batch[text_col]):

            sentences = splitter.split(document)
            if len(sentences) <= 1:
                add_sentence(
                    mapped_batch=mapped_batch,
                    batch=batch,
                    batch_index=i,
                    new_sentence=document
                )
                continue

            for sentence_gram in ngrams(sentences, sentence_count):
                new_sentence = ' '.join(sentence_gram)
                add_sentence(
                    mapped_batch=mapped_batch,
                    batch=batch,
                    batch_index=i,
                    new_sentence=new_sentence
                )

        return mapped_batch

    return dataset.map(map_fn, batched=True)


def filter_duplicate_rows(dataset: Union[Dataset, DatasetDict], text_col: str = TEXT_COL):
    texts = set()

    def filter_fn(ex):
        text = ex[text_col]
        if text in texts:
            return False
        texts.add(text)
        return True

    return dataset.filter(filter_fn)


def normalize_labels(
        dataset: Union[Dataset, DatasetDict],
        raw_label_col: str,
        processed_label_col: str = PROCESSED_LABEL_COL
) -> Union[Dataset, DatasetDict]:
    unique_labels = get_unique_labels(dataset, raw_label_col)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    def map_fn(example):
        example[processed_label_col] = label_mapping[example[raw_label_col]]
        return example

    dataset = dataset.map(map_fn)
    return dataset


def apply_tokenizer_formatting(dataset: Union[Dataset, DatasetDict],
                               raw_text_col: str,
                               tokenizer_name: str,
                               max_length: Optional[int] = None,
                               for_pair=False) -> Union[Dataset, DatasetDict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_length = tokenizer.model_max_length if max_length is None else max_length
    if max_length is None:
        raise ValueError("Please specify a max length!")
    if for_pair:
        max_length = max_length // 2

    def map_fn(example):
        all_input_ids = tokenizer(example[raw_text_col], truncation=True, max_length=max_length).input_ids
        decoded = [tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in all_input_ids]
        return {raw_text_col: decoded}

    dataset = dataset.map(map_fn, batched=True)
    return dataset


def truncate_labels_to_length(dataset: Union[Dataset, DatasetDict], raw_label_col: str,
                              length: int) -> Union[Dataset, DatasetDict]:
    def map_fn(example):
        label = str(example[raw_label_col])
        return {raw_label_col: label[:length]}

    dataset = dataset.map(map_fn)
    return dataset


def get_raw_label_mapping(dataset: Union[Dataset, DatasetDict], raw_label_col: str) -> Dict[int, Any]:
    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets([split for split in dataset.values()])
    unique_labels = get_unique_labels(dataset, PROCESSED_LABEL_COL)
    raw_label_mapping = {}
    for example in dataset:
        processed_label = example[PROCESSED_LABEL_COL]
        if processed_label not in unique_labels:
            continue

        raw_label_mapping[processed_label] = example[raw_label_col]

        unique_labels.remove(processed_label)
        if len(unique_labels) == 0:
            return raw_label_mapping

    raise AssertionError("This should not be reachable.")
