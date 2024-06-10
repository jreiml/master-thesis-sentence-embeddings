import math
import random
from typing import List, Iterable, NamedTuple, Optional, Union

import datasets
from datasets import DatasetDict
from sentence_transformers import InputExample
from torch.utils.data import Dataset
from transformers import set_seed

from data.constants import TEXT_COL, TEXT_A_COL, TEXT_B_COL, PROCESSED_LABEL_COL


class PairDataset(Dataset):
    def __init__(self, input_examples: List[InputExample]):
        self.input_examples = list(input_examples)

    @classmethod
    def from_dataset(
            cls, dataset: datasets.Dataset, text_a_col=TEXT_A_COL, text_b_col=TEXT_B_COL,
            label_col=PROCESSED_LABEL_COL, label_scale=None
    ):
        texts_a = dataset[text_a_col]
        texts_b = dataset[text_b_col]
        labels = [0.0] * len(texts_a)
        if label_col in dataset.features:
            labels = dataset[label_col]

        input_examples = [
            InputExample(texts=[text_a, text_b], label=label * label_scale if label_scale is not None else label)
            for text_a, text_b, label in zip(texts_a, texts_b, labels)
        ]
        return cls(input_examples)

    def __len__(self):
        return len(self.input_examples)

    def __iter__(self) -> Iterable[InputExample]:
        return iter(self.input_examples)

    def __getitem__(self, item) -> InputExample:
        return self.input_examples[item]

    def __str__(self):
        return f"PairDataset(" \
               f"input_examples={[str(example) for example in self.input_examples]}" \
               f")"

    def extend(self, examples: Union['PairDataset', List[InputExample]]):
        if isinstance(examples, PairDataset):
            examples = examples.input_examples
        self.input_examples.extend(examples)

    def as_symmetric_dataset(self):
        reversed_input_examples = [InputExample(
            guid=input_example.guid, texts=list(reversed(input_example.texts)), label=input_example.label
        ) for input_example in self]
        new_input_examples = [*self.input_examples, *reversed_input_examples]
        return PairDataset(new_input_examples)

    def flatten(self, filter_duplicates=True, return_labels=False, return_list=False):
        data_pair = [(text, input_example.label) for input_example in self for text in input_example.texts]
        if filter_duplicates:
            data_pair = list(set(data_pair))

        texts = [data[0] for data in data_pair]
        labels = [data[1] for data in data_pair]
        if return_list:
            if return_labels:
                return texts, labels
            return texts

        if return_labels:
            return datasets.Dataset.from_dict({
                TEXT_COL: texts,
                PROCESSED_LABEL_COL: labels
            })
        return datasets.Dataset.from_dict({
            TEXT_COL: texts
        })

    def train_validation_test_split(self, train_percent=0.7, validation_percent=0.1, seed=42):
        set_seed(seed)
        total_percent = train_percent + validation_percent
        if total_percent > 1:
            raise ValueError("percentage too high for splits!")

        new_input_examples = random.sample(self.input_examples, len(self))
        remaining_samples = len(self)

        train_samples = min(math.ceil(len(self) * train_percent), remaining_samples)
        remaining_samples -= train_samples
        new_train = new_input_examples[:train_samples]
        new_train = PairDataset(new_train)

        validation_samples = min(math.ceil(len(self) * validation_percent), remaining_samples)
        remaining_samples -= validation_samples
        new_validation = new_input_examples[train_samples:train_samples+validation_samples]
        new_validation = None if len(new_validation) == 0 else PairDataset(new_validation)

        new_test = new_input_examples[train_samples+validation_samples:]
        new_test = None if len(new_test) == 0 else PairDataset(new_test)

        return PairDatasetDict(
            train=new_train,
            validation=new_validation,
            test=new_test,
        )

    def get_unique_label_count(self):
        if len(self) == 0:
            return 0

        if isinstance(self[0].label, float):
            return 1

        return len(set([input_example.label for input_example in self]))


class PairDatasetDict(NamedTuple):
    train: PairDataset
    validation: Optional[PairDataset] = None
    test: Optional[PairDataset] = None

    @classmethod
    def from_dataset_dict(cls, dataset: datasets.DatasetDict, text_a_col=TEXT_A_COL, text_b_col=TEXT_B_COL,
            label_col=PROCESSED_LABEL_COL, label_scale=None):
        train = PairDataset.from_dataset(dataset["train"], text_a_col, text_b_col, label_col, label_scale)
        validation = PairDataset.from_dataset(dataset["validation"], text_a_col, text_b_col, label_col, label_scale) \
            if "validation" in dataset else None
        test = PairDataset.from_dataset(dataset["test"], text_a_col, text_b_col, label_col, label_scale) \
            if "test" in dataset else None

        return cls(
            train=train,
            validation=validation,
            test=test
        )

    def as_symmetric_dataset(self, include_validation=True, include_test=False):
        train = self.train.as_symmetric_dataset()

        validation = self.validation
        if include_validation and self.validation is not None:
            validation = self.validation.as_symmetric_dataset()

        test = self.test
        if include_test and self.test is not None:
            test = self.test.as_symmetric_dataset()

        return PairDatasetDict(
            train=train,
            validation=validation,
            test=test
        )

    def flatten(self, filter_duplicates=True, return_labels=False):
        train = self.train.flatten(filter_duplicates, return_labels=return_labels)

        validation = self.validation
        if self.validation is not None:
            validation = self.validation.flatten(filter_duplicates, return_labels=return_labels)

        test = self.test
        if self.test is not None:
            test = self.test.flatten(filter_duplicates, return_labels=return_labels)

        return DatasetDict({
            "train": train,
            "validation": validation,
            "test": test
        })

    def as_pair_dataset(self):
        input_examples = self.train.input_examples
        if self.validation is not None:
            input_examples = [*input_examples, *self.validation.input_examples]
        if self.test is not None:
            input_examples = [*input_examples, *self.test.input_examples]

        return PairDataset(input_examples)

    def get_unique_label_count(self):
        return self.train.get_unique_label_count()
