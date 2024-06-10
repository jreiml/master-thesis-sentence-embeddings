from typing import List, Union, NamedTuple, Optional, Iterable, Set, Dict, Tuple, Any, Callable

import datasets
import numpy as np
from datasets import DatasetDict
from sentence_transformers import InputExample
from torch.utils.data import Dataset

from data.constants import TEXT_COL, PROCESSED_LABEL_COL
from data.pair_dataset import PairDatasetDict, PairDataset


class MultitaskInputExample:
    """
    Structure for one input example with texts, the labels and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,
                 similarity_label: float = 0,
                 domain_labels: List[Union[float, int]] = None):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param similarity_label
            the cross-encoder label for the example
        :param domain_labels
            the individual label for each text of the example
        """
        self.guid = guid
        self.texts = texts
        self.similarity_label = similarity_label
        self.domain_labels = domain_labels

    @property
    def label(self) -> List[Union[int, float, np.ndarray]]:
        if self.domain_labels is None:
            return [self.similarity_label]
        return [self.similarity_label, *self.domain_labels]

    def __str__(self):
        texts = "; ".join(self.texts)
        domain_labels = "None"
        if self.domain_labels is not None:
            domain_labels = "; ".join([str(l) for l in self.domain_labels])
        return f"<MultitaskInputExample> similarity-label: {self.similarity_label}, " \
               f"domain-labels: {domain_labels}, texts: {texts}"


class MultitaskPairDataset(Dataset):
    def __init__(self, input_examples: List[MultitaskInputExample],
                 domain_label_pair_distribution: Dict[Tuple[Any, Any], float]):
        self.input_examples = input_examples
        self.domain_label_pair_distribution = domain_label_pair_distribution
        self.unique_domain_labels = set([
            label
            for input_example in input_examples
            if input_example.domain_labels is not None
            for label in input_example.domain_labels
        ])

    def __len__(self):
        return len(self.input_examples)

    def __iter__(self) -> Iterable[MultitaskInputExample]:
        return iter(self.input_examples)

    def __getitem__(self, item) -> MultitaskInputExample:
        return self.input_examples[item]

    def get_unique_domain_labels(self) -> Set[Union[float, int]]:
        return set(self.unique_domain_labels)

    def get_unique_domain_label_count(self) -> int:
        return len(self.unique_domain_labels)

    def get_input_example_domain_label_probabilities(self) -> np.ndarray:
        if self.get_unique_domain_label_count() == 0:
            return np.array([])

        return np.array(
            [self.domain_label_pair_distribution[tuple(input_example.domain_labels)]
             for input_example in self.input_examples]
        )

    def get_input_example_domain_label_weights_for_uniform_distribution(self) -> np.ndarray:
        return 1 / self.get_input_example_domain_label_probabilities()

    def as_pair_dataset(self) -> PairDataset:
        input_examples = [
            InputExample(guid=input_example.guid, texts=input_example.texts, label=input_example.similarity_label)
            for input_example in self.input_examples
        ]
        return PairDataset(input_examples)

    def as_dataset(self, filter_duplicates=True):
        texts = []
        labels = []
        unique_texts = set()
        for input_example in self:
            for text, label in zip(input_example.texts, input_example.domain_labels):
                if filter_duplicates:
                    if text in unique_texts:
                        continue
                    unique_texts.add(text)

                texts.append(text)
                labels.append(label)

        return datasets.Dataset.from_dict({
            TEXT_COL: texts,
            PROCESSED_LABEL_COL: labels
        })


class MultitaskPairDatasetDict(NamedTuple):
    train: MultitaskPairDataset
    validation: Optional[MultitaskPairDataset] = None
    test: Optional[MultitaskPairDataset] = None

    def get_unique_domain_labels(self) -> Set[Union[float, int]]:
        unique_domain_labels = self.train.get_unique_domain_labels()

        if self.validation is not None:
            unique_domain_labels.update(self.validation.get_unique_domain_labels())

        if self.test is not None:
            unique_domain_labels.update(self.test.get_unique_domain_labels())

        return unique_domain_labels

    def get_unique_domain_label_count(self) -> int:
        return len(self.get_unique_domain_labels())

    def as_pair_dataset_dict(self) -> PairDatasetDict:
        train = self.train.as_pair_dataset()
        validation = None if self.validation is None else self.validation.as_pair_dataset()
        test = None if self.test is None else self.test.as_pair_dataset()

        return PairDatasetDict(
            train=train,
            validation=validation,
            test=test
        )

    def map(self, map_fn: Callable[[MultitaskInputExample], MultitaskInputExample]) -> 'MultitaskPairDatasetDict':
        train_examples_mapped = [map_fn(input_example) for input_example in self.train.input_examples]
        train_mapped = MultitaskPairDataset(train_examples_mapped, self.train.domain_label_pair_distribution)
        validation_mapped = None
        if self.validation is not None:
            validation_examples_mapped = [map_fn(input_example) for input_example in self.validation.input_examples]
            validation_mapped = MultitaskPairDataset(
                validation_examples_mapped, self.validation.domain_label_pair_distribution
            )
        test_mapped = None
        if self.test is not None:
            test_examples_mapped = [map_fn(input_example) for input_example in self.test.input_examples]
            test_mapped = MultitaskPairDataset(test_examples_mapped, self.test.domain_label_pair_distribution)

        return MultitaskPairDatasetDict(
            train=train_mapped,
            validation=validation_mapped,
            test=test_mapped
        )

    def as_dataset_dict(self, filter_duplicates=True) -> DatasetDict:
        train = self.train.as_dataset(filter_duplicates)
        validation = None if self.validation is None else self.validation.as_dataset(filter_duplicates)
        test = None if self.test is None else self.test.as_dataset(filter_duplicates)

        return DatasetDict({
            "train": train,
            "validation": validation,
            "test": test
        })
