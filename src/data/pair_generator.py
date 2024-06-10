import logging
import random
from collections import defaultdict
from typing import Dict, Set, Union, Optional, Any, Callable, List, Iterable, Tuple

import numpy as np
import torch
from datasets import Dataset
from nltk import word_tokenize
from rank_bm25 import BM25Okapi, BM25
from sentence_transformers import SentenceTransformer, util, InputExample
from tqdm.auto import tqdm

from data.constants import TEXT_A_COL, TEXT_B_COL
from data.pair_dataset import PairDataset
from data.pair_preprocessing import map_to_pairs
from data.pair_set import PairSet

logger = logging.getLogger(__name__)


def get_label_to_indices(dataset: Dataset, label_col: str) -> Dict[Any, List[int]]:
    label_to_indices = defaultdict(list)
    for i, example in enumerate(dataset):
        label_to_indices[example[label_col]].append(i)
    return label_to_indices


def get_label_distribution(label_to_indices: Dict[Any, List[int]]) -> Dict[Any, float]:
    label_distribution = dict()
    total_indices = sum([len(indices) for indices in label_to_indices.values()])
    for label, indices in label_to_indices.items():
        label_distribution[label] = len(indices) / total_indices
    return label_distribution


def get_label_pair_distribution(label_to_indices: Dict[Any, List[int]]) -> Dict[Tuple[Any, Any], float]:
    label_distribution = get_label_distribution(label_to_indices)
    label_pair_distribution = dict()
    for label_a, probability_a in label_distribution.items():
        for label_b, probability_b in label_distribution.items():
            joint_probability = probability_a * probability_b
            label_pair_distribution[(label_a, label_b)] = joint_probability
            label_pair_distribution[(label_b, label_a)] = joint_probability

    return label_pair_distribution


class SentencePairGenerator:
    def __init__(self, sentences: List[str], labels: Optional[List[Any]] = None, return_pair_dataset=False):
        if labels is not None and len(labels) != len(sentences):
            logger.error(f"Length mismatch between sentences and labels. {len(sentences)} != {len(labels)}")

        if labels is not None and return_pair_dataset:
            logger.error("Cannot return pair dataset with labels! Returning dataset instead.")
            self.return_pair_dataset = False
        else:
            self.return_pair_dataset = return_pair_dataset
        self.sentences = sentences
        self.labels = labels

        self.model_cache: Dict[str, SentenceTransformer] = {}

    def _get_top_k_matches(self, skip_pairs: PairSet, index_a: int, sorted_best_indices_b: Iterable[int],
                           top_k: int, matches: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        matches = matches if matches is not None else list()
        found = 0
        for index_b in sorted_best_indices_b:
            if index_a == index_b:
                continue

            added = skip_pairs.add((index_a, index_b))
            if not added:
                continue

            matches.append((index_a, index_b))
            found += 1

            if found == top_k:
                break

        return matches

    def _find_best_matches_bm25(
            self, bm25_docs: BM25, skip_pairs: PairSet, docs: List[List[str]],
            indices_a: List[int], indices_b: List[int], top_k: int, least_similar: bool
    ) -> List[Tuple[int, int]]:
        matches = list()

        index_iter = tqdm(indices_a, desc=f"BM25 matching")

        for index_a in index_iter:
            doc_a = docs[index_a]
            scores = bm25_docs.get_scores(doc_a)
            best_matches = np.argsort(scores)
            best_matches = [indices_b[match] for match in best_matches]
            if not least_similar:
                best_matches = reversed(best_matches)
            self._get_top_k_matches(skip_pairs, index_a, best_matches, top_k, matches)

        return matches

    def _find_best_matches_semantic_search(
            self, best_matches: np.ndarray, skip_pairs: PairSet, indices_a: List[int], top_k: int
    ) -> List[Tuple[int, int]]:
        matches = list()
        index_iter = tqdm(indices_a, desc=f"Semantic Search matching")

        for index_a in index_iter:
            indices_b = best_matches[index_a]
            self._get_top_k_matches(skip_pairs, index_a, indices_b, top_k, matches)

        return matches

    def _find_random_matches(
            self, skip_pairs: PairSet, indices_a: List[int], indices_b: List[int], amount: int
    ) -> List[Tuple[int, int]]:
        matches = list()

        def get_b_candidates():
            while True:
                b_candidates_iter = list(indices_b)
                random.shuffle(b_candidates_iter)
                for candidate in b_candidates_iter:
                    yield candidate

        b_candidates = get_b_candidates()
        index_iter = tqdm(indices_a, desc=f"Random matching")

        for index_a in index_iter:
            self._get_top_k_matches(skip_pairs, index_a, b_candidates, amount, matches)

        return matches

    def _generate_and_gather(self, skip_pairs: PairSet, matching_fn: Callable) -> Union[Dataset, PairDataset]:
        matches = []

        indices_a = list(range(len(self.sentences)))
        indices_b = list(indices_a)
        new_matches = matching_fn(skip_pairs, indices_a, indices_b)
        matches.extend(new_matches)

        result = map_to_pairs(self.sentences, matches, self.labels)
        if self.return_pair_dataset:
            result = PairDataset(
                [InputExample(texts=[example[TEXT_A_COL], example[TEXT_B_COL]]) for example in result]
            )
        return result

    def generate_random(
            self, amount, skip_pairs: Optional[Union[Set, PairSet]] = None
    ) -> Union[Dataset, PairDataset]:
        skip_pairs = PairSet(skip_pairs)

        def matching_fn(local_skip_pairs: PairSet, indices_a: List[int], indices_b: List[int]):
            return self._find_random_matches(local_skip_pairs, indices_a, indices_b, amount=amount)

        return self._generate_and_gather(skip_pairs, matching_fn)

    def _get_semantic_search_matching_fn(self, top_k: int, least_similar: bool, semantic_model_name: str) -> Callable:
        if semantic_model_name not in self.model_cache:
            self.model_cache[semantic_model_name] = SentenceTransformer(semantic_model_name)
        embeddings = self.model_cache[semantic_model_name].encode(self.sentences, convert_to_tensor=True)
        cos_scores = util.cos_sim(embeddings, embeddings)
        del embeddings
        descending = not least_similar
        best_matches = torch.argsort(cos_scores, dim=1, descending=descending).cpu().numpy()

        def matching_fn(skip_pairs: PairSet, indices_a: List[int], indices_b: List[int]):
            return self._find_best_matches_semantic_search(best_matches, skip_pairs, indices_a, top_k=top_k)

        return matching_fn

    def generate_semantic_search(
            self, top_k: int, least_similar: bool = False, semantic_model_name: str = 'all-MiniLM-L6-v2',
            skip_pairs: Optional[Union[Set, PairSet]] = None
    ) -> Union[Dataset, PairDataset]:
        if top_k < 1:
            raise ValueError("Valid value range for top_k: [1, infinity[")

        skip_pairs = PairSet(skip_pairs)
        matching_fn = self._get_semantic_search_matching_fn(top_k, least_similar, semantic_model_name)
        return self._generate_and_gather(skip_pairs, matching_fn)

    def _get_bm25_matching_fn(self, top_k: int, least_similar: bool) -> Callable:
        docs = list()
        for example in self.sentences:
            doc = word_tokenize(example.lower())
            docs.append(doc)
        bm25 = BM25Okapi(docs)

        def matching_fn(skip_pairs: PairSet, indices_a: List[Any], indices_b: List[Any]):
            return self._find_best_matches_bm25(bm25, skip_pairs, docs, indices_a,
                                                indices_b, top_k=top_k, least_similar=least_similar)

        return matching_fn

    def generate_bm25(
            self, top_k: int, least_similar: bool = False, skip_pairs: Optional[Union[Set, PairSet]] = None
    ) -> Union[Dataset, PairDataset]:
        if top_k < 1:
            raise ValueError("Valid value range for top_k: [1, infinity[")

        skip_pairs = PairSet(skip_pairs)
        matching_fn = self._get_bm25_matching_fn(top_k, least_similar)
        return self._generate_and_gather(skip_pairs, matching_fn)

