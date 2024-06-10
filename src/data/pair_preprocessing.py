from typing import Tuple, List, Optional, Any

from datasets import Dataset

from data.constants import PROCESSED_LABEL_A_COL, PROCESSED_LABEL_B_COL, TEXT_A_COL, TEXT_B_COL


def map_to_pairs(sentences: List[str], matches: List[Tuple[int, int]], labels: Optional[List[Any]] = None) -> Dataset:
    text_a, text_b, label_a, label_b = [], [], [], []
    for idx_a, idx_b in matches:
        text_a.append(sentences[idx_a])
        text_b.append(sentences[idx_b])

        if labels is not None:
            label_a.append(labels[idx_a])
            label_b.append(labels[idx_b])

    if labels is None:
        return Dataset.from_dict({
            TEXT_A_COL: text_a,
            TEXT_B_COL: text_b,
        })

    return Dataset.from_dict({
        TEXT_A_COL: text_a,
        TEXT_B_COL: text_b,
        PROCESSED_LABEL_A_COL: label_a,
        PROCESSED_LABEL_B_COL: label_b
    })
