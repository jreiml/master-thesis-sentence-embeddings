import random
from typing import List

import numpy as np
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm


def renumerate(sequence, start=None):
    if start is None:
        start = len(sequence) - 1
    n = start
    for elem in sequence[::-1]:
        yield n, elem
        n -= 1


def sample_pair_dataset_via_kde(
        gold_scores: np.ndarray, silver_scores: np.ndarray, use_hard_cutoff_sampling: bool = True
) -> List[int]:
    gold_kde = gaussian_kde(gold_scores)
    silver_candidates = list(range(len(silver_scores)))
    random.shuffle(silver_candidates)

    # start with 2 points (minimum)
    initial_candidates = 2
    silver_dataset = silver_candidates[:initial_candidates]
    silver_candidates = silver_candidates[initial_candidates:]

    progress = tqdm(desc="KDE Sampling")
    silver_kde = gaussian_kde([silver_scores[candidate_idx] for candidate_idx in silver_dataset])
    converged = False
    while not converged:
        converged = True

        for i, candidate_idx in renumerate(list(silver_candidates)):
            score = silver_scores[candidate_idx]
            gold_density = gold_kde(score)
            silver_density = silver_kde(score)

            if gold_density < silver_density:
                # Sample with P = 0 if F_gold(s) < F_silver(s) (ours)
                if use_hard_cutoff_sampling:
                    continue

                # Sample with P = F_gold(s) / F_silver(s) if F_gold(s) < F_silver(s) (original AugSBERT)
                add_anyways_probability = gold_density / silver_density
                if add_anyways_probability <= random.random():
                    continue

            candidate = silver_candidates.pop(i)
            silver_dataset.append(candidate)
            progress.update()
            silver_kde = gaussian_kde([silver_scores[candidate_idx] for candidate_idx in silver_dataset])

            # Only one iteration if sampling with P = F_gold(s) / F_silver(s) instead of P = 0
            # for case F_gold(s) < F_silver(s)
            if use_hard_cutoff_sampling:
                converged = False

    return silver_dataset
