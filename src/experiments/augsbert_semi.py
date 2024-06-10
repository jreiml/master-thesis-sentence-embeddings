import enum
import logging
import os.path
import pickle
from typing import NamedTuple, Any, Dict

import numpy as np
from sentence_transformers import LoggingHandler, InputExample
from transformers import set_seed

from bi_encoder.train import train_bi_encoder
from cross_encoder.train import train_or_load_cross_bi_encoder, train_or_load_cross_encoder
from data.experiment_dataset_loader import load_bws_in_topic, load_bws_cross_topic, load_spanish_sts
from data.pair_dataset import PairDataset, PairDatasetDict
from data.pair_generator import SentencePairGenerator
from data.pair_set import PairSet
from experiments.sts_correlation import write_table
from experiments.utils import BATCH_SIZE, LEARNING_RATE, get_kde_dataset, \
    annotate_dataset, get_train_epochs, get_train_seeds, ensure_dir_exists, \
    get_english_cbe_model, get_spanish_cbe_model
from util.args import BiEncoderArguments, CrossEncoderArguments


class AugSBertSetting(enum.Enum):
    NONE = "NONE"
    RANDOM = "RANDOM"
    KDE = "KDE"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    BM25 = "BM25"


class AugSBertScoreCache(NamedTuple):
    bi_scores: Dict[Any, float]
    cbe_scores: Dict[Any, float]
    aug_scores: Dict[Any, float]


def save_cache(name, cache):
    with open(f'output/{name}/augsbert_semi_cache.pkl', 'wb') as fp:
        pickle.dump(cache, fp)


def load_cache(name):
    if os.path.exists(f'output/{name}/augsbert_semi_cache.pkl'):
        with open(f'output/{name}/augsbert_semi_cache.pkl', 'rb') as fp:
            return pickle.load(fp)

    ensure_dir_exists('output/augsbert-semi')
    return AugSBertScoreCache(
        bi_scores={},
        cbe_scores={},
        aug_scores={},
    )


def all_in_cache_for_seed_and_dataset(cache, seed, dataset_name, strategies):
    if (seed, dataset_name) not in cache.bi_scores:
        return False

    if (seed, dataset_name) not in cache.cbe_scores:
        return False

    for strategy in strategies:
        if (seed, dataset_name, strategy.value) not in cache.aug_scores:
            return False

    return True


def write_results(name, cache, seeds, strategies):
    output_path = f'output/{name}/augsbert_semi_scores.txt'
    dataset_display_names = [
        ("spanish-sts-fixed", "Spanish-STS"),
        ("bws-cross-topic", "BWS (cross-topic)"),
        ("bws-in-topic", "BWS (in-topic)"),
    ]

    header = ["Model / Dataset"]
    rows = []
    rows.append(["Baseline", "30.27", "5.53", "6.98"])
    rows.append(["Baseline", "86.86", "53.43", "57.23"])
    rows.append(["BERT (Upper-bound)"])
    rows.append(["SBERT (Lower-bound)"])
    rows.append(["AugSBERT-None"])
    rows.append(["AugSBERT-R.S."])
    rows.append(["AugSBERT-KDE"])
    rows.append(["AugSBERT-BM25"])
    rows.append(["AugSBERT-S.S."])

    for name, display_name in dataset_display_names:
        header.append(display_name)

        bi_scores = []
        cbe_scores = []
        for seed in seeds:
            base_key = (seed, name)
            bi_scores.append(cache.bi_scores[base_key])
            cbe_scores.append(cache.cbe_scores[base_key])

        cbe_score_mean = np.mean(cbe_scores)
        cbe_score_stdev = 0 if len(cbe_scores) == 1 else np.std(cbe_scores, ddof=1)
        rows[2].append(f"{cbe_score_mean:.2%} ± {cbe_score_stdev:.2%}")
        bi_score_mean = np.mean(bi_scores)
        bi_score_stdev = 0 if len(bi_scores) == 1 else np.std(bi_scores, ddof=1)
        rows[3].append(f"{bi_score_mean:.2%} ± {bi_score_stdev:.2%}")

        for i, strategy in enumerate(strategies, start=4):
            aug_scores = []
            for seed in seeds:
                aug_key = (seed, name, strategy.value)
                aug_scores.append(cache.aug_scores[aug_key])

            aug_score_mean = np.mean(aug_scores)
            aug_score_stdev = 0 if len(aug_scores) == 1 else np.std(aug_scores, ddof=1)
            rows[i].append(f"{aug_score_mean:.2%} ± {aug_score_stdev:.2%}")

    write_table(output_path, header, rows)


def do_run(experiment_name, base_model, multilingual_base_model, use_cross_bi_encoder, full):
    bws_in_topic = load_bws_in_topic().as_pair_dataset_dict()
    bws_cross_topic = load_bws_cross_topic().as_pair_dataset_dict()
    spanish_sts = load_spanish_sts()

    seed_optimization_steps = 10 if full else 2
    seeds = get_train_seeds(full)
    train_epochs = get_train_epochs(full)
    datasets = [
        ("spanish-sts-fixed", spanish_sts, True),
        ("bws-cross-topic", bws_cross_topic, False),
        ("bws-in-topic", bws_in_topic, False),
    ]
    strategies = [AugSBertSetting.NONE, AugSBertSetting.RANDOM, AugSBertSetting.KDE, AugSBertSetting.BM25, AugSBertSetting.SEMANTIC_SEARCH]
    cache = load_cache(experiment_name)

    for seed in seeds:
        for dataset_name, dataset, multilingual in datasets:
            if all_in_cache_for_seed_and_dataset(cache, seed, dataset_name, strategies):
                continue

            base_key = (seed, dataset_name)
            cbe_output_path = f"output/{experiment_name}/{dataset_name}/{seed}/cross-bi-encoder/"
            bi_encoder_output_path = f"output/{experiment_name}/{dataset_name}/{seed}/bi-encoder"
            aug_bi_encoder_output_path = f"output/{experiment_name}/{dataset_name}/{seed}/aug-bi-encoder"

            model = multilingual_base_model if multilingual else base_model
            cbe_args = CrossEncoderArguments(
                model_name=model,
                output_path=cbe_output_path,
                train_batch_size=BATCH_SIZE,
                eval_batch_size=BATCH_SIZE,
                train_epochs=train_epochs,
                max_length=512,
                learning_rate=LEARNING_RATE,
                use_cross_bi_encoder=use_cross_bi_encoder,
                seed_optimization_steps=seed_optimization_steps,
                train_seed=seed,
            )
            cbe_model, cbe_score = train_or_load_cross_encoder(cbe_args, dataset.as_symmetric_dataset())
            cache.cbe_scores[base_key] = cbe_score
            save_cache(experiment_name, cache)

            bi_args = BiEncoderArguments(
                model_name=model,
                output_path=bi_encoder_output_path,
                batch_size=BATCH_SIZE,
                train_epochs=train_epochs,
                max_length=512,
                learning_rate=LEARNING_RATE,
                save_checkpoints=False,
                seed_optimization_steps=seed_optimization_steps,
                train_seed=seed,
            )
            _, bi_score = train_bi_encoder(bi_args, dataset)
            cache.bi_scores[base_key] = bi_score
            save_cache(experiment_name, cache)

            train_sentences = dataset.train.flatten(filter_duplicates=True, return_list=True)
            generator = SentencePairGenerator(train_sentences, return_pair_dataset=True)
            for strategy in strategies:
                aug_key = (seed, dataset_name, strategy.value)
                if aug_key in cache.aug_scores:
                    continue

                set_seed(seed)
                skip_pairs = PairSet(dataset.train)
                if strategy == AugSBertSetting.NONE:
                    silver_dataset = PairDataset([])
                elif strategy == AugSBertSetting.RANDOM:
                    silver_dataset = generator.generate_random(5, skip_pairs=skip_pairs)
                elif strategy == AugSBertSetting.BM25:
                    silver_dataset = generator.generate_bm25(5, skip_pairs=skip_pairs)
                elif strategy == AugSBertSetting.SEMANTIC_SEARCH:
                    silver_dataset = generator.generate_semantic_search(
                        5, semantic_model_name=bi_encoder_output_path, skip_pairs=skip_pairs
                    )
                elif strategy == AugSBertSetting.KDE:
                    silver_dataset = get_kde_dataset(
                        cbe_model, cbe_args, generator, bi_encoder_output_path, dataset.train,
                        use_random=True, use_semantic_search=True, use_hard_cutoff_sampling=True
                    )
                else:
                    raise ValueError("Unimplemented!")

                # last parameter not needed for CE & CBE model
                dataset_pairs = [
                    [input_example.texts for input_example in d]
                    for d in [dataset.train, silver_dataset]
                ]
                gold_train_scores, silver_train_scores = annotate_dataset(
                    cbe_model, cbe_args, [dataset.train, silver_dataset], dataset_pairs, None
                )
                gold_train_samples = [
                    InputExample(texts=input_example.texts, label=score)
                    for input_example, score in zip(dataset.train, gold_train_scores)
                ]
                silver_samples = [
                    InputExample(texts=input_example.texts, label=score)
                    for input_example, score in zip(silver_dataset, silver_train_scores)
                ]
                augmented_train_dataset = PairDataset([*gold_train_samples, *silver_samples])
                augmented_dataset = PairDatasetDict(
                    train=augmented_train_dataset,
                    validation=dataset.validation,
                    test=dataset.test,
                )

                aug_bi_args = BiEncoderArguments(
                    model_name=model,
                    output_path=aug_bi_encoder_output_path,
                    batch_size=BATCH_SIZE,
                    train_epochs=train_epochs,
                    max_length=256,
                    learning_rate=LEARNING_RATE,
                    save_checkpoints=False,
                    seed_optimization_steps=seed_optimization_steps,
                    train_seed=seed
                )
                _, aug_score = train_bi_encoder(aug_bi_args, augmented_dataset)
                cache.aug_scores[aug_key] = aug_score
                save_cache(experiment_name, cache)

    write_results(experiment_name, cache, seeds, strategies)


def run_augsbert_semi_experiment_cbe(full):
    experiment_name = "augsbert-semi"
    base_model = get_english_cbe_model(full)
    multilingual_base_model = get_spanish_cbe_model(full)
    use_cross_bi_encoder = True

    do_run(experiment_name, base_model, multilingual_base_model, use_cross_bi_encoder, full)


def run_augsbert_semi_experiment_ce(full):
    experiment_name = "augsbert-semi-ce"
    base_model = get_english_cbe_model(full)
    multilingual_base_model = get_spanish_cbe_model(full)
    use_cross_bi_encoder = False

    do_run(experiment_name, base_model, multilingual_base_model, use_cross_bi_encoder, full)


def run_augsbert_semi_experiment(full):
    run_augsbert_semi_experiment_cbe(full)
    run_augsbert_semi_experiment_ce(full)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    run_augsbert_semi_experiment(True)
