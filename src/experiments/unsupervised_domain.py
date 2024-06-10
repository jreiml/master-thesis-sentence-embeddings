import logging
import os
import pickle
import sys

import numpy as np
import torch
from sentence_transformers import LoggingHandler, SentenceTransformer, models
from transformers import AutoTokenizer

from bi_encoder.multitask_distill import train_multitask_distill_bi_encoder
from bi_encoder.train import train_bi_encoder
from data.dataset_loader import load_processed_dataset
from data.experiment_dataset_loader import load_stsb
from data.useb import run_on
from data.useb.downloading import download
from experiments.distill_pipeline import train_multitask_distill_pipeline
from experiments.utils import get_train_seeds, get_english_default_model, ensure_dir_exists, get_english_cbe_model, \
    get_english_sbert_model, write_table
from pretrain.train import train_nm_adapted_encoder
from util.args import PretrainArguments, PretrainMode, PoolingStrategy, MultitaskDistillBiEncoderArguments, \
    CrossEncoderArguments, PairGenerationStrategy, DataArguments, BiEncoderArguments


def print_results(full):
    seeds = get_train_seeds(full)

    dataset_names = ["askubuntu", "cqadupstack", "twitter", "scidocs"]
    dataset_to_cache = {dataset_name: load_cache(dataset_name) for dataset_name in dataset_names}
    header = [
        "Method\nSubtask",
        "AskU.",
        "CQADup.",
        "Twitter\nTURL PIT Avg.",
        "SciDocs\nCite CC CR CV Avg.",
        "Avg."
    ]

    rows = []
    for train_mode in [
        "pretrained",
        "augsbert-default",
        "augsbert-domain",
        "augsbert-domain-full",
        PretrainMode.TSDAE.value + "-fixed",
        PretrainMode.NM.value, PretrainMode.NM.value + "-mean",
        PretrainMode.TSADE_NM.value, PretrainMode.TSADE_NM.value + "-mean",
        PretrainMode.MLM_NM.value, PretrainMode.MLM_NM.value + "-mean",
        PretrainMode.SIMCSE.value, PretrainMode.SIMCSE.value + "-mean",
        PretrainMode.MLM_SIMCSE.value, PretrainMode.MLM_SIMCSE.value + "-mean"
    ]:
        row = [train_mode]
        avg_scores = []

        for dataset_name in dataset_names:
            cache = dataset_to_cache[dataset_name]

            scores = []
            for seed in seeds:
                if train_mode == "pretrained":
                    seed = 0
                cache_key = (f"{train_mode}", seed)
                if cache_key not in cache:
                    logging.error(f"Key {cache_key} not computed!")
                    continue

                if dataset_name == "askubuntu":
                    scores.append([cache[cache_key]["map_askubuntu_title"]])
                elif dataset_name == "cqadupstack":
                    scores.append([cache[cache_key]["map@100_cqadupstack_avg"]])
                elif dataset_name == "twitter":
                    scores.append([
                        cache[cache_key]["ap_twitter_twitterurl"],
                        cache[cache_key]["ap_twitter_pit"],
                        cache[cache_key]["ap_twitter_avg"],
                    ])
                elif dataset_name == "scidocs":
                    scores.append([
                        cache[cache_key]["map_scidocs_cite_cosine"],
                        cache[cache_key]["map_scidocs_cocite_cosine"],
                        cache[cache_key]["map_scidocs_coread_cosine"],
                        cache[cache_key]["map_scidocs_coview_cosine"],
                        cache[cache_key]["map_scidocs_cosine_avg"],
                    ])

            if len(scores) == 0:
                continue

            scores = np.array(scores).mean(axis=0)
            if np.isnan(scores.any()):
                avg_scores.append(np.nan)
                scores_str = "nan"
            else:
                avg_scores.append(scores[-1])
                scores_str = "\t".join([f"{score/100:.1%}".replace("%", "") for score in scores])

            row.append(scores_str)

        avg_score = np.array(avg_scores).mean()
        avg_scores_str = f"{avg_score/100:.1%}".replace("%", "")
        row.append(avg_scores_str)
        rows.append(row)

    output_path = f"output/unsupervised-domain/results.txt"
    write_table(output_path, header, rows)



def save_cache(dataset_name, cache):
    with open(f'output/unsupervised-domain/{dataset_name}-cache.pkl', 'wb') as fp:
        pickle.dump(cache, fp)


def load_cache(dataset_name):
    file_name = f'output/unsupervised-domain/{dataset_name}-cache.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as fp:
            return pickle.load(fp)
    ensure_dir_exists('output/unsupervised-domain')

    return {}


def evaluate(model_path_or_name, dataset_name, pooling_mode):
    word_embedding_model = models.Transformer(model_path_or_name, max_seq_length=512)
    embedding_dim = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(embedding_dim, pooling_mode)
    encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    @torch.no_grad()
    def semb_fn(sentences):
        return torch.tensor(encoder.encode(sentences, show_progress_bar=False))

    eval_name = dataset_name if dataset_name != "twitter" else "twitterpara"
    return run_on(
        eval_name,
        semb_fn=semb_fn,
        eval_type='test',
        data_eval_path='datasets/usep/data-eval'
    )


def run_nm_experiments(dataset_name, cache, full, mean=False):
    # TODO undo
    bert_model = get_english_default_model(full)
    seeds = get_train_seeds(full)

    for seed in seeds:
        for train_mode in [PretrainMode.SIMCSE]: # [PretrainMode.NM, PretrainMode.TSADE_NM, PretrainMode.MLM_NM, PretrainMode.TSDAE]:
            train_mode_str = train_mode.value
            if mean:
                train_mode_str += "-mean"
            cache_key = (train_mode_str, seed)
            if cache_key in cache:
                continue

            dataset_path = f"datasets/usep/data-train/{dataset_name}/unsupervised/train.txt"
            data_args = DataArguments(
                dataset_path,
                is_hugging_face_dataset=False,
                text_col="text",
                label_col=None,
                is_text_dataset=True,
                strip_whitespaces=True,
                filter_duplicates=True,
                data_generation_seed=seed,
                dataset_size_limit=800000,
            )

            dataset = load_processed_dataset(data_args)
            output_path = f"output/unsupervised-domain/{dataset_name}/{train_mode_str}/{seed}"
            ensure_dir_exists(output_path)

            args = PretrainArguments(
                train_mode=train_mode,
                model_name=bert_model,
                noise_probability=0.6,
                pooling_strategy=PoolingStrategy.MEAN if mean else PoolingStrategy.CLS,
                output_path=output_path,
                train_batch_size=64 if train_mode != PretrainMode.TSDAE else 8,
                max_steps=12500 if train_mode != PretrainMode.TSDAE else 100000,
                lr_scheduler="linear" if train_mode != PretrainMode.TSDAE else "constant",
                warmup_percent=0.1 if train_mode != PretrainMode.TSDAE else 0.0,
                weight_decay=0.01 if train_mode != PretrainMode.TSDAE else 0.0,
                max_length=512,
                learning_rate=3e-5,
                train_seed=seed
            )
            train_nm_adapted_encoder(args, dataset)

            encoder_path = os.path.join(output_path, "encoder")
            result = evaluate(encoder_path, dataset_name, 'mean' if mean else 'cls')
            cache[cache_key] = result
            save_cache(dataset_name, cache)


def run_augsbert_domain_experiments(dataset_name, cache, full):
    seeds = get_train_seeds(full)
    roberta_model = get_english_cbe_model(full)

    for seed in seeds:
        cache_key = ("augsbert-domain", seed)
        cache_key_full = ("augsbert-domain-full", seed)
        if cache_key in cache and cache_key_full in cache:
            continue

        dataset_path = f"datasets/usep/data-train/{dataset_name}/unsupervised/train.txt"
        output_path_pretrain = f"output/unsupervised-domain/{dataset_name}/augsbert-domain/{seed}/pretrain"
        output_path_cross = f"output/unsupervised-domain/{dataset_name}/augsbert-domain/{seed}/cross-encoder"
        output_path_bi = f"output/unsupervised-domain/{dataset_name}/augsbert-domain/{seed}/bi-encoder"
        output_path_data_raw = f"output/unsupervised-domain/{dataset_name}/augsbert/{seed}/data-raw"
        output_path_data_pair = f"output/unsupervised-domain/{dataset_name}/augsbert/{seed}/data-pair"
        output_path_silver = f"output/unsupervised-domain/{dataset_name}/augsbert-domain/{seed}/data-silver.pkl"

        data_args = DataArguments(
            dataset_path,
            is_hugging_face_dataset=False,
            text_col="text",
            label_col=None,
            is_text_dataset=True,
            strip_whitespaces=True,
            filter_duplicates=True,
            pair_generation_strategies=[PairGenerationStrategy.BM25, PairGenerationStrategy.RANDOM],
            top_k_pairs=[1, 1],
            pair_generation_batch_size=2000,
            dataset_size_limit=400000, # 400000 samples => 800000 pairs => 64 batch size ==> 12.5k train steps
            raw_dataset_cache_path=output_path_data_raw,
            pair_dataset_cache_path=output_path_data_pair,
            silver_dataset_cache_path=output_path_silver,
            data_generation_seed=seed,
        )

        pretrain_args = PretrainArguments(
            train_mode=PretrainMode.MLM_PAIR,
            model_name=roberta_model,
            output_path=output_path_pretrain,
            train_batch_size=64,
            max_steps=12500,
            max_length=512,
            learning_rate=3e-5,
            train_seed=seed,
        )

        cross_encoder_args = CrossEncoderArguments(
            model_name=output_path_pretrain,
            output_path=output_path_cross,
            train_batch_size=16,
            eval_batch_size=64,
            max_length=512,
            train_seed=seed,
            train_epochs=5,
            learning_rate=2e-5,
            freeze_embeddings=True,
            use_cross_bi_encoder=True
        )
        bi_encoder_args = MultitaskDistillBiEncoderArguments(
            model_name=roberta_model,
            output_path=output_path_bi,
            domain_objective_types=[],
            batch_size=64,
            max_length=256,
            train_seed=seed,
            train_epochs=1,
            steps_per_epoch=12500,
            learning_rate=2e-5
        )

        train_multitask_distill_pipeline(data_args, bi_encoder_args, cross_encoder_args, pretrain_args)
        result = evaluate(output_path_bi, dataset_name, 'mean')
        cache[cache_key] = result
        save_cache(dataset_name, cache)

        # overwrite tokenizer, otherwise it will try to load the CBE tokenizer
        tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        tokenizer.save_pretrained(output_path_pretrain)
        output_path_bi_full = output_path_bi + "-full"
        bi_encoder_args = bi_encoder_args._replace(
            model_name=output_path_pretrain,
            output_path=output_path_bi_full
        )
        with open(output_path_silver, "rb") as f:
            silver_dataset = pickle.load(f)

        train_multitask_distill_bi_encoder(bi_encoder_args, silver_dataset)

        result = evaluate(output_path_bi_full, dataset_name, 'mean')
        cache[cache_key_full] = result
        save_cache(dataset_name, cache)


def run_augsbert_experiments(dataset_name, cache, full):
    seeds = get_train_seeds(full)
    roberta_model = get_english_cbe_model(full)

    for seed in seeds:
        cache_key = ("augsbert-default", seed)
        if cache_key in cache:
            continue

        dataset_path = f"datasets/usep/data-train/{dataset_name}/unsupervised/train.txt"
        output_path_cross = f"output/unsupervised-domain/{dataset_name}/augsbert-default/{seed}/cross-encoder"
        output_path_bi = f"output/unsupervised-domain/{dataset_name}/augsbert-default/{seed}/bi-encoder"
        output_path_data_raw = f"output/unsupervised-domain/{dataset_name}/augsbert/{seed}/data-raw"
        output_path_data_pair = f"output/unsupervised-domain/{dataset_name}/augsbert/{seed}/data-pair"
        output_path_silver = f"output/unsupervised-domain/{dataset_name}/augsbert-default/{seed}/data-silver.pkl"

        data_args = DataArguments(
            dataset_path,
            is_hugging_face_dataset=False,
            text_col="text",
            label_col=None,
            is_text_dataset=True,
            strip_whitespaces=True,
            filter_duplicates=True,
            pair_generation_strategies=[PairGenerationStrategy.BM25, PairGenerationStrategy.RANDOM],
            top_k_pairs=[1, 1],
            pair_generation_batch_size=2000,
            dataset_size_limit=400000, # 400000 samples => 800000 pairs => 64 batch size ==> 12.5k train steps
            raw_dataset_cache_path=output_path_data_raw,
            pair_dataset_cache_path=output_path_data_pair,
            silver_dataset_cache_path=output_path_silver,
            data_generation_seed=seed,
        )

        cross_encoder_args = CrossEncoderArguments(
            model_name=roberta_model,
            output_path=output_path_cross,
            train_batch_size=16,
            eval_batch_size=32,
            max_length=512,
            train_seed=seed,
            train_epochs=5,
            learning_rate=2e-5,
            freeze_embeddings=True,
            use_cross_bi_encoder=True
        )
        bi_encoder_args = MultitaskDistillBiEncoderArguments(
            model_name=roberta_model,
            output_path=output_path_bi,
            domain_objective_types=[],
            batch_size=64,
            max_length=256,
            train_seed=seed,
            train_epochs=1,
            steps_per_epoch=12500,
            learning_rate=2e-5
        )

        train_multitask_distill_pipeline(data_args, bi_encoder_args, cross_encoder_args, None)

        result = evaluate(output_path_bi, dataset_name, 'mean')
        cache[cache_key] = result
        save_cache(dataset_name, cache)


def run_pretrained_experiments(dataset_name, cache, full):
    sbert_model = get_english_sbert_model(full)
    cache_key = ("pretrained", 0)
    if cache_key in cache:
        return
    result = evaluate(sbert_model, dataset_name, 'mean')
    cache[cache_key] = result
    save_cache(dataset_name, cache)

def run_unsupervised_domain_experiments(dataset_id, full):
    dataset_names = ["askubuntu", "twitter", "scidocs", "cqadupstack"]
    dataset_name = dataset_names[dataset_id]
    download("all")
    cache = load_cache(dataset_name)

    # run_pretrained_experiments(dataset_name, cache, full)
    run_nm_experiments(dataset_name, cache, full)
    run_nm_experiments(dataset_name, cache, full, mean=True)
    # run_augsbert_experiments(dataset_name, cache, full)
    # run_augsbert_domain_experiments(dataset_name, cache, full)


def run_unsupervised_domain_experiments_all(full):
    for i in range(4):
        run_unsupervised_domain_experiments(i, full)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    # print_results(True)
    run_unsupervised_domain_experiments(int(sys.argv[1]), True)
