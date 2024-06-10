import enum
import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, Any, NamedTuple

import torch
from sentence_transformers import LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from tabulate import tabulate
from transformers import AutoTokenizer, TrainingArguments, IntervalStrategy, Trainer

from bi_encoder.multitask_distill import train_multitask_distill_bi_encoder
from bi_encoder.utils import create_bi_encoder
from data.constants import PROCESSED_LABEL_COL, TEXT_COL
from data.data_preprocessing import normalize_labels, filter_duplicate_rows
from data.experiment_dataset_loader import load_prompted_ukp_sentential_argument_mining, \
    load_bws_in_topic, load_prompted_bws_in_topic, load_prompted_20newsgroups
from data.multitask_pair_dataset import MultitaskInputExample
from evaluate.metrics import simple_classification_metric
from experiments.distill_pipeline import train_multitask_distill_pipeline
from experiments.frozen_encoder_classification import evaluate_classification_on_frozen_encoder
from experiments.run_senteval import run_senteval
from experiments.utils import get_english_sbert_model, get_english_cbe_model, get_train_seeds, ensure_dir_exists
from model import AutoModelForSentenceEmbeddingClassification
from util.args import PretrainArguments, PretrainMode, MultitaskDistillBiEncoderArguments, \
    CrossEncoderArguments, PairGenerationStrategy, DataArguments


class PromptMode(enum.Enum):
    ORIGINAL = "ORIGINAL"
    PROMPTED = "PROMPTED"


class SupervisedScoreCache(NamedTuple):
    sts_scores: Dict[Any, Any]
    domain_scores: Dict[Any, Any]


def save_cache(dataset_name, cache):
    cache_file = f'output/supervised/{dataset_name}-cache.pkl'
    with open(cache_file, 'wb') as fp:
        pickle.dump(cache, fp)


def load_cache(dataset_name):
    cache_file = f'output/supervised/{dataset_name}-cache.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            cache = pickle.load(fp)

            # TODO: remove me
            # bug fix for results, accidentally used PromptMode.ORIGINAL instead of the str for some values.
            for scores in [cache.sts_scores, cache.domain_scores]:
                for c in list(scores):
                    if c[2] == PromptMode.ORIGINAL:
                        new_c = list(c)
                        new_c[2] = PromptMode.ORIGINAL.value
                        scores[tuple(new_c)] = scores[c]
                        del scores[c]

            return cache

    ensure_dir_exists('output/supervised')
    return SupervisedScoreCache(
        sts_scores={},
        domain_scores={},
    )

def evaluate_newsgroup(model, full):
    dataset = load_prompted_20newsgroups()
    dataset = normalize_labels(dataset, "label_text")
    newsgroup_original = dataset.rename_column("original_text", TEXT_COL)
    newsgroup_original_topic_score = evaluate_classification_on_frozen_encoder(model, newsgroup_original, full)

    newsgroup_prompted = dataset.rename_column("prompted_text", TEXT_COL)
    newsgroup_prompted_topic_score = evaluate_classification_on_frozen_encoder(model, newsgroup_prompted, full)

    return {
        "newsgroup_original_topic": newsgroup_original_topic_score,
        "newsgroup_prompted_topic": newsgroup_prompted_topic_score,
    }


def evaluate_ukp_argument_mining(model, full):
    dataset = load_prompted_ukp_sentential_argument_mining()
    dataset = normalize_labels(dataset, "topic")
    ukp_argument_mining_original = dataset.rename_column("original_text", TEXT_COL)
    ukp_argument_mining_original_topic_score = evaluate_classification_on_frozen_encoder(
        model, ukp_argument_mining_original, full)

    ukp_argument_mining_prompted = dataset.rename_column("prompted_text", TEXT_COL)
    ukp_argument_mining_prompted_topic_score = evaluate_classification_on_frozen_encoder(
        model, ukp_argument_mining_prompted, full)

    bws_original = load_bws_in_topic().as_pair_dataset_dict()
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        bws_original.test.input_examples, name='test', main_similarity=SimilarityFunction.COSINE
    )
    bi_encoder = create_bi_encoder(model, None, "mean")
    if torch.cuda.is_available():
        bi_encoder = bi_encoder.cuda()
    bws_score_original = bi_encoder.evaluate(test_evaluator)

    bws_prompted = load_prompted_bws_in_topic().as_pair_dataset_dict()
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        bws_prompted.test.input_examples, name='test', main_similarity=SimilarityFunction.COSINE
    )
    bi_encoder = create_bi_encoder(model, None, "mean")
    if torch.cuda.is_available():
        bi_encoder = bi_encoder.cuda()
    bws_score_prompted = bi_encoder.evaluate(test_evaluator)


    return {
        "ukp_argument_mining_original_topic": ukp_argument_mining_original_topic_score,
        "ukp_argument_mining_prompted_topic": ukp_argument_mining_prompted_topic_score,
        "bws_original_topic": bws_score_original,
        "bws_prompted_topic": bws_score_prompted,
    }

def is_experiment_cached(cache, cache_key):
    if cache_key not in cache.sts_scores:
        return False

    if cache_key not in cache.domain_scores:
        return False

    return True


def run_pretrained_experiments(name, model, cache, dataset_name, full):
    cache_key = (0, name, PromptMode.ORIGINAL.value, dataset_name)
    if is_experiment_cached(cache, cache_key):
        return

    sts_score = run_senteval(model, "mean", full)
    cache.sts_scores[cache_key] = sts_score
    save_cache(dataset_name, cache)

    domain_score = evaluate_domain(model, dataset_name, full)
    cache.domain_scores[cache_key] = domain_score
    save_cache(dataset_name, cache)


def train_classifier(seed, prompt_mode, dataset_name, full):
    model_name = get_english_cbe_model(full)
    output_path = f"output/supervised/{dataset_name}/{prompt_mode.value}/classify/{seed}"

    if dataset_name == "newsgroups":
        dataset = load_prompted_20newsgroups()
        label_col = "supergroup"
    elif dataset_name == "ukp_argument_mining":
        dataset = load_prompted_ukp_sentential_argument_mining()
        label_col = "topic"
    else:
        raise ValueError(f"unknown dataset {dataset_name}!")

    if prompt_mode == PromptMode.ORIGINAL:
        text_col = "original_text"
    elif prompt_mode == PromptMode.PROMPTED:
        text_col = "prompted_text"
    else:
        raise ValueError(f"unknown prompt mode {prompt_mode}!")

    dataset = dataset.map(lambda ex: {text_col: ex[text_col].strip()})
    dataset = filter_duplicate_rows(dataset, text_col)
    dataset = normalize_labels(dataset, label_col)
    num_labels = len(set(dataset["train"][PROCESSED_LABEL_COL]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def model_init():
        model = AutoModelForSentenceEmbeddingClassification.from_pretrained(model_name, num_labels=num_labels)
        return model

    def data_collator(features):
        texts = [feature[text_col] for feature in features]
        labels = torch.tensor([feature[PROCESSED_LABEL_COL] for feature in features])
        new_features = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        new_features["labels"] = labels
        return new_features

    eval_metric = "f1_macro"
    args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        seed=seed,
        num_train_epochs=5 if full else 1,
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        greater_is_better=True
    )
    trainer = Trainer(
        model_init=model_init,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=simple_classification_metric
    )
    trainer.train()
    tokenizer.save_pretrained(output_path)
    trainer.model.save_pretrained(output_path)
    return output_path



def train_augsbert(seed, prompt_mode, dataset_name, pretrain, full):
    roberta_model = get_english_cbe_model(full)

    output_path = f"output/supervised/{dataset_name}/{prompt_mode.value}/augsbert/{seed}/pretrain-{pretrain}"
    output_path_pretrain = f"{output_path}/pretrain"
    output_path_cross = f"{output_path}/cross-encoder"
    output_path_bi = f"{output_path}/bi-encoder"
    output_path_data_raw = f"{output_path}/data-raw"
    output_path_data_pair = f"{output_path}/data-pair"
    output_path_silver = f"{output_path}/data-silver.pkl"
    output_path_dataset_raw = f"{output_path}/dataset"

    if prompt_mode == PromptMode.ORIGINAL:
        text_col = "original_text"
    elif prompt_mode == PromptMode.PROMPTED:
        text_col = "prompted_text"
    else:
        raise ValueError(f"unknown prompt mode {prompt_mode}!")

    if dataset_name == "newsgroups":
        dataset = load_prompted_20newsgroups()["train"]
        label_col = "supergroup"
        top_k_pairs = [4, 4] if full else [1, 1]
    elif dataset_name == "ukp_argument_mining":
        dataset = load_prompted_ukp_sentential_argument_mining()["train"]
        label_col = "topic"
        top_k_pairs = [2, 2] if full else [1, 1]
    else:
        raise ValueError(f"unknown dataset {dataset_name}!")

    epochs = 5 if full else 1
    dataset.save_to_disk(output_path_dataset_raw)
    data_args = DataArguments(
        output_path_dataset_raw,
        is_hugging_face_dataset=False,
        text_col=text_col,
        label_col=label_col,
        strip_whitespaces=True,
        filter_duplicates=True,
        pair_generation_strategies=[PairGenerationStrategy.BM25, PairGenerationStrategy.RANDOM],
        top_k_pairs=top_k_pairs,
        pair_generation_batch_size=2026 if dataset_name == "ukp_argument_mining" else 1885,
        raw_dataset_cache_path=output_path_data_raw,
        pair_dataset_cache_path=output_path_data_pair,
        silver_dataset_cache_path=output_path_silver,
        data_generation_seed=seed,
    )

    pretrain_args = None
    if pretrain:
        pretrain_args = PretrainArguments(
            train_mode=PretrainMode.MLM_PAIR,
            model_name=roberta_model,
            output_path=output_path_pretrain,
            train_batch_size=32,
            gradient_accumulation_steps=2,
            train_epochs=epochs,
            max_length=512,
            learning_rate=3e-5,
            train_seed=seed,
            prompt_delimiter="¥" if prompt_mode == PromptMode.PROMPTED else None
        )

    cross_encoder_args = CrossEncoderArguments(
        model_name=output_path_pretrain if pretrain else roberta_model,
        output_path=output_path_cross,
        train_batch_size=16,
        eval_batch_size=32,
        max_length=512,
        train_seed=seed,
        train_epochs=epochs,
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
        train_epochs=epochs,
        learning_rate=2e-5
    )

    train_multitask_distill_pipeline(data_args, bi_encoder_args, cross_encoder_args, pretrain_args)
    with open(output_path_silver, "rb") as f:
        silver_dataset = pickle.load(f)
    def silver_map_fn(input_example):
        new_texts = [text[text.find("¥") + 1:] for text in input_example.texts]
        return MultitaskInputExample(
            input_example.guid,
            new_texts,
            input_example.similarity_label,
            input_example.domain_labels,
        )


    if prompt_mode == PromptMode.PROMPTED:
        bi_encoder_args_hidden_prompt = bi_encoder_args._replace(
            model_name=roberta_model,
            output_path=output_path_bi + "-hidden-prompt"
        )
        silver_dataset_no_prompt = silver_dataset.map(silver_map_fn)
        train_multitask_distill_bi_encoder(bi_encoder_args_hidden_prompt, silver_dataset_no_prompt)

    if pretrain:
        # overwrite tokenizer, otherwise it will try to load the CBE tokenizer
        tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        tokenizer.save_pretrained(output_path_pretrain)
        bi_encoder_args_pretrained = bi_encoder_args._replace(
            model_name=output_path_pretrain,
            output_path=output_path_bi + "-pretrain"
        )
        train_multitask_distill_bi_encoder(bi_encoder_args_pretrained, silver_dataset)

        if prompt_mode == PromptMode.PROMPTED:
            bi_encoder_args_pretrained_hidden_prompt = bi_encoder_args._replace(
                model_name=output_path_pretrain,
                output_path=output_path_bi + "-pretrain-hidden-prompt"
            )
            silver_dataset_no_prompt = silver_dataset.map(silver_map_fn)
            train_multitask_distill_bi_encoder(bi_encoder_args_pretrained_hidden_prompt, silver_dataset_no_prompt)

    return output_path_bi

def evaluate_domain(model_name, dataset_name, full):
    if dataset_name == "newsgroups":
        return evaluate_newsgroup(model_name, full)
    else:
        return evaluate_ukp_argument_mining(model_name, full)

def classify(cache, seed, prompt_mode, dataset_name, full):
    prompt_mode_str = prompt_mode.value
    cache_key = (seed, "classify", prompt_mode_str, dataset_name)
    if is_experiment_cached(cache, cache_key):
        return

    encoder_output_path = train_classifier(seed, prompt_mode, dataset_name, full)
    cache.sts_scores[cache_key] = run_senteval(encoder_output_path, "mean", full)
    cache.domain_scores[cache_key] = evaluate_domain(encoder_output_path, dataset_name, full)
    save_cache(dataset_name, cache)

def augsbert(cache, seed, prompt_mode, dataset_name, full):
    prompt_mode_str = prompt_mode.value

    for pretrain in [False, True]:
        cache_key = (seed, "augsbert", prompt_mode_str, dataset_name, pretrain)
        if is_experiment_cached(cache, cache_key):
            continue

        model = train_augsbert(seed, prompt_mode, dataset_name, pretrain, full)
        cache.sts_scores[cache_key] = run_senteval(model, "mean", full)
        cache.domain_scores[cache_key] = evaluate_domain(model, dataset_name, full)

        if pretrain:
            cache_key = (seed, "augsbert-pretrain", prompt_mode_str, dataset_name, pretrain)
            model_pretrain = model + "-pretrain"
            cache.sts_scores[cache_key] = run_senteval(model_pretrain, "mean", full)
            cache.domain_scores[cache_key] = evaluate_domain(model_pretrain, dataset_name, full)

            if prompt_mode == PromptMode.PROMPTED:
                model_pretrain_hidden_prompt = model + "-pretrain-hidden-prompt"
                cache_key = (seed, "augsbert-pretrain-hidden-prompt", prompt_mode_str, dataset_name, pretrain)
                cache.sts_scores[cache_key] = run_senteval(model_pretrain_hidden_prompt, "mean", full)
                cache.domain_scores[cache_key] = evaluate_domain(model_pretrain_hidden_prompt, dataset_name, full)
        else:
            if prompt_mode == PromptMode.PROMPTED:
                model_hidden_prompt = model + "-hidden-prompt"
                cache_key = (seed, "augsbert-hidden-prompt", prompt_mode_str, dataset_name, pretrain)
                cache.sts_scores[cache_key] = run_senteval(model_hidden_prompt, "mean", full)
                cache.domain_scores[cache_key] = evaluate_domain(model_hidden_prompt, dataset_name, full)
        save_cache(dataset_name, cache)

def run_supervised_experiment(dataset_name, full):
    assert dataset_name in ["newsgroups", "ukp_argument_mining"]
    seeds = get_train_seeds(full)
    cache = load_cache(dataset_name)

    base_model = get_english_cbe_model(full)
    run_pretrained_experiments("pretrained-roberta", base_model, cache, dataset_name, full)

    sbert_model = get_english_sbert_model(full)
    run_pretrained_experiments("pretrained-sbert", sbert_model, cache, dataset_name, full)

    for seed in seeds:
        for prompt_mode in [PromptMode.ORIGINAL, PromptMode.PROMPTED]:
            for method in ["classify", "augsbert"]:
                if method == "classify":
                    classify(cache, seed, prompt_mode, dataset_name, full)
                else:
                    augsbert(cache, seed, prompt_mode, dataset_name, full)



from collections import defaultdict
from tabulate import tabulate

def average_domain_scores(results_dict):
    model_mapping = {
        "pretrained-sbert": "Pretrained SBERT",
        "classify": "Classifier",
        "pretrained-roberta": "Pretrained RoBERTa",
        "augsbert-pretrain": "FullDomainSBERT",
        "augsbert-pretrain-hidden-prompt": "FullDomainSBERT",
        "augsbert-hidden-prompt": "DomainSBERT",
        "augsbert": "DomainSBERT",
    }
    model_to_metric_to_value = defaultdict(lambda: defaultdict(list))
    unique_metrics = set()
    for (seed, *rest), metrics in results_dict.items():
        for metric, value in metrics.items():
            model_to_metric_to_value[tuple(rest[:-1])][metric].append(value)
            unique_metrics.add(metric)
    headers = ["Model", "Prompted"] + sorted(list(unique_metrics)) + ["Avg."]
    table = []
    for rest, metric_to_value in model_to_metric_to_value.items():
        row = [model_mapping[rest[0]], "HIDDEN" if "hidden-prompt" in rest[0] else rest[1]]
        avg_metric = 0.0
        num_metrics = 0
        for metric in sorted(list(unique_metrics)):
            if metric in metric_to_value:
                avg = sum(metric_to_value[metric]) / len(metric_to_value[metric])
                avg_metric += avg
                num_metrics += 1
                row.append("{:.2f}".format(avg * 100))
            else:
                row.append("N/A")
        if num_metrics > 0:
            avg_metric /= num_metrics
        row.append("{:.2f}".format(avg_metric * 100))
        table.append(row)
    return tabulate(sorted(table, key=lambda x: x[-1], reverse=True), headers, tablefmt="grid")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    import nltk
    nltk.download("punkt")

    print(average_domain_scores(load_cache("newsgroups").sts_scores))
    print(average_domain_scores(load_cache("ukp_argument_mining").sts_scores))

    if int(sys.argv[1]) == 0:
        run_supervised_experiment("newsgroups", True)
    elif int(sys.argv[1]) == 1:
        run_supervised_experiment("ukp_argument_mining", True)

