import logging
import os.path
import pickle
from collections import OrderedDict

import numpy as np
from sentence_transformers import LoggingHandler
from transformers import set_seed

from experiments.run_senteval import run_senteval
from util.args import CrossEncoderArguments, CrossEncoderActivationFunction, BiEncoderArguments
from data.experiment_dataset_loader import load_bws_in_topic, load_sts_test
from experiments.frozen_encoder_classification import evaluate_classification_on_frozen_encoder
from experiments.sts_correlation import write_table, print_correlation_tables
from experiments.utils import get_output_path, BATCH_SIZE, LEARNING_RATE, DistillSetting, \
    get_test_datasets, train_and_annotate_dataset, get_english_default_model, get_english_cbe_model


def print_all_correlation_tables(
        test_dataset_names, model_names, all_scores, actual_labels, figures_path
):
    for i in range(0, len(test_dataset_names)):
        augmented_correlation = OrderedDict()
        augmented_correlation["Actual Labels"] = actual_labels[i]
        for name, scores in zip(model_names, all_scores):
            augmented_correlation[name] = scores[i]

        print_correlation_tables(test_dataset_names[i], figures_path, augmented_correlation)


def print_distill_classification_tables(test_name, figures_path, name_to_scores):
    output_filename = test_name.replace("(", "").replace(")", "").replace(" ", "_").lower()

    header = [*name_to_scores.keys()]
    scores_row = []

    for name, scores in name_to_scores.items():
        scores_mean = np.mean(scores)
        scores_stdev = 0 if len(scores) == 1 else np.std(scores, ddof=1)
        scores_row.append(f"{scores_mean:.2%} ± {scores_stdev:.2%}")

    scores_output = os.path.join(figures_path, f"{output_filename}-classification-scores-table.txt")
    write_table(scores_output, header, [scores_row])


def start_distill_classification_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path=None):
    score_cache = {}
    if score_cache_path is not None and os.path.exists(score_cache_path):
        with open(score_cache_path, "rb") as score_cache_file:
            score_cache = pickle.load(score_cache_file)

    flat_dataset = dataset.as_dataset_dict()
    pair_dataset = dataset.as_pair_dataset_dict()
    test_dataset_names, test_datasets = load_sts_test()
    test_dataset_names.append("BWS")
    test_datasets.append(pair_dataset.test)

    test_dataset_gen_fns = [lambda test_dataset=test_dataset: test_dataset for test_dataset in test_datasets]
    all_sts_scores = [[[] for _ in range(len(test_dataset_names))] for _ in range(len(settings))]
    all_classification_scores = [[] for i in range(len(settings))]

    actual_labels = [[np.array([input_example.label for input_example in test_dataset])] * len(seeds) for test_dataset in test_datasets]

    for seed in seeds:
        set_seed(seed)
        test_datasets, test_dataset_pairs, test_dataset_sentences = get_test_datasets(test_dataset_gen_fns)

        for i, setting in enumerate(settings):
            if isinstance(setting, DistillSetting):
                setting = DistillSetting(
                    train_stsb=setting.train_stsb,
                    teacher=setting.teacher._replace(train_seed=seed),
                    student=setting.student._replace(train_seed=seed),
                )
                setting_no_output = DistillSetting(
                    train_stsb=setting.train_stsb,
                    teacher=setting.teacher._replace(output_path=""),
                    student=setting.student._replace(output_path=""),
                )
                trained_model_path = setting.student.output_path
            else:
                setting = setting._replace(train_seed=seed)
                setting_no_output = setting._replace(output_path="")
                trained_model_path = setting.output_path

            if setting_no_output not in score_cache:
                setting_score_cache = {}
                setting_score_cache["sts"] = train_and_annotate_dataset(
                    pair_dataset, setting, test_datasets, test_dataset_pairs, test_dataset_sentences
                )
                # setting_score_cache["senteval"] = run_senteval(trained_model_path, pooling_mode="mean", full=True)
                setting_score_cache["topic"] = evaluate_classification_on_frozen_encoder(
                    trained_model_path, flat_dataset
                )

                score_cache[setting_no_output] = setting_score_cache
                if score_cache_path is not None:
                    with open(score_cache_path, "wb") as score_cache_file:
                        pickle.dump(score_cache, score_cache_file)

            scores = score_cache[setting_no_output]
            for j, score in enumerate(scores["sts"]):
                all_sts_scores[i][j].append(score)
            all_classification_scores[i].append(scores["topic"])

    print_all_correlation_tables(test_dataset_names, model_names, all_sts_scores, actual_labels, figures_path)

    name_to_classification_scores = {name: scores for name, scores in zip(model_names, all_classification_scores)}
    print_distill_classification_tables("Simple", figures_path, name_to_classification_scores)


def compare_distilled_classification(experiment, model_name, distill_stsb, dataset, full, score_cache_path=None):
    cbe_distilled_bi_encoder_path = get_output_path(experiment, 'cbe-bi-encoder')
    ce_sigmoid_distilled_bi_encoder_path = get_output_path(experiment, 'ce-sigmoid-bi-encoder')
    bi_encoder_path = get_output_path(experiment, 'bi-encoder')
    cbe_path = get_output_path(experiment, 'cross-bi-encoder')
    ce_sigmoid_path = get_output_path(experiment, 'cross-encoder-sigmoid')
    figures_path = get_output_path(experiment, 'figures')

    if full:
        train_epochs = 5
        seeds = [42, 43, 44]
    else:
        train_epochs = 1
        seeds = [42]

    model_names = [
        "Bi-Encoder",
        "Bi-Encoder (distill CBE)",
        "Bi-Encoder (distill CE-Sigmoid)",
    ]
    settings = [
        BiEncoderArguments(
            model_name=model_name,
            output_path=bi_encoder_path,
            batch_size=BATCH_SIZE,
            train_epochs=train_epochs,
            max_length=256,
            learning_rate=LEARNING_RATE,
            save_checkpoints=False,
        ),
        DistillSetting(
            train_stsb=distill_stsb,
            teacher=CrossEncoderArguments(
                model_name=model_name,
                output_path=cbe_path,
                train_batch_size=BATCH_SIZE,
                eval_batch_size=BATCH_SIZE,
                train_epochs=train_epochs,
                max_length=512,
                learning_rate=LEARNING_RATE,
                use_cross_bi_encoder=True,
            ),
            student=BiEncoderArguments(
                model_name=model_name,
                output_path=cbe_distilled_bi_encoder_path,
                batch_size=BATCH_SIZE,
                train_epochs=train_epochs,
                max_length=256,
                learning_rate=LEARNING_RATE,
                save_checkpoints=False,
            ),
        ),
        DistillSetting(
            train_stsb=distill_stsb,
            teacher=CrossEncoderArguments(
                model_name=model_name,
                output_path=ce_sigmoid_path,
                train_batch_size=BATCH_SIZE,
                eval_batch_size=BATCH_SIZE,
                train_epochs=train_epochs,
                max_length=512,
                learning_rate=LEARNING_RATE,
                use_cross_bi_encoder=False,
                activation_function=CrossEncoderActivationFunction.SIGMOID,
            ),
            student=BiEncoderArguments(
                model_name=model_name,
                output_path=ce_sigmoid_distilled_bi_encoder_path,
                batch_size=BATCH_SIZE,
                train_epochs=train_epochs,
                max_length=256,
                learning_rate=LEARNING_RATE,
                save_checkpoints=False,
            ),
        ),
    ]

    start_distill_classification_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path)


def compare_bws_distill_classification(full):
    dataset = load_bws_in_topic()
    bert_model = get_english_default_model(full)
    roberta_model = get_english_cbe_model(full)

    experiment = "distill-classification/bert"
    score_cache_path = "output/distill-classification-bert.pkl"
    compare_distilled_classification(experiment, bert_model, False, dataset, full, score_cache_path)

    experiment = "distill-classification/roberta"
    score_cache_path = "output/distill-classification-roberta.pkl"
    compare_distilled_classification(experiment, roberta_model, False, dataset, full, score_cache_path)

    experiment = "distill-classification/bert-stsb"
    score_cache_path = "output/distill-classification-bert-stsb.pkl"
    compare_distilled_classification(experiment, bert_model, True, dataset, full, score_cache_path)

    experiment = "distill-classification/roberta-stsb"
    score_cache_path = "output/distill-classification-roberta-stsb.pkl"
    compare_distilled_classification(experiment, roberta_model, True, dataset, full, score_cache_path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    compare_bws_distill_classification(True)
