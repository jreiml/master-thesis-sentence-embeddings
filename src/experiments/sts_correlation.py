import logging
import os.path
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import LoggingHandler, InputExample
from transformers import set_seed

from data.experiment_dataset_loader import load_stsb
from data.pair_dataset import PairDataset
from data.pair_generator import SentencePairGenerator
from evaluate.metrics import spearman
from experiments.utils import get_output_path, get_english_default_model, BATCH_SIZE, LEARNING_RATE, DistillSetting, \
    get_test_datasets, train_and_annotate_dataset, write_table, get_train_seeds, get_train_epochs, get_english_cbe_model
from util.args import BiEncoderArguments, CrossEncoderArguments, CrossEncoderActivationFunction


def get_test_dataset_setup(test_dataset):
    test_sentences = list(set([
        text
        for input_example in test_dataset
        for text in input_example.texts
    ]))
    generator = SentencePairGenerator(test_sentences, return_pair_dataset=True)
    test_dataset_names = [
        "Original",
        "Same Sentence",
        "Random (random 5)",            # random
        "Semantic Search (top 5)",      # most similar
        "Semantic Search (bottom 5)",   # least similar
    ]
    test_dataset_gen_fn = [
        lambda: test_dataset,
        lambda: PairDataset([InputExample(texts=[test_sentence, test_sentence]) for test_sentence in test_sentences]),
        lambda: generator.generate_random(5),
        lambda: generator.generate_semantic_search(5),
        lambda: generator.generate_semantic_search(5, least_similar=True),
    ]
    return test_dataset_names, test_dataset_gen_fn


def print_correlation_tables(test_name, figures_path, name_to_scores):
    output_filename = test_name.replace("(", "").replace(")", "").replace(" ", "_").lower()

    header = ["", *name_to_scores.keys()]
    pearson_rows = []
    mse_rows = []

    for name_a, all_scores_a in name_to_scores.items():
        pearson_row = [name_a]
        mse_row = [name_a]

        for name_b, all_scores_b in name_to_scores.items():
            correlations = [spearman(scores_a, scores_b) for scores_a, scores_b in zip(all_scores_a, all_scores_b)]
            pearson_mean = np.mean(correlations)
            pearson_stdev = 0 if len(correlations) == 1 else np.std(correlations, ddof=1)
            pearson_row.append(f"{pearson_mean:.2%} ± {pearson_stdev:.2%}")

            mse = [np.square(scores_a - scores_b).mean() for scores_a, scores_b in zip(all_scores_a, all_scores_b)]
            mse_mean = np.mean(mse)
            mse_stdev = 0 if len(mse) == 1 else np.std(mse, ddof=1)
            mse_row.append(f"{mse_mean:.2%} ± {mse_stdev:.2%}")

        pearson_rows.append(pearson_row)
        mse_rows.append(mse_row)

    pearson_output = os.path.join(figures_path, f"{output_filename}-pearson-table.txt")
    write_table(pearson_output, header, pearson_rows)

    mse_output = os.path.join(figures_path, f"{output_filename}-mse-table.txt")
    write_table(mse_output, header, mse_rows)

    name_to_mean_scores = {name: np.mean(scores, axis=0) for name, scores in name_to_scores.items()}
    scores_df = pd.DataFrame.from_dict(name_to_mean_scores)
    sns.set_theme(style="darkgrid")

    for kind in ["kde", "ecdf"]:
        output_filepath = os.path.join(figures_path, f"{output_filename}-{kind}.png")
        plot = sns.displot(scores_df, kind=kind)
        plot.set_xlabels("Similarity Score")
        fig = plot.figure
        fig.savefig(output_filepath, bbox_inches="tight")


def print_all_correlation_tables(
        test_dataset_names, model_names, all_scores, actual_labels, figures_path
):
    actual_correlation = OrderedDict()
    actual_correlation["Actual Labels"] = actual_labels
    for name, scores in zip(model_names, all_scores):
        actual_correlation[name] = scores[0]

    print_correlation_tables("Original", figures_path, actual_correlation)

    for i in range(1, len(test_dataset_names)):
        augmented_correlation = OrderedDict()
        for name, scores in zip(model_names, all_scores):
            augmented_correlation[name] = scores[i]

        print_correlation_tables(test_dataset_names[i], figures_path, augmented_correlation)


def start_correlation_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path=None):
    score_cache = {}
    if score_cache_path is not None and os.path.exists(score_cache_path):
        with open(score_cache_path, "rb") as score_cache_file:
            score_cache = pickle.load(score_cache_file)
    
    test_dataset_names, test_dataset_gen_fn = get_test_dataset_setup(dataset.test)
    all_scores = [[[] for _ in range(len(test_dataset_names))] for _ in range(len(settings))]
    actual_labels = [np.array([input_example.label for input_example in dataset.test])] * len(seeds)

    for seed in seeds:
        set_seed(seed)
        test_datasets, test_dataset_pairs, test_dataset_sentences = get_test_datasets(test_dataset_gen_fn)

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
            else:
                setting = setting._replace(train_seed=seed)
                setting_no_output = setting._replace(output_path="")

            if setting_no_output not in score_cache:
                score_cache[setting_no_output] = train_and_annotate_dataset(
                    dataset, setting, test_datasets, test_dataset_pairs, test_dataset_sentences
                )
                if score_cache_path is not None:
                    with open(score_cache_path, "wb") as score_cache_file:
                        pickle.dump(score_cache, score_cache_file)

            scores = score_cache[setting_no_output]
            for j, score in enumerate(scores):
                all_scores[i][j].append(score)

    print_all_correlation_tables(test_dataset_names, model_names, all_scores, actual_labels, figures_path)


def compare_activation_correlation(dataset, full, score_cache_path=None):
    experiment = "compare-activation"
    ce_path = get_output_path(experiment, 'cross-encoder')
    ce_sigmoid_path = get_output_path(experiment, 'cross-encoder-sigmoid')
    ce_tanh_path = get_output_path(experiment, 'cross-encoder-tanh')
    figures_path = get_output_path(experiment, 'figures')

    model_name = get_english_default_model(full)
    seeds = get_train_seeds(full)
    train_epochs = get_train_epochs(full)

    model_names = [
        "Cross-Encoder No Activation",
        "Cross-Encoder Sigmoid",
        "Cross-Encoder Tanh",
    ]
    settings = [
        CrossEncoderArguments(
            model_name=model_name,
            output_path=ce_path,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            train_epochs=train_epochs,
            max_length=512,
            learning_rate=LEARNING_RATE,
            use_cross_bi_encoder=False,
            activation_function=CrossEncoderActivationFunction.IDENTITY,
        ),
        CrossEncoderArguments(
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
        CrossEncoderArguments(
            model_name=model_name,
            output_path=ce_tanh_path,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            train_epochs=train_epochs,
            max_length=512,
            learning_rate=LEARNING_RATE,
            use_cross_bi_encoder=False,
            activation_function=CrossEncoderActivationFunction.TANH,
        )
    ]

    start_correlation_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path)


def compare_simple_correlation(dataset, full, score_cache_path=None):
    experiment = "compare-simple"
    bi_encoder_path = get_output_path(experiment, 'bi-encoder')
    ce_sigmoid_path = get_output_path(experiment, 'cross-encoder-sigmoid')
    figures_path = get_output_path(experiment, 'figures')

    model_name = get_english_default_model(full)
    seeds = get_train_seeds(full)
    train_epochs = get_train_epochs(full)

    model_names = [
        "Bi-Encoder",
        "Cross-Encoder Sigmoid",
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
        CrossEncoderArguments(
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
    ]

    start_correlation_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path)


def compare_best_correlation(experiment, model_name, dataset, full, score_cache_path=None):
    bi_encoder_path = get_output_path(experiment, 'bi-encoder')
    cbe_path = get_output_path(experiment, 'cross-bi-encoder')
    ce_sigmoid_path = get_output_path(experiment, 'cross-encoder-sigmoid')
    figures_path = get_output_path(experiment, 'figures')

    seeds = get_train_seeds(full)
    train_epochs = get_train_epochs(full)

    model_names = [
        "Bi-Encoder",
        "Cross-Bi-Encoder",
        "Cross-Encoder Sigmoid",
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
        CrossEncoderArguments(
            model_name=model_name,
            output_path=cbe_path,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            train_epochs=train_epochs,
            max_length=512,
            learning_rate=LEARNING_RATE,
            use_cross_bi_encoder=True
        ),
        CrossEncoderArguments(
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
    ]

    start_correlation_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path)


def compare_best_correlation_bert(dataset, full, score_cache_path=None):
    experiment = "compare-best-bert"
    model_name = get_english_default_model(full)
    compare_best_correlation(experiment, model_name, dataset, full, score_cache_path)


def compare_best_correlation_roberta(dataset, full, score_cache_path=None):
    experiment = "compare-best-roberta"
    model_name = get_english_cbe_model(full)
    compare_best_correlation(experiment, model_name, dataset, full, score_cache_path)


def compare_distilled_roberta_correlation(dataset, full, score_cache_path=None):
    experiment = "compare-distilled-roberta"
    bi_encoder_path = get_output_path(experiment, 'bi-encoder')
    cbe_distilled_bi_encoder_path = get_output_path(experiment, 'cbe-bi-encoder')
    ce_sigmoid_distilled_bi_encoder_path = get_output_path(experiment, 'ce-sigmoid-bi-encoder')
    cbe_path = get_output_path(experiment, 'cross-bi-encoder')
    ce_sigmoid_path = get_output_path(experiment, 'cross-encoder-sigmoid')
    figures_path = get_output_path(experiment, 'figures')

    model_name = get_english_cbe_model(full)
    seeds = get_train_seeds(full)
    train_epochs = get_train_epochs(full)

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
            )
        ),
        DistillSetting(
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
            )
        ),
    ]

    start_correlation_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path)


def compare_distilled_bert_correlation(dataset, full, score_cache_path=None):
    experiment = "compare-distilled-bert"
    bi_encoder_path = get_output_path(experiment, 'bi-encoder')
    cbe_distilled_bi_encoder_path = get_output_path(experiment, 'cbe-bi-encoder')
    ce_sigmoid_distilled_bi_encoder_path = get_output_path(experiment, 'ce-sigmoid-bi-encoder')
    cbe_path = get_output_path(experiment, 'cross-bi-encoder')
    ce_sigmoid_path = get_output_path(experiment, 'cross-encoder-sigmoid')
    figures_path = get_output_path(experiment, 'figures')

    model_name = get_english_default_model(full)
    seeds = get_train_seeds(full)
    train_epochs = get_train_epochs(full)

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
            )
        ),
        DistillSetting(
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
            )
        ),
    ]

    start_correlation_experiment(seeds, model_names, dataset, settings, figures_path, score_cache_path)


def compare_stsb_correlation(full):
    dataset = load_stsb()
    score_cache_path = "output/correlation-stsb-score-cache.pkl"
    compare_simple_correlation(dataset, full, score_cache_path)
    compare_activation_correlation(dataset, full, score_cache_path)
    compare_best_correlation_roberta(dataset, full, score_cache_path)
    compare_best_correlation_bert(dataset, full, score_cache_path)
    compare_distilled_roberta_correlation(dataset, full, score_cache_path)
    compare_distilled_bert_correlation(dataset, full, score_cache_path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    compare_stsb_correlation(True)
