import logging
import os
from collections import OrderedDict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sentence_transformers import LoggingHandler
from transformers import set_seed

from util.args import CrossEncoderArguments, BiEncoderArguments
from bi_encoder.train import train_bi_encoder
from data.experiment_dataset_loader import load_stsb, load_spanish_sts
from data.pair_generator import SentencePairGenerator
from data.pair_set import PairSet
from experiments.utils import BATCH_SIZE, LEARNING_RATE, get_english_default_model, get_output_path, get_kde_dataset, \
    annotate_dataset, train, get_test_datasets, get_train_epochs, get_spanish_default_model


def get_silver_dataset_setup(test_dataset, base_model, semantic_model_name):
    test_sentences = [
        text
        for input_example in test_dataset
        for text in input_example.texts
    ]
    generator = SentencePairGenerator(test_sentences, return_pair_dataset=True)
    test_dataset_names = [
        "Test (Predicted)",
        "Random",
        "BM25",
        "Semantic Search (Pretrained)",
        "Semantic Search (Finetuned)",
        "KDE (S.S. + Random)",
        "KDE (S.S.)",
        "KDE (Random)",
        "KDE (Original)",
    ]
    test_dataset_gen_fn = [
        lambda _m, _s: test_dataset,
        lambda _m, _s: generator.generate_random(5, skip_pairs=PairSet(test_dataset)),
        lambda _m, _s: generator.generate_bm25(5, skip_pairs=PairSet(test_dataset)),
        lambda _m, _s: generator.generate_semantic_search(
            5, semantic_model_name=base_model, skip_pairs=PairSet(test_dataset)
        ),
        lambda _m, _s: generator.generate_semantic_search(
            5, semantic_model_name=semantic_model_name, skip_pairs=PairSet(test_dataset)
        ),
        lambda model, setting: get_kde_dataset(
            model, setting, generator, semantic_model_name, test_dataset,
            use_random=True, use_semantic_search=True, use_hard_cutoff_sampling=True
        ),
        lambda model, setting: get_kde_dataset(
            model, setting, generator, semantic_model_name, test_dataset,
            use_random=False, use_semantic_search=True, use_hard_cutoff_sampling=True
        ),
        lambda model, setting: get_kde_dataset(
            model, setting, generator, semantic_model_name, test_dataset,
            use_random=True, use_semantic_search=False, use_hard_cutoff_sampling=True
        ),
        lambda model, setting: get_kde_dataset(
            model, setting, generator, semantic_model_name, test_dataset,
            use_random=True, use_semantic_search=False, use_hard_cutoff_sampling=False
        ),
    ]
    return test_dataset_names, test_dataset_gen_fn


def print_sampling_tables(test_name, legend_on_left, figures_path, name_to_scores):
    output_filename = test_name.replace("(", "").replace(")", "").replace(" ", "_").lower()

    name_to_color = {
        "Test (Original)": "lightcoral",
        "Test (Predicted)": "royalblue",
        "Random": "gold",
        "BM25": "red",
        "Semantic Search (Pretrained)": "gold",
        "Semantic Search (Finetuned)": "darkviolet",
        "KDE (S.S. + Random)": "limegreen",
        "KDE (S.S.)": "gold",
        "KDE (Random)": "violet",
        "KDE (Original)": "red",
    }

    sns.set_theme(style="darkgrid")

    output_filepath = os.path.join(figures_path, f"{output_filename}.png")

    fig, ax = plt.subplots()
    for name, scores in name_to_scores.items():
        sns.kdeplot(data=scores.squeeze(), ax=ax, color=name_to_color[name], label=name)
    ax.legend(loc='upper left' if legend_on_left else 'upper right')

    plt.tight_layout()
    plt.show()
    fig.savefig(output_filepath, bbox_inches="tight")


def print_all_sampling_tables(test_dataset_names, all_scores, actual_labels, figures_path):
    groups = [
        (True, "model_approximation_test", [
            "Test (Original)",
            "Test (Predicted)",
        ]),
        (False, "similar_pairs", [
            "Test (Predicted)",
            "BM25",
            "Semantic Search (Pretrained)",
            "Semantic Search (Finetuned)",
        ]),
        (True, "kde", [
            "Test (Predicted)",
            "KDE (S.S. + Random)",
            "KDE (S.S.)",
            "KDE (Random)",
            "KDE (Original)",
        ]),
        (False, "overview", [
            "Test (Predicted)",
            "Random",
            "BM25",
            "Semantic Search (Finetuned)",
            "KDE (S.S. + Random)",
        ])
    ]

    for legend_on_left, test_name, sample_method_names in groups:
        name_to_scores = OrderedDict()
        if "Test (Original)" in sample_method_names:
            name_to_scores["Test (Original)"] = actual_labels

        for name, scores in zip(test_dataset_names, all_scores):
            if name in sample_method_names:
                name_to_scores[name] = np.array(scores)

        print_sampling_tables(test_name, legend_on_left, figures_path, name_to_scores)


def start_sampling_experiment(seed, dataset, setting, figures_path, base_model, semantic_model_name):
    silver_dataset_names, silver_dataset_gen_fn = get_silver_dataset_setup(
        dataset.test, base_model, semantic_model_name
    )
    actual_labels = np.array([input_example.label for input_example in dataset.test])

    set_seed(seed)
    setting = setting._replace(train_seed=seed)
    model = train(dataset, setting)
    test_datasets, test_dataset_pairs, test_dataset_sentences = get_test_datasets(
        silver_dataset_gen_fn, model, setting
    )
    all_scores = annotate_dataset(model, setting, test_datasets, test_dataset_pairs, test_dataset_sentences)
    print_all_sampling_tables(silver_dataset_names, all_scores, actual_labels, figures_path)


def do_compare_sampling(dataset, full, experiment, english, use_cross_bi_encoder):
    semantic_search_bi_encoder_path = get_output_path(experiment, 'semantic-search-bi-encoder')
    cbe_path = get_output_path(experiment, 'cross-bi-encoder')
    figures_path = get_output_path(experiment, 'figures')

    base_model = get_english_default_model(full) if english else get_spanish_default_model(full)
    train_epochs = get_train_epochs(full)
    seed = 42

    semantic_model_args = BiEncoderArguments(
        model_name=base_model,
        output_path=semantic_search_bi_encoder_path,
        batch_size=BATCH_SIZE,
        train_epochs=train_epochs,
        max_length=256,
        learning_rate=LEARNING_RATE,
        train_seed=42,
        save_checkpoints=False,
    )
    train_bi_encoder(semantic_model_args, dataset)

    setting = CrossEncoderArguments(
        model_name=base_model,
        output_path=cbe_path,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        train_epochs=train_epochs,
        max_length=512,
        learning_rate=LEARNING_RATE,
        use_cross_bi_encoder=use_cross_bi_encoder,
    )

    start_sampling_experiment(seed, dataset, setting, figures_path, base_model, semantic_search_bi_encoder_path)


def compare_stsb_sampling(full):
    experiment = "compare-sampling-stsb"
    dataset = load_stsb()
    do_compare_sampling(dataset, full, experiment=experiment + "-ce", english=True, use_cross_bi_encoder=False)
    do_compare_sampling(dataset, full, experiment=experiment + "-cbe", english=True, use_cross_bi_encoder=True)


def compare_spanish_sts_sampling(full):
    experiment = "compare-sampling-spanish-sts-fixed"
    dataset = load_spanish_sts()
    do_compare_sampling(dataset, full, experiment=experiment + "-ce", english=False, use_cross_bi_encoder=False)
    do_compare_sampling(dataset, full, experiment=experiment + "-cbe", english=False, use_cross_bi_encoder=True)

def compare_sampling(full):
    compare_spanish_sts_sampling(full)
    compare_stsb_sampling(full)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    compare_spanish_sts_sampling(True)
