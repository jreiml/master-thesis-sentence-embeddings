import os
from typing import NamedTuple, Union

import numpy as np
import torch
from sentence_transformers import InputExample
from tabulate import tabulate

from bi_encoder.train import train_bi_encoder
from cross_encoder.train import train_or_load_simple_cross_encoder, train_or_load_cross_bi_encoder
from data.experiment_dataset_loader import load_stsb
from data.kde_sampler import sample_pair_dataset_via_kde
from data.pair_dataset import PairDataset, PairDatasetDict
from data.pair_set import PairSet
from util.args import BiEncoderArguments, CrossEncoderArguments

BATCH_SIZE = 16
LEARNING_RATE = 2e-5


class DistillSetting(NamedTuple):
    teacher: Union[CrossEncoderArguments, BiEncoderArguments]
    student: BiEncoderArguments
    train_stsb: bool = False


def write_table(output_path, header, rows):
    print(f"Writing to file {output_path} ...")
    table = tabulate(rows, header, "grid")
    table_latex = tabulate(rows, header, "latex")
    with open(output_path, "w") as output_file:
        output_file.write("# Simple Table\n")
        output_file.write(table)
        output_file.write("\n# Latex Table\n")
        output_file.write(table_latex)
    print(table)


def get_test_datasets(test_dataset_gen_fn, *args):
    datasets = [gen_fn(*args) for gen_fn in test_dataset_gen_fn]
    dataset_pairs = [
        [input_example.texts for input_example in aug_dataset]
        for aug_dataset in datasets
    ]
    dataset_sentences = [
        [text for texts in aug_dataset_pair for text in texts]
        for aug_dataset_pair in dataset_pairs
    ]
    return datasets, dataset_pairs, dataset_sentences


def get_pair_dataset_with_scores(dataset, scores):
    return PairDataset([
        InputExample(texts=input_example.texts, label=score)
        for input_example, score in zip(dataset, scores)
    ])


def distill(dataset, setting):
    if isinstance(setting.teacher, BiEncoderArguments):
        bi_encoder, _ = train_bi_encoder(setting.teacher, dataset)
        sentences = [text for input_example in dataset.train for text in input_example.texts]
        train_student_scores = get_bi_encoder_similarities(bi_encoder, sentences)
    else:
        if setting.train_stsb:
            teacher_dataset = load_stsb().as_symmetric_dataset()
        else:
            teacher_dataset = dataset.as_symmetric_dataset()

        if setting.teacher.use_cross_bi_encoder:
            trainer, _ = train_or_load_cross_bi_encoder(setting.teacher, teacher_dataset)
            train_student_scores = trainer.predict(dataset.train).predictions
        else:
            ce, _ = train_or_load_simple_cross_encoder(setting.teacher, teacher_dataset)
            train_student_dataset_pairs = [input_example.texts for input_example in dataset.train]
            train_student_scores = ce.predict(train_student_dataset_pairs)

    new_train = get_pair_dataset_with_scores(dataset.train, train_student_scores)
    new_dataset = PairDatasetDict(
        train=new_train,
        validation=dataset.validation,
        test=dataset.test
    )

    bi_encoder, _ = train_bi_encoder(setting.student, new_dataset)
    return bi_encoder


def train(dataset, setting):
    if isinstance(setting, DistillSetting):
        return distill(dataset, setting)
    elif isinstance(setting, BiEncoderArguments):
        bi_encoder, _ = train_bi_encoder(setting, dataset)
        return bi_encoder
    elif isinstance(setting, CrossEncoderArguments):
        if setting.use_cross_bi_encoder:
            model, _ = train_or_load_cross_bi_encoder(setting, dataset.as_symmetric_dataset())
            return model
        else:
            ce, _ = train_or_load_simple_cross_encoder(setting, dataset.as_symmetric_dataset())
            return ce
    else:
        raise ValueError("Unknown setting!")


def train_and_annotate_dataset(dataset, setting, test_datasets, test_dataset_pairs, test_dataset_sentences):
    model = train(dataset, setting)
    scores = annotate_dataset(model, setting, test_datasets, test_dataset_pairs, test_dataset_sentences)
    return scores


def get_bi_encoder_similarities(bi_encoder, sentences):
    if len(sentences) == 0:
        return np.array([])

    embeddings_bi_encoder = bi_encoder.encode(sentences, convert_to_tensor=True)
    embeddings_bi_encoder = embeddings_bi_encoder.view(len(sentences) // 2, 2, -1)
    embeddings_a_bi_encoder = embeddings_bi_encoder[:, 0]
    embeddings_b_bi_encoder = embeddings_bi_encoder[:, 1]
    cosine_similarities = torch.cosine_similarity(embeddings_a_bi_encoder, embeddings_b_bi_encoder)
    return cosine_similarities.cpu().detach().numpy()


def annotate_dataset(model_or_trainer, setting, test_datasets, test_dataset_pairs, test_dataset_sentences):
    if isinstance(setting, DistillSetting):
        setting = setting.student

    if isinstance(setting, BiEncoderArguments):
        bi_encoder_scores = [
            get_bi_encoder_similarities(model_or_trainer, sentences)
            for sentences in test_dataset_sentences
        ]
        return bi_encoder_scores
    elif isinstance(setting, CrossEncoderArguments):
        if setting.use_cross_bi_encoder:
            cbe_scores = [
                model_or_trainer.predict(d).predictions if len(d) > 0 else np.array([])
                for d in test_datasets
            ]
            return cbe_scores
        else:
            ce_scores = [
                model_or_trainer.predict(dataset_pair) if len(dataset_pair) > 0 else np.array([])
                for dataset_pair in test_dataset_pairs
            ]
            return ce_scores
    else:
        raise ValueError("Unknown setting!")


def get_kde_dataset(
        model, setting, generator, semantic_model_name, gold_dataset,
        use_random, use_semantic_search, use_hard_cutoff_sampling
):
    skip_pairs = PairSet(gold_dataset)
    amount = 3 if use_random and use_semantic_search else 5

    if use_random:
        silver_dataset_random = generator.generate_random(
            amount, skip_pairs=skip_pairs
        )

    if use_semantic_search:
        silver_dataset_semantic_search = generator.generate_semantic_search(
            amount, semantic_model_name=semantic_model_name, skip_pairs=skip_pairs
        )

    if use_semantic_search:
        silver_dataset = silver_dataset_semantic_search
        if use_random:
            silver_dataset.extend(silver_dataset_random)
    elif use_random:
        silver_dataset = silver_dataset_random
    else:
        raise ValueError("Invalid configuration!")

    gold_pairs = [input_example.texts for input_example in gold_dataset]
    gold_sentences = [text for texts in gold_pairs for text in texts]
    silver_pairs = [input_example.texts for input_example in silver_dataset]
    silver_sentences = [text for texts in silver_pairs for text in texts]
    scores = annotate_dataset(model, setting, [gold_dataset, silver_dataset], [gold_pairs, silver_pairs],
                              [gold_sentences, silver_sentences])
    gold_scores, silver_scores = scores
    silver_indices = set(sample_pair_dataset_via_kde(gold_scores, silver_scores, use_hard_cutoff_sampling))
    filtered_silver = [input_example for idx, input_example in enumerate(silver_dataset) if idx in silver_indices]
    return PairDataset(filtered_silver)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_output_path(experiment, name, create=True):
    output_path = "output/" + experiment + "/" + name
    if create:
        ensure_dir_exists(output_path)
    return output_path


def get_english_sbert_model(full):
    if full:
        return "sentence-transformers/all-mpnet-base-v2"
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_english_cbe_model(full):
    if full:
        return "roberta-base"
    return "google/bert_uncased_L-2_H-128_A-2"


def get_english_default_model(full):
    if full:
        return "bert-base-uncased"
    return "google/bert_uncased_L-2_H-128_A-2"


def get_spanish_default_model(full):
    if full:
        return "bert-base-multilingual-cased"
    return "mrm8488/es-tinybert-v1-1"


def get_spanish_cbe_model(full):
    if full:
        return "xlm-roberta-base"
    return "mrm8488/es-tinybert-v1-1"


def get_train_seeds(full):
    if full:
        return [42, 43, 44]

    return [42]


def get_train_epochs(full):
    if full:
        return 3

    return 1
