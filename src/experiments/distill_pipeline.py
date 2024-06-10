import logging
import os
from typing import List, Callable

from sentence_transformers import SentenceTransformer

from data.multitask_pair_dataset import MultitaskInputExample
from util.args import DataArguments, MultitaskDistillBiEncoderArguments, CrossEncoderArguments, PretrainArguments
from bi_encoder.multitask_distill import visualize_and_train_multitask_distill_bi_encoder, \
    train_multitask_distill_bi_encoder
from cross_encoder.annotate import generate_multitask_silver_dataset
from cross_encoder.train import train_or_load_cross_bi_encoder
from data.experiment_dataset_loader import load_stsb
from pretrain.train import train_domain_adapted_encoder
from util.torch_utils import clear_memory


def train_multitask_distill_pipeline(
        data_args: DataArguments,
        bi_encoder_args: MultitaskDistillBiEncoderArguments,
        cross_encoder_args: CrossEncoderArguments,
        pretrain_args: PretrainArguments = None,
        visualize: bool = False,
        silver_map_fn: Callable[[MultitaskInputExample], MultitaskInputExample] = None,
        evaluation_hooks: List[Callable[[SentenceTransformer], None]] = None
):
    silver_dataset_exists = data_args.silver_dataset_cache_path is not None and \
                           os.path.exists(data_args.silver_dataset_cache_path)

    if not silver_dataset_exists:
        if pretrain_args is not None:
            clear_memory()
            if pretrain_args.train_epochs > 0:
                train_domain_adapted_encoder(data_args, pretrain_args)
        else:
            logging.warning("No pretraining parameters specified!")

        clear_memory()
        stsb_dataset = load_stsb().as_symmetric_dataset()

        if cross_encoder_args.train_epochs > 0:
            train_or_load_cross_bi_encoder(cross_encoder_args, stsb_dataset)

    clear_memory()
    silver_dataset = generate_multitask_silver_dataset(
        data_args, cross_encoder_args.output_path, max_length=bi_encoder_args.max_length
    )
    if silver_map_fn is not None:
        silver_dataset = silver_dataset.map(silver_map_fn)

    clear_memory()

    if visualize:
        visualize_and_train_multitask_distill_bi_encoder(data_args, bi_encoder_args, silver_dataset, evaluation_hooks)
    else:
        train_multitask_distill_bi_encoder(bi_encoder_args, silver_dataset, evaluation_hooks)
    clear_memory()
