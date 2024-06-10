import logging
import math
import os
from typing import Callable, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import set_seed

from bi_encoder.utils import create_bi_encoder
from data.constants import TEXT_COL
from data.dataset_loader import load_processed_dataset
from data.multitask_pair_dataset import MultitaskPairDatasetDict
from evaluate.evaluate_hook import HookedSentenceEvaluator
from evaluate.multitask_distill_evaluate import MultitaskDistillLossEvaluator
from loss.multitask_distill_cov_weighting import CoVWeightedMultitaskDistillLoss
from loss.multitask_distill_manual_weighting import ManualWeightedMultitaskDistillLoss
from util.args import DataArguments, MultitaskDistillBiEncoderArguments
from util.visualize import get_umap_hook


def get_distill_model_and_loss(
        args: MultitaskDistillBiEncoderArguments,
        domain_label_count: int,
        overwrite_model_name_or_path: Optional[str] = None
):
    model_name = overwrite_model_name_or_path if overwrite_model_name_or_path is not None else args.model_name
    bi_encoder = create_bi_encoder(model_name, args.max_length, args.pooling_mode)

    num_losses = 1
    use_manual_weight = True
    if len(args.domain_objective_types) > 0:
        num_losses = 1 + len(args.domain_objective_types)
        use_manual_weight = args.use_manual_weight_train

    loss_cls = ManualWeightedMultitaskDistillLoss if use_manual_weight else CoVWeightedMultitaskDistillLoss
    train_loss = loss_cls(model=bi_encoder,
                          num_losses=num_losses,
                          distill_objective_weight=args.distill_objective_weight,
                          domain_objective_types=args.domain_objective_types,
                          domain_objective_weights=args.domain_objective_weights,
                          domain_label_count=domain_label_count,
                          mcr2_eps=args.mcr2_eps,
                          mcr2_gamma=args.mcr2_gamma)
    if torch.cuda.is_available():
        bi_encoder = bi_encoder.cuda()
        train_loss = train_loss.cuda()
    return bi_encoder, train_loss


def train_multitask_distill_bi_encoder(args: MultitaskDistillBiEncoderArguments,
                                       dataset: MultitaskPairDatasetDict,
                                       evaluation_hooks: List[Callable[[SentenceTransformer], None]] = None):
    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.train_seed)
    domain_label_count = dataset.get_unique_domain_label_count()
    bi_encoder, train_loss = get_distill_model_and_loss(args, domain_label_count)

    train_dataloader = DataLoader(dataset.train, shuffle=True, batch_size=args.batch_size)
    train_steps = args.steps_per_epoch if args.steps_per_epoch is not None else len(train_dataloader)
    warmup_steps = math.ceil(train_steps * args.train_epochs * args.warmup_percent)
    checkpoint_save_steps = train_steps if args.save_checkpoints else 0
    logging.info(f"Warmup-steps: {warmup_steps}")

    evaluator = None
    if dataset.validation is not None:
        evaluator = MultitaskDistillLossEvaluator.from_multitask_dataset(train_loss, dataset.validation, name="dev")
        if evaluation_hooks is not None:
            evaluator = HookedSentenceEvaluator(evaluation_hooks, evaluator)
            evaluator(bi_encoder, comment='pretrained', output_path=args.output_path)
        else:
            evaluator(bi_encoder, output_path=args.output_path)
    elif evaluation_hooks is not None:
        evaluator = HookedSentenceEvaluator(evaluation_hooks)
        evaluator(bi_encoder, comment='pretrained', output_path=args.output_path)

    bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
                   weight_decay=args.weight_decay,
                   optimizer_params={'lr': args.learning_rate},
                   evaluator=evaluator,
                   epochs=args.train_epochs,
                   warmup_steps=warmup_steps,
                   steps_per_epoch=args.steps_per_epoch,
                   output_path=args.output_path,
                   save_best_model=False,
                   checkpoint_path=args.output_path,
                   checkpoint_save_steps=checkpoint_save_steps)
    bi_encoder.save(args.output_path)

    score = float("nan")
    if dataset.test is not None:
        bi_encoder, test_loss = get_distill_model_and_loss(args, domain_label_count, args.output_path)
        evaluator = MultitaskDistillLossEvaluator.from_multitask_dataset(test_loss, dataset.test, name="test")
        score = bi_encoder.evaluate(evaluator, output_path=args.output_path)
    return bi_encoder, score


def visualize_and_train_multitask_distill_bi_encoder(
        data_args: DataArguments,
        bi_encoder_args: MultitaskDistillBiEncoderArguments,
        silver_dataset: MultitaskPairDatasetDict,
        evaluation_hooks: List[Callable[[SentenceTransformer], None]] = None
):
    visual_dataset = load_processed_dataset(data_args)
    umap_evaluation_hooks = []
    if bi_encoder_args.visualize:
        umap_output_dir = os.path.join(bi_encoder_args.output_path, "umap")
        umap_evaluation_hooks = [
            get_umap_hook(visual_dataset[split], TEXT_COL, data_args.label_col, split, umap_output_dir)
            for split in ["train", "validation", "test"] if split in visual_dataset
        ]

    if evaluation_hooks is None:
        evaluation_hooks = umap_evaluation_hooks
    else:
        evaluation_hooks = [*evaluation_hooks, *umap_evaluation_hooks]
    train_multitask_distill_bi_encoder(bi_encoder_args, silver_dataset, evaluation_hooks)

