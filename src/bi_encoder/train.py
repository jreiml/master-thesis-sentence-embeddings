import logging
import math
import os
import random
from typing import Tuple

import torch
from sentence_transformers import losses, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
from transformers import set_seed

from bi_encoder.utils import create_bi_encoder
from data.pair_dataset import PairDatasetDict
from util.args import BiEncoderArguments
logger = logging.getLogger(__name__)


def get_simple_model_and_loss(args: BiEncoderArguments):
    bi_encoder = create_bi_encoder(args.model_name, args.max_length, args.pooling_mode)
    if args.use_multiple_negatives_ranking:
        loss = losses.MultipleNegativesRankingLoss(model=bi_encoder)
    else:
        loss = losses.CosineSimilarityLoss(model=bi_encoder)
    if torch.cuda.is_available():
        bi_encoder = bi_encoder.cuda()
        loss = loss.cuda()
    return bi_encoder, loss


def get_best_bi_encoder_seed(args: BiEncoderArguments, dataset: PairDatasetDict) -> int:
    if dataset.validation is None:
        raise ValueError("Cannot do seed optimization without validation split!")

    best_score = None
    best_seed = args.train_seed
    explored_seeds = set()

    set_seed(args.train_seed)

    for _ in range(args.seed_optimization_steps):
        # Seed must be between 0 and 2**32 - 1
        seed = random.randint(0, 2 ** 32 - 1)
        while seed in explored_seeds:
            seed = random.randint(0, 2 ** 32 - 1)

        explored_seeds.add(seed)

        logger.info("Early-stopping: 30% of the training-data")
        set_seed(seed)
        train_dataloader = DataLoader(dataset.train, shuffle=True, batch_size=args.batch_size)
        warmup_steps = math.ceil(len(train_dataloader) * args.train_epochs * args.warmup_percent)
        # Stopping and Evaluating after 30% of training data (less than 1 epoch)
        # We find from (Dodge et al.) that 20-30% is often ideal for convergence of random seed
        evaluation_steps = math.ceil(len(train_dataloader) * 0.3)
        logger.info(f"Warmup-steps: {warmup_steps}")

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            dataset.validation.input_examples, main_similarity=SimilarityFunction.COSINE
        )

        # Hack to finish training early
        score_ref = [0]

        def evaluator_wrap(*args, **kwargs):
            score_ref[0] = evaluator(*args, **kwargs)
            raise EOFError("training finished")

        set_seed(seed)
        bi_encoder, train_loss = get_simple_model_and_loss(args)
        set_seed(seed)
        try:
            bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
                           weight_decay=args.weight_decay,
                           optimizer_params={'lr': args.learning_rate},
                           epochs=args.train_epochs,
                           evaluation_steps=evaluation_steps,
                           warmup_steps=warmup_steps,
                           evaluator=evaluator_wrap)
        except EOFError:
            pass

        score = score_ref[0]
        if best_score is None or score > best_score:
            logger.info(f"New best seed {seed} reached a score of {score}!")
            best_score = score
            best_seed = seed

    return best_seed


def train_bi_encoder(args: BiEncoderArguments, dataset: PairDatasetDict) -> Tuple[SentenceTransformer, float]:
    train_seed = args.train_seed
    if args.seed_optimization_steps > 0:
        train_seed = get_best_bi_encoder_seed(args, dataset)

    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.train_seed)
    train_dataloader = DataLoader(dataset.train, shuffle=True, batch_size=args.batch_size)
    train_steps = len(train_dataloader)
    warmup_steps = math.ceil(train_steps * args.train_epochs * args.warmup_percent)
    checkpoint_save_steps = train_steps if args.save_checkpoints else 0
    checkpoint_path = args.output_path if args.save_checkpoints else None
    logger.info(f"Warmup-steps: {warmup_steps}")

    evaluator = None
    if dataset.validation is not None:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            dataset.validation.input_examples, name='dev', main_similarity=SimilarityFunction.COSINE
        )

    set_seed(train_seed)
    bi_encoder, train_loss = get_simple_model_and_loss(args)
    if evaluator is not None:
        bi_encoder.evaluate(evaluator, output_path=args.output_path)

    set_seed(train_seed)
    if args.train_epochs > 0:
        bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
                       weight_decay=args.weight_decay,
                       optimizer_params={'lr': args.learning_rate},
                       evaluator=evaluator,
                       epochs=args.train_epochs,
                       warmup_steps=warmup_steps,
                       output_path=args.output_path,
                       save_best_model=True,
                       checkpoint_path=checkpoint_path,
                       checkpoint_save_steps=checkpoint_save_steps)

    score = float("nan")
    if dataset.test is not None:
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            dataset.test.input_examples, name='test', main_similarity=SimilarityFunction.COSINE
        )
        bi_encoder = create_bi_encoder(args.output_path, args.max_length, args.pooling_mode)
        if torch.cuda.is_available():
            bi_encoder = bi_encoder.cuda()
        score = bi_encoder.evaluate(test_evaluator, output_path=args.output_path)
    return bi_encoder, score
