import json
import logging
import math
import os
import random
from pprint import pprint
from typing import Union, Tuple

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch import nn
from torch.utils.data import DataLoader
from transformers import set_seed, TrainingArguments, IntervalStrategy

from cross_encoder.collator import input_example_collator
from util.pair_tokenizer import get_tokenizer_for_cross_bi_encoder
from data.pair_dataset import PairDatasetDict
from evaluate.metrics import score_metric
from model import AutoModelForEmbeddingSimilarityCrossBiEncoder
from util.args import CrossEncoderArguments, CrossEncoderActivationFunction
from util.new_trainer import NewTrainer

logger = logging.getLogger(__name__)


def get_activation_function(activation_function):
    if activation_function is None:
        return None
    elif activation_function == CrossEncoderActivationFunction.IDENTITY:
        return nn.Identity()
    elif activation_function == CrossEncoderActivationFunction.SIGMOID:
        return nn.Sigmoid()
    elif activation_function == CrossEncoderActivationFunction.TANH:
        return nn.Tanh()
    else:
        raise ValueError(f"Unhandled activation function {activation_function}")


def freeze_embeddings(model):
    logger.info("Checking for embeddings to freeze ...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            logger.info(f"Freezing embeddings {name} ...")
            module.requires_grad_(False)


def get_best_simple_cross_encoder_seed(args: CrossEncoderArguments, dataset: PairDatasetDict):
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
        train_dataloader = DataLoader(dataset.train, shuffle=True, batch_size=args.train_batch_size)
        warmup_steps = math.ceil(len(train_dataloader) * args.train_epochs * args.warmup_percent)
        # Stopping and Evaluating after 30% of training data (less than 1 epoch)
        # We find from (Dodge et al.) that 20-30% is often ideal for convergence of random seed
        evaluation_steps = math.ceil(len(train_dataloader) * 0.3)
        logger.info(f"Warmup-steps: {warmup_steps}")

        evaluator = CECorrelationEvaluator.from_input_examples(dataset.validation.input_examples, name='seed-optim-dev')
        # Hack to finish training early
        score_ref = [0]

        def evaluator_wrap(*args, **kwargs):
            score_ref[0] = evaluator(*args, **kwargs)
            raise EOFError("training finished")

        activation_function = get_activation_function(args.activation_function)
        cross_encoder = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length,
                                     default_activation_function=activation_function,
                                     automodel_args={"ignore_mismatched_sizes": True})
        if args.freeze_embeddings:
            freeze_embeddings(cross_encoder.model)

        try:
            cross_encoder.fit(train_dataloader=train_dataloader,
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


def train_or_load_simple_cross_encoder(
        args: CrossEncoderArguments, dataset: PairDatasetDict
) -> Tuple[CrossEncoder, float]:
    logger.info(f"Loading cross encoder for model {args.model_name} ...")
    if args.output_path is None:
        raise ValueError("Output path must be specified if CrossEncoder training is enabled!")

    os.makedirs(args.output_path, exist_ok=True)
    train_seed = args.train_seed
    if args.seed_optimization_steps > 0:
        train_seed = get_best_simple_cross_encoder_seed(args, dataset)

    set_seed(train_seed)
    train_dataloader = DataLoader(dataset.train, shuffle=True, batch_size=args.train_batch_size)
    train_steps = len(train_dataloader)
    warmup_steps = math.ceil(train_steps * args.train_epochs * args.warmup_percent)

    dev_evaluator = None
    if dataset.validation is not None:
        dev_evaluator = CECorrelationEvaluator.from_input_examples(dataset.validation.input_examples, name='dev')

    activation_function = get_activation_function(args.activation_function)
    cross_encoder = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length,
                                 default_activation_function=activation_function,
                                 automodel_args={"ignore_mismatched_sizes": True})
    if args.freeze_embeddings:
        freeze_embeddings(cross_encoder.model)

    if args.train_epochs > 0:
        cross_encoder.fit(train_dataloader=train_dataloader,
                          weight_decay=args.weight_decay,
                          optimizer_params={'lr': args.learning_rate},
                          evaluator=dev_evaluator,
                          epochs=args.train_epochs,
                          warmup_steps=warmup_steps,
                          output_path=args.output_path)

    # Reload best at the end
    logger.debug(f"Loading best cross-encoder ...")
    cross_encoder = CrossEncoder(args.output_path, num_labels=1, max_length=args.max_length,
                                 default_activation_function=activation_function)
    score = float("nan")
    if dataset.test is not None:
        test_evaluator = CECorrelationEvaluator.from_input_examples(dataset.test.input_examples, name='test')
        score = test_evaluator(cross_encoder, output_path=args.output_path)
    return cross_encoder, score


def get_best_cross_bi_encoder_seed(
        args: CrossEncoderArguments,
        dataset: PairDatasetDict,
        tokenizer,
        data_collator,
        model_init
):
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

        eval_metric = "cosine_spearman"
        # Stopping and Evaluating after 30% of training data (less than 1 epoch)
        # We find from (Dodge et al.) that 20-30% is often ideal for convergence of random seed
        logger.info("Early-stopping: 30% of the training-data")
        evaluation_steps = math.ceil(0.3 * len(dataset.train) / args.train_batch_size)
        # Hack to finish training early
        score_ref = [0]

        def score_metric_wrap(eval_prediction):
            score_ref[0] = score_metric(eval_prediction)[eval_metric]
            raise EOFError("training finished")

        set_seed(seed)
        train_args = TrainingArguments(
            output_dir=args.output_path,
            num_train_epochs=args.train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_percent,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            seed=seed,
            eval_steps=evaluation_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            logging_strategy=IntervalStrategy.NO,
            save_strategy=IntervalStrategy.NO,
            remove_unused_columns=False,
        )
        trainer = NewTrainer(
            args=train_args,
            model_init=model_init,
            tokenizer=tokenizer,
            train_dataset=dataset.train,
            eval_dataset=dataset.validation,
            data_collator=data_collator,
            compute_metrics=score_metric_wrap
        )
        try:
            trainer.train()
        except EOFError:
            pass

        score = score_ref[0]
        if best_score is None or score > best_score:
            logger.info(f"New best seed {seed} reached a score of {score}!")
            best_score = score
            best_seed = seed

    return best_seed


def train_or_load_cross_bi_encoder(
        args: CrossEncoderArguments, dataset: PairDatasetDict
) -> Tuple[NewTrainer, float]:
    logger.info(f"Loading cross encoder for model {args.model_name} ...")
    if args.activation_function is not None:
        raise ValueError("Activation function only implemented for simple cross encoder!")

    if args.output_path is None:
        raise ValueError("Output path must be specified if CrossEncoder training is enabled!")

    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.train_seed)

    tokenizer = get_tokenizer_for_cross_bi_encoder(args.model_name)
    data_collator = input_example_collator(tokenizer, args.max_length)

    def model_init():
        model = AutoModelForEmbeddingSimilarityCrossBiEncoder.from_pretrained(args.model_name)
        if args.freeze_embeddings:
            freeze_embeddings(model)
        return model

    train_seed = args.train_seed
    if args.seed_optimization_steps > 0:
        train_seed = get_best_cross_bi_encoder_seed(args, dataset, tokenizer, data_collator, model_init)

    set_seed(train_seed)
    eval_metric = "cosine_spearman"
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_percent,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        seed=train_seed,
        evaluation_strategy=IntervalStrategy.EPOCH if dataset.validation is not None else IntervalStrategy.NO,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        metric_for_best_model=eval_metric,
        greater_is_better=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=2
    )
    trainer = NewTrainer(
        args=train_args,
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=dataset.train,
        eval_dataset=dataset.validation,
        data_collator=data_collator,
        compute_metrics=None if dataset.validation is None else score_metric
    )

    if args.train_epochs > 0:
        trainer.train()
        trainer.save_model(args.output_path)

    score = float("nan")
    if dataset.test is not None:
        prediction = trainer.predict(dataset.test)
        logger.info("Test results:")
        for key, score in prediction.metrics.items():
            logging.info(f"{key}: {score}")
        score = prediction.metrics[f"test_{eval_metric}"]
    return trainer, score


def train_or_load_cross_encoder(
        args: CrossEncoderArguments, dataset: PairDatasetDict
) -> Tuple[Union[CrossEncoder, NewTrainer], float]:
    if args.use_cross_bi_encoder:
        return train_or_load_cross_bi_encoder(args, dataset)
    else:
        return train_or_load_simple_cross_encoder(args, dataset)

