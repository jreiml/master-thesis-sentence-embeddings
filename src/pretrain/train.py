import logging
from typing import Union

from datasets import DatasetDict
from transformers import AutoTokenizer, BertForPreTraining, TrainingArguments, IntervalStrategy, Trainer

from util.args import DataArguments, PretrainArguments, PretrainMode
from data.constants import TEXT_COL
from data.dataset_loader import load_processed_dataset, load_processed_dataset_for_sop, load_pair_dataset_from_args
from model import AutoModelForFixedMaskedLM
from model.nm_model import NmTsdaeModel, TsdaeModel, NmMlmModel, NmModel, SimCseMlmModel, SimCseModel
from pretrain.collator import mlm_train_collator, mlm_eval_collator, prepare_for_sop, mlm_sop_train_collator, \
    mlm_sop_eval_collator, msd_delete_collator, mlm_pair_train_collator, mlm_pair_eval_collator
from util.new_trainer import NewTrainer
from util.pair_tokenizer import get_tokenizer_for_cross_bi_encoder

logger = logging.getLogger(__name__)


def train_nm_adapted_encoder(args: PretrainArguments, dataset: Union[DatasetDict, dict]):
    if args.output_path is None:
        raise ValueError("Output path must be specified for pretraining!")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = msd_delete_collator(
        tokenizer=tokenizer,
        text_col=TEXT_COL,
        del_ratio=args.noise_probability,
        max_length=args.max_length,
        # reproduce original TSDAE
        delete_at_least_one=args.train_mode != PretrainMode.TSDAE,
    )

    def model_init():
        if args.train_mode == PretrainMode.TSDAE:
            model_class = TsdaeModel
        elif args.train_mode == PretrainMode.TSADE_NM:
            model_class = NmTsdaeModel
        elif args.train_mode == PretrainMode.MLM_NM:
            model_class = NmMlmModel
        elif args.train_mode == PretrainMode.MLM_SIMCSE:
            model_class = SimCseMlmModel
        elif args.train_mode == PretrainMode.SIMCSE:
            model_class = SimCseModel
        elif args.train_mode == PretrainMode.NM:
            model_class = NmModel
        else:
            raise ValueError(f"Unhandled pretrain mode: {args.train_mode}!")

        return model_class.from_pretrained(
            args.model_name,
            pooling_strategy=args.pooling_strategy,
        )

    evaluation_strategy = IntervalStrategy.EPOCH if "validation" in dataset else IntervalStrategy.NO
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_percent,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.train_seed,
        evaluation_strategy=evaluation_strategy,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=evaluation_strategy,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=2
    )
    trainer = NewTrainer(
        args=train_args,
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_path)


def train_sop_mlm_adapted_encoder(args: PretrainArguments, dataset: DatasetDict):
    if args.output_path is None:
        raise ValueError("Output path must be specified if CrossEncoder training is enabled!")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = prepare_for_sop(
            features=dataset["validation"],
            tokenizer=tokenizer,
            text_col=TEXT_COL,
            max_length=args.max_length,
            return_special_tokens_mask=True,
            return_list=True
        )

    train_collator = mlm_sop_train_collator(
        tokenizer=tokenizer,
        mlm_probability=args.noise_probability,
        text_col=TEXT_COL,
        max_length=args.max_length
    )
    eval_collator = mlm_sop_eval_collator(
        tokenizer=tokenizer,
    )

    def model_init():
        # TODO more models than BERT
        return BertForPreTraining.from_pretrained(args.model_name)

    evaluation_strategy = IntervalStrategy.EPOCH if "validation" in dataset else IntervalStrategy.NO
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_percent,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.train_seed,
        evaluation_strategy=evaluation_strategy,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=evaluation_strategy,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=2
    )
    trainer = NewTrainer(
        args=train_args,
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        eval_data_collator=eval_collator
    )
    trainer.train()
    trainer.save_model(args.output_path)


def train_mlm_adapted_encoder(args: PretrainArguments, dataset: DatasetDict):
    if args.output_path is None:
        raise ValueError("Output path must be specified if CrossEncoder training is enabled!")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_collator = mlm_train_collator(
        tokenizer=tokenizer,
        mlm_probability=args.noise_probability,
        max_length=args.max_length
    )
    eval_collator = mlm_eval_collator(
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    def model_init():
        return AutoModelForFixedMaskedLM.from_pretrained(args.model_name)

    evaluation_strategy = IntervalStrategy.EPOCH if "validation" in dataset else IntervalStrategy.NO
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_percent,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.train_seed,
        evaluation_strategy=evaluation_strategy,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=evaluation_strategy,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=2
    )
    trainer = NewTrainer(
        args=train_args,
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=train_collator,
        eval_data_collator=eval_collator
    )
    trainer.train()
    trainer.save_model(args.output_path)


def train_mlm_pair_adapted_encoder(args: PretrainArguments, dataset: DatasetDict):
    if args.output_path is None:
        raise ValueError("Output path must be specified if CrossEncoder training is enabled!")

    # prompt delimiter used to set special tokens mask for prompt
    tokenizer = get_tokenizer_for_cross_bi_encoder(args.model_name, prompt_delimiter=args.prompt_delimiter)
    train_collator = mlm_pair_train_collator(
        tokenizer=tokenizer,
        mlm_probability=args.noise_probability,
        max_length=args.max_length
    )
    eval_collator = mlm_pair_eval_collator(
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    def model_init():
        return AutoModelForFixedMaskedLM.from_pretrained(args.model_name)

    evaluation_strategy = IntervalStrategy.EPOCH if "validation" in dataset else IntervalStrategy.NO
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_percent,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.train_seed,
        evaluation_strategy=evaluation_strategy,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=evaluation_strategy,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_total_limit=2
    )
    trainer = NewTrainer(
        args=train_args,
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=train_collator,
        eval_data_collator=eval_collator
    )
    trainer.train()
    trainer.save_model(args.output_path)


def train_domain_adapted_encoder(data_args: DataArguments, encoder_args: PretrainArguments):
    if encoder_args.train_epochs <= 0:
        return

    if encoder_args.prompt_delimiter is not None and encoder_args.train_mode != PretrainMode.MLM_PAIR:
        raise ValueError(f"Prompt delimiter not implemented for {encoder_args.train_mode}!")

    if encoder_args.train_mode == PretrainMode.TSDAE or \
            encoder_args.train_mode == PretrainMode.TSADE_NM or \
            encoder_args.train_mode == PretrainMode.MLM_NM or \
            encoder_args.train_mode == PretrainMode.MLM_SIMCSE or \
            encoder_args.train_mode == PretrainMode.SIMCSE or \
            encoder_args.train_mode == PretrainMode.NM:
        dataset = load_processed_dataset(data_args)
        train_nm_adapted_encoder(encoder_args, dataset)
    elif encoder_args.train_mode == PretrainMode.MLM_SOP:
        dataset = load_processed_dataset_for_sop(data_args)
        train_sop_mlm_adapted_encoder(encoder_args, dataset)
    elif encoder_args.train_mode == PretrainMode.MLM:
        dataset = load_processed_dataset(data_args)
        train_mlm_adapted_encoder(encoder_args, dataset)
    elif encoder_args.train_mode == PretrainMode.MLM_PAIR:
        dataset, _ = load_pair_dataset_from_args(data_args)
        train_mlm_pair_adapted_encoder(encoder_args, dataset)
    else:
        raise ValueError("Invalid training mode!")
