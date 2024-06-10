from typing import List

import torch
from sentence_transformers import InputExample
from transformers import Trainer, TrainingArguments, IntervalStrategy, AutoTokenizer

from data.pair_dataset import PairDatasetDict
from experiments.frozen_encoder_classification import simple_classification_metric
from experiments.utils import get_output_path
from model.sentence_embedding_model import SentenceEmbeddingPairClassificationModel
from util.args import PoolingStrategy


def data_collator(tokenizer):
    def inner(features: List[InputExample]):
        texts = [feature.texts[text_col] for feature in features for text_col in [0, 1]]
        labels = torch.tensor([feature.label for feature in features])
        new_features = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        new_features["labels"] = labels
        return new_features

    return inner


def evaluate_pair_classification_on_frozen_encoder(model_name: str, dataset: PairDatasetDict, full: bool = True) -> float:
    experiment = "frozen_pair_encoder_classification"
    output_model_name = model_name.replace("/", "_")
    output_dir = get_output_path(experiment, output_model_name, create=False)
    num_labels = dataset.get_unique_label_count()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model_init():
        model = SentenceEmbeddingPairClassificationModel(
            model_name, pooling_strategy=PoolingStrategy.MEAN, num_labels=num_labels
        )
        encoder = model.sentence_encoder.model
        for name, param in encoder.named_parameters():
            param.requires_grad = False

        return model

    eval_metric = "f1_macro"
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.NO,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=1e-3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=10 if full else 1,
        remove_unused_columns=False,
        metric_for_best_model=eval_metric,
        greater_is_better=True
    )
    trainer = Trainer(
        model_init=model_init,
        args=args,
        tokenizer=tokenizer,
        compute_metrics=simple_classification_metric,
        train_dataset=dataset.train,
        eval_dataset=dataset.validation,
        data_collator=data_collator(tokenizer)
    )
    trainer.train()

    prediction = trainer.predict(dataset.test)
    f1_macro = prediction.metrics[f"test_{eval_metric}"]
    return f1_macro
