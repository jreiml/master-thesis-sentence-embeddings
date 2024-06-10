import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, IntervalStrategy, AutoTokenizer

from data.constants import PROCESSED_LABEL_COL, TEXT_COL
from evaluate.metrics import simple_classification_metric
from experiments.utils import get_output_path
from model import AutoModelForSentenceEmbeddingClassification


def data_collator(tokenizer, text_col, label_col):
    def inner(features):
        texts = [feature[text_col] for feature in features]
        labels = torch.tensor([feature[label_col] for feature in features])
        new_features = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        new_features["labels"] = labels
        return new_features

    return inner


def evaluate_classification_on_frozen_encoder(model_name: str, dataset: DatasetDict, full: bool = True) -> float:
    experiment = "frozen_encoder_classification"
    output_model_name = model_name.replace("/", "_")
    output_dir = get_output_path(experiment, output_model_name, create=False)
    num_labels = len(set(dataset["train"][PROCESSED_LABEL_COL]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model_init():
        model = AutoModelForSentenceEmbeddingClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        encoder = getattr(model, model.base_model_prefix)

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
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator(tokenizer, TEXT_COL, PROCESSED_LABEL_COL)
    )
    trainer.train()

    prediction = trainer.predict(dataset["test"])
    f1_macro = prediction.metrics[f"test_{eval_metric}"]
    return f1_macro
