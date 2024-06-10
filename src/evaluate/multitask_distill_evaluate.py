import collections
import csv
import logging
import os
from typing import List, Union, OrderedDict

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch import Tensor

from data.multitask_pair_dataset import MultitaskPairDataset
from loss.multitask_distill_cov_weighting import CoVWeightedMultitaskDistillLoss
from loss.multitask_distill_manual_weighting import ManualWeightedMultitaskDistillLoss

logger = logging.getLogger(__name__)


class MultitaskDistillLossEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the multi-task loss.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self,
                 loss_model: Union[ManualWeightedMultitaskDistillLoss, CoVWeightedMultitaskDistillLoss],
                 sentences1: List[str],
                 sentences2: List[str],
                 labels: List[List[Union[float, int]]],
                 batch_size: int = 16,
                 name: str = '',
                 show_progress_bar: bool = False,
                 write_csv: bool = True):

        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param labels: Similarity score between sentences1[i] and sentences2[i] and extra domain labels
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = torch.tensor(labels)

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)

        self.loss_model = loss_model
        self.distill_labels = self.labels[:, 0]
        _, label_count = self.labels.shape
        if label_count > 1:
            self.domain_labels = self.labels[:, range(1, 3)]
        else:
            self.domain_labels = None
        self.write_csv = write_csv
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "multitask_distill_loss_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = [
            "epoch", "steps",
            "total_loss", "distill_loss",
            *[f"{str(objective_type.value).lower()}_{i}_loss"
              for i, objective_type in enumerate(loss_model.loss.domain_objective_types)],
        ]

    @classmethod
    def from_multitask_dataset(cls,
                               loss_model: Union[ManualWeightedMultitaskDistillLoss, CoVWeightedMultitaskDistillLoss],
                               dataset: MultitaskPairDataset,
                               **kwargs):
        sentences1 = []
        sentences2 = []
        labels = []

        for example in dataset.input_examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            labels.append(example.label)
        return cls(loss_model, sentences1, sentences2, labels, **kwargs)

    def get_distill_metrics(self, embeddings1: Tensor, embeddings2: Tensor) -> float:
        distill_labels = self.distill_labels
        if torch.cuda.is_available():
            distill_labels = distill_labels.cuda()
        distill_loss = self.loss_model.loss.compute_distill_loss(
            [embeddings1, embeddings2], distill_labels).item()

        logger.info("Distill Loss :\t {:.4f}".format(distill_loss))
        return distill_loss

    def get_domain_metrics(self, embeddings1: Tensor, embeddings2: Tensor) -> OrderedDict[str, float]:
        domain_metrics = collections.OrderedDict()

        if self.domain_labels is None:
            return domain_metrics

        domain_labels = self.domain_labels
        if torch.cuda.is_available():
            domain_labels = domain_labels.cuda()
        domain_losses = self.loss_model.loss.compute_domain_losses(
            [embeddings1, embeddings2], domain_labels)

        for i, (domain_loss, domain_type) in enumerate(
                zip(domain_losses, self.loss_model.loss.domain_objective_types)):
            if torch.isnan(domain_loss):
                logging.error("nan loss for domain loss in evaluation! "
                              "Try increasing eps if you are using MCR².")

            domain_loss_item = domain_loss.item()
            domain_metrics[f"eval_{str(domain_type.value).lower()}_{i}_loss"] = domain_loss_item
            logger.info("{} [{}] Loss :\t {:.4f}".format(domain_type, i, domain_loss_item))

        return domain_metrics

    def __call__(self, model: SentenceTransformer,
                 output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("MultitaskDistillLossEvaluator: Evaluating the model on " +
                    self.name + " dataset" + out_txt)
        self.loss_model.eval()
        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_tensor=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_tensor=True)
        distill_loss = self.get_distill_metrics(embeddings1, embeddings2)
        domain_metrics = self.get_domain_metrics(embeddings1, embeddings2)
        total_loss = sum([distill_loss, *domain_metrics.values()])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([
                    epoch,
                    steps,
                    total_loss,
                    distill_loss,
                    *domain_metrics.values(),
                ])

        return total_loss
