import logging
import os
from typing import Optional, Union, Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from loss.cov_weighting import CoVWeightedLoss
from loss.mcr2 import MaximalCodingRateReduction
from model import SentenceEncoderOutput
from util.args import PoolingStrategy
from util.torch_utils import pool_sequence_output, get_sequence_classification_loss

logger = logging.getLogger(__name__)

class SentenceEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, pooling_strategy: PoolingStrategy):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.pooling_strategy = pooling_strategy

    @property
    def base_model(self):
        return self.model

    @property
    def config(self):
        return self.model.config

    def save_pretrained(self, output_path, **kwargs):
        return self.model.save_pretrained(output_path, **kwargs)

    def encode(self, *args, attention_mask=None, **kwargs):
        outputs = self.model(*args, **kwargs, attention_mask=attention_mask)

        sequence_output = outputs[0]
        if self.pooling_strategy == PoolingStrategy.CLS:
            pooled_output = sequence_output[:, 0]
        elif self.pooling_strategy == PoolingStrategy.MEAN:
            pooled_output = pool_sequence_output(
                sequence_output=sequence_output,
                attention_mask=attention_mask,
                use_mean_pooling=True
            )
        elif self.pooling_strategy == PoolingStrategy.MAX:
            pooled_output = pool_sequence_output(
                sequence_output=sequence_output,
                attention_mask=attention_mask,
                use_mean_pooling=False
            )
        else:
            raise ValueError(f"Unimplemented pooling strategy {self.pooling_strategy}")

        return outputs, pooled_output

    def forward(
            self,
            *args,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Any, SentenceEncoderOutput]:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs, pooled_output = self.encode(*args, **kwargs)
        sequence_output = outputs[0]

        if not return_dict:
            return (sequence_output, pooled_output,) + outputs[2:]

        return SentenceEncoderOutput(
            logits=sequence_output,
            pooled_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SentenceEmbeddingPairClassificationModel(nn.Module):
    def __init__(self, model_name_or_path: str, pooling_strategy: PoolingStrategy, num_labels: int):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(model_name_or_path, pooling_strategy)
        self.num_labels = num_labels

        config = self.sentence_encoder.config
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None else
            config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.activation = nn.SiLU()
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
            self,
            *args,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Any, SentenceEncoderOutput]:
        return_dict = return_dict if return_dict is not None else self.sentence_encoder.config.use_return_dict
        outputs = self.sentence_encoder(*args, return_dict=return_dict, **kwargs)
        pooled_output = outputs.pooled_output

        batch_size = pooled_output.shape[0]
        pooled_output_split = pooled_output.view(batch_size // 2, 2, -1)
        pooled_output_a = pooled_output_split[:, 0]
        pooled_output_b = pooled_output_split[:, 1]
        pooled_output_ab = torch.abs(pooled_output_a - pooled_output_b)

        logits = torch.concat((pooled_output_a, pooled_output_b, pooled_output_ab), dim=1)
        logits = self.dense(logits)
        logits = self.dropout(logits)
        logits = self.activation(logits)
        logits = self.classifier(logits)

        loss = get_sequence_classification_loss(logits, labels, self.sentence_encoder.config, self.num_labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
