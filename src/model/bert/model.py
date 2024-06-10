from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import MSELoss
from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from transformers.modeling_outputs import SequenceClassifierOutput

from model import CrossBiEncoderOutput
from util.torch_utils import mean_pool_sequence_outputs_for_cross_encoder, \
    pool_sequence_output, get_sequence_classification_loss, determine_bos_token_id


class BertForContrastiveCrossBiEncoder(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, contrastive_margin=0.5):
        super().__init__(config)
        self.contrastive_margin = contrastive_margin
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        bos_token_id = determine_bos_token_id(self.config, input_ids)
        pooled_output_a, pooled_output_b = mean_pool_sequence_outputs_for_cross_encoder(
            bos_token_id=bos_token_id,
            input_ids=input_ids,
            sequence_output=sequence_output,
            attention_mask=attention_mask
        )

        cosine_similarities = torch.cosine_similarity(pooled_output_a, pooled_output_b)
        loss = None
        if labels is not None:
            distances = 1 - cosine_similarities
            loss = 0.5 * (labels.float() * distances.pow(2) + (1 - labels.float()) *
                          torch.relu(self.contrastive_margin - distances).pow(2))
            loss = loss.mean()

        if not return_dict:
            output = (cosine_similarities,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrossBiEncoderOutput(
            loss=loss,
            logits=cosine_similarities,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForEmbeddingSimilarityCrossBiEncoder(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        bos_token_id = determine_bos_token_id(self.config, input_ids)
        pooled_output_a, pooled_output_b = mean_pool_sequence_outputs_for_cross_encoder(
            bos_token_id=bos_token_id,
            input_ids=input_ids,
            sequence_output=sequence_output,
            attention_mask=attention_mask
        )

        cosine_similarities = torch.cosine_similarity(pooled_output_a, pooled_output_b)
        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(cosine_similarities, labels)

        if not return_dict:
            output = (cosine_similarities, ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CrossBiEncoderOutput(
            loss=loss,
            logits=cosine_similarities,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Just here for auto-class
class BertForFixedMaskedLM(BertForMaskedLM):
    pass


class BertForSentenceEmbeddingClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = pool_sequence_output(
            sequence_output=sequence_output,
            attention_mask=attention_mask
        )
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = get_sequence_classification_loss(logits, labels, self.config, self.num_labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
