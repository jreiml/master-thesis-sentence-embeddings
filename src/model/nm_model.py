import logging
import os.path

import torch
from sentence_transformers import util
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel, AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig

from model import SentenceEncoderOutput
from model.sentence_embedding_model import SentenceEncoder
from pretrain.collator import apply_masking, get_mlm_eval_labels
from util.args import PoolingStrategy

logger = logging.getLogger(__name__)


class TsdaeModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            tie_encoder_auxiliary: bool = True,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
    ):
        """
        :param model_name_or_path: Model name or path for initializing the model
        :param tie_encoder_auxiliary: whether to tie the trainable parameters of encoder and auxiliary
        :param pooling_strategy: the pooling strategy to use for the sentence embedding in the encoder
        """
        super().__init__()

        encoder_path = os.path.join(model_name_or_path, "encoder")
        if os.path.exists(encoder_path):
            encoder_name_or_path = encoder_path
        else:
            encoder_name_or_path = model_name_or_path

        auxiliary_path = os.path.join(model_name_or_path, "auxiliary")
        if os.path.exists(auxiliary_path):
            auxiliary_name_or_path = auxiliary_path
        else:
            auxiliary_name_or_path = model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # This will be the final model used during the inference time.
        self.encoder = SentenceEncoder(encoder_name_or_path, pooling_strategy)

        auxiliary_config = AutoConfig.from_pretrained(
            auxiliary_name_or_path, is_decoder=True, add_cross_attention=True
        )
        self.auxiliary = AutoModelForCausalLM.from_pretrained(auxiliary_name_or_path, config=auxiliary_config)

        if tie_encoder_auxiliary:
            PreTrainedModel._tie_encoder_decoder_weights(
                self.encoder.base_model,
                self.auxiliary.base_model,
                self.auxiliary.base_model_prefix
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path, **encoder_kwargs):
        return cls(model_name_or_path, **encoder_kwargs)

    def save_pretrained(self, output_path, **kwargs):
        encoder_output_path = os.path.join(output_path, "encoder")
        auxiliary_output_path = os.path.join(output_path, "auxiliary")

        self.encoder.save_pretrained(encoder_output_path, **kwargs)
        self.auxiliary.save_pretrained(auxiliary_output_path, **kwargs)
        self.tokenizer.save_pretrained(encoder_output_path)
        self.tokenizer.save_pretrained(auxiliary_output_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            special_tokens_mask=None,
    ):
        lm_labels, _ = get_mlm_eval_labels(self.tokenizer, input_ids, special_tokens_mask)
        lm_labels = lm_labels[:, 1:].clone()

        outputs_noised = self.encoder(input_ids=noised_input_ids, attention_mask=noised_attention_mask)
        noised_pooled = outputs_noised.pooled_output
        outputs_lm = self.auxiliary(
            input_ids=input_ids[:, :-1],
            encoder_hidden_states=noised_pooled.unsqueeze(0)
        )
        lm_logits = outputs_lm.logits
        
        # Calculate loss
        ce_loss_fct = nn.CrossEntropyLoss()
        lm_loss = ce_loss_fct(lm_logits.view(-1, self.encoder.config.vocab_size), lm_labels.view(-1))

        return SentenceEncoderOutput(
            loss=lm_loss,
            pooled_output=outputs_noised.pooled_output,
            logits=outputs_lm.logits,
            hidden_states=outputs_lm.hidden_states,
            attentions=outputs_lm.attentions
        )


class NmTsdaeModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            tie_encoder_auxiliary: bool = True,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
            scale: float = 20.0
    ):
        """
        :param model_name_or_path: Model name or path for initializing the model
        :param tie_encoder_auxiliary: whether to tie the trainable parameters of encoder and auxiliary
        :param pooling_strategy: the pooling strategy to use for the sentence embedding in the encoder
        :param scale: the multiplicative value for the contrastive loss
        """
        super().__init__()

        encoder_path = os.path.join(model_name_or_path, "encoder")
        if os.path.exists(encoder_path):
            encoder_name_or_path = encoder_path
        else:
            encoder_name_or_path = model_name_or_path

        auxiliary_path = os.path.join(model_name_or_path, "auxiliary")
        if os.path.exists(auxiliary_path):
            auxiliary_name_or_path = auxiliary_path
        else:
            auxiliary_name_or_path = model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.scale = scale
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # This will be the final model used during the inference time.
        self.encoder = SentenceEncoder(encoder_name_or_path, pooling_strategy)

        auxiliary_config = AutoConfig.from_pretrained(
            auxiliary_name_or_path, is_decoder=True, add_cross_attention=True
        )
        self.auxiliary = AutoModelForCausalLM.from_pretrained(auxiliary_name_or_path, config=auxiliary_config)

        if tie_encoder_auxiliary:
            PreTrainedModel._tie_encoder_decoder_weights(
                self.encoder.base_model,
                self.auxiliary.base_model,
                self.auxiliary.base_model_prefix
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path, **encoder_kwargs):
        return cls(model_name_or_path, **encoder_kwargs)

    def save_pretrained(self, output_path, **kwargs):
        encoder_output_path = os.path.join(output_path, "encoder")
        auxiliary_output_path = os.path.join(output_path, "auxiliary")

        self.encoder.save_pretrained(encoder_output_path, **kwargs)
        self.auxiliary.save_pretrained(auxiliary_output_path, **kwargs)
        self.tokenizer.save_pretrained(encoder_output_path)
        self.tokenizer.save_pretrained(auxiliary_output_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            special_tokens_mask=None,
    ):
        lm_labels, _ = get_mlm_eval_labels(self.tokenizer, input_ids, special_tokens_mask)
        lm_labels = lm_labels[:, 1:].clone()
        
        outputs_noised = self.encoder(input_ids=noised_input_ids, attention_mask=noised_attention_mask)
        noised_pooled = outputs_noised.pooled_output
        with torch.no_grad():
            outputs_base = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            base_pooled = outputs_base.pooled_output

        outputs_lm = self.auxiliary(
            input_ids=input_ids[:, :-1],
            encoder_hidden_states=noised_pooled.unsqueeze(0)
        )
        lm_logits = outputs_lm.logits

        # Calculate loss
        ce_loss_fct = nn.CrossEntropyLoss()
        batch_size = len(base_pooled)
        labels = torch.arange(batch_size, device=noised_pooled.device)

        similarity = util.cos_sim(base_pooled, noised_pooled) * self.scale
        similarity_loss = ce_loss_fct(similarity, labels)
        similarity_t_loss = ce_loss_fct(similarity.t(), labels)
        lm_loss = ce_loss_fct(lm_logits.view(-1, self.encoder.config.vocab_size), lm_labels.view(-1))

        loss = similarity_loss + similarity_t_loss + lm_loss
        return SentenceEncoderOutput(
            loss=loss,
            pooled_output=outputs_base.pooled_output,
            logits=outputs_lm.logits,
            hidden_states=outputs_lm.hidden_states,
            attentions=outputs_lm.attentions
        )
    

class NmMlmModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            tie_encoder_auxiliary: bool = True,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
            scale: float = 20.0,
            mlm_probability: float = 0.20
    ):
        """
        :param model_name_or_path: Model name or path for initializing the model
        :param tie_encoder_auxiliary: whether to tie the trainable parameters of encoder and auxiliary
        :param pooling_strategy: the pooling strategy to use for the sentence embedding in the encoder
        :param scale: the multiplicative value for the contrastive loss
        """
        super().__init__()

        encoder_path = os.path.join(model_name_or_path, "encoder")
        if os.path.exists(encoder_path):
            encoder_name_or_path = encoder_path
        else:
            encoder_name_or_path = model_name_or_path

        auxiliary_path = os.path.join(model_name_or_path, "auxiliary")
        if os.path.exists(auxiliary_path):
            auxiliary_name_or_path = auxiliary_path
        else:
            auxiliary_name_or_path = model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.scale = scale
        self.mlm_probability = mlm_probability
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # This will be the final model used during the inference time.
        self.encoder = SentenceEncoder(encoder_name_or_path, pooling_strategy)
        self.auxiliary = AutoModelForMaskedLM.from_pretrained(auxiliary_name_or_path)

        if tie_encoder_auxiliary:
            PreTrainedModel._tie_encoder_decoder_weights(
                self.encoder.base_model,
                self.auxiliary.base_model,
                self.auxiliary.base_model_prefix
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path, **encoder_kwargs):
        return cls(model_name_or_path, **encoder_kwargs)

    def save_pretrained(self, output_path, **kwargs):
        encoder_output_path = os.path.join(output_path, "encoder")
        auxiliary_output_path = os.path.join(output_path, "auxiliary")

        self.encoder.save_pretrained(encoder_output_path, **kwargs)
        self.auxiliary.save_pretrained(auxiliary_output_path, **kwargs)
        self.tokenizer.save_pretrained(encoder_output_path)
        self.tokenizer.save_pretrained(auxiliary_output_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            special_tokens_mask=None,
    ):
        outputs_noised = self.encoder(input_ids=noised_input_ids, attention_mask=noised_attention_mask)
        noised_pooled = outputs_noised.pooled_output
        with torch.no_grad():
            outputs_base = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            base_pooled = outputs_base.pooled_output
        
        mlm_input_ids = input_ids.clone()
        mlm_labels, _ = apply_masking(self.tokenizer, self.mlm_probability, mlm_input_ids, special_tokens_mask)
        outputs_lm = self.auxiliary(
            input_ids=mlm_input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels
        )

        # Calculate loss
        ce_loss_fct = nn.CrossEntropyLoss()
        batch_size = len(base_pooled)
        labels = torch.arange(batch_size, device=noised_pooled.device)

        similarity = util.cos_sim(base_pooled, noised_pooled) * self.scale
        similarity_loss = ce_loss_fct(similarity, labels)
        similarity_t_loss = ce_loss_fct(similarity.t(), labels)
        lm_loss = outputs_lm.loss
        loss = similarity_loss + similarity_t_loss + lm_loss
        return SentenceEncoderOutput(
            loss=loss,
            pooled_output=outputs_base.pooled_output,
            logits=outputs_lm.logits,
            hidden_states=outputs_lm.hidden_states,
            attentions=outputs_lm.attentions
        )


class SimCseMlmModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            tie_encoder_auxiliary: bool = True,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
            scale: float = 20.0,
            mlm_probability: float = 0.20
    ):
        """
        :param model_name_or_path: Model name or path for initializing the model
        :param tie_encoder_auxiliary: whether to tie the trainable parameters of encoder and auxiliary
        :param pooling_strategy: the pooling strategy to use for the sentence embedding in the encoder
        :param scale: the multiplicative value for the contrastive loss
        """
        super().__init__()

        encoder_path = os.path.join(model_name_or_path, "encoder")
        if os.path.exists(encoder_path):
            encoder_name_or_path = encoder_path
        else:
            encoder_name_or_path = model_name_or_path

        auxiliary_path = os.path.join(model_name_or_path, "auxiliary")
        if os.path.exists(auxiliary_path):
            auxiliary_name_or_path = auxiliary_path
        else:
            auxiliary_name_or_path = model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.scale = scale
        self.mlm_probability = mlm_probability
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # This will be the final model used during the inference time.
        self.encoder = SentenceEncoder(encoder_name_or_path, pooling_strategy)
        self.auxiliary = AutoModelForMaskedLM.from_pretrained(auxiliary_name_or_path)

        if tie_encoder_auxiliary:
            PreTrainedModel._tie_encoder_decoder_weights(
                self.encoder.base_model,
                self.auxiliary.base_model,
                self.auxiliary.base_model_prefix
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path, **encoder_kwargs):
        return cls(model_name_or_path, **encoder_kwargs)

    def save_pretrained(self, output_path, **kwargs):
        encoder_output_path = os.path.join(output_path, "encoder")
        auxiliary_output_path = os.path.join(output_path, "auxiliary")

        self.encoder.save_pretrained(encoder_output_path, **kwargs)
        self.auxiliary.save_pretrained(auxiliary_output_path, **kwargs)
        self.tokenizer.save_pretrained(encoder_output_path)
        self.tokenizer.save_pretrained(auxiliary_output_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            special_tokens_mask=None,
    ):
        outputs_base_a = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        base_pooled_a = outputs_base_a.pooled_output
        outputs_base_b = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        base_pooled_b = outputs_base_b.pooled_output

        mlm_input_ids = input_ids.clone()
        mlm_labels, _ = apply_masking(self.tokenizer, self.mlm_probability, mlm_input_ids, special_tokens_mask)
        outputs_lm = self.auxiliary(
            input_ids=mlm_input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels
        )

        # Calculate loss
        ce_loss_fct = nn.CrossEntropyLoss()
        batch_size = len(base_pooled_b)
        labels = torch.arange(batch_size, device=base_pooled_a.device)

        similarity = util.cos_sim(base_pooled_b, base_pooled_a) * self.scale
        similarity_loss = ce_loss_fct(similarity, labels)
        similarity_t_loss = ce_loss_fct(similarity.t(), labels)
        lm_loss = outputs_lm.loss
        loss = similarity_loss + similarity_t_loss + lm_loss
        return SentenceEncoderOutput(
            loss=loss,
            pooled_output=outputs_base_b.pooled_output,
            logits=outputs_lm.logits,
            hidden_states=outputs_lm.hidden_states,
            attentions=outputs_lm.attentions
        )


class NmModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
            scale: float = 20.0
    ):
        """
        :param model_name_or_path: Model name or path for initializing the model
        :param pooling_strategy: the pooling strategy to use for the sentence embedding in the encoder
        :param scale: the multiplicative value for the contrastive loss
        """
        super().__init__()

        encoder_path = os.path.join(model_name_or_path, "encoder")
        if os.path.exists(encoder_path):
            encoder_name_or_path = encoder_path
        else:
            encoder_name_or_path = model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.scale = scale
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # This will be the final model used during the inference time.
        self.encoder = SentenceEncoder(encoder_name_or_path, pooling_strategy)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **encoder_kwargs):
        return cls(model_name_or_path, **encoder_kwargs)

    def save_pretrained(self, output_path, **kwargs):
        encoder_output_path = os.path.join(output_path, "encoder")
        self.encoder.save_pretrained(encoder_output_path, **kwargs)
        self.tokenizer.save_pretrained(encoder_output_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            special_tokens_mask=None,
    ):
        outputs_noised = self.encoder(input_ids=noised_input_ids, attention_mask=noised_attention_mask)
        noised_pooled = outputs_noised.pooled_output
        with torch.no_grad():
            outputs_base = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            base_pooled = outputs_base.pooled_output

        # Calculate loss
        ce_loss_fct = nn.CrossEntropyLoss()
        batch_size = len(base_pooled)
        labels = torch.arange(batch_size, device=noised_pooled.device)

        similarity = util.cos_sim(base_pooled, noised_pooled) * self.scale
        similarity_loss = ce_loss_fct(similarity, labels)
        similarity_t_loss = ce_loss_fct(similarity.t(), labels)

        loss = similarity_loss + similarity_t_loss
        return SentenceEncoderOutput(
            loss=loss,
            pooled_output=outputs_base.pooled_output,
            logits=outputs_base.logits,
            hidden_states=outputs_base.hidden_states,
            attentions=outputs_base.attentions
        )


class SimCseModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
            scale: float = 20.0
    ):
        """
        :param model_name_or_path: Model name or path for initializing the model
        :param pooling_strategy: the pooling strategy to use for the sentence embedding in the encoder
        :param scale: the multiplicative value for the contrastive loss
        """
        super().__init__()

        encoder_path = os.path.join(model_name_or_path, "encoder")
        if os.path.exists(encoder_path):
            encoder_name_or_path = encoder_path
        else:
            encoder_name_or_path = model_name_or_path

        self.pooling_strategy = pooling_strategy
        self.scale = scale
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # This will be the final model used during the inference time.
        self.encoder = SentenceEncoder(encoder_name_or_path, pooling_strategy)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **encoder_kwargs):
        return cls(model_name_or_path, **encoder_kwargs)

    def save_pretrained(self, output_path, **kwargs):
        encoder_output_path = os.path.join(output_path, "encoder")
        self.encoder.save_pretrained(encoder_output_path, **kwargs)
        self.tokenizer.save_pretrained(encoder_output_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            special_tokens_mask=None,
    ):
        outputs_base_a = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        base_pooled_a = outputs_base_a.pooled_output
        outputs_base_b = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        base_pooled_b = outputs_base_b.pooled_output

        # Calculate loss
        ce_loss_fct = nn.CrossEntropyLoss()
        batch_size = len(base_pooled_b)
        labels = torch.arange(batch_size, device=base_pooled_a.device)

        similarity = util.cos_sim(base_pooled_b, base_pooled_a) * self.scale
        similarity_loss = ce_loss_fct(similarity, labels)
        similarity_t_loss = ce_loss_fct(similarity.t(), labels)

        loss = similarity_loss + similarity_t_loss
        return SentenceEncoderOutput(
            loss=loss,
            pooled_output=outputs_base_b.pooled_output,
            logits=outputs_base_b.logits,
            hidden_states=outputs_base_b.hidden_states,
            attentions=outputs_base_b.attentions
        )
