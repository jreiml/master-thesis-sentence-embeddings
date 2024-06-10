import importlib
from collections import OrderedDict

from transformers.models.auto.auto_factory import auto_class_update, _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, model_type_to_module_name


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("model")
    return getattribute_from_module(transformers_module, attr)


class CustomLazyAutoMapping(_LazyAutoMapping):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:

        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):

        super().__init__(config_mapping, model_mapping)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "model")
        return getattribute_from_module(self._modules[module_name], attr)


MODEL_FOR_EMBEDDING_SIMILARITY_CROSS_ENCODER_MAPPING_NAMES = OrderedDict(
    [
        ("deberta", "DebertaForEmbeddingSimilarityCrossBiEncoder"),
        ("bert", "BertForEmbeddingSimilarityCrossBiEncoder"),
        ("roberta", "RobertaForEmbeddingSimilarityCrossBiEncoder"),
        ("xlm-roberta", "XLMRobertaForEmbeddingSimilarityCrossBiEncoder"),
    ]
)

MODEL_FOR_EMBEDDING_SIMILARITY_CROSS_ENCODER_MAPPING = CustomLazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_EMBEDDING_SIMILARITY_CROSS_ENCODER_MAPPING_NAMES
)


class AutoModelForEmbeddingSimilarityCrossBiEncoder(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_EMBEDDING_SIMILARITY_CROSS_ENCODER_MAPPING


AutoModelForEmbeddingSimilarityCrossBiEncoder = auto_class_update(
    AutoModelForEmbeddingSimilarityCrossBiEncoder, head_doc="embedding similarity cross encoder"
)

###
MODEL_FOR_CROSS_ENCODER_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("deberta", "DebertaForContrastiveCrossBiEncoder"),
        ("bert", "BertForContrastiveCrossBiEncoder"),
        ("roberta", "RobertaForContrastiveCrossBiEncoder"),
        ("xlm-roberta", "XLMRobertaForContrastiveCrossBiEncoder"),
    ]
)

MODEL_FOR_CROSS_ENCODER_CLASSIFICATION_MAPPING = CustomLazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CROSS_ENCODER_CLASSIFICATION_MAPPING_NAMES
)


class AutoModelForContrastiveCrossBiEncoder(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CROSS_ENCODER_CLASSIFICATION_MAPPING


AutoModelForContrastiveCrossBiEncoder = auto_class_update(
    AutoModelForContrastiveCrossBiEncoder, head_doc="pair classification cross encoder"
)

###
MODEL_FOR_FIXED_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        ("deberta", "DebertaForFixedMaskedLM"),
        ("bert", "BertForFixedMaskedLM"),
        ("roberta", "RobertaForFixedMaskedLM"),
        ("xlm-roberta", "XLMRobertaForFixedMaskedLM"),
    ]
)

MODEL_FOR_FIXED_MASKED_LM_MAPPING = CustomLazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_FIXED_MASKED_LM_MAPPING_NAMES
)


class AutoModelForFixedMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_FIXED_MASKED_LM_MAPPING


AutoModelForFixedMaskedLM = auto_class_update(
    AutoModelForFixedMaskedLM, head_doc="fixed masked language model"
)


###
MODEL_FOR_SENTENCE_EMBEDDING_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("deberta", "DebertaForSentenceEmbeddingClassification"),
        ("bert", "BertForSentenceEmbeddingClassification"),
        ("roberta", "RobertaForSentenceEmbeddingClassification"),
        ("xlm-roberta", "XLMRobertaForSentenceEmbeddingClassification"),
        ("mpnet", "MPNetForSentenceEmbeddingClassification"),
    ]
)

MODEL_FOR_SENTENCE_EMBEDDING_CLASSIFICATION_MAPPING = CustomLazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SENTENCE_EMBEDDING_CLASSIFICATION_MAPPING_NAMES
)


class AutoModelForSentenceEmbeddingClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SENTENCE_EMBEDDING_CLASSIFICATION_MAPPING


AutoModelForSentenceEmbeddingClassification = auto_class_update(
    AutoModelForSentenceEmbeddingClassification, head_doc="classification for mean pooled sentence embeddings"
)

