import enum
from typing import NamedTuple, Optional, List, Union, Dict, Callable


class PoolingStrategy(enum.Enum):
    CLS = "CLS"
    MEAN = "MEAN"
    MAX = "MAX"


class CrossEncoderActivationFunction(enum.Enum):
    IDENTITY = "IDENTITY"
    SIGMOID = "SIGMOID"
    TANH = "TANH"


class PairGenerationStrategy(enum.Enum):
    RANDOM = "RANDOM"
    BM25 = "BM25"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    REVERSE_BM25 = "REVERSE_BM25"
    REVERSE_SEMANTIC_SEARCH = "REVERSE_SEMANTIC_SEARCH"


class DomainObjectiveType(enum.Enum):
    CONTRASTIVE = "CONTRASTIVE"
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    MCR2 = "MCR2"
    MCR2_DISCRIM = "MCR2_DISCRIM"
    MCR2_COMPRESS = "MCR2_COMPRESS"


class PretrainMode(enum.Enum):
    MLM = "MLM"
    MLM_PAIR = "MLM_PAIR"
    MLM_SOP = "MLM_SOP"
    TSDAE = "TSADE"
    TSADE_NM = "TSADE_NM"
    MLM_NM = "MLM_NM"
    MLM_SIMCSE = "MLM_SIMCSE"
    SIMCSE = "SIMCSE"
    NM = "NM"


class DataArguments(NamedTuple):
    dataset_path: str
    is_hugging_face_dataset: bool
    text_col: str
    label_col: Optional[str] = None
    raw_dataset_cache_path: Optional[str] = None
    pair_dataset_cache_path: Optional[str] = None
    silver_dataset_cache_path: Optional[str] = None
    validation_percent: float = 0.0
    test_percent: float = 0.0
    pair_generation_strategies: List[PairGenerationStrategy] = [PairGenerationStrategy.RANDOM]
    pair_generation_batch_size: int = 0
    top_k_pairs: List[int] = [1]
    data_generation_seed: int = 42
    csv_delimiter: str = ","
    split_into_sentence_grams: Optional[int] = None
    tokenizer_name: Optional[str] = None
    max_length: Optional[int] = None
    strip_whitespaces: Optional[bool] = True
    do_lowercase: Optional[bool] = False
    filter_duplicates: Optional[bool] = False
    truncate_hierarchical_label_to_length: Optional[int] = None
    minimum_required_samples_per_class: Optional[int] = None
    new_label_for_filtered_samples: Optional[bool] = False
    use_data_percentage: Optional[float] = None
    dataset_size_limit: Optional[float] = None
    raw_filter_fn: Optional[Callable[[Dict], Dict]] = None
    raw_map_fn: Optional[Callable[[Dict], Dict]] = None
    semantic_search_model: Optional[str] = 'all-MiniLM-L6-v2'
    is_text_dataset: bool = False
    is_csv_dataset: bool = False


class BiEncoderArguments(NamedTuple):
    model_name: str
    output_path: str
    batch_size: int = 16
    train_epochs: int = 3
    warmup_percent: float = 0.1
    max_length: Optional[int] = None
    train_seed: int = 42
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    seed_optimization_steps: int = 0
    save_checkpoints: bool = True
    use_multiple_negatives_ranking: bool = False
    pooling_mode: str = 'mean'


class MultitaskDistillBiEncoderArguments(NamedTuple):
    model_name: str
    output_path: str
    # Will use the same labels for now
    domain_objective_types: Union[DomainObjectiveType, List[DomainObjectiveType]]
    distill_objective_weight: float = 1
    domain_objective_weights: Optional[Union[float, List[float]]] = None
    use_manual_weight_train: bool = False
    batch_size: int = 16
    steps_per_epoch: Optional[int] = None
    train_epochs: int = 3
    warmup_percent: float = 0.1
    max_length: Optional[int] = None
    train_seed: int = 42
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    mcr2_gamma: float = 1.0
    mcr2_eps: float = 0.01
    visualize: bool = True
    save_checkpoints: bool = True
    pooling_mode: str = 'mean'


class CrossEncoderArguments(NamedTuple):
    model_name: str
    output_path: Optional[str] = None
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    train_epochs: int = 3
    warmup_percent: float = 0.1
    max_length: Optional[int] = None
    train_seed: int = 42
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    seed_optimization_steps: int = 0
    use_cross_bi_encoder: bool = True
    freeze_embeddings: bool = False
    # Only for Cross-Encoder
    activation_function: Optional[CrossEncoderActivationFunction] = None


class PretrainArguments(NamedTuple):
    train_mode: PretrainMode
    model_name: str
    output_path: Optional[str] = None
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = "linear"
    max_steps: int = -1
    train_epochs: int = 3
    warmup_percent: float = 0.1
    max_length: Optional[int] = None
    train_seed: int = 42
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    noise_probability: float = 0.2
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    prompt_delimiter: Optional[str] = None
