import logging
import os
import pickle

from sentence_transformers import LoggingHandler

from data.experiment_dataset_loader import load_wikipedia
from experiments.run_senteval import run_senteval
from experiments.utils import ensure_dir_exists, get_english_default_model, get_train_seeds, get_english_sbert_model
from pretrain.train import train_nm_adapted_encoder
from util.args import PretrainMode, PretrainArguments, PoolingStrategy


def save_cache(cache):
    with open('output/unsupervised-wikipedia/cache.pkl', 'wb') as fp:
        pickle.dump(cache, fp)


def load_cache():
    if os.path.exists('output/unsupervised-wikipedia/cache.pkl'):
        with open('output/unsupervised-wikipedia/cache.pkl', 'rb') as fp:
            return pickle.load(fp)
    ensure_dir_exists('output/unsupervised-wikipedia')

    return {}


def run_pretrained_experiment(cache, full):
    sbert_model = get_english_sbert_model(full)
    cache_key = ("pretrained", 0)
    if cache_key in cache:
        return

    result = run_senteval(sbert_model, 'mean')
    cache[cache_key] = result
    save_cache(cache)


def run_nm_experiments(cache, full):
    bert_model = get_english_default_model(full)
    seeds = get_train_seeds(False)
    for seed in seeds:
        dataset = load_wikipedia(test_size=0.0, size_limit=800000, simple=False)

        for train_mode in [PretrainMode.MLM_SIMCSE]:  # [PretrainMode.NM, PretrainMode.TSADE_NM, PretrainMode.MLM_NM, PretrainMode.TSDAE]:
            train_mode_str = train_mode.value
            cache_key = (train_mode_str, seed)
            if cache_key in cache and train_mode != PretrainMode.TSDAE:
               continue

            output_path = f"output/unsupervised-wikipedia/{train_mode_str}/{seed}"
            args = PretrainArguments(
                train_mode=train_mode,
                model_name=bert_model,
                noise_probability=0.6,
                output_path=output_path,
                train_batch_size=64 if train_mode != PretrainMode.TSDAE else 8,
                train_epochs=1,
                max_length=128,
                learning_rate=3e-5,
                pooling_strategy=PoolingStrategy.CLS
            )

            # train 12500 steps
            train_nm_adapted_encoder(args, dataset)

            encoder_path = os.path.join(output_path, "encoder")
            result = run_senteval(encoder_path, 'cls')
            cache[cache_key] = result
            save_cache(cache)


def run_unsupervised_wikipedia_experiment(full):
    cache = load_cache()
    run_pretrained_experiment(cache, full)
    run_nm_experiments(cache, full)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    run_unsupervised_wikipedia_experiment(True)
