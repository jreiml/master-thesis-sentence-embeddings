import logging

from sentence_transformers import LoggingHandler

from experiments.augsbert_semi import run_augsbert_semi_experiment
from experiments.distill_classification import compare_bws_distill_classification
from experiments.unsupervised_domain import run_unsupervised_domain_experiments, print_results
from experiments.sampling_distribution import compare_sampling
from experiments.sts_correlation import compare_stsb_correlation
from experiments.unsupervised_wikipedia import run_unsupervised_wikipedia_experiment


def run_all(full):
    compare_stsb_correlation(full)
    compare_stsb_sampling(full)
    compare_bws_distill_classification(full)
    print_results(full)
    run_unsupervised_wikipedia_experiment(full)
    run_augsbert_semi_experiment(full)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    import nltk
    nltk.download("punkt")

    run_all(True)
