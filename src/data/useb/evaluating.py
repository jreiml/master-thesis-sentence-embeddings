import json
import logging
import os
from typing import Dict, List, Tuple, Callable

import torch

from .evaluators import AskubuntuEvaluator, CQADupStackEvaluator, TwitterParaEvaluator, SciDocsEvaluator

EVALUATOR_MAP = {evaluator_class.name: evaluator_class for evaluator_class in [AskubuntuEvaluator, CQADupStackEvaluator, TwitterParaEvaluator, SciDocsEvaluator]}
logger = logging.getLogger(__name__)


def run_on(
    dataset_name:str, 
    semb_fn: Callable[[List[str],], torch.Tensor], 
    eval_type:str = 'test', 
    data_eval_path:str = './datasets/usep/data-eval'
) -> Dict[str, float]:
    """
    Run on one single dataset.
    :param dataset_name: Which target dataset from ['AskUbuntu', 'CQADupStack', 'TwitterPara', 'SciDocs'] (the lower case is also acceptable)
    :param semb_fn: The sentence embedding function which changes list of strings into scores of the torch.Tensor type
    :eval_type: Evaluation on either 'valid' or 'test' set
    :data_eval_path: The path to the usep eval datasets
    :return: Returns scores in the format of Dict[str, float] (the scores are post-processed with round(score x 100, 2))
    """
    assert dataset_name.lower() in EVALUATOR_MAP, f"'dataset_name' should be one of these: {list(EVALUATOR_MAP)}"
    assert eval_type in ['valid', 'test'], f"'eval_type' should be one of ['valid', 'test']"
    evaluator_class = EVALUATOR_MAP[dataset_name.lower()]
    return evaluator_class(semb_fn, os.path.join(data_eval_path, evaluator_class.name)).run(eval_type)


def run(
    semb_fn_askubuntu: Callable[[List[str],], torch.Tensor], 
    semb_fn_cqadupstack: Callable[[List[str],], torch.Tensor], 
    semb_fn_twitterpara: Callable[[List[str],], torch.Tensor], 
    semb_fn_scidocs: Callable[[List[str],], torch.Tensor], 
    eval_type:str = 'test', 
    data_eval_path:str = './datasets/usep/data-eval'
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Run on one single dataset.
    :param semb_fn_xxx: The sentence embedding function for dataset xxx, which changes list of strings into scores of the torch.Tensor type
    :eval_type: Evaluation on either 'valid' or 'test' set
    :data_eval_path: The path to the usep eval datasets
    :return: Returns both detailed scores and main scores (using Average Precision)
    """
    assert eval_type in ['valid', 'test'], f"'eval_type' should be one of ['valid', 'test']"
    results = {}
    results_main_metric = {}
    for semb_fn, evaluator_class in zip(
        [semb_fn_askubuntu, semb_fn_cqadupstack, semb_fn_twitterpara, semb_fn_scidocs], 
        [AskubuntuEvaluator, CQADupStackEvaluator, TwitterParaEvaluator, SciDocsEvaluator]
    ):
        evaluator = evaluator_class(semb_fn, os.path.join(data_eval_path, evaluator_class.name))
        result = evaluator.run(eval_type)
        results[evaluator_class.name] = result  # all the detailed scores for the dataset
        results_main_metric[evaluator_class.name] = result[evaluator_class.main_metric]  # the score for the main metric for the dataset
    results_main_metric['avg'] = sum(results_main_metric.values()) / len(results_main_metric.values())
    logger.info('============ evaluation done ============')
    logger.info(f'Main evaluation scores (average precision): {results_main_metric}')

    with open('results.detailed.json', 'w') as f:
        json.dump(results, f, indent=4)
        logger.info('saved detailed scores to ./results.detailed.json')
    with open('results.average_precision.json', 'w') as f:
        json.dump(results_main_metric, f, indent=4)
        logger.info('saved main scores to ./results.average_precision.json')
    return results, results_main_metric