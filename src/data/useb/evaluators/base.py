import logging
from typing import Dict, List

import torch
from torch.nn import functional as F
from tqdm.auto import trange

logger = logging.getLogger(__name__)


class BaseEvaluator(object):

    def __init__(self, semb_fn, bsz, show):
        self.semb_fn = semb_fn
        self.show = show
        self.bsz = bsz
    
    @property
    def metric_names(self) -> List[str]:
        raise NotImplementedError

    def set_semb_fn(self, semb_fn):
        self.semb_fn = semb_fn

    def _round_percentage(self, results: Dict[str, float]):
        return {k: round(v * 100, 2) for k, v in results.items()}

    def _text2se(self, texts, normalize=False) -> torch.Tensor:
        bsz = self.bsz
        show = self.show
        embs = []
        texts = list(map(lambda text:text.strip(), texts))
        for i in trange(0, len(texts), bsz, disable=not show):
            batch = texts[i:i+bsz]
            emb: torch.Tensor = self.semb_fn(batch)
            if normalize:
                emb = F.normalize(emb, dim=-1)
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def _run(self, eval_type) -> Dict[str, float]:
        raise NotImplementedError

    def run(self, eval_type):
        if self.show: logger.info(f'============ evaluation on {self.name} ({eval_type}) ============')
        results: Dict[str, float] = self._run(eval_type)
        if self.show: logger.info(f'Final results: {results}')
        return self._round_percentage(results)
