import torch
from sentence_transformers import SentenceTransformer

from data.useb import run_on

sbert = SentenceTransformer('bert-base-nli-mean-tokens')

@torch.no_grad()
def semb_fn(sentences) -> torch.Tensor:
    return torch.Tensor(sbert.encode(sentences, show_progress_bar=False))

result = run_on(
    'askubuntu',
    semb_fn=semb_fn,
    eval_type='test',
    data_eval_path='datasets/usep/data-eval'
)

assert round(result['map_askubuntu_title'], 1) == 52.6