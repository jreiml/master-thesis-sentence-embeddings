import senteval
from sentence_transformers import SentenceTransformer, models


def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) for sent in batch]
    embeddings = params['encoder'].encode(batch, show_progress_bar=False)
    return embeddings


def run_senteval(model_path_or_name, pooling_mode, full=True):
    # Set params for SentEval
    params = {'task_path': "senteval/data", 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    # Load SkipThought model
    word_embedding_model = models.Transformer(model_path_or_name, max_seq_length=512)
    embedding_dim = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(embedding_dim, pooling_mode)
    encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    params['encoder'] = encoder

    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark'] if full else ['STS12']
    return process_senteval(se.eval(tasks), full)


def process_senteval(result, full=True):
    processed = {}
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark'] if full else ['STS12']
    for task in tasks:
        if 'all' in result[task]:
            processed[task] = result[task]['all']['spearman']['mean']
        else:
            processed[task] = result[task]['spearman']
    return processed
