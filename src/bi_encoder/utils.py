from sentence_transformers import models, SentenceTransformer


def create_bi_encoder(model_name, max_length, pooling_mode):
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_length)
    embedding_dim = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(embedding_dim, pooling_mode)
    bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return bi_encoder
