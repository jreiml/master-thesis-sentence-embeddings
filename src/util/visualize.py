import os
import re
from typing import Optional, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datasets import Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from util.args import DataArguments
from data.constants import PROCESSED_LABEL_COL, TEXT_COL
from data.data_preprocessing import get_raw_label_mapping, normalize_labels
from data.dataset_loader import load_raw_dataset


def save_figure(fig, png_dir, title):
    """ Saves a figure as a png. file. """
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    file_name = 'image.png'
    if title is not None and title != '':
        file_name = f"{title.replace(' ', '_')}.png"

    fig.write_image(os.path.join(png_dir, file_name))


def get_projection_values(manifold, plot, embeddings):
    """ Return the projected values for a given set of embeddings (optionally, using on a given model). """
    if not manifold:
        if plot == "pca":
            manifold = PCA(n_components=2)
        elif plot == "tsne":
            manifold = TSNE(n_components=2)
        elif plot == "umap":
            manifold = UMAP(n_components=2)
        else:
            raise Exception(f"Reduction {plot} not supported")
        # Fit model.
        manifold = manifold.fit(embeddings)
    # Get transformed values.
    new_values = manifold.transform(embeddings)
    return manifold, new_values


def plot_embeddings(ids, embeddings, labels, projection="umap", title="", manifold=None, png_dir=None,
                    interactive=True, width=550, height=550):
    """Plots the embedding in a lower dimensional space  as a scatter plot."""
    palette = px.colors.qualitative.Dark24
    pio.templates.default = "plotly"
    projection = projection.lower()
    manifold, new_values = get_projection_values(manifold, projection, embeddings)
    hover_data = ["label"]
    x, y = new_values[:, 0], new_values[:, 1]
    if labels is not None:
        df = pd.DataFrame(zip(ids, labels, x, y), columns=["name", "label", "n1", "n2"])
        df = df.sort_values(by=['label'], ascending=False)
        fig = px.scatter(df, x='n1', y='n2', color="label", hover_name='name', hover_data=hover_data,
                         color_discrete_sequence=palette)
    else:
        df = pd.DataFrame(zip(ids, x, y), columns=["name", "n1", "n2"])
        fig = px.scatter(df, x='n1', y='n2',hover_name='name', hover_data=hover_data,
                         color_discrete_sequence=palette)

    fig.update_layout(autosize=False, width=width, height=height, title=title, legend={'traceorder': 'reversed'})
    if interactive:
        fig.show()
    # Save figure as png.
    if png_dir:
        save_figure(fig, png_dir, title)
    return manifold


def get_umap_hook(dataset: Dataset,
                  raw_text_col: str,
                  raw_label_col: Optional[str] = None,
                  name: Optional[str] = None,
                  output_dir: Optional[str] = None,
                  width: int = 550,
                  height: int = 550) -> Callable:
    manifolds = {}
    texts = dataset[raw_text_col]
    texts_with_line_breaks = [' '.join(
        [word if i % 10 != 0 else word + "<br>" for i, word in enumerate(re.split(r"\s+", text), start=1)]
    ) for text in texts]

    labels = None
    if raw_label_col is not None:
        dataset: Dataset = normalize_labels(dataset, raw_label_col)
        label_text_mapping = get_raw_label_mapping(dataset, raw_label_col)
        labels = [str(label_text_mapping[label]) for label in dataset[PROCESSED_LABEL_COL]]

    def visualize_umap(model: SentenceTransformer, epoch: int = -1, steps: int = -1,
                       interactive: bool = False, comment: Optional[str] = None,):
        title = name if name is not None else ''
        if epoch != -1:
            if steps == -1:
                title += f" epoch {epoch}"
            else:
                title += f" epoch {epoch} steps {steps} "
        if comment is not None:
            title += f" {comment}"
        title = re.sub(r'\s+', ' ', title)
        print(f"UMAP visualization {title}")
        embeddings = np.array(model.encode(texts))

        # For same comment/epoch: use same umap reduction
        manifold_key = f"{comment if comment is not None else ''}_{epoch}"
        manifold = None
        if manifold_key in manifolds:
            manifold = manifolds[manifold_key]
        manifolds[manifold_key] = plot_embeddings(ids=texts_with_line_breaks, embeddings=embeddings, labels=labels,
                                                  title=title, manifold=manifold, png_dir=output_dir,
                                                  interactive=interactive, width=width, height=height)

    return visualize_umap


def visualize_encoder(data_args: DataArguments, model_name, max_seq_length: Optional[int] = None,
                      width=550, height=550):
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    embedding_dim = word_embedding_model.get_word_embedding_dimension()
    
    pooling_model = models.Pooling(embedding_dim,
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    dataset = load_raw_dataset(data_args)
    dataset = concatenate_datasets(list(dataset.values()))
    hook = get_umap_hook(dataset, TEXT_COL, data_args.label_col, width=width, height=height)
    hook(bi_encoder, interactive=True)
