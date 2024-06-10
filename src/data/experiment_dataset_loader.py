import csv
import gzip
import os
import random
from collections import defaultdict, Counter
from typing import Tuple, List

from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
from sentence_transformers import util, InputExample
from transformers import set_seed

from data.constants import TEXT_A_COL, TEXT_B_COL, PROCESSED_LABEL_COL, PROCESSED_LABEL_B_COL, PROCESSED_LABEL_A_COL
from data.data_preprocessing import normalize_labels
from data.multitask_pair_dataset import MultitaskInputExample, MultitaskPairDatasetDict, MultitaskPairDataset
from data.pair_dataset import PairDataset, PairDatasetDict


def load_nli(size_limit=-1) -> DatasetDict:
    rename_dict = {
        "premise": TEXT_A_COL,
        "hypothesis": TEXT_B_COL,
        "label": PROCESSED_LABEL_COL
    }

    snli = load_dataset("snli")
    snli = snli.rename_columns(rename_dict)
    mnli = load_dataset("glue", "mnli")
    mnli = mnli.rename_columns(rename_dict)
    mnli = mnli.remove_columns("idx")
    mnli["validation"] = mnli["validation_mismatched"]
    mnli["test"] = mnli["test_mismatched"]
    del mnli["validation_mismatched"], mnli["test_mismatched"], mnli["validation_matched"], mnli["test_matched"]
    assert sorted(mnli.keys()) == sorted(snli.keys()), "Dataset split mismatch fo mnli/snli!"

    dataset = DatasetDict({
        key: concatenate_datasets([snli[key], mnli[key]]) for key in snli
    })
    if size_limit != -1:
        dataset = DatasetDict({key: Dataset.from_dict(split[:size_limit]) for key, split in dataset.items()})
    return dataset


def get_samples_for_input_score_pair(texts_filepath, score_filepath, max_score):
    samples = []

    with open(texts_filepath, "r", encoding="utf-8") as texts_file:
        with open(score_filepath, "r", encoding="utf-8") as score_file:
            for raw_texts, score in zip(texts_file.readlines(), score_file.readlines()):
                if len(raw_texts.strip()) == 0:
                    continue

                texts = raw_texts.split("\t")
                if len(texts) != 2:
                    raise ValueError(f"Unexpected texts value: {texts}")
                label = float(score) / max_score
                sample = InputExample(texts=texts, label=label)
                samples.append(sample)

    return samples


def load_spanish_sts() -> PairDatasetDict:
    train_dataset_files = [
        ("sts14-li65-input.txt", "sts14-li65-score.txt", 4),
        ("sts14-news-input.txt", "sts14-news-score.txt", 4),
        ("sts14-wikipedia-input.txt", "sts14-wikipedia-score.txt", 4),
        ("sts15-newswire-input.txt", "sts15-newswire-score.txt", 4),
        ("sts15-wikipedia-input.txt", "sts15-wikipedia-score.txt", 4),
    ]
    test_dataset_files = [
        ("sts17-input.txt", "sts17-score.txt", 5)
    ]
    gold_samples = [
        sample
        for texts_fp, score_fp, score_range in train_dataset_files
        for sample in get_samples_for_input_score_pair(
            f"datasets/es-sts/{texts_fp}", f"datasets/es-sts/{score_fp}", score_range
        )
    ]
    test_samples = [
        sample
        for texts_fp, score_fp, score_range in test_dataset_files
        for sample in get_samples_for_input_score_pair(
            f"datasets/es-sts/{texts_fp}", f"datasets/es-sts/{score_fp}", score_range
        )
    ]

    set_seed(42)
    random.shuffle(gold_samples)
    dev_samples = gold_samples[:220]
    gold_samples = gold_samples[220:]
    assert len(gold_samples) == 1400 and len(dev_samples) == 220 and len(test_samples) == 250, \
        "Unexpected dataset size!"

    return PairDatasetDict(
        train=PairDataset(gold_samples),
        validation=PairDataset(dev_samples),
        test=PairDataset(test_samples)
    )


def load_mrpc() -> PairDatasetDict:
    mrpc = load_dataset("glue", "mrpc")

    pair_datasets = {}
    for key, split in mrpc.items():
        sentences1 = split["sentence1"]
        sentences2 = split["sentence2"]
        labels = split["label"]
        pair_datasets[key] = PairDataset([
            InputExample(texts=[sentence1, sentence2], label=label)
            for sentence1, sentence2, label in zip(sentences1, sentences2, labels)
        ])

    return PairDatasetDict(**pair_datasets)


def load_qqp() -> PairDatasetDict:
    qqp = load_dataset("glue", "qqp")

    pair_datasets = {}

    for key, split in qqp.items():
        questions1 = split["question1"]
        questions2 = split["question2"]
        labels = split["label"]
        pair_datasets[key] = PairDataset([
            InputExample(texts=[question1, question2], label=label)
            for question1, question2, label in zip(questions1, questions2, labels)
        ])

    return PairDatasetDict(**pair_datasets)


def load_sts_test() -> Tuple[List[str], List[PairDataset]]:
    names = ["sts12-sts", "sts13-sts", "sts14-sts", "sts15-sts", "sts16-sts", "stsbenchmark-sts", "sickr-sts"]
    datasets = [
        PairDataset.from_dataset(load_dataset(f"mteb/{name}", split="test"),
                                 text_a_col="sentence1", text_b_col="sentence2", label_col="score", label_scale=1/5)
        for name in names
    ]

    return names, datasets


def load_stsb(size_limit=None) -> PairDatasetDict:
    # Check if dataset exists. If not, download and extract it
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    gold_samples = []
    dev_samples = []
    test_samples = []

    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        rows = list(reader)
        import random
        random.seed(42)
        random.shuffle(rows)
        if size_limit is not None:
            rows = rows[:size_limit]

        for row in rows:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

            if row['split'] == 'dev':
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            elif row['split'] == 'test':
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
            else:
                gold_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    return PairDatasetDict(
        train=PairDataset(gold_samples),
        validation=PairDataset(dev_samples),
        test=PairDataset(test_samples)
    )


# See https://arxiv.org/pdf/2010.08240.pdf
def load_bws_cross_topic() -> MultitaskPairDatasetDict:
    bws_dataset_path = "datasets/BWS_Argument_Similarity_Corpus.csv"
    if not os.path.exists(bws_dataset_path):
        raise ValueError(
            f"BWS dataset not found! "
            f"Please download it at https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2496.2 "
            f"and place it in {bws_dataset_path}"
        )

    gold_samples = []
    dev_samples = []
    test_samples = []

    topic_to_split = {
        "Cloning": gold_samples,
        "Abortion": gold_samples,
        "Minimum wage": gold_samples,
        "Marijuana legalization": gold_samples,
        "Nuclear energy": gold_samples,
        "Death penalty": dev_samples,
        "Gun control": test_samples,
        "School uniforms": test_samples,
    }
    topic_to_id = {topic: i for i, topic in enumerate(sorted(topic_to_split))}
    id_to_count = defaultdict(int)

    with open(bws_dataset_path, "r", encoding="utf-8") as bws_dataset_file:
        reader = csv.DictReader(bws_dataset_file, delimiter=',')
        rows = list(reader)
        set_seed(42)
        random.shuffle(rows)

        for row in rows:
            score = float(row['score'])
            topic = row["topic"]
            topic_id = topic_to_id[topic]
            id_to_count[topic_id] += 1

            input_example = MultitaskInputExample(
                texts=[row['argument1'], row['argument2']],
                similarity_label=score,
                domain_labels=[topic_id, topic_id]
            )
            split = topic_to_split[topic]
            split.append(input_example)

    total_count = sum(id_to_count.values())
    domain_label_pair_distribution = {topic_id: count/total_count for topic_id, count in id_to_count.items()}

    return MultitaskPairDatasetDict(
        train=MultitaskPairDataset(gold_samples, domain_label_pair_distribution),
        validation=MultitaskPairDataset(dev_samples, domain_label_pair_distribution),
        test=MultitaskPairDataset(test_samples, domain_label_pair_distribution)
    )


# See https://arxiv.org/pdf/2010.08240.pdf
def load_bws_in_topic() -> MultitaskPairDatasetDict:
    bws_dataset_path = "datasets/BWS_Argument_Similarity_Corpus.csv"
    if not os.path.exists(bws_dataset_path):
        raise ValueError(
            f"BWS dataset not found! "
            f"Please download it at https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2496.2 "
            f"and place it in {bws_dataset_path}"
        )

    gold_samples = []
    dev_samples = []
    test_samples = []

    topic_to_split_count = {
        "Cloning": [266, 53, 106],
        "Abortion": [266, 53, 106],
        "Minimum wage": [266, 53, 106],
        "Marijuana legalization": [266, 53, 106],
        "Nuclear energy": [266, 53, 106],
        "Death penalty": [265, 54, 106],
        "Gun control": [265, 53, 107],
        "School uniforms": [265, 53, 107],
    }

    assert sum([sum(split_count) for split_count in topic_to_split_count.values()]) == 3400, "Unhandled topic count!"
    topic_to_id = {topic: i for i, topic in enumerate(sorted(topic_to_split_count))}
    id_to_count = defaultdict(int)

    with open(bws_dataset_path, "r", encoding="utf-8") as bws_dataset_file:
        reader = csv.DictReader(bws_dataset_file, delimiter=',')
        rows = list(reader)
        set_seed(42)
        random.shuffle(rows)

        for row in rows:
            score = float(row['score'])
            topic = row["topic"]
            topic_id = topic_to_id[topic]
            id_to_count[topic_id] += 1

            input_example = MultitaskInputExample(
                texts=[row['argument1'], row['argument2']],
                similarity_label=score,
                domain_labels=[topic_id, topic_id]
            )
            split_count = topic_to_split_count[topic]
            if split_count[0] > 0:
                split_count[0] -= 1
                gold_samples.append(input_example)
            elif split_count[1] > 0:
                split_count[1] -= 1
                dev_samples.append(input_example)
            elif split_count[2] > 0:
                split_count[2] -= 1
                test_samples.append(input_example)
            else:
                raise ValueError("Unhandled topic count! Did you load the correct dataset?")

    total_count = sum(id_to_count.values())
    domain_label_pair_distribution = {topic_id: count/total_count for topic_id, count in id_to_count.items()}

    return MultitaskPairDatasetDict(
        train=MultitaskPairDataset(gold_samples, domain_label_pair_distribution),
        validation=MultitaskPairDataset(dev_samples, domain_label_pair_distribution),
        test=MultitaskPairDataset(test_samples, domain_label_pair_distribution)
    )


def load_wikipedia(test_size=0.0, size_limit=-1, simple=False):
    dataset = load_dataset("wikipedia", "20220301.simple" if simple else "20220301.en")
    dataset = dataset["train"]
    dataset = dataset.remove_columns(["id", "url", "title"])

    def process(example):
        new_texts = []
        for text in example["text"]:
            for sentence in text.split("\n"):
                sentence = sentence.strip()
                if len(sentence) == 0:
                    continue
                new_texts.append(sentence)

        return {"text": new_texts}

    dataset = dataset.map(process, batched=True)

    unique_texts = set()

    def deduplicate(example):
        text = example["text"]
        if text in unique_texts:
            return False
        unique_texts.add(text)
        return True

    dataset = dataset.filter(deduplicate)
    if size_limit != -1:
        dataset = dataset.shuffle(seed=42)
        dataset = Dataset.from_dict(dataset[:size_limit])
    if test_size > 0.0:
        dataset = dataset.train_test_split(test_size=test_size)
        dataset["validation"] = dataset["test"]
        del dataset["test"]
    else:
        dataset = DatasetDict({"train": dataset})

    return dataset

def load_prompted_20newsgroups(validation_size=0.1):
    # groups from http://qwone.com/~jason/20Newsgroups/
    label_text_to_supergroup = {
        # Computers
        "comp.graphics": "Computers",
        "comp.os.ms-windows.misc": "Computers",
        "comp.sys.ibm.pc.hardware": "Computers",
        "comp.sys.mac.hardware": "Computers",
        "comp.windows.x": "Computers",
        # Recreational
        "rec.autos": "Recreational",
        "rec.motorcycles": "Recreational",
        "rec.sport.baseball": "Recreational",
        "rec.sport.hockey": "Recreational",
        # Science
        "sci.crypt": "Science",
        "sci.electronics": "Science",
        "sci.med": "Science",
        "sci.space": "Science",
        # For Sale
        "misc.forsale": "For Sale",
        # Politics
        "talk.politics.misc": "Politics",
        "talk.politics.guns": "Politics",
        "talk.politics.mideast": "Politics",
        # Religion
        "talk.religion.misc": "Religion",
        "alt.atheism": "Religion",
        "soc.religion.christian": "Religion",
    }

    dataset = load_dataset("SetFit/20_newsgroups")

    if validation_size > 0.0:
        validation_size = validation_size / 0.6  # get validation data from train_data
        dataset_train_validation = dataset["train"].train_test_split(test_size=validation_size, seed=42)
        dataset["train"] = dataset_train_validation["train"]
        dataset["validation"] = dataset_train_validation["test"]

    dataset = dataset.rename_column("text", "original_text")

    # [PROMPT] ¥[TEXT]
    def map_fn(example):
        supergroup = label_text_to_supergroup[example["label_text"]]
        text = example['original_text'].replace('¥', '')
        prompted_text = f"{supergroup} ¥{text}"
        return {
            "prompted_text": prompted_text,
            "supergroup": supergroup
        }

    dataset = dataset.map(map_fn)
    return dataset


def load_prompted_ukp_sentential_argument_mining():
    ukp_sentential_dataset = 'datasets/ukp_sentential_argument_mining.csv'

    if not os.path.exists(ukp_sentential_dataset):
        raise ValueError("Unable to find UKP sentential argument mining dataset! Please use BA thesis scripts.")

    dataset = load_dataset("csv", data_files=ukp_sentential_dataset)["train"]
    dataset = dataset.rename_column("sentence", "original_text")
    # [PROMPT] ¥[TEXT]
    dataset = dataset.map(lambda ex: {
        "prompted_text": f"{ex['topic']} ¥{ex['original_text'].replace('¥', '')}",
    })
    dataset = DatasetDict({
        "train": dataset.filter(lambda ex: ex["set"] == "train"),
        "validation": dataset.filter(lambda ex: ex["set"] == "dev"),
        "test": dataset.filter(lambda ex: ex["set"] == "test")
    })
    dataset = dataset.remove_columns(["id", "is_argument", "set", "stance"])
    return dataset

def load_prompted_bws_in_topic():
    topic_to_text = {
        # Evidences sentences
        "Abortion": "We should ban abortions",
        # Arg Rank 30k, Evidences sentences
        "Cloning": "We should ban human cloning",
        # Arg Rank 30k, Evidences sentences
        "Death penalty": "We should abolish capital punishment",
        # Evidences sentences
        "Gun control": "We should increase gun control",
        # Arg Rank 30k, Evidences sentences
        "Marijuana legalization": "We should legalize cannabis",
        # Custom
        "Minimum wage": "We should introduce a minimum wage",
        # Evidences sentences
        "Nuclear energy": "We should further exploit nuclear power",
        # Evidences sentences
        "School uniforms": "We should ban school uniforms",
    }

    id_to_topic_text = {i: topic_to_text[topic] for i, topic in enumerate(sorted(topic_to_text))}
    bws = load_bws_in_topic()

    def map_example(input_example: MultitaskInputExample):
        topic_text = id_to_topic_text[input_example.domain_labels[0]]
        new_texts = [f"{topic_text} ¥{text.replace('¥', '')}" for text in input_example.texts]
        return MultitaskInputExample(
            input_example.guid,
            new_texts,
            input_example.similarity_label,
            input_example.domain_labels,
        )

    dataset = bws.map(map_example)
    return dataset
