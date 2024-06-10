# Sentence Embeddings in Various Supervision Settings

This repository contains the code and results for my master’s thesis:

**Title:** Sentence Embeddings in Various Supervision Settings  
**Author:** Johanna Reiml  
**University:** Ludwig-Maximilians-Universität München
**Supervised by:** Prof. Dr. Thomas Seidl  
**Advised by:** Dr. Michael Fromm  
**Submission Date:** February 13, 2023

## Abstract

This thesis explores techniques for enhancing sentence embeddings, which are vectors that capture the meaning of sentences. 
The focus is on improving embeddings under different supervision settings. The key contributions are:
- 
- **Cross-Bi-Encoder (CBE):** Enhances Augmented SBERT for better sentence-pair task performance.
- **Noise Matching (NM):** A contrastive training objective to create deletion noise-invariant embeddings.

Experiments were conducted on established datasets and evaluated using SentEval and USEB.

## Getting Started
### Prerequisites

- Python 3.10+
- Install dependencies using:

```bash
  pip3 install -r requirements.txt
```

### Usage

1. **Download Datasets:**
   Follow the instructions in `datasets/README.md` to download and prepare the datasets.

2. **Run Experiments:**
   Use the following command to run all experiments:
    ```bash
   python3 src/experiments/run_all.py
    ```

## Results

Detailed results and evaluations can be found in the `output/` directory after running the experiments.

## Citing

If you use this work, please cite it as:

```bibtex
@mastersthesis{reiml2023sentence,
  title={Sentence Embeddings in Various Supervision Settings},
  author={Johanna Reiml},
  school={Ludwig-Maximilians-Universität München},
  year={2023},
  note={\url{https://github.com/jreiml/master-thesis-sentence-embeddings}}
}
```

## License

This work is licensed under the CC BY-SA 4.0 License.
