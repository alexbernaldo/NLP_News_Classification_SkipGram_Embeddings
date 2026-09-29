# NLP_News_Classification_SkipGram_Embeddings

Complete NLP pipeline for **news article classification (120k articles, 4 classes: World,
Sports, Business, Sci/Tech)**. Word embeddings are **trained from scratch** (Skip-Gram +
negative sampling, PyTorch) and then used **frozen** inside BiLSTM / GRU / Transformer
recurrent classifiers.

## Dataset

**AG News** (`news_dataset.csv`) — 120,000 samples, balanced 30k/class. E1 trains on a
2,000-article subset; E2 uses a stratified 70/15/15 split (Keras Tokenizer fit on train
only, `MAX_LEN=150`). Generalization test on 20 real 2024–2026 news articles.

## Approach

- **E1 — Embeddings from scratch:** custom Skip-Gram + Negative Sampling (PyTorch), grid of
  **162 configs** (6 vocab sizes × 3 windows × 3 dims × 3 neg-samples), evaluated by mean
  intra-class cosine similarity, t-SNE, bias-variance map.
- **E2 — Classification:** frozen embeddings (`trainable=False`) → Hyperband (Keras Tuner)
  over BiLSTM, GRU, and Transformer+BiGRU; early stopping + LR scheduling.

## Results

| Architecture | Test accuracy | Macro F1 | Params |
|---|---|---|---|
| **GRU** | **0.8844** | **0.8844** | 640K |
| BiLSTM | 0.8674 | 0.8669 | 1.214M |
| Transformer + BiGRU | 0.5375 | 0.5333 | 429K |

Winning embedding: `vocab5000, window 2, dim 1024, neg 1` (avg intra-class cosSim 0.4985),
which also beat the vocab7500 variant by up to 3.6pp on the Transformer. Real-world news:
**18/20 (90%)** correct.

## Reproduce

Download AG News as `news_dataset.csv`, run `PW3_E1_GROUP2.ipynb` (embedding training) then
`PW3_E2_GROUP2.ipynb` (classification) in Colab with GPU.
