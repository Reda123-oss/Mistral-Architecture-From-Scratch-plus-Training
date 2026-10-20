# Mistral-Inspired Decoder-Only Language Model from Scratch

**A research-oriented PyTorch implementation of a Mistral-inspired decoder-only Transformer, built from scratch with Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), Sliding-Window Attention, RMSNorm, and KV-cached autoregressive decoding.**

**Training environment:** Approximately **8 hours on Kaggle using 2 × NVIDIA Tesla T4 GPUs (15 GB VRAM per GPU)**. The notebook uses PyTorch `DataParallel` when more than one GPU is available. These are experiment run details supplied by the project author; independently verified benchmark metrics are not claimed.

> **Scope:** This repository implements a **Mistral-inspired architecture trained from random initialization**. It is **not** the official pretrained Mistral 7B model, does **not** reproduce its full architecture, and does **not** contain verified 7-billion-parameter weights. The `Mistral7B` class name is retained from the notebook.

## Model weights on Hugging Face

The model artifacts associated with this project are available on Hugging Face:

**[View pretrained model / weights — reda12Malki/Mistral7B](https://huggingface.co/reda12Malki/Mistral7B/tree/main)**

These are project-hosted model artifacts; the repository is a custom Mistral-inspired implementation, **not** the official Mistral 7B checkpoint. Check the Hugging Face repository for the available files and any loading instructions before use.

## Research objective

This project studies how selected architectural choices from modern autoregressive language models can be implemented directly in PyTorch and integrated into an end-to-end causal language-modeling experiment. Rather than loading pretrained model weights or delegating attention to a high-level model library, the notebook constructs its own embedding, normalization, attention, transformer blocks, inference cache, training loop, and evaluation routines.

The central questions explored are:

- How do rotary positional embeddings encode token positions inside query and key representations?
- How does grouped-query attention share a smaller set of key/value heads across query heads?
- How does causal sliding-window masking restrict the attention receptive field?
- How can rolling key/value caches support incremental autoregressive inference?
- How can a custom Transformer be trained and assessed using cross-entropy loss, perplexity, and generated text?

## Architecture

The notebook implements the following processing pipeline:

```text
Token IDs
   │
Trainable token embeddings
   │
32 × Decoder block
   ├── RMSNorm → Sliding-window grouped-query self-attention + RoPE → Residual
   └── RMSNorm → Feed-forward network (Linear → SiLU → Linear) → Residual
   │
Final RMSNorm
   │
Linear vocabulary projection
   │
Next-token logits
```

### Configuration used in the notebook

| Component | Setting |
|---|---|
| Transformer blocks | 32 (constructor default) |
| Hidden dimension | 768 |
| Query attention heads | 12 |
| Key/value attention heads | 4 |
| Attention head dimension | 64 |
| Causal sliding-window size | 64 tokens |
| Feed-forward hidden dimension | 4 × 768 = 3,072 |
| Tokenizer | GPT-2 tokenizer (`gpt2`) |
| Tokenizer vocabulary size | 50,257 |
| Training context length | 128 tokens |
| Training batch size | 24 |
| Validation batch size | 20 |
| Optimizer | AdamW; learning rate `3e-4`, weight decay `0.1` |
| Training epochs configured | 7 |

These settings describe **the configured notebook**, not the published Mistral 7B configuration. Parameter count and benchmark performance are not independently established in this repository.

### Core implementation details

**Rotary positional embeddings (RoPE).** Complex-valued rotations are applied to pairs of features in the query and key vectors. The decoding path uses absolute position offsets to maintain positional consistency as generation advances.

**Grouped-query attention (GQA).** The model projects to 12 query heads but only four key/value heads. Key/value representations are expanded to match query-head groups during attention computation, reducing the number of distinct K/V projections relative to standard multi-head attention.

**Sliding-window causal attention.** A causal mask restricts each token to at most the configured 64 most recent positions, including itself. Attention is calculated using explicit matrix multiplication and softmax, making the mechanism inspectable.

**Rolling KV cache.** The custom `forward_chunk` method maintains a fixed-window key/value cache and allows prompt prefill followed by incremental token generation. Cache state is explicitly reset between generation calls.

**Pre-normalized decoder blocks.** Each layer applies RMSNorm before attention and feed-forward transformations, with residual connections around both operations. The implemented feed-forward module uses SiLU and two linear layers; it is **not** Mistral's gated SwiGLU MLP.

## Dataset and preprocessing

The notebook reads the **TinyStories Narrative Classification** CSV files through Kaggle-mounted input paths, samples up to 150,000 training rows and 5,000 validation rows, removes missing text entries, concatenates stories, and tokenizes them with the GPT-2 tokenizer.

A custom `TextDataset` divides the token stream into 128-token inputs and shifted next-token targets. PyTorch DataLoaders provide batching and worker-based prefetching.

The notebook expects paths under:

```text
/kaggle/input/datasets/thedevastator/tinystories-narrative-classification/
├── train.csv
└── validation.csv
```

These paths are Kaggle-specific and must be adapted for other environments.

## Compute environment and experimental setup

| Resource | Experiment setup |
|---|---|
| Platform | Kaggle Notebooks |
| Accelerator | 2 × NVIDIA Tesla T4 GPUs |
| VRAM | 15 GB per GPU (30 GB aggregate, **not** a single shared 30 GB pool) |
| Training duration | Approximately 8 hours |
| Multi-GPU mechanism | PyTorch `nn.DataParallel` when multiple GPUs are detected |
| Dataset | TinyStories (Kaggle CSV version) |
| Training type | From-scratch next-token language modeling |

This documents the reported training environment, not a claim of achieving a specific benchmark score or reproduction of the official pretrained Mistral 7B.

## Training and evaluation

Training uses next-token categorical cross-entropy, AdamW, gradient clipping (maximum norm 1.0), and CUDA mixed precision when available. The training function records epoch-wise training and validation losses, computes validation perplexity as `exp(loss)`, saves the best-validation-loss checkpoint, and generates a qualitative sample after each epoch.

Generation supports prompt prefill and incremental token sampling with the custom KV cache. Two `generate` implementations appear in the notebook: a temperature-controlled version and a later simplified version that overwrites the function name if run.

**No final validation results, perplexity scores, or text-generation quality claims are included here**, since they are not established by the source notebook code alone.

## Important experimental limitations

Before reporting numerical results, the following implementation issues should be resolved and tests repeated:

1. **Validation-loader data leakage:** the notebook constructs `validation_dataset` but currently passes the *training* `dataset` into `val_loader`. Change `DataLoader(dataset, ...)` to `DataLoader(validation_dataset, ...)` so evaluation uses held-out stories.
2. **Last-sequence alignment:** the `TextDataset` length calculation can permit a final input slice whose one-token-shifted target is shorter by one token. Adjust the indexing or dataset length so every `(x, y)` pair has matching shapes.
3. **Corpus boundaries:** stories are concatenated directly with `"".join(...)`, without explicit story separators; this can create artificial boundaries between neighboring documents.
4. **Reproducibility:** pandas sampling is not seeded, and complete experiment seeds/configuration are not recorded.
5. **Architectural fidelity:** the project is inspired by Mistral attention design; its feed-forward block, attention code, and overall model size should not be presented as a verified reproduction of Mistral 7B.
6. **Generation interface:** executing the later simplified `generate` definition removes the earlier `temperature` argument from the notebook namespace.

These are experimental caveats, not claims that benchmark results have already been validated.

## Running the experiment

1. Create a Python environment with PyTorch, Transformers, pandas, psutil, and tqdm, with a CUDA-enabled PyTorch installation if using a GPU.
2. Make the TinyStories CSV files available at the notebook paths or update the paths to your local dataset.
3. Open `mistral-7b-commented.ipynb` (or the original `mistral-7b.ipynb`) in Jupyter or Kaggle.
4. Resolve the validation-loader and dataset-alignment issues above before interpreting validation metrics.
5. Execute the notebook in order to initialize the tokenizer and model, construct the loaders, then train and generate text.

The notebook is a hands-on research prototype, not a packaged library or pretrained inference service. Larger configurations can require substantial GPU memory and compute.

## Suggested further experiments

- Benchmark cross-entropy and perplexity on a correctly separated held-out split.
- Compare different attention-window sizes and KV-head counts at matched model sizes.
- Profile throughput and peak memory of full-sequence inference versus KV-cached decoding.
- Compare the implemented SiLU FFN against a gated SwiGLU alternative.
- Evaluate sampling strategies, output diversity, and repetition on fixed prompts and checkpoints.
- Add deterministic seeds, parameter-count reporting, and explicit training configuration logging.

## References

- Jiang et al., **Mistral 7B** (2023): https://arxiv.org/abs/2310.06825
- Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (2023): https://arxiv.org/abs/2305.13245
- Su et al., **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021): https://arxiv.org/abs/2104.09864
- Zhang and Sennrich, **Root Mean Square Layer Normalization** (2019): https://arxiv.org/abs/1910.07467
- Eldan and Li, **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** (2023): https://arxiv.org/abs/2305.07759

## Project status

**Research / educational prototype.** The repository is intended to expose the internals of transformer training and inference rather than make state-of-the-art language modeling claims.
