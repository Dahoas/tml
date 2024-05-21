# Transformer Manifold Learning

Tokenize OpenWebText and split into `train` and `val` subsets:

```bash
python data/openwebtext/prepare.py
```

Example script to train LLM (adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)):

```bash
torchrun --standalone --nproc_per_node=4 train.py \
--init_from=hybrid \
--log_path=logs/owt \
--dataset=openwebtext \
--n_embd=384
```

Example script to estimate intrinsic dimension of LLM embeddings:

```bash
python embeddings.py \
--model_path logs/owt \
--model_mode oai \
--tokenizer_type oai \
--dataset_path Dahoas/openwebtext_val \
--split validation \
--dataset_mode hf \
--context_len 1024 \
--dataset_upper 4007 \
--num_dataset_subsample 4000 \
--shuffle_embeddings_per_sample \
--max_embeddings_per_sample 32 \
--shuffle_embeddings \
--max_embeddings 1000000
```