Transformers have recently emerged as a cornerstone of modern natural language processing, it offers a ro-
bust and powerful alternative to recurrent neural networks by leveraging self-attention mechanisms to model
long-range dependencies more effectively and in parallel. First introduced by Vaswani et al. in “Attention
Is All You Need” [1], the transformer architecture replaces sequential processing with an encoder–decoder
framework built around scaled dot-product and multi-head attention, enabling substantial gains in tasks
ranging from machine translation to summarization.

This project investigates two principal variants of the transformer family: an encoder-only model for sen-
timent classification and a decoder-only model for text generation. In Part 1, we implement the masked
self-attention mechanism from scratch in PyTorch and apply it to IMDb movie reviews, training a classifier
that uses the final [CLS] token embedding to predict positive or negative sentiment. In Part 2, we build
a GPT-style, decoder-only transformer—complete with causal masking and a byte-pair encoding tokenizer
and train it on question–answer pairs from the GooAQ dataset, culminating in a simple chatbot interface
for interactive text generation.

The report is organized as follows. Section 2 details the design and implementation of the encoder-only
sentiment classifier, including data preprocessing, model architecture, and evaluation results. Section 3 de-
scribes the decoder-only text generator, covering tokenizer training, model construction, sampling strategies,
and qualitative analysis of generated outputs.

