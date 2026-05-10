---
title: "Language Models in NLP"
subtitle: "From probabilistic n-grams to the age of Large Language Models"

summary: "A walkthrough of language models in NLP — what they are, the probabilistic foundations (n-grams), and the breakthroughs that defined the LLM era: BERT, T5, and GPT-3."

authors:
  - me

date: '2023-10-20T00:00:00Z'
lastmod: '2023-10-20T00:00:00Z'

featured: true

tags:
  - NLP
  - Language Models
  - LLMs
  - Deep Learning
  - BERT
  - GPT-3
  - T5

categories:
  - NLP

image:
  caption: 'Language Models in NLP'
  focal_point: 'Smart'
  preview_only: false

projects: []
---

## Introduction

Language models have been part of NLP since the early days of probabilistic statistical models. At their core, **language models aim to estimate the probability of different linguistic units** — symbols, tokens, or token sequences. In short, an LM estimates the probability of these events.

In the last few years we've witnessed an explosion of **Large Language Models (LLMs)** — a real breakthrough that lets machines understand chat conversations, summarize long documents in minutes, and generate text that's nearly indistinguishable from a human's. Let's deep dive in.

## What are Language Models?

A language model is a computer algorithm that can understand and recognize text by estimating the probability of different linguistic traits. The simplest example in the literature is the **probabilistic language model**, also called the **N-Gram model**, which predicts the next word given the previous ones in a sequence.

A familiar application: the autocomplete bar on your smartphone keyboard. Behind the scenes, a language model is predicting the next word given the sequence you've typed so far — what people commonly call **autocomplete typing**.

With the rise of deep learning, a new generation of models marked the transition to the **age of Large Language Models** — language models built on neural networks instead of pure probabilistic methods. These power a much wider spread of applications: text classification, machine translation, generative writing, and so on.

## Probabilistic Language Models

The first approach to statistical language models was probabilistic: build a **probability distribution over word sequences** that assigns a probability to a whole sequence of words.

$$
P(w_1, \ldots, w_m)
$$

The problem with this approach is **data sparsity** — the most likely word may simply not appear in the training data, so the prediction breaks. The fix is to make a word's probability depend on a window of previous words. Take *"I'm going to travel to Italy"* — to predict the country, you have to take the previous words into account. That's exactly what the **N-Gram** approach does: it provides a probability distribution over words conditioned on a sequence.

This is the **relative likelihood** concept, which is useful across many NLP applications — speech recognition, machine translation, information retrieval, and more.

Given tokens $y_1, y_2, y_3, \ldots, y_n$ representing words, by the **chain rule of probability**:

$$
P(y_1, y_2, \ldots, y_n) = P(y_1) \cdot P(y_2 \mid y_1) \cdot P(y_3 \mid y_1, y_2) \cdots P(y_n \mid y_1, \ldots, y_{n-1})
$$

Or more compactly:

$$
\sum_t P(y_t \mid y_{<t})
$$

We've decomposed the probability of a text into conditional probabilities of each token given its previous context.

A simple Python function to compute the probability of a single word in a sentence:

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sentence = "I saw an airplane over and Paris more than one time"

def word_proba(sentence, word):
    tokens = word_tokenize(sentence)
    word_count = sum(1 for w in tokens if w == word)
    total_words = len(tokens)
    proba = word_count / total_words
    print("This word probability is:", proba)

word_proba(sentence, "over")
# Output:
# This word probability is: 0.0909090909090909
```

The formula for the unigram probability of a word in a sequence:

$$
P(\text{word}) = \frac{\text{count}(\text{word})}{\text{total number of words}}
$$

## N-Grams

N-gram language models also rely on global statistics from a text corpus. As we've seen, the probability of a word depends on a fixed number of previous words. Different values of `n` give us different models:

- **n = 1 (unigram):** $P(y_t \mid y_1, \ldots, y_{t-1}) = P(y_t)$
- **n = 2 (bigram):** $P(y_t \mid y_1, \ldots, y_{t-1}) = P(y_t \mid y_{t-1})$
- **n = 3 (trigram):** $P(y_t \mid y_1, \ldots, y_{t-1}) = P(y_t \mid y_{t-2}, y_{t-1})$

A short Python example:

```python
import nltk
nltk.download('punkt')
from nltk import ngrams

sentence = "I saw an airplane over and Paris more than one time"

# unigram model
for gram in ngrams(sentence.split(), 1):
    print(gram)

# bigram model
for gram in ngrams(sentence.split(), 2):
    print(gram)

# trigram model
for gram in ngrams(sentence.split(), 3):
    print(gram)
```

## The Age of Large Language Models

The age of Large Language Models began at the start of the past decade with the first large-scale models that successfully tackled NLP tasks. The first to point out is **BERT** (Bidirectional Encoder Representations from Transformers) — a model that used a masked language modeling objective and a few clever tricks to achieve state-of-the-art results.

A key trade-off being explored by AI researchers is **model performance vs. model size**. Each year, a new architecture overperforms the previous SOTA, with active research on alternatives like **distillation**, **pruning**, and **quantization** to keep models efficient.

Reducing parameters while preserving performance matters because training large models from scratch is *extremely* expensive. Megatron-Turing NLG, with 530B parameters, requires dozens of GPUs and roughly $100M of infrastructure to train.

Let's look at three breakthroughs that defined the LLM era.

### BERT

**Bidirectional Encoder Representations from Transformers (BERT)** is a transformer-based pre-training technique developed by Google. The original English BERT comes in two flavors:

1. **BERT-Base:** 12 encoder layers, 12 bidirectional self-attention heads
2. **BERT-Large:** 24 encoder layers, 16 bidirectional self-attention heads

Both were pre-trained on unlabeled text from the BooksCorpus (800M words) and English Wikipedia (2,500M words).

BERT was pre-trained on two tasks:

- **Masked Language Modeling** — 15% of tokens are masked and the model is trained to predict them from context.
- **Next Sentence Prediction** — the model learns whether one sentence is a plausible continuation of another.

After pre-training (which is computationally expensive), BERT can be **fine-tuned** with far fewer resources on smaller, task-specific datasets. When published, BERT achieved SOTA on:

- **GLUE** — General Language Understanding Evaluation (9 tasks)
- **SQuAD** — Stanford Question Answering Dataset (v1.1 and v2.0)
- **SWAG** — Situations With Adversarial Generations

### T5

**T5 (Text-to-Text Transfer Transformer)** was introduced in *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"*. Its core idea: cast **every** task — translation, question answering, classification — as feeding the model text as input and training it to generate target text.

T5 was trained on slices of cloud TPU pods using a combination of model and data parallelism. Each pod is a multi-rack ML supercomputer with 1,024 TPU v3 chips connected via a high-speed 2D mesh.

The most striking idea behind T5 is the **text-to-text framing**: even tasks that would normally be modeled as classification or regression produce text as output. It unifies the API across tasks at the cost of some engineering convenience.

### GPT-3

**Generative Pre-Trained Transformer 3 (GPT-3)** is an autoregressive language model from OpenAI that uses deep learning to produce human-like text. It's the third generation of the GPT-n series, with **175 billion parameters**. Introduced in May 2020, GPT-3 marked a clear shift in how the field thought about NLP systems.

The quality of GPT-3's generated text is high enough that telling it apart from human writing is hard — which is both an opportunity and a risk. Thirty-one researchers and engineers from OpenAI presented the original paper on May 28, 2020.

Unlike previous models, GPT-3 is applied **without any gradient updates or fine-tuning** — tasks and few-shot demonstrations are specified purely via text interaction with the model. It achieves strong performance on translation, question answering, cloze tasks, and many other benchmarks.

In practice, GPT-3 powers a wide range of applications: rewriting text in a different tone, generating layout code from natural-language descriptions, and powering chatbots that better simulate human interaction.

## Conclusion

Language models have been growing fast and generating real impact in our society. Throughout this post we've walked through the strides NLP has made via language models — from probabilistic n-grams to massive neural architectures, with research actively trying to **shrink models** while **preserving performance**. Breakthroughs like GPT-3 are pushing toward **generalist models** capable of solving many tasks under a single interface.

What's exciting is that the NLP community has been remarkably collaborative — startups and labs like **HuggingFace**, **AssemblyAI**, and **EleutherAI** are pushing open-source research forward. The evolution of NLP is happening across cultures and perspectives. Stay tuned for what's coming next.

## References

1. [Language Modeling course](https://lena-voita.github.io/nlp_course/language_modeling.html) — Lena Voita
2. [Stanford n-gram studies](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
3. [Large Language Models](https://huggingface.co/blog/large-language-models) — HuggingFace
4. [Language Model Survey](https://syncedreview.com/) — Synced
5. [How Language Models Will Transform Science](https://hai.stanford.edu/news/how-foundation-models-will-transform-science) — Stanford
