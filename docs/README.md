# Welcome to the Sentiment Analysis 
***Author:*** Juan David Arroyave Ramirez

### Introduction

# Sentiment Analysis in Spanish Reviews  (🥹😡😍😭)
*Author: Juan David Arroyave Ramirez*
### **Generative AI**

### Overview

This project develops and evaluates a Spanish sentiment classification solution (ternary: negative / neutral / positive), training models entirely from scratch and without leveraging pre-trained parameters. The process emphasizes a foundational understanding of how recurrent neural networks and Transformers, built and initialized de novo, learn Spanish language representations from the available corpus. Special focus is given to preprocessing, modelling, and training strategies necessary for robust performance amid data noise and ambiguity.

The analysis utilizes the Amazon Reviews Multi dataset (version: mexwell/amazon-reviews-multi), filtered to retain only Spanish entries. Core fields include review_id, product_id, reviewer_id, stars (1–5), review_title, review_body, language, and product_category. For classification, star ratings are mapped to ternary sentiment: 0 (negative: 1–2), 1 (neutral: 3), 2 (positive: 4–5). All work proceeds using stratified or pre-defined splits for training, validation, and testing to preserve class proportions.

Building from scratch introduces both practical hurdles (e.g., larger data needs, overfitting risk) and pedagogical benefits—granting full transparency and control across the workflow. Our pipeline incorporates: comprehensive text cleaning and normalization, word/subword tokenization, embeddings trained from scratch, and two architectural lines—(1) bidirectional RNNs with attention (BiLSTM/BiGRU+Attention), and (2) a Transformer encoder entirely implemented from the ground up.

---

### Dataset Summary (Amazon Reviews Multi, Spanish)

**Source & Structure:**  
A multilingual Amazon reviews dataset, each entry paired with metadata. Main fields in use are review_title, review_body, and stars.

**Language Filtering:**  
Entries are restricted to those where language == 'es', with additional random sampling (+ langdetect) to ensure linguistic integrity.

**Label Engineering:**  
Star ratings are converted into three classes: 0 (1–2 stars, negative), 1 (3 stars, neutral), and 2 (4–5 stars, positive). Entries lacking star data are excluded or marked accordingly.

**Preprocessing:**  
HTML tags and URLs are removed, excessive whitespace normalized, text lowercased, and the review_title concatenated with review_body. Very short reviews (by word count) are filtered out for training, but set aside for subsequent analysis.

**Data Challenges:**  
Notable obstacles include short text fragments, multilingual noise, metadata remnants, and an ambiguous neutral class. Rigorous cleaning, label curation, and targeted error analysis are prioritized throughout.

---

### Theoretical Foundations

**Natural Language Processing (NLP):**  
The field enabling computational understanding and generation of human language, with text classification being a central task.

**Embeddings:**  
Dense, trainable vector representations (e.g., word2vec, GloVe, fastText) that encode word semantics and enable downstream learning.

**Recurrent Neural Networks (RNNs):**  
Models well-suited to sequence data. Variants like LSTM and GRU address long-range dependencies; bidirectional arcs capture contextual cues from both past and future tokens. Attention mechanisms assign dynamic weights to input elements.

**Transformers:**  
Vaswani et al. (2017) introduced self-attention architectures, discarding recurrence for scalable, long-range contextualization. Fine-tuned, pre-trained Transformers (e.g., BERT, BETO) are state-of-the-art, but here, models are trained from scratch for experimentation.

**Training Strategies:**  
Fine-tuning leverages existing knowledge from large-scale pretraining for efficiency. In contrast, training from scratch ensures no external bias but requires more data and tuning.

**Evaluation Metrics:**  
For multiclass and imbalanced data, macro F1 (averaging class F1s), classwise precision and recall, confusion matrices, and accuracy are monitored.

---

## Project Goals

### 1. Primary Objective

Train two distinct models—one RNN (BiGRU or BiLSTM with attention) and one Transformer encoder—on the Spanish dataset.

- RNN: Bidirectional GRU or LSTM with attention for ternary classification.
- Transformer: Custom implementation comprising token and positional embeddings, and stacked MultiHeadAttention blocks.

Both are evaluated by accuracy, per-class precision, recall, F1, and macro-F1, with special emphasis on optimizing the neutral class F1 through balancing, augmentation, and post-hoc error analysis.

---

### 2. Data Pipeline

- **Loading:** Utilize pre-defined or stratified splits for train/validation/test.
- **Cleaning:** Remove HTML, unescape entities, strip URLs, normalize whitespace, and standardize case. Optionally map emojis or emoticons to tokens.
- **Concatenation:** Merge title and body, with fallback handling for missing data.
- **Filtering:** Exclude reviews below a minimum word threshold or flag them for error analysis.
- **Tokenization:**
  - Option 1: Keras word-level tokenizer (simple, fast).
  - Option 2: Subword tokenization (e.g., SentencePiece) for robust handling of rare words or OOV.
- **Sequences:** Convert tokens to IDs and pad to a fixed maximum length.
- **Labels:** Encode categorical outcomes and, where needed, apply class_weight to compensate for imbalance.

---

### 3. Model Architectures

#### A. RNN with Attention

- **Embedding:** Learnable (train from scratch), dimensions typically 128–300.
- **Encoder:** Bidirectional GRU or LSTM (units=128), outputting sequences for attention.
- **Attention:** Contextual aggregation via a trainable weight vector.
- **Classifier Head:** Dropout, dense layers (ReLU), final softmax layer.
- **Training:** Sparse categorical cross-entropy, Adam optimizer, learning rate scheduling, dropout, L2 regularization, and early stopping. Apply class weights as needed.

#### B. Transformer Encoder

- **Embedding:** Token + positional embeddings (trainable).
- **Transformer Block:** MultiHeadAttention → Add & Norm → Positionwise FFN → Add & Norm; repeated N=2–4 times.
- **Pooling:** Global max/average pooling or attention-based. Optionally use a special [CLS] embedding.
- **Classifier Head:** Dense layers and softmax.
- **Training:** Adam optimizer with clipnorm, learning rate warmup and decay if possible, dropout, L2 regularization, batch size 32–64, and early stopping by validation loss or F1.

---

### 4. Generalization and Neutral Class Strategies

- **Class Weights:** Enhance learning on minority/ambiguous classes by adjusting the loss function.
- **Data Augmentation:**  
  - Random token deletion  
  - Synonym replacement (cautious with noise)  
  - Backtranslation targeting neutral class improvements
- **Oversampling:** Duplicate neutral examples when underrepresented.
- **Focal Loss:** Downweights easy/clean instances to focus on harder classes.
- **Ensembling:** Average RNN and Transformer model outputs to enhance macro-F1.
- **Threshold Tuning:** Custom class thresholds for desired recall or precision emphasis.

---

### 5. Validation and Diagnostics

- **Key Metrics:** Macro-F1 (primary), overall accuracy, classwise precision and recall.
- **Visualization:** Track and plot loss and accuracy across epochs; include F1 if logged.
- **Confusion Matrix:** Diagnose patterns of class confusion, especially among borderline categories.
- **Error Analysis:** Highlight frequent misclassifications, grouping by ambiguity, brevity, linguistic noise, or code-mixing, to guide targeted re-labelling or further augmentation.
- **Reproducibility:** Set seeds (numpy, random, tf); persist tokenizers, model weights, and dataset splits.
