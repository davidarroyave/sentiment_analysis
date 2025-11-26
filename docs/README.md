# Customer Satisfaction Sentiment Analysis System 
# (😡😐🤩) **http://www.thesentimentcheck.com/** 
## public IP: **(http://159.89.145.212:8501/)**

***Author:*** Juan David Arroyave Ramirez

## **Discriminative Modeling AI** 🤖⚡
#### *"AI-powered customer feedback classification"*

## 🎯 Project Overview

This project develops a production-ready sentiment analysis system for automated customer satisfaction (CSAT) survey analysis. By fine-tuning state-of-the-art transformer models on domain-specific feedback data, the solution delivers accurate, real-time sentiment classification to enable proactive customer success interventions and data-driven business insights.

**Key Features:**
- **Ternary sentiment classification:** Negative, Neutral, Positive
- **Domain-adapted models:** Fine-tuned on customer satisfaction language patterns
- **Production-ready deployment:** REST API with near real-time processing (<5 minutes)
- **Scalable architecture:** Supports batch and real-time processing workflows

---

## 📊 Business Impact

**Primary Use Cases:**
- Automated triage and escalation of negative customer feedback
- Real-time customer success team alerts for at-risk accounts
- Trend analysis and executive dashboards for customer sentiment tracking

**Expected Outcomes:**
- 85-95% sentiment classification accuracy on customer feedback
- Sub-second inference latency per survey response
- Reduced manual review time by 60-80%
- Faster identification of critical customer issues (<1 hour)

---

## 🏗️ Architecture Overview

### Model Pipeline

**Base Model:** RoBERTa-base transformer (125M parameters)  
**Pre-trained Checkpoint:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
**Fine-tuning Approach:** QLoRA (4-bit quantization) for efficient training  
**Inference Latency:** ~100-300ms per response

### Deployment Architecture
Survey Submission → Message Queue (SQS/RabbitMQ) →
Sentiment API (FastAPI) → Model Inference (GPU) →
Database Storage → Real-time Dashboard + Alerts


**Processing Modes:**
- **Real-time triage:** <5 minute processing for immediate escalation
- **Batch analytics:** Hourly/daily aggregation for trend reporting

---

## 📁 Dataset & Data Pipeline

### Data Sources

**Primary:** McAuley-Lab/Amazon-Reviews-2023 organized by product categories. 
**Secondary:** Internal CSAT survey responses (500-2,000 labeled examples)

**Data Fields:**
- `review_id`, `product_id`, `customer_id`
- `stars` (1-5) → Mapped to sentiment labels
- `review_title`, `review_body` (concatenated as input text)
- `language`, `product_category`, `timestamp`

### Label Mapping

| Star Rating | Sentiment Label | Class ID |
|-------------|-----------------|----------|
| 1-2 stars   | Negative        | 0        |
| 3 stars     | Neutral         | 1        |
| 4-5 stars   | Positive        | 2        |

### Preprocessing Pipeline

1. **Language Filtering:** Retain only English responses (`language == 'en'`)
2. **Cleaning:** Remove HTML tags, URLs, excessive whitespace
3. **Normalization:** Lowercase conversion, emoticon tokenization
4. **Concatenation:** Merge `review_title` + `review_body`
5. **Quality Filtering:** Exclude responses <5 words
6. **Splitting:** 70% train / 15% validation / 15% test (stratified)

**Data Challenges Addressed:**
- Short, ambiguous text fragments
- Class imbalance (neutral underrepresented)
- Mixed language noise
- Survey-specific terminology

---

## 🧠 Model Architecture & Training

### Fine-Tuning Strategy

**Base Architecture:** RoBERTa transformer encoder
- **Token Embeddings:** 768-dimensional learned representations
- **Positional Embeddings:** Absolute positional encoding
- **Classification Head:** Dense layers + softmax (3 classes)


**Training Optimizations:**
- **QLoRA (4-bit quantization):** Reduces VRAM usage by 60%
- **Gradient accumulation:** Effective batch size of 64
- **Mixed precision (FP16):** 2x training speedup
- **Class weights:** Compensate for neutral class imbalance
- **Early stopping:** Patience=3 epochs on validation F1-macro

**Hardware Requirements:**
- Inference: 2-3GB VRAM, ~150ms per response

---

## 🚀 Deployment & Production
Automated with Gitlab CI/CD Pipeline


### Processing Workflows

**1. Real-Time Triage (Recommended)**
- **Trigger:** Survey submission event
- **Processing:** <5 minutes via message queue
- **Action:** Auto-escalate negative sentiment (>85% confidence) to customer success team
- **Use case:** Immediate intervention for at-risk customers

**2. Batch Analytics**
- **Schedule:** Hourly/daily aggregation
- **Processing:** 50-500 responses per batch
- **Output:** Trend reports, executive dashboards
- **Use case:** Strategic insights and performance tracking

### Monitoring & Metrics

**Model Performance (Tracked Weekly):**
- Macro F1-score (target: >90%)
- Per-class precision/recall
- Confusion matrix (monitor neutral misclassifications)
- Inference latency (target: <500ms)

**Business Metrics (Tracked Daily):**
- Survey response rate (target: 30-50%)
- Time-to-action on negative feedback (target: <1 hour)
- Sentiment trend correlation with churn/retention
- Escalation accuracy (false positive rate <10%)

---

## 📈 Roadmap & Future Enhancements

### Phase 1: MVP (Current)
- ✅ RoBERTa fine-tuning on English CSAT data
- ✅ REST API deployment
- ✅ Real-time triage workflow

### Phase 2: Enhanced Analytics (Q1 2026)
- 🔄 Aspect-based sentiment analysis (product, support, pricing)
- 🔄 Emotion detection (frustration, delight, urgency)

### Phase 3: Advanced Features (Q2 2026)
- 📋 Automated response generation for common issues
- 📋 Predictive churn modeling based on sentiment trends

---

## 🤝 Best Practices & Recommendations

### Data Collection
- **Survey Timing:** Send CSAT surveys 5-30 minutes post-interaction
- **Survey Length:** 3-5 minutes (5-10 questions max)
- **Response Rate Goal:** 30-50%

### Model Maintenance
- **Retraining Cadence:** Quarterly with new survey data
- **Drift Monitoring:** Weekly performance checks
- **Human-in-the-Loop:** Manual review for low-confidence predictions (<70%)

### Privacy & Compliance
- Remove PII during preprocessing
- Secure data storage (encryption at rest)
- GDPR/CCPA compliance for customer data handling

---

## 📚 References & Resources

**Key Papers:**
- Vaswani et al. (2017) - "Attention Is All You Need" (Transformer architecture)
- Liu et al. (2019) - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- Dettmers et al. (2023) - "QLoRA: Efficient Finetuning of Quantized LLMs"

**Hugging Face Models:**
- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

**Dataset:**
- [McAuley-Lab/Amazon-Reviews-2023 organized by product categories.](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

---

## 📨 Contact

**Maintainer:** Juan David Arroyave Ramirez  
**Email:** juan.arroyaver@outlook.com
**Last Updated:** November 25, 2025  
**License:** Apache 2.0


