# SAMSum Conversational Summarization using Sentence-BERT and GPT-2

This project implements an end-to-end conversational summarization pipeline using the SAMSum dataset. It combines **Sentence-BERT (SBERT)** for extractive summarization and **GPT-2** for abstractive summarization to generate concise summaries of human-to-human dialogues.

## 📌 Project Highlights

- ✅ Fine-tunes a Sentence-BERT model (`all-MiniLM-L6-v2`) using the SAMSum dataset to learn high-quality semantic representations of dialogue-summary pairs.
- ✅ Performs extractive summarization by ranking the most relevant sentences using cosine similarity in embedding space.
- ✅ Fine-tunes GPT-2 on the same dataset to generate abstractive summaries.
- ✅ Evaluates generated summaries using ROUGE and cosine similarity metrics.

---

## 🧠 Model Overview

### 1. Sentence-BERT (SBERT)
- Trained to learn semantic similarity between dialogues and their summaries.
- Used for extractive summarization by selecting top-K semantically relevant sentences.

### 2. GPT-2
- Fine-tuned for generating abstractive conversational summaries.
- Utilizes tokenizer from Hugging Face Transformers.

---

## 📚 Dataset

- **SAMSum Dataset**
  - Contains ~16k English dialogues with manually written summaries.
  - Source: [`huggingface/datasets`](https://huggingface.co/datasets/samsum)

---

## 📦 Installation


pip install sentence-transformers transformers datasets rouge-score py7zr
## 🚀 Usage
1. Fine-tune Sentence-BERT
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset

# Load and preprocess data
dataset = load_dataset('samsum')
train_data = [
    InputExample(texts=[dialogue, summary])
    for dialogue, summary in zip(dataset['train']['dialogue'], dataset['train']['summary'])
]

# Load and train model
model = SentenceTransformer('all-MiniLM-L6-v2')
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_data, train_loss)], epochs=1, warmup_steps=100)

# Save model
model.save('./fine_tuned_sbert')
2. Fine-tune GPT-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize data and define Trainer...
📊 Evaluation
ROUGE-1, ROUGE-2, ROUGE-L for comparing summaries to reference.
Cosine Similarity using Sentence-BERT embeddings.
🛠️ Future Work
Integrate T5 or BART for improved abstractive summaries.
Add Gradio/Streamlit interface for demo.
Experiment with multi-lingual summarization.
🤝 Acknowledgments
Hugging Face 🤗 for transformers, datasets, and pre-trained models.
SAMSum dataset authors.
SentenceTransformers for SBERT models.
📄 License
This project is licensed under the MIT License.

---

Let me know if you'd like me to generate the `requirements.txt`, `.gitignore`, or Gradio demo interface too.
