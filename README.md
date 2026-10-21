# NLP-Based Question Answering and Chatbot System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-DistilBERT-yellow.svg)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF-orange.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-green.svg)](https://www.sqlite.org/)

An undergraduate-grade, end-to-end **Natural Language Processing (NLP) and Dialogue System** combining:
1. **Information Retrieval (IR)**: Unigram & bigram TF-IDF cosine ranking across passage corpora.
2. **Extractive Question Answering**: Pretrained Hugging Face DistilBERT fine-tuned on SQuAD for exact text span extraction with confidence scoring and anti-hallucination thresholding.
3. **Structured Knowledge-Based QA**: SQLite relational store providing deterministic, zero-hallucination factoid querying with SQL provenance.
4. **Dialogue Management**: Finite-state machine and **Dialogue Frame** tracking active discourse entities, intent classification, and anaphoric follow-up pronoun resolution (*he*, *she*, *it*).
5. **Empirical Evaluation**: Genuine, uninvented metrics calculating SQuAD Exact Match (EM), Token-Level F1, Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).

---

## 🏗️ Project Structure

```text
nlp2/
│
├── app.py                      # Main Streamlit Multi-Page Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation & setup guide
│
├── data/
│   ├── passages_corpus.csv     # Reading comprehension corpus (AI, Science, History, Geography)
│   ├── test_eval_set.json      # Labeled ground-truth QA evaluation test set
│   ├── knowledge_base.db       # Relational SQLite database with entities, facts, and relations
│   └── download_kaggle_dataset.py # Automated importer for official Kaggle SQuAD 1.1/2.0
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # Data ingestion, text cleaning, and corpus loaders
│   ├── retriever.py            # TF-IDF vectorizer, cosine ranking, and IR metrics
│   ├── qa_engine.py            # Hugging Face DistilBERT extractive reader & confidence filter
│   ├── knowledge_base.py       # SQLite connection, schema, seed facts, and slot-filling query parser
│   ├── dialogue_manager.py     # Intent classification, state tracking, and follow-up resolution
│   └── evaluator.py            # SQuAD Exact Match, Token F1, Recall@K, and MRR metrics
│
├── tests/
│   ├── test_data_loader.py     # Unit tests for loading and preprocessing
│   ├── test_retriever.py       # Unit tests for TF-IDF indexing and top-K ranking
│   ├── test_knowledge_base.py  # Unit tests for SQLite schema and fact lookups
│   ├── test_dialogue_manager.py# Unit tests for intents, frame tracking, and follow-ups
│   └── test_evaluator.py       # Unit tests for EM and Token F1 calculations
│
└── docs/
    ├── methodology.md          # Academic methodology, aim, objectives, and formulations
    ├── dialogue_flow.md        # State machine transition graph and frame design
    └── evaluation_summary.md   # Benchmark results, SQuAD standards, and viva prep
```

---

## ⚡ Quick Start & Installation (Windows / VS Code)

### Step 1: Open Terminal in Project Directory
Open VS Code in `c:\Users\User\OneDrive\Desktop\nlp2`.

### Step 2: Create and Activate Virtual Environment
```powershell
# Create Python virtual environment
py -3.9 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Run Unit Tests
Verify all system modules and metric calculations:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Step 5: Launch Streamlit Application
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📚 Dataset & Kaggle Setup

The project runs **immediately out of the box** using the included sample corpus (`data/passages_corpus.csv`) covering topics in Artificial Intelligence, Computing, Astronomy, Biology, Geography, and History.

### Adding the Full Kaggle SQuAD Dataset

1. **Direct Official SQuAD Sample (No Kaggle account needed)**:
   ```powershell
   .\.venv\Scripts\python.exe data/download_kaggle_dataset.py --source url --limit 100
   ```
   This automatically downloads official SQuAD dev data and creates `passages_squad_imported.csv`.

2. **Using Kaggle CLI**:
   - Download the Kaggle dataset: [Stanford Question Answering Dataset (SQuAD)](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset)
   - Ensure your `kaggle.json` API key is placed at `C:\Users\<User>\.kaggle\kaggle.json`.
   - Run:
     ```powershell
     kaggle datasets download -d stanfordu/stanford-question-answering-dataset
     ```

---

## 🧠 System Architecture & Routing Flow

```
                      +-----------------------------+
                      |      User Query Input       |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Intent & Frame Tracker    |
                      +-----------------------------+
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
 [Chit-Chat Intent]        [KB Factoid Intent]          [Passage QA Intent]
 (Greeting, Help, etc.)     (Capitals, Founders)         (Reading Comprehension)
        |                            |                            |
        v                            v                            v
Conversational Response       SQLite Database             TF-IDF Retriever
(FSM State Transition)      (Deterministic Match)        (Cosine Similarity)
                                     |                            |
                                     |                   Top-K Passages
                                     |                            |
                                     |                            v
                                     |                  DistilBERT QA Model
                                     |                  (Softmax Span Prob)
                                     |                            |
                                     +----------------------------+
                                                  |
                                                  v
                                     +--------------------------+
                                     |  Confidence Filter &     |
                                     |  Provenance Formatter    |
                                     +--------------------------+
                                                  |
                                                  v
                                     +--------------------------+
                                     |   Streamlit Interface    |
                                     +--------------------------+
```

---

## 📊 Evaluation & Real Benchmarking

The evaluation module implements standard SQuAD metrics without invented scores:
- **Exact Match (EM)**: $1.0$ if normalized predicted string matches ground truth exactly, else $0.0$.
- **Token-Level F1**: Harmonic mean of token precision and recall across normalized tokens.
- **Precision@1 & Recall@K**: Measures IR accuracy in surfacing the target passage.
- **Unanswerable Detection**: Verifies that the model outputs a fallback instead of hallucinating on impossible questions.

---

## 🎓 Academic Viva Voce Highlights

- **Extractive vs. Generative QA**: Extractive models (DistilBERT) enforce factual grounding by picking exact spans $[i, j]$ from source text, eliminating hallucinations.
- **Dialogue State Frame**: Tracks `current_entity` across turns to resolve follow-ups like *"When was he born?"*.
- **Anti-Hallucination Thresholding**: If span probability $P(start) \times P(end) < \tau$ (default $0.15$), the system refuses to fabricate an answer.
