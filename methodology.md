# NLP-Based Question Answering and Chatbot System: Methodology & Architecture

## 1. Project Aim & Objectives

### 1.1 Academic Project Aim
To design, implement, and rigorously evaluate an end-to-end Natural Language Processing (NLP) and Information Retrieval (IR) system capable of answering natural language questions through a hybrid architecture combining structured Knowledge Base queries (SQLite) and unstructured text reading comprehension (TF-IDF retrieval + Pretrained Hugging Face Transformer extractive QA).

### 1.2 Specific Objectives
1. **Corpus & Data Management**: Build an indexing and loading pipeline supporting Kaggle reading comprehension formats (e.g., SQuAD 1.1 / 2.0) alongside a domain-specific passage corpus.
2. **Information Retrieval (IR)**: Implement TF-IDF vectorization with n-gram representations and cosine similarity ranking to select top-$K$ candidate passages.
3. **Extractive Question Answering**: Deploy a pretrained Hugging Face Transformer model (`distilbert-base-cased-distilled-squad`) to identify the exact span $[i, j]$ containing the answer, computing start/end token probabilities.
4. **Structured Knowledge Base (KB)**: Create an SQLite relational knowledge store supporting entity-attribute lookup for instantaneous, deterministic factoid QA.
5. **Dialogue Management**: Develop an intent classification and dialogue frame engine to track conversational state, resolve anaphoric follow-up pronouns (e.g., *he*, *it*), and manage graceful fallbacks.
6. **Empirical Evaluation**: Evaluate the system using standard SQuAD metrics (Exact Match, Token-level F1) and IR metrics (Precision@K, Recall@K, MRR) on labeled test sets, avoiding hardcoded or fabricated scores.

---

## 2. System Architecture & Pipeline

```
                                      +--------------------------+
                                      |     User Query Input     |
                                      +--------------------------+
                                                   |
                                                   v
                                      +--------------------------+
                                      |     Dialogue Manager     |
                                      | (Intent & Frame Tracker) |
                                      +--------------------------+
                                                   |
                   +-------------------------------+-------------------------------+
                   |                               |                               |
                   v                               v                               v
          [Chit-Chat Intent]             [KB Factoid Intent]             [Passage QA Intent]
          (Greeting, Thanks,             (Direct Entity/Attr)            (General Question)
           Goodbye, Help)                         |                               |
                   |                              v                               v
                   |                    +--------------------+          +--------------------+
                   |                    |  SQLite Knowledge  |          |  TF-IDF Retriever  |
                   |                    |     Base Engine    |          |   (Cosine Rank)    |
                   |                    +--------------------+          +--------------------+
                   |                              |                               |
                   |                              |                     [Top-K Passages]
                   |                              |                               v
                   |                              |                     +--------------------+
                   |                              |                     | Hugging Face Model |
                   |                              |                     | (DistilBERT SQuAD) |
                   |                              |                     +--------------------+
                   |                              |                               |
                   |                              |                     [Answer Span & Conf]
                   |                              |                               |
                   +------------------------------+-------------------------------+
                                                  |
                                                  v
                                      +--------------------------+
                                      |  Confidence Threshold &  |
                                      |    Provenance Formatter  |
                                      +--------------------------+
                                                  |
                                                  v
                                      +--------------------------+
                                      | Streamlit Chatbot & UI   |
                                      +--------------------------+
```

---

## 3. Detailed Component Methodology

### 3.1 Information Retrieval (IR) Engine
Given a query $q$ and a passage corpus $D = \{d_1, d_2, \dots, d_N\}$:
1. **Vectorization**: Both the query and the documents are projected into a TF-IDF vector space:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   using sublinear term frequency scaling:
   $$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if count}(t, d) > 0$$
2. **Similarity Scoring**: Cosine similarity between query vector $\mathbf{q}$ and passage vector $\mathbf{d}_i$ is computed:
   $$\text{CosineSim}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\|_2 \|\mathbf{d}_i\|_2}$$
3. **Top-$K$ Selection**: Documents are sorted in descending order of similarity, and the top $K$ passages (default $K=3$) are passed to the reader.

### 3.2 Extractive Question Answering Model
The reading comprehension module uses **DistilBERT** fine-tuned on the Stanford Question Answering Dataset (SQuAD):
- DistilBERT operates via knowledge distillation from BERT-base, retaining 97% of language understanding performance with 40% fewer parameters and 60% faster inference.
- The input sequence is formatted as:
  $$\text{[CLS]} \circ q_1 \dots q_m \circ \text{[SEP]} \circ p_1 \dots p_n \circ \text{[SEP]}$$
- Two linear classification heads output start logits $s_i$ and end logits $e_j$.
- Span probability is computed via softmax:
  $$P(\text{start}=i) = \frac{e^{s_i}}{\sum_k e^{s_k}}, \quad P(\text{end}=j) = \frac{e^{e_j}}{\sum_k e^{e_k}}$$
- The candidate span $(i, j)$ with $i \le j$ maximizing $P(\text{start}=i) \times P(\text{end}=j)$ is extracted.

### 3.3 Confidence Thresholding & Anti-Hallucination
Extractive QA models can output erroneous spans if forced to answer unanswerable questions. To prevent fabricated answers:
- The system checks if the top span confidence satisfies:
  $$\text{Confidence}(i, j) \ge \tau \quad (\text{default } \tau = 0.15)$$
- If the score is below $\tau$, the system marks the output as low-confidence and issues a verified fallback message stating insufficient evidence.

### 3.4 Structured Knowledge Base (SQLite)
The knowledge base schema consists of:
- `entities`: `(id, name, category, description)`
- `facts`: `(id, entity_name, attribute, value, source)`
- `relations`: `(id, subject, relation, object)`

Queries such as *"What is the capital of France?"* or *"Who founded Python?"* trigger regex slot-filling that generates parameterized SQL queries:
```sql
SELECT value, source FROM facts
WHERE LOWER(entity_name) LIKE '%france%'
  AND LOWER(attribute) LIKE '%capital%';
```
This guarantees 100% precision, zero hallucination, and millisecond latency for verified facts.
