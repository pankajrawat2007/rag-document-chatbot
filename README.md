# RAG Document Chatbot with Automated Evaluation

A chatbot that answers questions from your own PDF and text documents using Retrieval-Augmented Generation (RAG), plus an evaluation suite that measures retrieval and answer quality instead of relying on impressions.

> **Status:** working prototype. Next: Docker, CI, deployment.

## Features

- **Document ingestion:** upload `.txt` and `.pdf` files, with incremental indexing (already-indexed files are skipped)
- **Retrieval:** local embeddings (`all-MiniLM-L6-v2`) stored in ChromaDB, with MMR search
- **History-aware chat:** follow-up questions such as "what is it used for?" are rewritten into standalone queries
- **Grounded answers:** the model answers only from retrieved context, and says so when the answer isn't in the documents
- **Source citations:** every answer shows the file, page and passage it used
- **Streaming UI:** built with Streamlit
- **Evaluation harness:** scores retrieval hit rate, answer correctness (LLM judge) and refusal behavior, with a pass/fail quality gate

## Architecture

```
Documents -> split into chunks -> embeddings -> ChromaDB
                                                   |
Question -> rewrite with chat history -> retrieve top-k chunks -> LLM answer + sources
```

## Tech stack

Python, LangChain, ChromaDB, Hugging Face sentence-transformers, Groq LLM API, Streamlit

## Quick start

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create a virtual environment (Python 3.10+; 3.12 recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
copy .env.example .env          # then edit .env  (macOS/Linux: cp .env.example .env)

# 5. Run the app
python -m streamlit run app.py
```

In the sidebar, upload documents or click **Index everything in ./data**, then ask questions.

## Evaluation

```bash
python evaluate.py                                   # baseline
python evaluate.py --k 6 --mmr --label k6_mmr        # experiment
python evaluate.py --min-accuracy 0.8 --min-retrieval 0.8   # pass/fail quality gate
```

The test set (`eval_questions.json`) covers factual, explanatory, multi-part, follow-up and out-of-scope questions.

### Results

| Run | Settings | Accuracy | Retrieval hit rate | Avg latency |
|---|---|---|---|---|
| baseline | k=4 | _fill in_ | _fill in_ | _fill in_ |
| k6 | k=6 | _fill in_ | _fill in_ | _fill in_ |
| k6_mmr | k=6, MMR | _fill in_ | _fill in_ | _fill in_ |

Replace the placeholders with numbers from `eval_results/history.csv`.

## Project structure

```
app.py          Streamlit app
evaluate.py          Evaluation script with quality gate
eval_questions.json  Test questions with reference answers
data/                Documents to index (private files are git-ignored)
eval_results/        Evaluation history
.env.example         Template for required settings
```

## Limitations and next steps

- Scanned PDFs need OCR before they can be indexed
- Small test set: results are indicative, not statistically strong
- The LLM judge is the same model family as the answerer, so grading may be lenient
- Planned: hybrid search (BM25 + vectors), reranking, Docker, GitHub Actions, deployment

## Privacy

Never upload confidential or restricted documents to a public deployment or to an external LLM API. The sample document included here is generic.
