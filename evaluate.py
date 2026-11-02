"""
evaluate.py - automatic evaluation for the RAG chatbot.

Run from the project folder (same place as .env and chroma_db):

    python evaluate.py                       # baseline run
    python evaluate.py --k 6                 # try retrieving 6 chunks
    python evaluate.py --k 6 --mmr           # use MMR search
    python evaluate.py --label chunk500 --db ./chroma_db_500
    python evaluate.py --limit 5             # quick test on first 5 questions

Change ONE setting at a time and compare the summary lines in
eval_results/history.csv.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

# Show output immediately, even inside IDE consoles (otherwise Python buffers it)
sys.stdout.reconfigure(line_buffering=True)
print("evaluate.py started - loading libraries (first run can take 30-90 seconds)...")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

REFUSAL_PHRASE = "don't have enough information"
RESULTS_DIR = "eval_results"


# ---------- helpers ----------
def norm(text):
    """Lowercase, unify apostrophes, collapse whitespace."""
    return " ".join(str(text).replace("\u2019", "'").lower().split())


def with_retry(fn, tries=3):
    """Call fn(); retry with backoff (helps with API rate limits)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def build_chain(llm, db_dir, k, use_mmr):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    if not db.get()["ids"]:
        sys.exit(
            f"No indexed documents found in: {os.path.abspath(db_dir)}\n"
            "Fix: run the app from this folder, click 'Index everything in ./data', "
            "stop the app, then run this again. Or pass --db with the path to your chroma_db folder."
        )

    if use_mmr:
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(20, k * 4)})
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might "
                   "reference context in the chat history, formulate a standalone "
                   "question which can be understood without the chat history. "
                   "Do NOT answer the question, just reformulate it if needed, "
                   "otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based ONLY on the provided context below. "
                   "If the answer cannot be found in the context, say "
                   "\"I don't have enough information in the provided context.\"\n\n"
                   "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def judge(llm, question, reference, candidate):
    """LLM-as-judge: returns (is_correct, reason)."""
    prompt = (
        "You are a strict grader. Compare the CANDIDATE answer with the REFERENCE answer.\n"
        "The candidate is correct if it contains the key facts of the reference and does not "
        "contradict it. Extra correct detail is fine. Wording may differ.\n"
        "Reply with ONLY a JSON object like "
        '{"correct": true, "reason": "one short sentence"}\n\n'
        f"QUESTION: {question}\n"
        f"REFERENCE: {reference}\n"
        f"CANDIDATE: {candidate}\n"
    )
    text = with_retry(lambda: llm.invoke(prompt)).content
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return False, "judge output not parseable"
    try:
        data = json.loads(match.group(0))
        return bool(data.get("correct")), str(data.get("reason", ""))[:200]
    except json.JSONDecodeError:
        return False, "judge output not parseable"


def rate(values):
    """Share of True among non-None values, or None if there are none."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval_questions.json")
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--k", type=int, default=4, help="chunks to retrieve")
    ap.add_argument("--mmr", action="store_true", help="use MMR search")
    ap.add_argument("--label", default="baseline", help="name for this run")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between questions")
    ap.add_argument("--limit", type=int, default=0, help="only run first N questions")
    ap.add_argument("--min-accuracy", type=float, default=None,
                    help="quality gate: exit with code 1 if overall accuracy is below this (0-1)")
    ap.add_argument("--min-retrieval", type=float, default=None,
                    help="quality gate: exit with code 1 if retrieval hit rate is below this (0-1)")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is missing. Check your .env file.")

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    print("Libraries loaded. Opening database and loading embedding model...")
    chain = build_chain(llm, args.db, args.k, args.mmr)
    print("Ready.")

    print(f"\nRunning {len(questions)} questions | label={args.label} | k={args.k} | mmr={args.mmr}\n")
    rows, history = [], []

    for item in questions:
        if not item.get("follow_up"):
            history = []  # fresh conversation unless this is a follow-up

        q = item["question"]
        row = {"id": item["id"], "question": q, "answer": "", "passed": False,
               "retrieval_hit": None, "keyword_ok": None, "refused": None,
               "judge_reason": "", "latency_s": 0.0, "error": ""}
        start = time.time()

        try:
            result = with_retry(lambda: chain.invoke({"input": q, "chat_history": history[-6:]}))
            answer = result["answer"]
            contexts = norm(" ".join(d.page_content for d in result["context"]))
            row["answer"] = answer
            row["refused"] = REFUSAL_PHRASE in norm(answer)

            if item.get("should_refuse"):
                row["passed"] = row["refused"]
                row["judge_reason"] = "expected a refusal"
            else:
                ctx_kw = item.get("context_keywords", [])
                row["retrieval_hit"] = all(norm(k) in contexts for k in ctx_kw) if ctx_kw else None
                ans_kw = item.get("answer_keywords", [])
                row["keyword_ok"] = all(norm(k) in norm(answer) for k in ans_kw) if ans_kw else None
                correct, reason = judge(llm, q, item["reference"], answer)
                row["passed"] = correct
                row["judge_reason"] = reason

            history += [HumanMessage(content=q), AIMessage(content=answer)]
        except Exception as e:  # keep going even if one question fails
            row["error"] = str(e)[:300]

        row["latency_s"] = round(time.time() - start, 2)
        rows.append(row)

        status = "PASS" if row["passed"] else "FAIL"
        ret = {True: "hit", False: "MISS", None: "-"}[row["retrieval_hit"]]
        print(f"[{status}] {item['id']:>4} | retrieval: {ret:<4} | {row['latency_s']:>5.1f}s | {q[:55]}")
        if row["error"]:
            print(f"        error: {row['error']}")
        time.sleep(args.sleep)

    # ---------- summary ----------
    answerable = [r for r, it in zip(rows, questions) if not it.get("should_refuse")]
    refusal_rows = [r for r, it in zip(rows, questions) if it.get("should_refuse")]

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": args.label, "k": args.k, "mmr": args.mmr, "db": args.db,
        "total": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "accuracy": rate([r["passed"] for r in rows]),
        "answer_accuracy": rate([r["passed"] for r in answerable]),
        "retrieval_hit_rate": rate([r["retrieval_hit"] for r in answerable]),
        "refusal_accuracy": rate([r["passed"] for r in refusal_rows]),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(len(rows), 1), 2),
    }

    print("\n" + "=" * 52)
    print(f"Overall accuracy     : {summary['passed']}/{summary['total']} ({pct(summary['accuracy'])})")
    print(f"Answerable questions : {pct(summary['answer_accuracy'])} correct")
    print(f"Retrieval hit rate   : {pct(summary['retrieval_hit_rate'])}")
    print(f"Refusal accuracy     : {pct(summary['refusal_accuracy'])}")
    print(f"Average latency      : {summary['avg_latency_s']}s")
    print("=" * 52)

    failed = [r for r in rows if not r["passed"]]
    if failed:
        print("\nFailed questions (look at these first):")
        for r in failed:
            print(f"- [{r['id']}] {r['question']}")
            print(f"    answer : {r['answer'][:160]!r}")
            print(f"    why    : {r['judge_reason'] or r['error']}")

    # ---------- save results ----------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"run_{stamp}_{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)

    history_path = os.path.join(RESULTS_DIR, "history.csv")
    is_new = not os.path.exists(history_path)
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(summary)

    print(f"\nDetails saved to {path}")
    print(f"Run history in   {history_path}\n")

    # ---------- quality gate (for automatic testing) ----------
    gate_problems = []
    if args.min_accuracy is not None and (summary["accuracy"] or 0) < args.min_accuracy:
        gate_problems.append(f"accuracy {pct(summary['accuracy'])} is below the minimum {pct(args.min_accuracy)}")
    if args.min_retrieval is not None and (summary["retrieval_hit_rate"] or 0) < args.min_retrieval:
        gate_problems.append(
            f"retrieval hit rate {pct(summary['retrieval_hit_rate'])} is below the minimum {pct(args.min_retrieval)}")

    if gate_problems:
        print("QUALITY GATE FAILED:")
        for p in gate_problems:
            print(f"  - {p}")
        sys.exit(1)
    if args.min_accuracy is not None or args.min_retrieval is not None:
        print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
