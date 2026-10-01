"""Run eval/questions.yaml against a running API and report retrieval, faithfulness and refusals.

Usage:
    python eval/run_eval.py                      # new collection from eval/sample_docs
    python eval/run_eval.py --collection-id ID   # reuse an existing collection
    python eval/run_eval.py --no-judge           # skip the LLM-as-judge faithfulness check

The questions go through the HTTP API, exactly as the UI uses it. The judge calls
the LLM directly via app.llm, using LLM_MODEL unless --judge-model is given.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

JUDGE_PROMPT = """You are grading a retrieval-augmented answer for faithfulness.

Decide whether every factual claim in the ANSWER is supported by the CONTEXT. Ignore \
style and citation formatting. An answer that says the context lacks information is \
supported only if the context really lacks it.

Reply with a first line of exactly "VERDICT: SUPPORTED" or "VERDICT: UNSUPPORTED", then \
one sentence explaining why."""


def wait_for_job(api: str, job_id: str, timeout: float = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = requests.get(f"{api}/jobs/{job_id}", timeout=30).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(1)
    raise TimeoutError(f"ingestion job {job_id} did not finish in {timeout}s")


def build_collection(api: str, docs_dir: Path) -> str:
    resp = requests.post(f"{api}/collections", json={"name": f"eval {time.strftime('%Y-%m-%d %H:%M')}"}, timeout=30)
    resp.raise_for_status()
    cid = resp.json()["collection_id"]
    paths = sorted(p for p in docs_dir.iterdir() if p.is_file())
    files = [("files", (p.name, p.read_bytes())) for p in paths]
    resp = requests.post(f"{api}/collections/{cid}/documents", files=files, timeout=120)
    resp.raise_for_status()
    job = wait_for_job(api, resp.json()["job_id"])
    for f in job["files"]:
        print(f"  ingested {f['filename']}: {f['status']} ({f['chunks']} chunks) {f['error'] or ''}")
    if job["status"] != "done":
        raise RuntimeError("ingestion failed, see errors above")
    return cid


def judge(llm, answer: str, sources: list[dict]) -> tuple[bool, str]:
    context = "\n\n".join(f"[{s['filename']} p.{s['page']}]\n{s['text']}" for s in sources)
    reply = llm.complete(
        JUDGE_PROMPT,
        [{"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"}],
        max_tokens=2048,
        effort="low",
    ).strip()
    first, _, reason = reply.partition("\n")
    return first.strip().upper() == "VERDICT: SUPPORTED", reason.strip()


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--questions", type=Path, default=ROOT / "questions.yaml")
    parser.add_argument("--docs", type=Path, default=ROOT / "sample_docs")
    parser.add_argument("--collection-id")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", help="defaults to LLM_MODEL")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "latest.json")
    args = parser.parse_args()
    api = args.api_url.rstrip("/")

    llm = None
    if not args.no_judge:
        from app.config import get_settings
        from app.llm import create_llm

        settings = get_settings()
        if args.judge_model:
            settings = settings.model_copy(update={"llm_model": args.judge_model})
        llm = create_llm(settings)

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))["questions"]
    cid = args.collection_id
    if not cid:
        print(f"Building collection from {args.docs} ...")
        cid = build_collection(api, args.docs)
    print(f"Collection: {cid}\n")

    results = []
    for i, q in enumerate(questions, start=1):
        resp = requests.post(f"{api}/collections/{cid}/query", json={"question": q["question"]}, timeout=300)
        if not resp.ok:
            print(f"[{i}] ERROR {resp.status_code}: {resp.text[:200]}")
            results.append({**q, "error": resp.text})
            continue
        r = resp.json()
        row = {**q, "answer": r["answer"], "grounded": r["grounded"], "sources": r["sources"]}

        if q["answerable"]:
            hits = [s for s in r["sources"] if s["filename"] == q["expected_source"]]
            row["retrieval_hit"] = bool(hits)
            if "expected_page" in q:
                row["page_hit"] = any(s["page"] == q["expected_page"] for s in hits)
            if q.get("must_include"):
                row["contains_expected"] = all(m.lower() in r["answer"].lower() for m in q["must_include"])
        row["refusal_correct"] = r["grounded"] == q["answerable"]

        if llm and r["grounded"]:
            row["faithful"], row["judge_reason"] = judge(llm, r["answer"], r["sources"])

        flags = []
        for key in ("retrieval_hit", "page_hit", "contains_expected", "faithful", "refusal_correct"):
            if key in row:
                flags.append(f"{key}={'✓' if row[key] else '✗'}")
        print(f"[{i}] {q['question']}\n    -> {r['answer'][:160].replace(chr(10), ' ')}\n    {'  '.join(flags)}")
        results.append(row)

    ok = [r for r in results if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    judged = [r for r in ok if "faithful" in r]
    paged = [r for r in answerable if "page_hit" in r]
    keyword = [r for r in answerable if "contains_expected" in r]

    summary = {
        "retrieval_hit_rate": pct(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "page_hit_rate": pct(sum(r["page_hit"] for r in paged), len(paged)),
        "answered_when_answerable": pct(sum(r["grounded"] for r in answerable), len(answerable)),
        "expected_facts_in_answer": pct(sum(r["contains_expected"] for r in keyword), len(keyword)),
        "faithfulness": pct(sum(r["faithful"] for r in judged), len(judged)) if llm else "skipped",
        "refusal_correct_on_unanswerable": pct(sum(r["refusal_correct"] for r in unanswerable), len(unanswerable)),
        "errors": len(results) - len(ok),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:34} {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"collection_id": cid, "summary": summary, "results": results}, indent=2))
    print(f"\nDetailed results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
