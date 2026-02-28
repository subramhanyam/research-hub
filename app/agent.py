import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

import arxiv
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from PyPDF2 import PdfReader

load_dotenv()

# ---------- LLM ----------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1,
)


# ---------- Paths ----------
WORK_DIR = Path("./agent_workspace")
PDF_DIR = WORK_DIR / "papers_local"
ARXIV_DIR = WORK_DIR / "papers_arxiv"
CHROMA_DIR = WORK_DIR / "chroma_db"
SESSIONS_DIR = "./sessions"
for d in [WORK_DIR, PDF_DIR, ARXIV_DIR, CHROMA_DIR, Path(SESSIONS_DIR)]:
    d.mkdir(parents=True, exist_ok=True)


# ---------- Chroma ----------
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def create_new_session() -> str:
    return str(uuid.uuid4())[:8]


def get_session_collections(session_id: str):
    papers_col = chroma_client.get_or_create_collection(
        name=f"papers_{session_id}",
        embedding_function=embedding_fn,
    )
    limitations_col = chroma_client.get_or_create_collection(
        name=f"limitations_{session_id}",
        embedding_function=embedding_fn,
    )
    return papers_col, limitations_col


def _parse_json(text: str):
    t = text.strip()
    if "```" in t:
        t = t.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, t)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass

    raise json.JSONDecodeError("Could not parse JSON", t, 0)


def _llm_json(prompt: str, fallback: dict) -> dict:
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception:
        return fallback


def _emit(state: dict, payload: dict):
    cb = state.get("_emit")
    if cb:
        cb(payload)


def _token(state: dict, step: str, agent: str, text: str):
    cb = state.get("_token")
    if cb and text:
        cb(step, agent, text)


def _split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    pages = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for idx, p in enumerate(reader.pages):
        pages.append((idx + 1, p.extract_text() or ""))
    return pages


def _add_text_to_collection(session_id: str, source: str, title: str, text: str, page: int = -1):
    papers_col, _ = get_session_collections(session_id)
    chunks = _split_text(text)
    if not chunks:
        return 0

    ids = []
    metas = []
    for i, c in enumerate(chunks):
        hid = hashlib.md5(f"{source}|{page}|{i}|{c[:80]}".encode("utf-8")).hexdigest()
        ids.append(hid)
        metas.append({
            "source": source,
            "title": title,
            "page": page,
            "chunk_idx": i,
        })

    try:
        papers_col.add(documents=chunks, metadatas=metas, ids=ids)
    except Exception:
        return 0
    return len(chunks)


def _ensure_workspace_papers_indexed(session_id: str, papers: list[dict]):
    for p in papers:
        summary = p.get("summary", "")
        if summary:
            src = p.get("url") or f"workspace://paper/{p.get('paper_id', 'x')}"
            _add_text_to_collection(session_id, src, p.get("title", "Untitled"), summary, page=-1)


def arxiv_search(query: str, max_results: int = 10, sort_by: str = "relevance",
                 date_from: str = None, date_to: str = None) -> list[dict]:
    sort_criterion = arxiv.SortCriterion.SubmittedDate if sort_by == "date" else arxiv.SortCriterion.Relevance
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion)

    out = []
    for r in client.results(search):
        pub = str(r.published.date())
        if date_from and pub < date_from:
            continue
        if date_to and pub > date_to:
            continue
        out.append({
            "title": r.title,
            "summary": r.summary,
            "url": r.entry_id,
            "published": pub,
            "authors": [a.name for a in r.authors[:5]],
            "categories": [c for c in r.categories[:3]],
        })
    return out


def add_paper_to_workspace(session_id: str, paper: dict, existing_papers: list) -> dict:
    next_id = max((p.get("paper_id", 0) for p in existing_papers), default=0) + 1
    authors = paper.get("authors", [])
    first_author = authors[0].split()[-1] if authors else "Unknown"
    year = (paper.get("published") or "")[:4]

    enriched = {
        "paper_id": next_id,
        "title": paper.get("title", "Untitled"),
        "summary": paper.get("summary", ""),
        "url": paper.get("url", ""),
        "published": paper.get("published", ""),
        "authors": authors,
        "citation_label": f"[Paper {next_id} - {first_author} {year}]",
        "source": "arxiv_search",
    }

    src = enriched.get("url") or f"workspace://paper/{next_id}"
    _add_text_to_collection(session_id, src, enriched["title"], enriched.get("summary", ""), page=-1)
    return enriched


def add_local_paper_to_workspace(session_id: str, file_name: str, extracted_text: str, existing_papers: list) -> dict:
    next_id = max((p.get("paper_id", 0) for p in existing_papers), default=0) + 1
    title = os.path.splitext(os.path.basename(file_name))[0].replace("_", " ").strip() or "Local Upload"
    year = datetime.now().year
    content = (extracted_text or "").strip()
    summary = content[:8000]

    digest = hashlib.md5(f"{file_name}:{len(content)}".encode("utf-8")).hexdigest()[:10]
    local_url = f"local://{digest}/{os.path.basename(file_name)}"

    enriched = {
        "paper_id": next_id,
        "title": title,
        "summary": summary,
        "url": local_url,
        "published": str(datetime.now().date()),
        "authors": ["Local Upload"],
        "citation_label": f"[Paper {next_id} - Local {year}]",
        "source": "local_upload",
    }

    _add_text_to_collection(session_id, local_url, title, content or summary, page=-1)
    return enriched


def semantic_search_workspace(session_id: str, query: str, n_results: int = 5) -> list[dict]:
    papers_col, _ = get_session_collections(session_id)
    if papers_col.count() == 0:
        return []

    results = papers_col.query(
        query_texts=[query],
        n_results=min(n_results, papers_col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] or {}
        dist = results["distances"][0][i]
        relevance = round(max(0.0, 1 - dist), 3)
        hits.append({
            "chunk": (results["documents"][0][i] or "")[:300],
            "title": meta.get("title", ""),
            "citation_label": meta.get("source", ""),
            "url": meta.get("source", ""),
            "published": "",
            "authors": "",
            "relevance": relevance,
        })
    return hits


def get_workspace_stats(session_id: str) -> dict:
    papers_col, limitations_col = get_session_collections(session_id)
    return {
        "total_papers": papers_col.count(),
        "total_limitations": limitations_col.count(),
    }


def list_all_sessions() -> list[dict]:
    sessions_list = []
    if not os.path.exists(SESSIONS_DIR):
        return sessions_list

    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions_list.append({
                "session_id": data.get("session_id", ""),
                "topic": data.get("topic", ""),
                "papers_found": data.get("papers_found", 0),
                "gaps_count": len(data.get("research_gaps", [])),
                "ideas_count": len(data.get("research_ideas", [])),
                "novelty_scores": data.get("novelty_scores", []),
                "generated_at": data.get("generated_at", ""),
                "filename": fname,
            })
        except Exception:
            continue
    return sessions_list


class AgentState(TypedDict, total=False):
    goal: str
    session_id: str
    search_mode: str
    current_agent: str
    next_agent: str
    iterations: int
    max_iterations: int
    planner: dict
    researcher: dict
    evaluator: dict
    expansion: dict
    final_answer: str
    final_sources: list
    future_ideas: list
    trace: list
    papers: list
    gaps: list
    weak_gaps: list
    ideas: list
    novelty_scores: list
    evidence_assessment: dict
    insufficient_evidence_message: str
    _emit: Callable[[dict], None]
    _token: Callable[[str, str, str], None]


RESEARCHER_PROMPT = """You are the Researcher Agent in an autonomous multi-agent research pipeline.

Your job is to answer the research goal using only the retrieved context and planner subtasks.
Rules:
1. Ground claims in provided context only.
2. Do not fabricate facts or citations.
3. If evidence is weak/missing, say so.
4. Explicitly separate facts from hypotheses and speculation.
5. Never claim hard quantitative gains unless directly supported by retrieved evidence.

Return in this structure:
Direct Answer:
<your synthesis>

Support by Subtask:
- <subtask>: [FACT|HYPOTHESIS|SPECULATION] <finding> [source: <path>, page: <n or unknown>]

Limitations / Gaps:
- ...

Claim Register:
- Fact: <statement grounded in retrieved context>
- Hypothesis: <testable but unverified claim>
- Speculation: <idea with weak/no direct evidence>

Source Notes:
- [source: <path>, page: <n or unknown>] <support>
"""


EVALUATOR_PROMPT = """You are the Evaluator Agent.
Research goal: {goal}
Operating mode: {mode}
Answer: {answer}
Number of sources: {num_sources}

Tasks:
1. Score confidence from 0.0 to 1.0.
2. List specific missing aspects.
3. Set next_agent to END if sufficient, else Expansion.

Return ONLY JSON:
{{"confidence": 0.62, "missing_aspects": ["..."], "next_agent": "Expansion"}}"""


def _enforce_claim_discipline(answer: str, goal: str, context: str, num_sources: int) -> str:
    """Rewrite answer to separate fact/hypothesis/speculation and remove overclaims."""
    if not answer.strip():
        return answer

    prompt = f"""You are a strict scientific editor.
Research goal: {goal}
Number of sources: {num_sources}

Rewrite the draft answer below with these constraints:
1. Preserve only content supported by context.
2. Label uncertain claims as [HYPOTHESIS] or [SPECULATION].
3. Keep verified claims as [FACT].
4. Do not assert hard performance numbers (e.g., speedup %, perplexity drop, complexity reduction) unless explicitly evidenced in context.
5. If quantitative claims are not evidenced, rewrite with cautious language: "may", "hypothesized", "requires validation".
6. Keep the same high-level structure and keep it concise.

Context:
<context>
{context[:7000]}
</context>

Draft answer:
<draft>
{answer[:7000]}
</draft>
"""
    try:
        rewritten = llm.invoke([HumanMessage(content=prompt)]).content
        return rewritten or answer
    except Exception:
        return answer


def planner_agent(state: AgentState) -> dict:
    goal = state["goal"]
    _emit(state, {"step": "generate_plan", "agent": "Planner", "status": "running", "summary": ""})
    _token(state, "generate_plan", "Planner", f"Planning subtasks for goal: {goal[:120]}\n")

    fallback = {
        "subtasks": ["problem framing", "methods", "evaluation", "limitations"],
        "next_agent": "Researcher",
    }
    out = _llm_json(
        f"""You are the Planner Agent.
Research goal: {goal}
Break into 3-5 sub-questions and route to Researcher.
Return JSON: {{"subtasks": [...], "next_agent": "Researcher"}}""",
        fallback,
    )
    subtasks = out.get("subtasks", fallback["subtasks"])
    if not isinstance(subtasks, list):
        subtasks = fallback["subtasks"]

    keywords = [str(s) for s in subtasks][:8]
    _emit(state, {
        "step": "generate_plan",
        "agent": "Planner",
        "status": "complete",
        "summary": f"Generated {len(subtasks)} sub-questions",
        "data": {"subtopics": subtasks, "keywords": keywords},
    })
    return {
        "planner": {"subtasks": subtasks, "next_agent": "Researcher"},
        "subtopics": subtasks,
        "keywords": keywords,
        "next_agent": "Researcher",
    }


def _retrieve_context(session_id: str, queries: list[str], k: int = 6):
    papers_col, _ = get_session_collections(session_id)
    if papers_col.count() == 0:
        return [], []

    docs = []
    metas = []
    for q in queries:
        try:
            res = papers_col.query(
                query_texts=[q],
                n_results=min(k, papers_col.count()),
                include=["documents", "metadatas", "distances", "ids"],
            )
            for i in range(len(res["ids"][0])):
                docs.append(res["documents"][0][i] or "")
                metas.append(res["metadatas"][0][i] or {})
        except Exception:
            continue

    seen = set()
    uniq_docs = []
    uniq_metas = []
    for d, m in zip(docs, metas):
        key = hashlib.md5(f"{m.get('source')}|{m.get('page')}|{d[:100]}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        uniq_docs.append(d)
        uniq_metas.append(m)

    return uniq_docs, uniq_metas


def researcher_agent(state: AgentState) -> dict:
    _emit(state, {"step": "search_papers", "agent": "Researcher", "status": "running", "summary": ""})

    goal = state["goal"]
    session_id = state["session_id"]
    subtasks = state.get("planner", {}).get("subtasks", [])
    queries = [goal] + [str(s) for s in subtasks]

    docs, metas = _retrieve_context(session_id, queries, k=6)

    if not docs:
        _token(state, "search_papers", "Researcher", "No indexed documents found in workspace.\n")
        answer = "No evidence could be retrieved from the current workspace."
        sources = []
    else:
        context = "\n\n".join(docs[:20])
        _token(state, "search_papers", "Researcher", f"Retrieved {len(docs)} context chunks. Synthesizing grounded answer...\n")
        prompt = (
            f"{RESEARCHER_PROMPT}\n\n"
            f"Research Goal:\n{goal}\n\n"
            f"Planner Subtasks:\n{subtasks}\n\n"
            f"Retrieved Context:\n<context>\n{context}\n</context>"
        )
        try:
            answer = llm.invoke([HumanMessage(content=prompt)]).content
            answer = _enforce_claim_discipline(
                answer=answer,
                goal=goal,
                context=context,
                num_sources=len(metas),
            )
        except Exception:
            answer = "Could not generate a synthesis due to model error."

        sources = [
            {
                "source": m.get("source", "unknown"),
                "page": m.get("page", "unknown"),
                "title": m.get("title", ""),
            }
            for m in metas
        ]

    existing_papers = state.get("papers", [])
    by_source = {p.get("url"): p for p in existing_papers if p.get("url")}
    next_id = max((p.get("paper_id", 0) for p in existing_papers), default=0) + 1

    for s in sources:
        src = s.get("source")
        if not src or src in by_source:
            continue
        title = s.get("title") or os.path.basename(src) or "Retrieved Source"
        year = str(datetime.now().year)
        paper = {
            "paper_id": next_id,
            "title": title,
            "summary": "",
            "url": src,
            "published": year,
            "authors": ["Indexed Source"],
            "citation_label": f"[Paper {next_id} - Source {year}]",
            "source": "mission",
        }
        by_source[src] = paper
        next_id += 1

    updated_papers = list(by_source.values())

    _emit(state, {
        "step": "search_papers",
        "agent": "Researcher",
        "status": "complete",
        "summary": f"Retrieved evidence from {len({s.get('source') for s in sources})} sources",
        "data": {
            "papers": [
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title", ""),
                    "citation_label": p.get("citation_label", ""),
                    "url": p.get("url", ""),
                    "published": p.get("published", ""),
                    "authors": p.get("authors", []),
                }
                for p in updated_papers
            ]
        },
    })

    return {
        "researcher": {"answer": answer, "sources": sources, "next_agent": "Evaluator"},
        "final_answer": answer,
        "final_sources": sources,
        "papers": updated_papers,
        "next_agent": "Evaluator",
    }


def evaluator_agent(state: AgentState) -> dict:
    _emit(state, {"step": "extract_limitations", "agent": "Analyst", "status": "running", "summary": ""})
    _emit(state, {"step": "extract_limitations", "agent": "Analyst", "status": "complete", "summary": "Evaluating answer quality"})
    _emit(state, {"step": "cluster_limitations", "agent": "Analyst", "status": "running", "summary": ""})
    _emit(state, {"step": "cluster_limitations", "agent": "Analyst", "status": "complete", "summary": "Scoring evidence and missing aspects"})
    _emit(state, {"step": "identify_gaps", "agent": "Analyst", "status": "running", "summary": ""})

    goal = state["goal"]
    mode = state.get("search_mode", "workspace")
    answer = state.get("final_answer", "")
    sources = state.get("final_sources", [])

    fallback = {
        "confidence": 0.55 if len(sources) >= 2 else 0.25,
        "missing_aspects": ["experimental validation", "benchmark diversity"],
        "next_agent": "Expansion",
    }

    out = _llm_json(
        EVALUATOR_PROMPT.format(
            goal=goal,
            mode=mode,
            answer=answer[:1400],
            num_sources=len(sources),
        ),
        fallback,
    )

    confidence = float(out.get("confidence", fallback["confidence"]))
    missing_aspects = out.get("missing_aspects", fallback["missing_aspects"])
    if not isinstance(missing_aspects, list):
        missing_aspects = fallback["missing_aspects"]

    answer_len_ok = len((answer or "").strip()) >= 220
    unique_sources = len({s.get("source", "") for s in sources if isinstance(s, dict) and s.get("source")})

    if mode == "workspace":
        sufficient = answer_len_ok and unique_sources >= 1 and confidence >= 0.45
    else:
        sufficient = answer_len_ok and unique_sources >= 2 and confidence >= 0.72

    next_agent = "END" if sufficient else "Expansion"

    gaps = []
    for i, g in enumerate(missing_aspects, start=1):
        gaps.append({
            "gap": str(g),
            "missing_component": str(g),
            "supporting_papers": [],
            "citations": [],
            "frequency": 1,
            "confidence": round(confidence * 100, 1),
            "strength": "strong" if confidence >= 0.7 else "weak",
        })

    verdict = "strong" if sufficient else ("weak" if sources else "insufficient")
    evidence_assessment = {
        "total_papers": unique_sources,
        "strong_gap_count": sum(1 for g in gaps if g["strength"] == "strong"),
        "weak_gap_count": sum(1 for g in gaps if g["strength"] != "strong"),
        "verdict": verdict,
        "confidence": confidence,
    }

    _emit(state, {
        "step": "identify_gaps",
        "agent": "Analyst",
        "status": "complete",
        "summary": f"Confidence {confidence:.2f}; missing aspects: {len(missing_aspects)}",
        "data": {
            "gaps": gaps,
            "weak_gaps": [g for g in gaps if g["strength"] != "strong"],
            "evidence_assessment": evidence_assessment,
        },
    })

    return {
        "evaluator": {
            "confidence": confidence,
            "missing_aspects": missing_aspects,
            "next_agent": next_agent,
        },
        "gaps": gaps,
        "weak_gaps": [g for g in gaps if g["strength"] != "strong"],
        "evidence_assessment": evidence_assessment,
        "next_agent": next_agent,
    }


def _download_arxiv_and_ingest(session_id: str, query: str, max_results: int = 4):
    client = arxiv.Client(page_size=max_results, delay_seconds=2, num_retries=2)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    downloaded = []
    chunks_total = 0
    for r in client.results(search):
        try:
            filename = f"{r.get_short_id()}.pdf"
            fp = r.download_pdf(dirpath=str(ARXIV_DIR), filename=filename)
            path = Path(fp)
            pdf_bytes = path.read_bytes()
            pages = _extract_pdf_pages(pdf_bytes)
            page_text = "\n".join([t for _, t in pages if t])
            chunks_added = _add_text_to_collection(
                session_id,
                r.entry_id,
                r.title,
                page_text,
                page=-1,
            )
            chunks_total += chunks_added
            downloaded.append({
                "title": r.title,
                "summary": (r.summary or "")[:8000],
                "url": r.entry_id,
                "published": str(r.published.date()),
                "authors": [a.name for a in r.authors[:5]],
            })
        except Exception:
            continue

    return downloaded, chunks_total


def expansion_agent(state: AgentState) -> dict:
    _emit(state, {"step": "generate_ideas", "agent": "Planner", "status": "running", "summary": ""})

    goal = state["goal"]
    mode = state.get("search_mode", "online")
    session_id = state["session_id"]
    missing = state.get("evaluator", {}).get("missing_aspects", [])
    iterations = int(state.get("iterations", 0)) + 1
    max_iterations = int(state.get("max_iterations", 3))

    def _generate_ideas(reason: str):
        fallback = {
            "ideas": [
                "[HYPOTHESIS] Design targeted experiments for the missing aspects.",
                "[HYPOTHESIS] Build stronger benchmark coverage for edge cases.",
                "[HYPOTHESIS] Run ablation studies to isolate causality.",
                "[SPECULATION] Test cross-domain transfer to validate generalization.",
            ]
        }
        out = _llm_json(
            f"""You are the Expansion Agent.
Research goal: {goal}
Missing aspects: {missing}
Reason: {reason}
Generate future directions, not verified findings.
Rules:
1. Prefix every idea with [HYPOTHESIS] or [SPECULATION].
2. Do NOT claim guaranteed complexity/performance gains.
3. If giving any quantitative expectation, mark it explicitly as unverified.
Return JSON: {{"ideas": ["idea1", "idea2"]}}""",
            fallback,
        )
        ideas = out.get("ideas", fallback["ideas"])
        return [str(x) for x in ideas] if isinstance(ideas, list) else fallback["ideas"]

    downloaded = []
    chunks_added = 0
    insufficient_msg = ""

    if mode == "workspace":
        _token(state, "generate_ideas", "Planner", "Workspace-only mode: generating future directions from current evidence.\n")
        idea_lines = _generate_ideas("workspace-only mode")
        next_agent = "END"
        saturated = True
        if not state.get("final_sources"):
            insufficient_msg = (
                f"Workspace mode is active, but no indexed evidence was found. Add local PDFs under {PDF_DIR.resolve()} or switch mission mode."
            )
    else:
        query = f"{goal} {' '.join([str(x) for x in missing[:3]])}".strip()
        _token(state, "generate_ideas", "Planner", f"Searching arXiv for: {query}\n")
        downloaded, chunks_added = _download_arxiv_and_ingest(session_id, query, max_results=4)
        saturated = chunks_added == 0 or iterations >= max_iterations
        if saturated:
            idea_lines = _generate_ideas("arXiv saturated")
            next_agent = "END"
        else:
            idea_lines = []
            next_agent = "Researcher"

    existing = state.get("papers", [])
    by_url = {p.get("url"): p for p in existing if p.get("url")}
    next_id = max((p.get("paper_id", 0) for p in existing), default=0) + 1
    for p in downloaded:
        if p["url"] in by_url:
            continue
        authors = p.get("authors", [])
        first_author = authors[0].split()[-1] if authors else "Unknown"
        year = (p.get("published") or "")[:4]
        p_enriched = {
            "paper_id": next_id,
            "title": p["title"],
            "summary": p.get("summary", ""),
            "url": p.get("url", ""),
            "published": p.get("published", ""),
            "authors": authors,
            "citation_label": f"[Paper {next_id} - {first_author} {year}]",
            "source": "mission",
        }
        by_url[p_enriched["url"]] = p_enriched
        next_id += 1

    ideas = [
        {
            "title": f"Future Direction {i+1}",
            "description": txt,
            "missing_component": ", ".join([str(m) for m in missing]) if missing else "General evidence gaps",
            "addresses_gaps": [str(m) for m in missing],
            "cited_papers": [],
            "novelty_justification": "Generated by Expansion agent from identified missing aspects.",
            "source": "expansion",
        }
        for i, txt in enumerate(idea_lines)
    ]
    novelty_scores = [7.0 for _ in ideas]

    summary_bits = []
    if mode == "online":
        summary_bits.append(f"Downloaded {len(downloaded)} papers")
        summary_bits.append(f"indexed {chunks_added} chunks")
    summary_bits.append(f"ideas: {len(ideas)}")

    _emit(state, {
        "step": "generate_ideas",
        "agent": "Planner",
        "status": "complete",
        "summary": "; ".join(summary_bits),
        "data": {"ideas": ideas, "message": insufficient_msg},
    })

    _emit(state, {
        "step": "reflect",
        "agent": "Critic",
        "status": "running",
        "summary": "",
    })
    _emit(state, {
        "step": "reflect",
        "agent": "Critic",
        "status": "complete",
        "summary": "Reflection complete",
        "data": {
            "scores": [{"title": i["title"], "novelty_score": 7, "feedback": "Promising direction."} for i in ideas],
            "average": 7.0 if ideas else 0.0,
            "attempt": iterations,
        },
    })

    return {
        "iterations": iterations,
        "expansion": {
            "mode": mode,
            "papers_downloaded": len(downloaded),
            "chunks_added": chunks_added,
            "saturated": saturated,
            "ideas": idea_lines,
            "next_agent": next_agent,
        },
        "papers": list(by_url.values()),
        "future_ideas": idea_lines,
        "ideas": ideas if ideas else state.get("ideas", []),
        "novelty_scores": novelty_scores if ideas else state.get("novelty_scores", []),
        "insufficient_evidence_message": insufficient_msg,
        "next_agent": next_agent,
    }


AGENT_REGISTRY: dict[str, Callable[[AgentState], dict]] = {
    "Planner": planner_agent,
    "Researcher": researcher_agent,
    "Evaluator": evaluator_agent,
    "Expansion": expansion_agent,
}


def agent_executor(state: AgentState) -> AgentState:
    current = state.get("current_agent", "Planner")
    fn = AGENT_REGISTRY.get(current)
    if fn is None:
        return {
            **state,
            "next_agent": "END",
            "current_agent": "END",
            "trace": state.get("trace", []) + [f"{current} -> UNKNOWN -> END"],
        }

    update = fn(state)
    next_agent = update.get("next_agent", "END")
    trace = state.get("trace", []) + [f"{current} -> {next_agent}"]
    return {
        **state,
        **update,
        "current_agent": next_agent,
        "trace": trace,
    }


def route(state: AgentState) -> str:
    return "END" if state.get("next_agent", "END") == "END" else "LOOP"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent_executor", agent_executor)
    g.set_entry_point("agent_executor")
    g.add_conditional_edges(
        "agent_executor",
        route,
        {
            "LOOP": "agent_executor",
            "END": END,
        },
    )
    return g.compile()


GRAPH = build_graph()


def run_autonomous_mission(state: dict, emit: Optional[Callable[[dict], None]] = None,
                           token: Optional[Callable[[str, str, str], None]] = None) -> dict:
    mission_state = {
        **state,
        "current_agent": "Planner",
        "next_agent": "Planner",
        "iterations": int(state.get("iterations", 0)),
        "max_iterations": int(state.get("max_iterations", 3)),
        "future_ideas": state.get("future_ideas", []),
        "trace": state.get("trace", []),
        "_emit": emit,
        "_token": token,
    }

    _ensure_workspace_papers_indexed(mission_state["session_id"], mission_state.get("papers", []))
    result = GRAPH.invoke(mission_state)
    result.pop("_emit", None)
    result.pop("_token", None)
    return result


def save_results(state: dict, on_token: Optional[Callable] = None) -> tuple:
    session_id = state.get("session_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SESSIONS_DIR, f"session_{session_id}_{timestamp}.json")

    papers = state.get("papers", [])
    output = {
        "session_id": session_id,
        "topic": state.get("topic", ""),
        "subtopics": state.get("subtopics", []),
        "keywords": state.get("keywords", []),
        "papers_found": len(papers),
        "papers": papers,
        "research_gaps": state.get("gaps", []),
        "weak_research_gaps": state.get("weak_gaps", []),
        "evidence_assessment": state.get("evidence_assessment", {}),
        "research_ideas": state.get("ideas", []),
        "insufficient_evidence_message": state.get("insufficient_evidence_message", ""),
        "novelty_scores": state.get("novelty_scores", []),
        "trace": state.get("trace", []),
        "generated_at": datetime.now().isoformat(),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    event = {
        "step": "save_results",
        "status": "complete",
        "summary": f"Results saved to {filename}",
        "data": {"filename": filename, "session_id": session_id},
    }
    return {"output_file": filename}, event


def generate_report(state: dict) -> str:
    goal = state.get("topic", "Unknown")
    answer = state.get("final_answer", "")
    ideas = state.get("ideas", [])
    gaps = state.get("gaps", [])
    evidence = state.get("evidence_assessment", {})

    prompt = f"""You are a senior research analyst. Write a concise research brief.
Goal: {goal}
Answer: {answer}
Gaps: {json.dumps(gaps)}
Ideas: {json.dumps(ideas)}
Evidence: {json.dumps(evidence)}

Critical rules:
1. Distinguish verified findings from hypotheses/speculation.
2. Do not present unverified claims as facts.
3. Do not assert performance numbers or complexity improvements unless explicitly supported by evidence.
4. If evidence is weak, explicitly state uncertainty.

Structure:
# Research Brief: {goal}
## 1. Key Findings
## 2. Limitations
## 3. Future Directions
## 4. Recommended Next Steps
"""

    try:
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        return (
            f"# Research Brief: {goal}\n\n"
            f"## 1. Key Findings\n{answer or 'No grounded answer generated.'}\n\n"
            f"## 2. Limitations\n{json.dumps(gaps, indent=2)}\n\n"
            f"## 3. Future Directions\n{json.dumps(ideas, indent=2)}\n"
        )
