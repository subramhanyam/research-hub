"""
Research Agent — extracted from research_agent.ipynb
Each node function returns (state_update, sse_event) tuple.
Streaming variants use callback to push partial tokens via SSE.
"""

import os
import re
import json
import uuid
import hashlib
import arxiv
import numpy as np
from datetime import datetime
from typing import Literal, Callable, Optional
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

llm_streaming = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    streaming=True,
)

MIN_GAP_CLUSTER_FREQUENCY = 2
MIN_SUPPORTING_SOURCES_FOR_IDEA = 3
MIN_GAP_CONFIDENCE = 60.0
MAX_NOVELTY_SIMILARITY = 0.85


def llm_stream(prompt: str, on_token: Optional[Callable[[str], None]] = None) -> str:
    """Invoke LLM with streaming. Calls on_token for each chunk, returns full text."""
    if on_token is None:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    full_text = ""
    for chunk in llm_streaming.stream([HumanMessage(content=prompt)]):
        token = chunk.content
        if token:
            full_text += token
            on_token(token)
    return full_text

# ---------- ChromaDB ----------
CHROMA_DB_PATH = "./chroma_db"
SESSIONS_DIR = "./sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    return session_id


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


# ---------- Helper: parse LLM JSON (robust) ----------
def _parse_json(text: str):
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the first [ ... ] or { ... } block via regex
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Attempt 3: fix common LLM issues — trailing commas, single quotes
    cleaned = text
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)   # remove trailing commas
    cleaned = cleaned.replace("'", '"')                  # single to double quotes
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 4: for arrays, try extracting from first [ to last ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Attempt 5: for objects, try extracting from first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(f"Could not parse LLM response as JSON", text, 0)


def _first_author_lastname(paper: dict) -> str:
    authors = paper.get("authors", [])
    if not authors:
        return "Unknown"
    return authors[0].split()[-1]


def _compute_gap_confidence(supporting_paper_ids: list[int], papers: list[dict]) -> tuple[float, float]:
    total_papers = max(len(papers), 1)
    support_count = len(set(supporting_paper_ids))
    if support_count == 0:
        return 0.0, 0.0

    paper_lookup = {p["paper_id"]: p for p in papers}
    supporting = [paper_lookup[pid] for pid in set(supporting_paper_ids) if pid in paper_lookup]
    if not supporting:
        return 0.0, 0.0

    unique_authors = {_first_author_lastname(p).lower() for p in supporting}
    diversity_score = len(unique_authors) / max(len(supporting), 1)
    confidence = min(100.0, ((support_count / total_papers) * diversity_score) * 100.0)
    return round(confidence, 1), round(diversity_score, 3)


def _embedding_novelty_similarity(text: str, papers: list[dict]) -> float:
    corpus = [p.get("summary", "") for p in papers if p.get("summary")]
    if not text.strip() or not corpus:
        return 0.0

    try:
        idea_embedding = np.array(embedding_fn([text])[0]).reshape(1, -1)
        paper_embeddings = np.array(embedding_fn(corpus))
        sims = cosine_similarity(idea_embedding, paper_embeddings)[0]
        return float(np.max(sims)) if sims.size else 0.0
    except Exception:
        return 0.0


# ======================================================================
#  NODE FUNCTIONS — each returns (state_update_dict, sse_event_dict)
# ======================================================================

def generate_plan(state: dict, on_token: Optional[Callable] = None) -> tuple:
    topic = state["topic"]

    prompt = f"""You are a research assistant. Given the topic \"{topic}\", generate:
1. A list of 3-5 specific subtopics to investigate
2. A list of 5-8 search keywords/phrases for finding papers on arXiv

Return ONLY valid JSON in this format:
{{
  "subtopics": ["subtopic1", "subtopic2", ...],
  "keywords": ["keyword1", "keyword2", ...]
}}"""

    response_text = llm_stream(prompt, on_token)
    data = _parse_json(response_text)

    state_update = {"subtopics": data["subtopics"], "keywords": data["keywords"]}
    event = {
        "step": "generate_plan",
        "status": "complete",
        "summary": f"Generated {len(data['subtopics'])} subtopics, {len(data['keywords'])} keywords",
        "data": {"subtopics": data["subtopics"], "keywords": data["keywords"]},
    }
    return state_update, event


def search_papers(state: dict, on_token: Optional[Callable] = None) -> tuple:
    keywords = state["keywords"]
    session_id = state["session_id"]
    mission_mode = state.get("mission_mode", "autonomous_search")
    papers_col, _ = get_session_collections(session_id)

    existing_papers = state.get("papers", [])
    all_papers = list(existing_papers)
    seen_ids = set(p.get("url") for p in existing_papers if p.get("url"))
    paper_id_counter = max((p.get("paper_id", 0) for p in existing_papers), default=0) + 1

    if mission_mode == "workspace_only":
        if on_token:
            on_token("\n[Mission mode: workspace_only. Skipping arXiv search and using workspace papers only.]\n")
        state_update = {"papers": all_papers}
        event = {
            "step": "search_papers",
            "status": "complete",
            "summary": f"Workspace-only mode: using {len(all_papers)} existing papers",
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
                    for p in all_papers
                ]
            },
        }
        return state_update, event

    client = arxiv.Client()

    for kw in keywords:
        if on_token:
            on_token(f"\n[Searching arXiv for: \"{kw}\"]\n")
        search = arxiv.Search(
            query=kw,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            if result.entry_id not in seen_ids:
                seen_ids.add(result.entry_id)
                authors_list = [a.name for a in result.authors[:3]]
                paper = {
                    "paper_id": paper_id_counter,
                    "title": result.title,
                    "summary": result.summary,
                    "url": result.entry_id,
                    "published": str(result.published.date()),
                    "authors": authors_list,
                    "citation_label": f"[Paper {paper_id_counter} - {authors_list[0].split()[-1]} {result.published.year}]",
                }
                all_papers.append(paper)
                if on_token:
                    on_token(f"  Found: {result.title[:80]}\n")
                paper_id_counter += 1

                safe_id = result.entry_id.replace("/", "_").replace(":", "_")
                papers_col.add(
                    documents=[result.summary],
                    metadatas=[{
                        "paper_id": paper["paper_id"],
                        "title": result.title,
                        "url": result.entry_id,
                        "published": str(result.published.date()),
                        "authors": ", ".join(authors_list),
                        "citation_label": paper["citation_label"],
                    }],
                    ids=[safe_id],
                )

    state_update = {"papers": all_papers}
    event = {
        "step": "search_papers",
        "status": "complete",
        "summary": f"Found {len(all_papers)} unique papers from arXiv",
        "data": {
            "papers": [
                {
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "citation_label": p["citation_label"],
                    "url": p["url"],
                    "published": p["published"],
                    "authors": p["authors"],
                }
                for p in all_papers
            ]
        },
    }
    return state_update, event


def extract_limitations(state: dict, on_token: Optional[Callable] = None) -> tuple:
    papers = state["papers"]

    all_limitations = []

    batch_size = 5
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        if on_token:
            on_token(f"\n[Analyzing batch {i//batch_size + 1}/{(len(papers)-1)//batch_size + 1}: papers {i+1}-{min(i+batch_size, len(papers))}]\n")
        papers_text = ""
        for j, p in enumerate(batch):
            papers_text += f"\n--- {p['citation_label']}: {p['title']} ---\n{p['summary']}\n"

        prompt = f"""Analyze these paper abstracts and extract specific limitations,
weaknesses, and future work suggestions mentioned or implied.

{papers_text}

Return ONLY valid JSON — an array of objects. For each limitation, include the paper_id it came from.
Example:
[
  {{"limitation": "The method fails on small datasets.", "paper_id": 1}},
  {{"limitation": "No evaluation on real-world data.", "paper_id": 2}}
]"""

        text = llm_stream(prompt, on_token).strip()

        try:
            limitations_data = _parse_json(text)
            for item in limitations_data:
                if isinstance(item, dict) and "limitation" in item:
                    raw_pid = item.get("paper_id")
                    try:
                        pid = int(raw_pid) if raw_pid is not None else None
                    except (ValueError, TypeError):
                        pid = None
                    all_limitations.append({
                        "text": item["limitation"],
                        "paper_id": pid,
                    })
                elif isinstance(item, str):
                    all_limitations.append({"text": item, "paper_id": None})
        except (json.JSONDecodeError, Exception):
            for line in text.split("\n"):
                line = line.strip().strip('",-[]')
                if len(line) > 20:
                    all_limitations.append({"text": line, "paper_id": None})

    paper_lookup = {p["paper_id"]: p["citation_label"] for p in papers}
    for lim in all_limitations:
        pid = lim.get("paper_id")
        lim["citation_label"] = paper_lookup.get(pid, "[Unknown]") if pid else "[Unknown]"

    state_update = {"limitations": all_limitations}
    event = {
        "step": "extract_limitations",
        "status": "complete",
        "summary": f"Extracted {len(all_limitations)} limitations from {len(papers)} papers",
        "data": {
            "count": len(all_limitations),
            "sample": [
                {"text": l["text"][:120], "citation_label": l["citation_label"]}
                for l in all_limitations[:8]
            ],
        },
    }
    return state_update, event


def cluster_limitations(state: dict, on_token: Optional[Callable] = None) -> tuple:
    limitations = state["limitations"]
    topic = state.get("topic", "research")
    session_id = state["session_id"]
    _, limitations_col = get_session_collections(session_id)

    lim_texts = [l["text"] if isinstance(l, dict) else l for l in limitations]

    if len(lim_texts) < 3:
        clusters = [{"theme": "General", "limitations": limitations, "count": len(limitations), "cited_paper_ids": []}]
        state_update = {"clusters": clusters}
        event = {
            "step": "cluster_limitations",
            "status": "complete",
            "summary": "Too few limitations to cluster",
            "data": {"clusters": [{"theme": "General", "count": len(limitations), "cited_paper_ids": []}]},
        }
        return state_update, event

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ids = [f"lim_{run_id}_{i}" for i in range(len(lim_texts))]

    limitations_col.add(
        documents=lim_texts,
        metadatas=[
            {
                "topic": topic,
                "paper_id": str(l.get("paper_id", "")) if isinstance(l, dict) else "",
                "citation_label": l.get("citation_label", "") if isinstance(l, dict) else "",
            }
            for l in limitations
        ],
        ids=ids,
    )

    result = limitations_col.get(ids=ids, include=["embeddings"])
    embeddings = np.array(result["embeddings"])

    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, None)

    n_clusters = min(max(2, len(lim_texts) // 3), 8)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    clusters_dict = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(int(label), []).append(limitations[idx])

    clusters = []
    if on_token:
        on_token(f"\n[Formed {len(clusters_dict)} clusters, naming themes...]\n")
    for label, lims in clusters_dict.items():
        sample = "; ".join(l["text"] if isinstance(l, dict) else l for l in lims[:3])
        theme_prompt = f"Give a short 3-5 word theme name for these limitations: {sample}. Reply with ONLY the theme name."
        theme_text = llm_stream(theme_prompt, on_token)
        theme = theme_text.strip().strip('"')
        if on_token:
            on_token(f"\n  Cluster {label + 1}: \"{theme}\" ({len(lims)} limitations)\n")

        cited_papers = list(
            set(l.get("paper_id") for l in lims if isinstance(l, dict) and l.get("paper_id"))
        )

        clusters.append({
            "theme": theme,
            "limitations": lims,
            "count": len(lims),
            "cited_paper_ids": cited_papers,
        })

    clusters.sort(key=lambda c: c["count"], reverse=True)

    state_update = {"clusters": clusters}
    event = {
        "step": "cluster_limitations",
        "status": "complete",
        "summary": f"Formed {len(clusters)} limitation clusters",
        "data": {
            "clusters": [
                {"theme": c["theme"], "count": c["count"], "cited_paper_ids": c["cited_paper_ids"]}
                for c in clusters
            ]
        },
    }
    return state_update, event


def identify_gaps(state: dict, on_token: Optional[Callable] = None) -> tuple:
    clusters = state["clusters"]
    papers = state["papers"]

    paper_lookup = {p["paper_id"]: p["citation_label"] for p in papers}
    if on_token:
        on_token("\n[Scoring limitation clusters and extracting evidence-grounded missing components...]\n")

    gaps = []
    weak_gaps = []

    for c in clusters:
        supporting_ids = sorted(set(c.get("cited_paper_ids", [])))
        frequency = len(supporting_ids)
        if frequency < MIN_GAP_CLUSTER_FREQUENCY:
            continue

        sample_limitations = []
        for lim in c.get("limitations", [])[:4]:
            if isinstance(lim, dict):
                sample_limitations.append(lim.get("text", ""))
            else:
                sample_limitations.append(str(lim))

        prompt = f"""You are an evidence-bound research analyst.
Given this cluster of recurring limitations, produce ONE missing component statement.
Do not speculate beyond the provided limitations.

Theme: {c.get('theme', 'General')}
Sample limitations:
{json.dumps(sample_limitations, indent=2)}

Return ONLY valid JSON:
{{
  "gap": "single concise missing component statement",
  "evidence_summary": "one sentence stating what is missing"
}}"""
        try:
            parsed = _parse_json(llm_stream(prompt, on_token))
            gap_text = parsed.get("gap", c.get("theme", "Unspecified missing component"))
            evidence_summary = parsed.get("evidence_summary", "")
        except Exception:
            gap_text = c.get("theme", "Unspecified missing component")
            evidence_summary = "Derived from clustered recurring limitations."

        confidence, diversity_score = _compute_gap_confidence(supporting_ids, papers)
        gap_obj = {
            "gap": gap_text,
            "missing_component": gap_text,
            "evidence_summary": evidence_summary,
            "supporting_papers": supporting_ids,
            "citations": [paper_lookup.get(pid, f"[Paper {pid}]") for pid in supporting_ids],
            "frequency": frequency,
            "diversity_score": diversity_score,
            "confidence": confidence,
            "strength": "strong" if (frequency >= MIN_SUPPORTING_SOURCES_FOR_IDEA and confidence >= MIN_GAP_CONFIDENCE) else "weak",
        }
        if gap_obj["strength"] == "strong":
            gaps.append(gap_obj)
        else:
            weak_gaps.append(gap_obj)

    total_papers = len(papers)
    evidence_verdict = "insufficient"
    if gaps:
        evidence_verdict = "strong"
    elif weak_gaps:
        evidence_verdict = "weak"

    state_update = {
        "gaps": gaps,
        "weak_gaps": weak_gaps,
        "evidence_assessment": {
            "total_papers": total_papers,
            "strong_gap_count": len(gaps),
            "weak_gap_count": len(weak_gaps),
            "verdict": evidence_verdict,
            "rules": {
                "min_cluster_frequency": MIN_GAP_CLUSTER_FREQUENCY,
                "min_supporting_sources_for_idea": MIN_SUPPORTING_SOURCES_FOR_IDEA,
                "min_confidence_percent": MIN_GAP_CONFIDENCE,
            },
        },
    }
    event = {
        "step": "identify_gaps",
        "status": "complete",
        "summary": f"Evidence verdict: {evidence_verdict}. Strong gaps: {len(gaps)}, weak gaps: {len(weak_gaps)}",
        "data": {
            "gaps": gaps,
            "weak_gaps": weak_gaps,
            "evidence_assessment": state_update["evidence_assessment"],
        }
    }
    return state_update, event


def generate_ideas(state: dict, on_token: Optional[Callable] = None) -> tuple:
    gaps = state["gaps"]
    topic = state["topic"]
    papers = state["papers"]
    evidence_assessment = state.get("evidence_assessment", {})
    if not gaps:
        message = "Insufficient evidence to propose a validated research direction."
        if on_token:
            on_token(f"\n[{message}]\n")
        state_update = {
            "ideas": [],
            "reflection_attempts": 0,
            "novelty_scores": [],
            "insufficient_evidence_message": message,
        }
        event = {
            "step": "generate_ideas",
            "status": "complete",
            "summary": message,
            "data": {
                "ideas": [],
                "message": message,
                "evidence_assessment": evidence_assessment,
            },
        }
        return state_update, event

    paper_lookup = {p["paper_id"]: p for p in papers}
    gap_payload = []
    for g in gaps:
        gap_payload.append({
            "missing_component": g.get("missing_component", g.get("gap", "")),
            "supporting_papers": g.get("supporting_papers", []),
            "confidence": g.get("confidence", 0),
            "evidence_summary": g.get("evidence_summary", ""),
        })

    prompt = f"""You are an evidence-grounded research designer.
You must ONLY use the provided gaps and citations.
For each strong missing component, propose one novelty solution that directly addresses that missing component.
Do not introduce unrelated innovations.

Topic: {topic}
Strong missing components:
{json.dumps(gap_payload, indent=2)}

Return ONLY valid JSON array:
[
  {{
    "missing_component": "...",
    "title": "...",
    "description": "2-3 sentences",
    "novelty_justification": "why this is different from existing workspace papers",
    "addresses_gaps": ["..."],
    "cited_papers": [1,2,3]
  }}
]"""
    ideas_raw = _parse_json(llm_stream(prompt, on_token))

    ideas = []
    novelty_scores = []
    for idea in ideas_raw:
        if not isinstance(idea, dict):
            continue
        desc = idea.get("description", "")
        max_similarity = _embedding_novelty_similarity(desc, papers)
        novelty_score_10 = round(max(1.0, (1.0 - max_similarity) * 10.0), 1)
        novelty_scores.append(novelty_score_10)
        ideas.append({
            "missing_component": idea.get("missing_component", ""),
            "title": idea.get("title", "Untitled idea"),
            "description": desc,
            "novelty_justification": idea.get("novelty_justification", ""),
            "addresses_gaps": idea.get("addresses_gaps", []),
            "cited_papers": [int(x) for x in idea.get("cited_papers", []) if str(x).isdigit()],
            "max_similarity_to_workspace": round(max_similarity, 3),
            "novelty_status": "low" if max_similarity > MAX_NOVELTY_SIMILARITY else "accepted",
            "novelty_score": novelty_score_10,
        })

    state_update = {
        "ideas": ideas,
        "reflection_attempts": 0,
        "novelty_scores": novelty_scores,
    }
    event = {
        "step": "generate_ideas",
        "status": "complete",
        "summary": f"Generated {len(ideas)} research ideas",
        "data": {
            "ideas": [{
                **idea,
                "citations": [
                    paper_lookup[pid]["citation_label"]
                    for pid in idea.get("cited_papers", [])
                    if pid in paper_lookup
                ],
            } for idea in ideas],
            "evidence_assessment": evidence_assessment,
        },
    }
    return state_update, event


def reflect_on_ideas(state: dict, on_token: Optional[Callable] = None) -> tuple:
    ideas = state["ideas"]
    attempts = state.get("reflection_attempts", 0)
    if not ideas:
        state_update = {"novelty_scores": [], "reflection_attempts": attempts + 1}
        event = {
            "step": "reflect",
            "status": "complete",
            "summary": "No ideas generated due to insufficient evidence.",
            "data": {"scores": [], "average": 0, "attempt": attempts + 1},
        }
        return state_update, event

    ideas_text = json.dumps(ideas, indent=2)

    prompt = f"""Rate each of these research ideas for novelty on a scale of 1-10.
Consider: Is the idea truly new? Does it go beyond incremental improvement?

{ideas_text}

Return ONLY a JSON array of objects:
[
  {{"title": "...", "novelty_score": 8, "feedback": "why this score"}}
]"""

    response_text = llm_stream(prompt, on_token)
    scores_data = _parse_json(response_text)
    scores = [s["novelty_score"] for s in scores_data]

    state_update = {
        "novelty_scores": scores,
        "reflection_attempts": attempts + 1,
    }
    event = {
        "step": "reflect",
        "status": "complete",
        "summary": f"Novelty scores: {scores} (avg: {sum(scores)/len(scores):.1f})",
        "data": {
            "scores": [
                {"title": s["title"], "novelty_score": s["novelty_score"], "feedback": s["feedback"]}
                for s in scores_data
            ],
            "average": round(sum(scores) / len(scores), 1),
            "attempt": attempts + 1,
        },
    }
    return state_update, event


def should_regenerate(state: dict) -> bool:
    scores = state.get("novelty_scores", [])
    if not state.get("ideas"):
        return False
    attempts = state.get("reflection_attempts", 0)
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score < 7 and attempts < 3


def save_results(state: dict, on_token: Optional[Callable] = None) -> tuple:
    session_id = state.get("session_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SESSIONS_DIR, f"session_{session_id}_{timestamp}.json")

    papers = state.get("papers", [])
    papers_col, limitations_col = get_session_collections(session_id)

    references = []
    for p in papers:
        references.append({
            "paper_id": p["paper_id"],
            "citation_label": p["citation_label"],
            "title": p["title"],
            "authors": p["authors"],
            "published": p["published"],
            "url": p["url"],
        })

    output = {
        "session_id": session_id,
        "topic": state["topic"],
        "subtopics": state.get("subtopics", []),
        "keywords": state.get("keywords", []),
        "papers_found": len(papers),
        "papers": papers,
        "limitations_extracted": len(state.get("limitations", [])),
        "limitations": state.get("limitations", []),
        "clusters": [
            {
                "theme": c["theme"],
                "count": c["count"],
                "cited_paper_ids": c.get("cited_paper_ids", []),
                "limitations": [
                    {
                        "text": l["text"] if isinstance(l, dict) else l,
                        "citation_label": l.get("citation_label", "") if isinstance(l, dict) else "",
                    }
                    for l in c["limitations"]
                ],
            }
            for c in state.get("clusters", [])
        ],
        "research_gaps": state.get("gaps", []),
        "weak_research_gaps": state.get("weak_gaps", []),
        "evidence_assessment": state.get("evidence_assessment", {}),
        "research_ideas": state.get("ideas", []),
        "insufficient_evidence_message": state.get("insufficient_evidence_message", ""),
        "novelty_scores": state.get("novelty_scores", []),
        "references": references,
        "generated_at": datetime.now().isoformat(),
        "chroma_db": {
            "session_id": session_id,
            "path": CHROMA_DB_PATH,
            "papers_collection": f"papers_{session_id}",
            "limitations_collection": f"limitations_{session_id}",
            "total_papers_stored": papers_col.count(),
            "total_limitations_stored": limitations_col.count(),
        },
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    state_update = {"output_file": filename}
    event = {
        "step": "save_results",
        "status": "complete",
        "summary": f"Results saved to {filename}",
        "data": {"filename": filename, "session_id": session_id},
    }
    return state_update, event


def generate_report(state: dict) -> str:
    """Generate a structured research brief from the current state."""
    topic = state.get("topic", "Unknown")
    papers = state.get("papers", [])
    gaps = state.get("gaps", [])
    ideas = state.get("ideas", [])
    scores = state.get("novelty_scores", [])
    clusters = state.get("clusters", [])
    evidence_assessment = state.get("evidence_assessment", {})

    paper_lookup = {p["paper_id"]: p for p in papers}

    prompt = f"""You are a research report writer. Generate a structured research brief based on this data.

Topic: {topic}

Papers Analyzed: {len(papers)}

Research Gaps:
{json.dumps(gaps, indent=2)}

Research Ideas:
{json.dumps(ideas, indent=2)}

Novelty Scores: {scores}
Evidence Assessment: {json.dumps(evidence_assessment, indent=2)}

Write the report in this format:
# Research Brief: {{topic}}

## 1. Background
Brief overview of the research area.

## 2. Current State of Research
Summary of what the {len(papers)} analyzed papers cover.

## 3. Identified Research Gaps
List each gap with supporting citations.

## 4. Proposed Research Ideas
For each idea: title, description, novelty score, and which gaps it addresses.

## 5. Recommended Next Steps
Actionable recommendations for researchers.

## References
List all cited papers.

Write in academic but accessible language. Use the paper citations provided."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Pipeline step definitions: (step_name, function, agent_role)
PIPELINE_STEPS = [
    ("generate_plan", generate_plan, "Planner"),
    ("search_papers", search_papers, "Researcher"),
    ("extract_limitations", extract_limitations, "Researcher"),
    ("cluster_limitations", cluster_limitations, "Analyst"),
    ("identify_gaps", identify_gaps, "Analyst"),
    ("generate_ideas", generate_ideas, "Planner"),
    ("reflect", reflect_on_ideas, "Critic"),
]


# ======================================================================
#  ARXIV STANDALONE SEARCH (for the search tab, independent of missions)
# ======================================================================

def arxiv_search(query: str, max_results: int = 10, sort_by: str = "relevance",
                 date_from: str = None, date_to: str = None) -> list[dict]:
    """Search arXiv directly — returns papers with summaries for display."""
    sort_criterion = (
        arxiv.SortCriterion.SubmittedDate if sort_by == "date"
        else arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    results = []
    for r in client.results(search):
        pub_date = str(r.published.date())

        # Date filtering
        if date_from and pub_date < date_from:
            continue
        if date_to and pub_date > date_to:
            continue

        results.append({
            "title": r.title,
            "summary": r.summary,
            "url": r.entry_id,
            "published": pub_date,
            "authors": [a.name for a in r.authors[:5]],
            "categories": [c for c in r.categories[:3]],
        })

    return results


def add_paper_to_workspace(session_id: str, paper: dict, existing_papers: list) -> dict:
    """Add a paper to a workspace's ChromaDB collection and return the augmented paper."""
    papers_col, _ = get_session_collections(session_id)

    # Assign next paper_id
    next_id = max((p.get("paper_id", 0) for p in existing_papers), default=0) + 1
    authors = paper.get("authors", [])
    first_author_last = authors[0].split()[-1] if authors else "Unknown"
    year = paper.get("published", "")[:4]

    enriched = {
        "paper_id": next_id,
        "title": paper["title"],
        "summary": paper.get("summary", ""),
        "url": paper.get("url", ""),
        "published": paper.get("published", ""),
        "authors": authors,
        "citation_label": f"[Paper {next_id} - {first_author_last} {year}]",
        "source": "arxiv_search",
    }

    safe_id = paper.get("url", str(next_id)).replace("/", "_").replace(":", "_")
    try:
        papers_col.add(
            documents=[enriched["summary"]],
            metadatas=[{
                "paper_id": next_id,
                "title": enriched["title"],
                "url": enriched["url"],
                "published": enriched["published"],
                "authors": ", ".join(authors),
                "citation_label": enriched["citation_label"],
            }],
            ids=[safe_id],
        )
    except Exception:
        pass  # duplicate id — paper already in workspace

    return enriched


def add_local_paper_to_workspace(session_id: str, file_name: str, extracted_text: str, existing_papers: list) -> dict:
    """Add a local uploaded paper (pdf/txt/md) into workspace and ChromaDB."""
    papers_col, _ = get_session_collections(session_id)

    next_id = max((p.get("paper_id", 0) for p in existing_papers), default=0) + 1
    title = os.path.splitext(os.path.basename(file_name))[0].replace("_", " ").strip() or "Local Upload"
    year = datetime.now().year
    content = (extracted_text or "").strip()
    summary = content[:8000] if content else ""

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

    safe_id = f"local_{digest}_{next_id}"
    papers_col.add(
        documents=[summary or title],
        metadatas=[{
            "paper_id": next_id,
            "title": enriched["title"],
            "url": enriched["url"],
            "published": enriched["published"],
            "authors": ", ".join(enriched["authors"]),
            "citation_label": enriched["citation_label"],
        }],
        ids=[safe_id],
    )
    return enriched


def semantic_search_workspace(session_id: str, query: str, n_results: int = 5) -> list[dict]:
    """Search within a workspace's papers using vector similarity."""
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
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        relevance = round(max(0, 1 - distance), 3)
        hits.append({
            "chunk": results["documents"][0][i][:300],
            "title": meta.get("title", ""),
            "citation_label": meta.get("citation_label", ""),
            "url": meta.get("url", ""),
            "published": meta.get("published", ""),
            "authors": meta.get("authors", ""),
            "relevance": relevance,
        })

    return hits


def get_workspace_stats(session_id: str) -> dict:
    """Get workspace overview stats."""
    papers_col, limitations_col = get_session_collections(session_id)
    return {
        "total_papers": papers_col.count(),
        "total_limitations": limitations_col.count(),
    }


def list_all_sessions() -> list[dict]:
    """List all saved session JSON files from the sessions directory."""
    sessions_list = []
    if not os.path.exists(SESSIONS_DIR):
        return sessions_list

    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(SESSIONS_DIR, fname)
            try:
                with open(fpath, "r") as f:
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
                pass

    return sessions_list
