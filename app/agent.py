"""
Research Agent — extracted from research_agent.ipynb
Each node function returns (state_update, sse_event) tuple.
"""

import os
import re
import json
import uuid
import arxiv
import numpy as np
from datetime import datetime
from typing import Literal
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


# ======================================================================
#  NODE FUNCTIONS — each returns (state_update_dict, sse_event_dict)
# ======================================================================

def generate_plan(state: dict) -> tuple:
    topic = state["topic"]

    prompt = f"""You are a research assistant. Given the topic \"{topic}\", generate:
1. A list of 3-5 specific subtopics to investigate
2. A list of 5-8 search keywords/phrases for finding papers on arXiv

Return ONLY valid JSON in this format:
{{
  "subtopics": ["subtopic1", "subtopic2", ...],
  "keywords": ["keyword1", "keyword2", ...]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    data = _parse_json(response.content)

    state_update = {"subtopics": data["subtopics"], "keywords": data["keywords"]}
    event = {
        "step": "generate_plan",
        "status": "complete",
        "summary": f"Generated {len(data['subtopics'])} subtopics, {len(data['keywords'])} keywords",
        "data": {"subtopics": data["subtopics"], "keywords": data["keywords"]},
    }
    return state_update, event


def search_papers(state: dict) -> tuple:
    keywords = state["keywords"]
    session_id = state["session_id"]
    papers_col, _ = get_session_collections(session_id)

    all_papers = []
    seen_ids = set()
    paper_id_counter = 1

    client = arxiv.Client()

    for kw in keywords:
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


def extract_limitations(state: dict) -> tuple:
    papers = state["papers"]

    all_limitations = []

    batch_size = 5
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
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

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

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


def cluster_limitations(state: dict) -> tuple:
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
    for label, lims in clusters_dict.items():
        sample = "; ".join(l["text"] if isinstance(l, dict) else l for l in lims[:3])
        resp = llm.invoke(
            [HumanMessage(content=f"Give a short 3-5 word theme name for these limitations: {sample}. Reply with ONLY the theme name.")]
        )
        theme = resp.content.strip().strip('"')

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


def identify_gaps(state: dict) -> tuple:
    clusters = state["clusters"]
    papers = state["papers"]

    paper_lookup = {p["paper_id"]: p["citation_label"] for p in papers}

    cluster_summary = ""
    for c in clusters:
        cited = ", ".join(paper_lookup.get(pid, f"[Paper {pid}]") for pid in c.get("cited_paper_ids", []))
        cluster_summary += f"\nCluster '{c['theme']}' ({c.get('count', len(c['limitations']))} papers, cited by: {cited}):\n"
        for lim in c["limitations"][:3]:
            lim_text = lim["text"] if isinstance(lim, dict) else lim
            lim_cite = lim.get("citation_label", "") if isinstance(lim, dict) else ""
            cluster_summary += f"  - {lim_cite} {lim_text}\n"

    prompt = f"""Based on these clustered research limitations (with paper citations), identify the top 3-5 most
significant and recurring research gaps. For each gap, list which paper numbers support it.

{cluster_summary}

Return ONLY a JSON array:
[
  {{"gap": "description of the gap", "supporting_papers": [1, 3, 5]}}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    gaps_data = _parse_json(response.content)

    gaps = []
    for g in gaps_data:
        if isinstance(g, str):
            gaps.append({"gap": g, "supporting_papers": []})
        else:
            gaps.append({
                "gap": g.get("gap", str(g)),
                "supporting_papers": [int(x) for x in g.get("supporting_papers", []) if str(x).isdigit()],
            })

    state_update = {"gaps": gaps}
    event = {
        "step": "identify_gaps",
        "status": "complete",
        "summary": f"Identified {len(gaps)} key research gaps",
        "data": {
            "gaps": [
                {
                    "gap": g["gap"],
                    "supporting_papers": g["supporting_papers"],
                    "citations": [paper_lookup.get(pid, f"[Paper {pid}]") for pid in g["supporting_papers"]],
                }
                for g in gaps
            ]
        },
    }
    return state_update, event


def generate_ideas(state: dict) -> tuple:
    gaps = state["gaps"]
    topic = state["topic"]
    papers = state["papers"]

    paper_lookup = {p["paper_id"]: p for p in papers}

    gaps_text = ""
    all_cited_ids = set()
    for g in gaps:
        gap_str = g["gap"] if isinstance(g, dict) else g
        supporting = g.get("supporting_papers", []) if isinstance(g, dict) else []
        cited = ", ".join(f"[Paper {pid}]" for pid in supporting)
        gaps_text += f"- {gap_str} (supported by: {cited})\n"
        all_cited_ids.update(supporting)

    ref_context = "\n--- Reference Papers ---\n"
    for pid in sorted(all_cited_ids, key=lambda x: int(x) if str(x).isdigit() else 0):
        p = paper_lookup.get(pid)
        if p:
            ref_context += f"{p['citation_label']}: \"{p['title']}\" by {', '.join(p['authors'])}\n"

    prompt = f"""You are a creative research scientist. Based on these recurring research
gaps in the field of \"{topic}\":

{gaps_text}

{ref_context}

Propose 3 novel research ideas. For each idea:
- Provide a clear title
- A 2-3 sentence description of the approach
- Which gap(s) it addresses
- Cite the relevant paper numbers that inform this idea

Return ONLY a JSON array:
[
  {{
    "title": "...",
    "description": "...",
    "addresses_gaps": ["gap1", ...],
    "cited_papers": [1, 3]
  }}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    ideas = _parse_json(response.content)

    state_update = {"ideas": ideas, "reflection_attempts": 0}
    event = {
        "step": "generate_ideas",
        "status": "complete",
        "summary": f"Generated {len(ideas)} research ideas",
        "data": {
            "ideas": [
                {
                    "title": idea["title"],
                    "description": idea["description"],
                    "addresses_gaps": idea.get("addresses_gaps", []),
                    "cited_papers": idea.get("cited_papers", []),
                    "citations": [
                        paper_lookup[pid]["citation_label"]
                        for pid in idea.get("cited_papers", [])
                        if pid in paper_lookup
                    ],
                }
                for idea in ideas
            ]
        },
    }
    return state_update, event


def reflect_on_ideas(state: dict) -> tuple:
    ideas = state["ideas"]
    attempts = state.get("reflection_attempts", 0)

    ideas_text = json.dumps(ideas, indent=2)

    prompt = f"""Rate each of these research ideas for novelty on a scale of 1-10.
Consider: Is the idea truly new? Does it go beyond incremental improvement?

{ideas_text}

Return ONLY a JSON array of objects:
[
  {{"title": "...", "novelty_score": 8, "feedback": "why this score"}}
]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    scores_data = _parse_json(response.content)
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
    attempts = state.get("reflection_attempts", 0)
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score < 7 and attempts < 3


def save_results(state: dict) -> tuple:
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
        "research_ideas": state.get("ideas", []),
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

    paper_lookup = {p["paper_id"]: p for p in papers}

    prompt = f"""You are a research report writer. Generate a structured research brief based on this data.

Topic: {topic}

Papers Analyzed: {len(papers)}

Research Gaps:
{json.dumps(gaps, indent=2)}

Research Ideas:
{json.dumps(ideas, indent=2)}

Novelty Scores: {scores}

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
