"""
ResearchPilot — FastAPI Backend
Workspace-based autonomous research workflow engine.
"""

import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agent import (
    create_new_session,
    get_session_collections,
    generate_plan,
    search_papers,
    extract_limitations,
    cluster_limitations,
    identify_gaps,
    generate_ideas,
    reflect_on_ideas,
    should_regenerate,
    save_results,
    generate_report,
    arxiv_search,
    add_paper_to_workspace,
    semantic_search_workspace,
    get_workspace_stats,
    list_all_sessions,
    PIPELINE_STEPS,
    SESSIONS_DIR,
)
from .models import ResearchRequest, ArxivSearchRequest, AddPaperRequest, SemanticSearchRequest

app = FastAPI(title="ResearchPilot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory workspace store: session_id -> state dict
workspaces: dict[str, dict] = {}


# ===== PAGES =====

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===== WORKSPACE MANAGEMENT =====

@app.post("/api/workspace/create")
async def create_workspace(req: Request):
    body = await req.json()
    name = body.get("name", "Untitled Research")
    session_id = create_new_session()
    state = {
        "session_id": session_id,
        "name": name,
        "topic": "",
        "papers": [],
        "limitations": [],
        "clusters": [],
        "gaps": [],
        "ideas": [],
        "novelty_scores": [],
        "missions": [],
    }
    workspaces[session_id] = state
    return {"session_id": session_id, "name": name}


@app.get("/api/workspace/{session_id}")
async def get_workspace(session_id: str):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    db_stats = get_workspace_stats(session_id)
    papers = state.get("papers", [])

    return {
        "session_id": session_id,
        "name": state.get("name", ""),
        "topic": state.get("topic", ""),
        "total_papers": len(papers),
        "total_papers_embedded": db_stats["total_papers"],
        "total_limitations": db_stats["total_limitations"],
        "total_missions": len(state.get("missions", [])),
        "papers": [
            {
                "paper_id": p.get("paper_id"),
                "title": p["title"],
                "authors": p.get("authors", []),
                "published": p.get("published", ""),
                "url": p.get("url", ""),
                "citation_label": p.get("citation_label", ""),
                "source": p.get("source", "mission"),
            }
            for p in papers
        ],
        "gaps": state.get("gaps", []),
        "ideas": state.get("ideas", []),
        "novelty_scores": state.get("novelty_scores", []),
        "missions": state.get("missions", []),
    }


@app.get("/api/workspace/{session_id}/papers")
async def get_papers(session_id: str):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    return {"papers": state.get("papers", [])}


@app.delete("/api/workspace/{session_id}/papers/{paper_id}")
async def delete_paper(session_id: str, paper_id: int):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    state["papers"] = [p for p in state["papers"] if p.get("paper_id") != paper_id]
    return {"ok": True, "remaining": len(state["papers"])}


# ===== ARXIV SEARCH (standalone) =====

@app.post("/api/arxiv/search")
async def search_arxiv(req: ArxivSearchRequest):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, arxiv_search, req.query, req.max_results, req.sort_by, req.date_from, req.date_to
    )
    return {"results": results, "count": len(results)}


@app.post("/api/workspace/{session_id}/papers/add")
async def add_paper(session_id: str, req: AddPaperRequest):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    # Check duplicate by URL
    existing_urls = {p.get("url") for p in state.get("papers", [])}
    if req.paper.get("url") in existing_urls:
        return JSONResponse({"error": "Paper already in workspace"}, status_code=409)

    loop = asyncio.get_event_loop()
    enriched = await loop.run_in_executor(
        None, add_paper_to_workspace, session_id, req.paper, state.get("papers", [])
    )
    state.setdefault("papers", []).append(enriched)
    return {"paper": enriched, "total_papers": len(state["papers"])}


# ===== SEMANTIC SEARCH =====

@app.post("/api/workspace/{session_id}/search")
async def workspace_search(session_id: str, req: SemanticSearchRequest):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    loop = asyncio.get_event_loop()
    hits = await loop.run_in_executor(
        None, semantic_search_workspace, session_id, req.query, req.n_results
    )
    return {"results": hits}


# ===== MISSIONS (autonomous research runs) =====

@app.post("/api/workspace/{session_id}/mission/start")
async def start_mission(session_id: str, req: Request):
    body = await req.json()
    topic = body.get("topic", "")
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    state["topic"] = topic
    mission_id = len(state.get("missions", [])) + 1
    state.setdefault("missions", []).append({
        "mission_id": mission_id,
        "goal": topic,
        "status": "running",
        "date": __import__("datetime").datetime.now().isoformat(),
    })
    workspaces[session_id] = state
    return {"session_id": session_id, "mission_id": mission_id, "topic": topic}


@app.get("/api/workspace/{session_id}/mission/stream")
async def stream_mission(session_id: str):
    async def event_generator():
        state = workspaces.get(session_id)
        if not state:
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'summary': 'Workspace not found'})}\n\n"
            return

        loop = asyncio.get_event_loop()

        for step_name, step_fn, agent_role in PIPELINE_STEPS:
            yield f"data: {json.dumps({'step': step_name, 'agent': agent_role, 'status': 'running', 'summary': ''})}\n\n"
            await asyncio.sleep(0.05)

            try:
                state_update, event_payload = await loop.run_in_executor(None, step_fn, state)
                state.update(state_update)
                workspaces[session_id] = state

                event_payload["agent"] = agent_role
                yield f"data: {json.dumps(event_payload)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'step': step_name, 'agent': agent_role, 'status': 'error', 'summary': str(e)})}\n\n"
                # Mark mission as failed
                if state.get("missions"):
                    state["missions"][-1]["status"] = "failed"
                return

            if step_name == "reflect":
                while should_regenerate(state):
                    yield f"data: {json.dumps({'step': 'generate_ideas', 'agent': 'Planner', 'status': 'running', 'summary': 'Low novelty — regenerating...'})}\n\n"
                    await asyncio.sleep(0.05)
                    su, ep = await loop.run_in_executor(None, generate_ideas, state)
                    state.update(su)
                    ep["agent"] = "Planner"
                    yield f"data: {json.dumps(ep)}\n\n"

                    yield f"data: {json.dumps({'step': 'reflect', 'agent': 'Critic', 'status': 'running', 'summary': 'Re-evaluating...'})}\n\n"
                    await asyncio.sleep(0.05)
                    su, ep = await loop.run_in_executor(None, reflect_on_ideas, state)
                    state.update(su)
                    ep["agent"] = "Critic"
                    yield f"data: {json.dumps(ep)}\n\n"

        # Save
        yield f"data: {json.dumps({'step': 'save_results', 'agent': 'Planner', 'status': 'running', 'summary': ''})}\n\n"
        await asyncio.sleep(0.05)
        su, ep = await loop.run_in_executor(None, save_results, state)
        state.update(su)
        ep["agent"] = "Planner"
        yield f"data: {json.dumps(ep)}\n\n"

        # Update mission status
        if state.get("missions"):
            state["missions"][-1]["status"] = "completed"
            scores = state.get("novelty_scores", [])
            if scores:
                state["missions"][-1]["novelty_avg"] = round(sum(scores) / len(scores), 1)
            state["missions"][-1]["gaps_found"] = len(state.get("gaps", []))
            state["missions"][-1]["ideas_generated"] = len(state.get("ideas", []))

        yield f"data: {json.dumps({'step': 'done', 'status': 'complete', 'session_id': session_id, 'summary': 'Mission complete!'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===== REPORTS =====

@app.post("/api/workspace/{session_id}/report")
async def get_report(session_id: str):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    loop = asyncio.get_event_loop()
    report_md = await loop.run_in_executor(None, generate_report, state)
    return {"report": report_md}


@app.get("/api/workspace/{session_id}/download")
async def download_workspace(session_id: str):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    output_file = state.get("output_file")
    if output_file and os.path.exists(output_file):
        return FileResponse(output_file, media_type="application/json", filename=os.path.basename(output_file))
    return JSONResponse({"error": "No output file yet. Run a mission first."}, status_code=404)


# ===== SESSION RESUME =====

@app.post("/api/workspace/resume")
async def resume_workspace(request: Request):
    body = await request.json()
    session_data = body.get("session_data", {})
    session_id = session_data.get("session_id")
    if not session_id:
        return JSONResponse({"error": "No session_id in JSON"}, status_code=400)

    state = {
        "session_id": session_id,
        "name": session_data.get("topic", "Resumed Workspace"),
        "topic": session_data.get("topic", ""),
        "subtopics": session_data.get("subtopics", []),
        "keywords": session_data.get("keywords", []),
        "papers": session_data.get("papers", []),
        "limitations": session_data.get("limitations", []),
        "clusters": session_data.get("clusters", []),
        "gaps": session_data.get("research_gaps", []),
        "ideas": session_data.get("research_ideas", []),
        "novelty_scores": session_data.get("novelty_scores", []),
        "reflection_attempts": 0,
        "missions": [{
            "mission_id": 1,
            "goal": session_data.get("topic", ""),
            "status": "completed",
            "date": session_data.get("generated_at", ""),
            "gaps_found": len(session_data.get("research_gaps", [])),
            "ideas_generated": len(session_data.get("research_ideas", [])),
        }],
    }
    workspaces[session_id] = state
    return {
        "session_id": session_id,
        "topic": state["topic"],
        "papers_count": len(state["papers"]),
        "gaps_count": len(state["gaps"]),
        "ideas_count": len(state["ideas"]),
    }


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": list_all_sessions()}
