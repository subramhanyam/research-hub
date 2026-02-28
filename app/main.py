"""
ResearchPilot — FastAPI Backend
Workspace-based autonomous research workflow engine.
"""

import asyncio
import json
import os
import io
import urllib.request
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader

from .agent import (
    create_new_session,
    run_autonomous_mission,
    save_results,
    generate_report,
    arxiv_search,
    add_paper_to_workspace,
    add_local_paper_to_workspace,
    semantic_search_workspace,
    get_workspace_stats,
    list_all_sessions,
)
from .models import ArxivSearchRequest, AddPaperRequest, SemanticSearchRequest

app = FastAPI(title="ResearchPilot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory workspace store: session_id -> state dict
workspaces: dict[str, dict] = {}


def _local_papers_dir() -> str:
    return os.path.join(os.path.dirname(BASE_DIR), "agent_workspace", "papers_local")


def _arxiv_papers_dir() -> str:
    return os.path.join(os.path.dirname(BASE_DIR), "agent_workspace", "papers_arxiv")


def _safe_filename(name: str, default: str = "file.pdf") -> str:
    cleaned = "".join(ch for ch in (name or "").strip() if ch not in '<>:"/\\|?*').strip()
    return cleaned or default


def _resolve_pdf_url(arxiv_url: str) -> str:
    url = (arxiv_url or "").strip()
    if not url:
        return ""
    if "/abs/" in url:
        base = url.replace("/abs/", "/pdf/")
        return base if base.endswith(".pdf") else f"{base}.pdf"
    if url.endswith(".pdf"):
        return url
    tail = os.path.basename(url.rstrip("/"))
    return f"https://arxiv.org/pdf/{tail}.pdf" if tail else ""


def _download_arxiv_pdf_to_disk(arxiv_url: str) -> tuple[str, str]:
    pdf_url = _resolve_pdf_url(arxiv_url)
    if not pdf_url:
        raise ValueError("Invalid arXiv URL")

    arxiv_dir = _arxiv_papers_dir()
    os.makedirs(arxiv_dir, exist_ok=True)
    tail = os.path.basename(pdf_url.split("?")[0]) or "paper.pdf"
    filename = _safe_filename(tail, default="paper.pdf")
    save_path = os.path.join(arxiv_dir, filename)

    if not os.path.exists(save_path):
        with urllib.request.urlopen(pdf_url, timeout=30) as resp:
            content = resp.read()
        if not content:
            raise ValueError("Downloaded arXiv PDF is empty")
        with open(save_path, "wb") as f:
            f.write(content)

    return filename, save_path


# ===== PAGES =====

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/test", status_code=307)


@app.get("/main", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test.html", {"request": request, "active_page": "dashboard"})

@app.get("/test/papers", response_class=HTMLResponse)
async def test_papers_page(request: Request):
    return templates.TemplateResponse("test_papers.html", {"request": request, "active_page": "papers"})

@app.get("/test/search", response_class=HTMLResponse)
async def test_search_page(request: Request):
    return templates.TemplateResponse("test_search.html", {"request": request, "active_page": "search"})

@app.get("/test/research-ai", response_class=HTMLResponse)
async def test_research_ai_page(request: Request):
    return templates.TemplateResponse("test_research_ai.html", {"request": request, "active_page": "research_ai"})


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
        "weak_gaps": [],
        "evidence_assessment": {},
        "insufficient_evidence_message": "",
        "insufficient_data": False,
        "insufficiency_reason": "",
        "mission_mode": "autonomous_search",
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
        "mission_mode": state.get("mission_mode", "autonomous_search"),
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
        "weak_gaps": state.get("weak_gaps", []),
        "evidence_assessment": state.get("evidence_assessment", {}),
        "insufficient_evidence_message": state.get("insufficient_evidence_message", ""),
        "insufficient_data": bool(state.get("insufficient_data", False)),
        "insufficiency_reason": state.get("insufficiency_reason", ""),
        "ideas": state.get("ideas", []),
        "novelty_scores": state.get("novelty_scores", []),
        "missions": state.get("missions", []),
        "final_answer": state.get("final_answer", ""),
        "final_sources": state.get("final_sources", []),
        "trace": state.get("trace", []),
        "iterations": state.get("iterations", 0),
        "expansion": state.get("expansion", {}),
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

    papers = state.get("papers", [])
    target = next((p for p in papers if p.get("paper_id") == paper_id), None)
    if not target:
        return JSONResponse({"error": "Paper not found"}, status_code=404)

    # Remove persisted local/arXiv file when deleting uploaded records.
    if target.get("source") in {"local_upload", "arxiv_search"}:
        source = target.get("source")
        base_dir = _local_papers_dir() if source == "local_upload" else _arxiv_papers_dir()
        candidates: list[str] = []

        direct_path = target.get("local_file_path" if source == "local_upload" else "arxiv_file_path")
        if isinstance(direct_path, str) and direct_path.strip():
            candidates.append(direct_path)

        file_name = target.get("local_file_name" if source == "local_upload" else "arxiv_file_name")
        if isinstance(file_name, str) and file_name.strip():
            candidates.append(os.path.join(base_dir, os.path.basename(file_name)))

        url = target.get("url", "")
        url_name = ""
        if (
            (source == "local_upload" and isinstance(url, str) and url.startswith("local://"))
            or (source == "arxiv_search" and isinstance(url, str))
        ):
            url_name = os.path.basename(url)
            if url_name:
                candidates.append(os.path.join(base_dir, url_name))
                if source == "arxiv_search" and not url_name.lower().endswith(".pdf"):
                    candidates.append(os.path.join(base_dir, f"{url_name}.pdf"))

        # Fallback: match suffixed variants like "<name>_1.pdf" for older records.
        if url_name:
            stem, ext = os.path.splitext(url_name)
            try:
                pattern_hits = []
                for fname in os.listdir(base_dir):
                    if not fname.lower().endswith(ext.lower()):
                        continue
                    fstem = os.path.splitext(fname)[0]
                    if fstem == stem or fstem.startswith(f"{stem}_"):
                        pattern_hits.append(os.path.join(base_dir, fname))
                if len(pattern_hits) == 1:
                    candidates.append(pattern_hits[0])
            except Exception:
                pass

        for path in candidates:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    break
            except Exception:
                # Do not fail paper deletion if file delete fails unexpectedly.
                pass

    state["papers"] = [p for p in papers if p.get("paper_id") != paper_id]
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
    # Persist a local copy of added arXiv papers for parity with local uploads.
    try:
        arxiv_file_name, arxiv_file_path = await loop.run_in_executor(
            None, _download_arxiv_pdf_to_disk, enriched.get("url", "")
        )
    except Exception as e:
        return JSONResponse({"error": f"Failed to save arXiv paper locally: {str(e)}"}, status_code=502)
    enriched["arxiv_file_name"] = arxiv_file_name
    enriched["arxiv_file_path"] = arxiv_file_path
    state.setdefault("papers", []).append(enriched)
    return {"paper": enriched, "total_papers": len(state["papers"])}


@app.post("/api/workspace/{session_id}/papers/upload")
async def upload_local_paper(session_id: str, file: UploadFile = File(...)):
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    filename = file.filename or "uploaded_paper"
    ext = os.path.splitext(filename)[1].lower()
    allowed = {".pdf", ".txt", ".md"}
    if ext not in allowed:
        return JSONResponse({"error": "Unsupported file type. Use PDF, TXT, or MD."}, status_code=400)

    content_bytes = await file.read()
    if not content_bytes:
        return JSONResponse({"error": "Uploaded file is empty"}, status_code=400)

    # Persist raw uploaded files under agent_workspace/papers_local as requested.
    local_papers_dir = _local_papers_dir()
    os.makedirs(local_papers_dir, exist_ok=True)
    safe_name = os.path.basename(filename).strip() or f"uploaded_paper{ext}"
    base_name, ext_name = os.path.splitext(safe_name)
    saved_name = safe_name
    idx = 1
    while os.path.exists(os.path.join(local_papers_dir, saved_name)):
        saved_name = f"{base_name}_{idx}{ext_name}"
        idx += 1
    save_path = os.path.join(local_papers_dir, saved_name)
    try:
        with open(save_path, "wb") as f:
            f.write(content_bytes)
    except Exception as e:
        return JSONResponse({"error": f"Failed to save uploaded file: {str(e)}"}, status_code=500)

    extracted_text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(content_bytes))
            extracted_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            extracted_text = content_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return JSONResponse({"error": f"Failed to parse file: {str(e)}"}, status_code=400)

    if not extracted_text.strip():
        return JSONResponse({"error": "No extractable text found in file."}, status_code=400)

    loop = asyncio.get_event_loop()
    enriched = await loop.run_in_executor(
        None, add_local_paper_to_workspace, session_id, saved_name, extracted_text, state.get("papers", [])
    )
    enriched["local_file_name"] = saved_name
    enriched["local_file_path"] = save_path
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
    mission_mode = body.get("mission_mode", "autonomous_search")
    if mission_mode not in {"autonomous_search", "workspace_only"}:
        return JSONResponse({"error": "Invalid mission_mode"}, status_code=400)
    state = workspaces.get(session_id)
    if not state:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    state["topic"] = topic
    state["mission_mode"] = mission_mode
    mission_id = len(state.get("missions", [])) + 1
    state.setdefault("missions", []).append({
        "mission_id": mission_id,
        "goal": topic,
        "mode": mission_mode,
        "status": "running",
        "date": __import__("datetime").datetime.now().isoformat(),
    })
    workspaces[session_id] = state
    return {"session_id": session_id, "mission_id": mission_id, "topic": topic, "mission_mode": mission_mode}


@app.get("/api/workspace/{session_id}/mission/stream")
async def stream_mission(session_id: str):
    async def event_generator():
        state = workspaces.get(session_id)
        if not state:
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'summary': 'Workspace not found'})}\n\n"
            return

        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue = asyncio.Queue()
        result_box: dict = {}
        yield f"data: {json.dumps({'step': 'generate_plan', 'agent': 'Planner', 'status': 'running', 'summary': ''})}\n\n"

        def emit(payload: dict):
            loop.call_soon_threadsafe(token_queue.put_nowait, {"type": "event", "payload": payload})

        def on_token(step: str, agent: str, token: str):
            loop.call_soon_threadsafe(
                token_queue.put_nowait,
                {"type": "token", "payload": {"type": "token", "step": step, "agent": agent, "token": token}},
            )

        def _run_mission():
            try:
                search_mode = "workspace" if state.get("mission_mode") == "workspace_only" else "online"
                mission_state = {
                    "session_id": state["session_id"],
                    "topic": state.get("topic", ""),
                    "goal": state.get("topic", ""),
                    "search_mode": search_mode,
                    "papers": state.get("papers", []),
                    "gaps": state.get("gaps", []),
                    "weak_gaps": state.get("weak_gaps", []),
                    "ideas": state.get("ideas", []),
                    "novelty_scores": state.get("novelty_scores", []),
                    "evidence_assessment": state.get("evidence_assessment", {}),
                    "insufficient_evidence_message": state.get("insufficient_evidence_message", ""),
                    "insufficient_data": bool(state.get("insufficient_data", False)),
                    "insufficiency_reason": state.get("insufficiency_reason", ""),
                    "iterations": 0,
                    "max_iterations": 3,
                }
                result_box["state"] = run_autonomous_mission(mission_state, emit=emit, token=on_token)
            except Exception as e:
                result_box["error"] = str(e)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, {"type": "done"})

        import threading
        thread = threading.Thread(target=_run_mission, daemon=True)
        thread.start()

        while True:
            msg = await token_queue.get()
            if msg["type"] == "done":
                break
            payload = msg["payload"]
            yield f"data: {json.dumps(payload)}\n\n"

        thread.join(timeout=15)

        if "error" in result_box:
            err = result_box["error"]
            yield f"data: {json.dumps({'step': 'mission', 'agent': 'Planner', 'status': 'error', 'summary': err})}\n\n"
            if state.get("missions"):
                state["missions"][-1]["status"] = "failed"
            return

        state.update(result_box["state"])
        workspaces[session_id] = state

        yield f"data: {json.dumps({'step': 'save_results', 'agent': 'Planner', 'status': 'running', 'summary': ''})}\n\n"
        await asyncio.sleep(0.05)
        try:
            su, ep = save_results(state)
            state.update(su)
            ep["agent"] = "Planner"
            yield f"data: {json.dumps(ep)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'step': 'save_results', 'agent': 'Planner', 'status': 'error', 'summary': str(e)})}\n\n"
            if state.get("missions"):
                state["missions"][-1]["status"] = "failed"
            return

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
        "weak_gaps": session_data.get("weak_research_gaps", []),
        "evidence_assessment": session_data.get("evidence_assessment", {}),
        "insufficient_evidence_message": session_data.get("insufficient_evidence_message", ""),
        "insufficient_data": bool(session_data.get("insufficient_data", False)),
        "insufficiency_reason": session_data.get("insufficiency_reason", ""),
        "mission_mode": session_data.get("mission_mode", "autonomous_search"),
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
