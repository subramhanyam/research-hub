// ===== ResearchPilot — Workspace Frontend =====

let sessionId = null;
let evtSource = null;
let workspaceData = { papers: [], gaps: [], ideas: [], novelty_scores: [], missions: [] };

// Activity log state
let currentActivityStep = null;
let currentActivityEl = null;
let activityAutoScroll = true;
let activityStepCounter = 0;

// ===== NAVIGATION =====
function showSection(name) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const sec = document.getElementById('sec-' + name);
    if (sec) sec.classList.add('active');

    const nav = document.querySelector(`.nav-item[data-section="${name}"]`);
    if (nav) nav.classList.add('active');

    document.getElementById('section-title').textContent = {
        dashboard: 'Dashboard',
        papers: 'Papers Library',
        search: 'arXiv Search',
        missions: 'Mission History',
        gaps: 'Research Gaps',
        ideas: 'Research Ideas',
        reports: 'Reports',
    }[name] || name;

    // Refresh section data
    if (name === 'papers') renderPapersList();
    if (name === 'gaps') renderGapsList();
    if (name === 'ideas') renderIdeasList();
    if (name === 'missions') renderMissionsList();
}

// ===== STATUS =====
function setStatus(status, text) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-text');
    dot.className = 'status-dot ' + status;
    label.textContent = text || status;
}

// ===== WORKSPACE =====
async function createWorkspace() {
    const name = prompt('Workspace name:');
    if (!name) return;

    const resp = await fetch('/api/workspace/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    const data = await resp.json();
    sessionId = data.session_id;
    workspaceData = { papers: [], gaps: [], ideas: [], novelty_scores: [], missions: [] };

    document.getElementById('ws-name').textContent = name;
    document.getElementById('ws-id').textContent = sessionId;
    updateOverview();
    setStatus('ready', 'Workspace Ready');
}

async function refreshWorkspace() {
    if (!sessionId) return;
    try {
        const resp = await fetch(`/api/workspace/${sessionId}`);
        const data = await resp.json();
        workspaceData.papers = data.papers || [];
        workspaceData.gaps = data.gaps || [];
        workspaceData.ideas = data.ideas || [];
        workspaceData.novelty_scores = data.novelty_scores || [];
        workspaceData.missions = data.missions || [];
        updateOverview();
    } catch (e) { /* ignore */ }
}

function updateOverview() {
    document.getElementById('ov-papers').textContent = workspaceData.papers.length;
    document.getElementById('ov-missions').textContent = workspaceData.missions.length;
    document.getElementById('ov-gaps').textContent = workspaceData.gaps.length;
    document.getElementById('ov-ideas').textContent = workspaceData.ideas.length;

    const hasResults = workspaceData.gaps.length > 0;
    document.getElementById('gen-report-btn').disabled = !hasResults;
    document.getElementById('download-btn').disabled = !hasResults;
}

// ===== MISSION =====
async function runMission() {
    const input = document.getElementById('mission-topic');
    const topic = input.value.trim();
    if (!topic) { input.focus(); return; }

    if (!sessionId) {
        await createWorkspaceAuto(topic);
    }

    const btn = document.getElementById('run-mission-btn');
    btn.disabled = true;

    // Start mission
    await fetch(`/api/workspace/${sessionId}/mission/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic }),
    });

    resetTimeline();
    setStatus('running', 'Mission running...');
    connectSSE();
}

async function createWorkspaceAuto(name) {
    const resp = await fetch('/api/workspace/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    const data = await resp.json();
    sessionId = data.session_id;
    workspaceData = { papers: [], gaps: [], ideas: [], novelty_scores: [], missions: [] };
    document.getElementById('ws-name').textContent = name;
    document.getElementById('ws-id').textContent = sessionId;
}

function connectSSE() {
    if (evtSource) evtSource.close();
    evtSource = new EventSource(`/api/workspace/${sessionId}/mission/stream`);
    evtSource.onmessage = (e) => {
        try { handleEvent(JSON.parse(e.data)); } catch (err) { console.error(err); }
    };
    evtSource.onerror = () => { evtSource.close(); evtSource = null; };
}

function handleEvent(p) {
    const { type, step, status, agent, summary, data, token } = p;

    // Handle streaming token events
    if (type === 'token') {
        appendActivityToken(step, agent, token);
        return;
    }

    if (step === 'done') {
        setStatus('complete', 'Mission Complete');
        document.getElementById('run-mission-btn').disabled = false;
        if (evtSource) evtSource.close();
        refreshWorkspace();
        finalizeActivityLog();
        return;
    }

    if (status === 'error') {
        setStatus('error', summary || 'Error');
        updateTimeline(step, 'error', summary);
        document.getElementById('run-mission-btn').disabled = false;
        if (evtSource) evtSource.close();
        appendActivityEntry(step, agent, 'error', summary || 'Error occurred');
        return;
    }

    updateTimeline(step, status, summary);
    if (agent) updateAgent(agent, status);

    if (status === 'running') {
        setStatus('running', `${agent}: ${stepLabel(step)}...`);
        startActivityStep(step, agent);
    }

    if (status === 'complete' && data) {
        completeActivityStep(step, agent, summary);
        switch (step) {
            case 'generate_plan': onPlan(data); break;
            case 'search_papers': onPapers(data); break;
            case 'extract_limitations': break;
            case 'cluster_limitations': onClusters(data); break;
            case 'identify_gaps': onGaps(data); break;
            case 'generate_ideas': onIdeas(data); break;
            case 'reflect': onReflect(data); break;
        }
    }
}

// ===== TIMELINE =====
function updateTimeline(step, status, summary) {
    const el = document.getElementById('step-' + step);
    if (!el) return;
    const icon = el.querySelector('.tl-icon');
    const sumEl = el.querySelector('.tl-summary');

    el.classList.remove('running', 'complete');
    icon.classList.remove('pending', 'running', 'complete');

    if (status === 'running') {
        el.classList.add('running');
        icon.classList.add('running');
        icon.innerHTML = '&#8635;';
    } else if (status === 'complete') {
        el.classList.add('complete');
        icon.classList.add('complete');
        icon.innerHTML = '&#9745;';
    }
    if (summary) sumEl.textContent = summary;
}

function resetTimeline() {
    document.querySelectorAll('.tl-step').forEach(el => {
        el.classList.remove('running', 'complete');
        const icon = el.querySelector('.tl-icon');
        icon.className = 'tl-icon pending';
        icon.innerHTML = '&#9744;';
        el.querySelector('.tl-summary').textContent = '';
    });
    document.querySelectorAll('.agent-card').forEach(c => {
        c.classList.remove('active', 'done');
        c.querySelector('.agent-status').textContent = 'Idle';
    });
    // Reset activity log for new mission
    const log = getActivityLog();
    log.innerHTML = '';
    currentActivityStep = null;
    currentActivityEl = null;
    activityStepCounter = 0;
    updateActivityBadge('LIVE');
}

function stepLabel(s) {
    return { generate_plan:'Planning', search_papers:'Searching', extract_limitations:'Extracting',
        cluster_limitations:'Clustering', identify_gaps:'Analyzing', generate_ideas:'Ideating',
        reflect:'Reflecting', save_results:'Saving' }[s] || s;
}

function updateAgent(agent, status) {
    document.querySelectorAll('.agent-card').forEach(c => {
        if (status === 'running' && c.id !== 'agent-' + agent) c.classList.remove('active');
    });
    const card = document.getElementById('agent-' + agent);
    if (!card) return;
    const st = card.querySelector('.agent-status');
    if (status === 'running') { card.classList.remove('done'); card.classList.add('active'); st.textContent = 'Working...'; }
    else if (status === 'complete') { card.classList.remove('active'); card.classList.add('done'); st.textContent = 'Done'; }
}

// ===== ACTIVITY LOG =====
function getActivityLog() {
    return document.getElementById('activity-log');
}

function clearActivityLog() {
    const log = getActivityLog();
    log.innerHTML = '<div class="activity-placeholder">Agent activity will appear here when a mission is running...</div>';
    currentActivityStep = null;
    currentActivityEl = null;
    updateActivityBadge('IDLE');
}

function updateActivityBadge(text) {
    const badge = document.getElementById('activity-badge');
    if (badge) badge.textContent = text;
}

function startActivityStep(step, agent) {
    const log = getActivityLog();
    // Remove placeholder if present
    const placeholder = log.querySelector('.activity-placeholder');
    if (placeholder) placeholder.remove();

    activityStepCounter++;
    const stepId = `${step}-${activityStepCounter}`;
    currentActivityStep = stepId;
    updateActivityBadge('LIVE');

    // Create step header
    const stepEntry = document.createElement('div');
    stepEntry.className = 'activity-step';
    stepEntry.dataset.step = step;
    stepEntry.id = `activity-step-${stepId}`;
    stepEntry.innerHTML = `
        <div class="activity-step-header">
            <span class="activity-agent-tag agent-${agent.toLowerCase()}">${esc(agent)}</span>
            <span class="activity-step-label">${stepLabel(step)}</span>
            <span class="activity-spinner">&#8635;</span>
        </div>
        <div class="activity-stream" id="activity-stream-${stepId}"></div>
    `;
    log.appendChild(stepEntry);

    currentActivityEl = document.getElementById(`activity-stream-${stepId}`);
    scrollActivityLog();
}

function getLatestStreamEl(step) {
    const allSteps = document.querySelectorAll(`.activity-step[data-step="${step}"]`);
    if (allSteps.length === 0) return null;
    const latest = allSteps[allSteps.length - 1];
    return latest.querySelector('.activity-stream');
}

function appendActivityToken(step, agent, token) {
    let streamEl = getLatestStreamEl(step);
    if (!streamEl) {
        startActivityStep(step, agent);
        streamEl = getLatestStreamEl(step);
    }

    // Append token text, preserving newlines
    const text = token;
    if (text.includes('\n')) {
        const parts = text.split('\n');
        for (let i = 0; i < parts.length; i++) {
            if (i > 0) {
                streamEl.appendChild(document.createElement('br'));
            }
            if (parts[i]) {
                streamEl.appendChild(document.createTextNode(parts[i]));
            }
        }
    } else {
        streamEl.appendChild(document.createTextNode(text));
    }

    scrollActivityLog();
}

function completeActivityStep(step, agent, summary) {
    const allSteps = document.querySelectorAll(`.activity-step[data-step="${step}"]`);
    const stepEl = allSteps.length ? allSteps[allSteps.length - 1] : null;
    if (stepEl) {
        stepEl.classList.add('completed');
        const spinner = stepEl.querySelector('.activity-spinner');
        if (spinner) spinner.innerHTML = '&#10003;';

        if (summary) {
            const streamEl = stepEl.querySelector('.activity-stream');
            if (streamEl) {
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'activity-summary';
                summaryDiv.textContent = summary;
                streamEl.appendChild(summaryDiv);
            }
        }
    }

    scrollActivityLog();
}

function appendActivityEntry(step, agent, status, text) {
    const log = getActivityLog();
    const placeholder = log.querySelector('.activity-placeholder');
    if (placeholder) placeholder.remove();

    const entry = document.createElement('div');
    entry.className = `activity-entry activity-${status}`;
    entry.innerHTML = `<span class="activity-agent-tag">${esc(agent || '?')}</span> <span>${esc(text)}</span>`;
    log.appendChild(entry);
    scrollActivityLog();
}

function finalizeActivityLog() {
    updateActivityBadge('DONE');
    const log = getActivityLog();
    const doneEntry = document.createElement('div');
    doneEntry.className = 'activity-done';
    doneEntry.textContent = 'Mission completed successfully.';
    log.appendChild(doneEntry);
    scrollActivityLog();
}

function scrollActivityLog() {
    if (!activityAutoScroll) return;
    const log = getActivityLog();
    log.scrollTop = log.scrollHeight;
}

// ===== EVENT DATA HANDLERS =====
function onPlan(data) {
    // Show plan details in activity log
    if (data.subtopics && data.subtopics.length) {
        const streamEl = getLatestStreamEl('generate_plan');
        if (streamEl) {
            const planDiv = document.createElement('div');
            planDiv.className = 'activity-plan-detail';
            planDiv.innerHTML = `
                <div class="plan-section"><strong>Subtopics:</strong> ${data.subtopics.map(s => esc(s)).join(' | ')}</div>
                <div class="plan-section"><strong>Keywords:</strong> ${data.keywords.map(k => '<span class="plan-keyword">' + esc(k) + '</span>').join(' ')}</div>
            `;
            streamEl.appendChild(planDiv);
            scrollActivityLog();
        }
    }
}

function onPapers(data) {
    if (data.papers) {
        data.papers.forEach(p => {
            p.source = 'mission';
            if (!workspaceData.papers.find(x => x.url === p.url)) {
                workspaceData.papers.push(p);
            }
        });
    }
    updateOverview();
}

function onClusters(data) { /* stored via refreshWorkspace */ }

function onGaps(data) {
    if (data.gaps) workspaceData.gaps = data.gaps;
    updateOverview();
}

function onIdeas(data) {
    if (data.ideas) workspaceData.ideas = data.ideas;
    updateOverview();
}

function onReflect(data) {
    if (data.scores) {
        workspaceData.novelty_scores = data.scores.map(s => s.novelty_score);
    }
}

// ===== PAPERS LIBRARY =====
function renderPapersList() {
    const container = document.getElementById('papers-list');
    const empty = document.getElementById('papers-empty');
    const badge = document.getElementById('papers-lib-count');
    const papers = getFilteredPapers();

    badge.textContent = papers.length;
    if (!papers.length) { container.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');

    container.innerHTML = papers.map(p => `
        <div class="paper-row" data-paper-id="${p.paper_id || ''}">
            <div class="paper-row-body">
                <div class="paper-row-cite">${esc(p.citation_label || '')}</div>
                <div class="paper-row-title"><a href="${esc(p.url)}" target="_blank">${esc(p.title)}</a></div>
                <div class="paper-row-meta">${esc((p.authors||[]).join(', '))} &middot; ${esc(p.published || '')}
                    <span class="paper-row-source ${p.source === 'arxiv_search' ? 'source-manual' : 'source-mission'}">${p.source === 'arxiv_search' ? 'Added' : 'Mission'}</span>
                </div>
            </div>
            <div class="paper-row-actions">
                <button onclick="deletePaper(${p.paper_id})">Remove</button>
            </div>
        </div>
    `).join('');
}

function getFilteredPapers() {
    const filter = (document.getElementById('papers-filter')?.value || '').toLowerCase();
    let papers = workspaceData.papers.filter(p =>
        !filter || p.title.toLowerCase().includes(filter) || (p.authors||[]).join(' ').toLowerCase().includes(filter)
    );
    const sort = document.getElementById('papers-sort')?.value || 'id';
    if (sort === 'year') papers.sort((a, b) => (b.published || '').localeCompare(a.published || ''));
    else if (sort === 'title') papers.sort((a, b) => a.title.localeCompare(b.title));
    return papers;
}

function filterPapers() { renderPapersList(); }
function sortPapers() { renderPapersList(); }

async function deletePaper(paperId) {
    if (!sessionId || !paperId) return;
    await fetch(`/api/workspace/${sessionId}/papers/${paperId}`, { method: 'DELETE' });
    workspaceData.papers = workspaceData.papers.filter(p => p.paper_id !== paperId);
    updateOverview();
    renderPapersList();
}

// ===== ARXIV SEARCH =====
async function searchArxiv() {
    const query = document.getElementById('arxiv-query').value.trim();
    if (!query) return;

    const sort = document.getElementById('arxiv-sort').value;
    const dateFrom = document.getElementById('arxiv-date-from').value || null;
    const dateTo = document.getElementById('arxiv-date-to').value || null;
    const max = parseInt(document.getElementById('arxiv-max').value) || 10;

    document.getElementById('arxiv-loading').classList.remove('hidden');
    document.getElementById('arxiv-empty').classList.add('hidden');
    document.getElementById('arxiv-results').innerHTML = '';

    try {
        const resp = await fetch('/api/arxiv/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, max_results: max, sort_by: sort, date_from: dateFrom, date_to: dateTo }),
        });
        const data = await resp.json();
        renderArxivResults(data.results || []);
    } catch (err) {
        document.getElementById('arxiv-results').innerHTML = '<div class="empty-state">Search failed.</div>';
    }

    document.getElementById('arxiv-loading').classList.add('hidden');
}

function renderArxivResults(results) {
    const container = document.getElementById('arxiv-results');
    if (!results.length) {
        container.innerHTML = '<div class="empty-state">No results found.</div>';
        return;
    }

    container.innerHTML = results.map((r, i) => `
        <div class="arxiv-card" id="arxiv-${i}">
            <div class="arxiv-card-header">
                <div class="arxiv-card-title"><a href="${esc(r.url)}" target="_blank">${esc(r.title)}</a></div>
                <button class="arxiv-add-btn" id="arxiv-add-${i}" onclick="addPaperToWorkspace(${i})">+ Add</button>
            </div>
            <div class="arxiv-card-meta">${esc((r.authors||[]).join(', '))} &middot; ${esc(r.published)}</div>
            <div class="arxiv-card-summary">${esc(r.summary)}</div>
            ${r.categories && r.categories.length ? `<div class="arxiv-card-cats">${r.categories.map(c => `<span class="cat-tag">${esc(c)}</span>`).join('')}</div>` : ''}
        </div>
    `).join('');

    // Store for add-to-workspace
    window._arxivResults = results;
}

async function addPaperToWorkspace(idx) {
    if (!sessionId) {
        await createWorkspaceAuto('Research Workspace');
    }

    const paper = window._arxivResults[idx];
    if (!paper) return;

    const btn = document.getElementById('arxiv-add-' + idx);
    btn.textContent = 'Adding...';
    btn.disabled = true;

    try {
        const resp = await fetch(`/api/workspace/${sessionId}/papers/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, paper }),
        });
        const data = await resp.json();

        if (data.paper) {
            data.paper.source = 'arxiv_search';
            if (!workspaceData.papers.find(p => p.url === data.paper.url)) {
                workspaceData.papers.push(data.paper);
            }
            updateOverview();
            btn.textContent = 'Added';
            btn.classList.add('added');
        } else {
            btn.textContent = data.error || 'Error';
        }
    } catch (e) {
        btn.textContent = 'Failed';
    }
}

// ===== SEMANTIC SEARCH =====
async function semanticSearch() {
    if (!sessionId) return;
    const query = document.getElementById('sem-search-input').value.trim();
    if (!query) return;

    const container = document.getElementById('sem-search-results');
    container.innerHTML = '<div class="empty-state">Searching...</div>';

    try {
        const resp = await fetch(`/api/workspace/${sessionId}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, query, n_results: 5 }),
        });
        const data = await resp.json();

        if (!data.results || !data.results.length) {
            container.innerHTML = '<div class="empty-state">No results found.</div>';
            return;
        }

        container.innerHTML = data.results.map(h => `
            <div class="sem-hit">
                <div class="sem-hit-header">
                    <span class="sem-hit-title">${esc(h.title)}</span>
                    <span class="sem-hit-score">${(h.relevance * 100).toFixed(0)}% match</span>
                </div>
                <div class="sem-hit-cite">${esc(h.citation_label)}</div>
                <div class="sem-hit-chunk">${esc(h.chunk)}</div>
            </div>
        `).join('');
    } catch (e) {
        container.innerHTML = '<div class="empty-state">Search failed.</div>';
    }
}

// ===== MISSIONS LIST =====
function renderMissionsList() {
    const container = document.getElementById('missions-list');
    const empty = document.getElementById('missions-empty');
    const badge = document.getElementById('missions-count');
    const missions = workspaceData.missions || [];

    badge.textContent = missions.length;
    if (!missions.length) { container.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');

    container.innerHTML = missions.map(m => `
        <div class="mission-row" onclick="showSection('dashboard')">
            <div class="mission-row-header">
                <span class="mission-row-id">Mission ${m.mission_id}</span>
                <span class="mission-status ${m.status}">${m.status}</span>
            </div>
            <div class="mission-row-goal">Goal: "${esc(m.goal)}"</div>
            <div class="mission-row-stats">
                <span>Gaps: ${m.gaps_found || 0}</span>
                <span>Ideas: ${m.ideas_generated || 0}</span>
                ${m.novelty_avg ? `<span>Novelty: ${m.novelty_avg}/10</span>` : ''}
                <span>${m.date ? m.date.split('T')[0] : ''}</span>
            </div>
        </div>
    `).join('');
}

// ===== GAPS LIST =====
function renderGapsList() {
    const container = document.getElementById('gaps-list');
    const empty = document.getElementById('gaps-empty');
    const badge = document.getElementById('gaps-badge');
    const gaps = workspaceData.gaps || [];

    badge.textContent = gaps.length;
    if (!gaps.length) { container.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');

    container.innerHTML = gaps.map((g, i) => {
        const gap = typeof g === 'string' ? g : g.gap;
        const cites = g.citations || (g.supporting_papers || []).map(pid => `[Paper ${pid}]`);
        return `
            <div class="gap-card">
                <div class="gap-card-text"><strong>${i + 1}.</strong> ${esc(gap)}</div>
                ${cites.length ? `<div class="gap-card-papers">Sources: ${cites.map(c => `<span>${esc(c)}</span>`).join(', ')}</div>` : ''}
            </div>
        `;
    }).join('');
}

// ===== IDEAS LIST =====
function renderIdeasList() {
    const container = document.getElementById('ideas-list');
    const empty = document.getElementById('ideas-empty');
    const ideas = workspaceData.ideas || [];
    const scores = workspaceData.novelty_scores || [];

    if (!ideas.length) { container.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');

    container.innerHTML = ideas.map((idea, i) => {
        const score = scores[i] || (idea.novelty_score);
        const scoreVal = typeof score === 'object' ? score.novelty_score : score;
        const cls = scoreVal >= 8 ? 'novelty-high' : scoreVal >= 6 ? 'novelty-mid' : 'novelty-low';
        const confCls = scoreVal >= 8 ? 'confidence-high' : scoreVal >= 6 ? 'confidence-medium' : 'confidence-low';
        const confLabel = scoreVal >= 8 ? 'High' : scoreVal >= 6 ? 'Medium' : 'Low';
        const cites = idea.citations || (idea.cited_papers || []).map(pid => `[Paper ${pid}]`);

        return `
            <div class="idea-card">
                <div class="idea-card-header">
                    <div class="idea-card-title">${esc(idea.title)}</div>
                    ${scoreVal ? `<span class="novelty-badge ${cls}">${scoreVal}/10</span>` : ''}
                </div>
                <div class="idea-card-desc">${esc(idea.description)}</div>
                ${idea.addresses_gaps?.length ? `<div class="idea-card-gaps">Addresses: ${idea.addresses_gaps.map(g => esc(g)).join('; ')}</div>` : ''}
                ${cites.length ? `<div class="idea-card-cites">${cites.map(c => esc(c)).join(', ')}</div>` : ''}
                ${scoreVal ? `<span class="idea-card-confidence ${confCls}">Confidence: ${confLabel}</span>` : ''}
            </div>
        `;
    }).join('');
}

// ===== REPORT =====
async function generateReport() {
    if (!sessionId) return;
    const btn = document.getElementById('gen-report-btn');
    btn.disabled = true;
    btn.textContent = 'Generating...';
    setStatus('running', 'Generating report...');
    document.getElementById('report-empty').classList.add('hidden');

    try {
        const resp = await fetch(`/api/workspace/${sessionId}/report`, { method: 'POST' });
        const data = await resp.json();
        const container = document.getElementById('report-content');

        let html = esc(data.report);
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\n/g, '<br>');
        container.innerHTML = html;

        setStatus('complete', 'Report ready');
        showSection('reports');
    } catch (e) {
        setStatus('error', 'Report failed');
    }

    btn.disabled = false;
    btn.textContent = 'Generate Research Brief';
}

function downloadJSON() {
    if (!sessionId) return;
    window.open(`/api/workspace/${sessionId}/download`, '_blank');
}

// ===== RESUME =====
async function resumeSession() {
    const fileInput = document.getElementById('resume-file');
    if (!fileInput.files.length) return;

    setStatus('running', 'Restoring...');
    try {
        const text = await fileInput.files[0].text();
        const sessionData = JSON.parse(text);

        const resp = await fetch('/api/workspace/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_data: sessionData }),
        });
        const result = await resp.json();

        if (result.error) { setStatus('error', result.error); return; }

        sessionId = result.session_id;
        document.getElementById('ws-name').textContent = result.topic || 'Resumed';
        document.getElementById('ws-id').textContent = sessionId;
        document.getElementById('mission-topic').value = result.topic || '';

        // Populate workspace data from JSON
        workspaceData.papers = (sessionData.papers || []).map(p => ({ ...p, source: 'mission' }));
        workspaceData.gaps = sessionData.research_gaps || [];
        workspaceData.ideas = sessionData.research_ideas || [];
        workspaceData.novelty_scores = sessionData.novelty_scores || [];
        workspaceData.missions = [{
            mission_id: 1,
            goal: sessionData.topic || '',
            status: 'completed',
            date: sessionData.generated_at || '',
            gaps_found: workspaceData.gaps.length,
            ideas_generated: workspaceData.ideas.length,
        }];

        updateOverview();
        setStatus('complete', 'Session Restored');
    } catch (e) {
        setStatus('error', 'Failed to parse JSON');
    }
}

// ===== UTILS =====
function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.id === 'mission-topic') runMission();
    if (e.key === 'Enter' && e.target.id === 'arxiv-query') searchArxiv();
    if (e.key === 'Enter' && e.target.id === 'sem-search-input') semanticSearch();
});
