const RHX_ACTIVE_SESSION_KEY = "researchpilot.active_session_id";
let testSearchResults = [];
let addedPaperUrls = new Set();
let papersTableData = [];
let papersSearchQuery = "";
let papersFilterMode = "all";
let papersSortMode = "newest";
const RHX_WORKSPACE_NAME_KEY = "researchhubx.active_workspace_name";
const RHX_WORKSPACES_KEY = "researchhubx.workspaces";
const RHX_DEEP_READ_KEY = "researchpilot.deep_read_timer_state";
const RHX_DEEP_READ_TARGET_MS = 2 * 60 * 60 * 1000;
let rhxDeepReadTicker = null;

function loadDeepReadState() {
    const fallback = { elapsed_ms: 0, is_running: false, started_at: null };
    try {
        const raw = localStorage.getItem(RHX_DEEP_READ_KEY);
        if (!raw) return fallback;
        const parsed = JSON.parse(raw);
        const elapsed = Number(parsed?.elapsed_ms);
        const running = Boolean(parsed?.is_running);
        const startedAt = Number(parsed?.started_at);
        return {
            elapsed_ms: Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0,
            is_running: running,
            started_at: running && Number.isFinite(startedAt) && startedAt > 0 ? startedAt : null,
        };
    } catch (e) {
        return fallback;
    }
}

function saveDeepReadState(state) {
    try {
        localStorage.setItem(RHX_DEEP_READ_KEY, JSON.stringify(state));
    } catch (e) {
        // ignore storage issues
    }
}

function getDeepReadElapsedMs(state) {
    if (!state.is_running || !state.started_at) return state.elapsed_ms;
    return state.elapsed_ms + Math.max(0, Date.now() - state.started_at);
}

function startDeepReadSession(reset = false) {
    const state = loadDeepReadState();
    const next = reset
        ? { elapsed_ms: 0, is_running: true, started_at: Date.now() }
        : {
            elapsed_ms: state.is_running ? getDeepReadElapsedMs(state) : state.elapsed_ms,
            is_running: true,
            started_at: Date.now(),
        };
    saveDeepReadState(next);
    return next;
}

function pauseDeepReadSession() {
    const state = loadDeepReadState();
    if (!state.is_running) return state;
    const next = {
        elapsed_ms: getDeepReadElapsedMs(state),
        is_running: false,
        started_at: null,
    };
    saveDeepReadState(next);
    return next;
}

function stopDeepReadSession() {
    const next = { elapsed_ms: 0, is_running: false, started_at: null };
    saveDeepReadState(next);
    return next;
}

function formatClockTime(ms) {
    const totalSeconds = Math.floor(Math.max(0, ms) / 1000);
    const hours = Math.floor(totalSeconds / 3600).toString().padStart(2, "0");
    const minutes = Math.floor((totalSeconds % 3600) / 60).toString().padStart(2, "0");
    const seconds = (totalSeconds % 60).toString().padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
}

function renderDeepReadTimer(state) {
    const timerEl = document.getElementById("test-deepread-timer");
    const statusEl = document.getElementById("test-deepread-status");
    const progressEl = document.getElementById("test-deepread-progress");
    const toggleIconEl = document.getElementById("test-deepread-toggle-icon");
    if (!timerEl || !statusEl || !progressEl || !toggleIconEl) return;

    const elapsedMs = getDeepReadElapsedMs(state);
    timerEl.textContent = formatClockTime(elapsedMs);
    statusEl.textContent = state.is_running ? "Running" : (elapsedMs > 0 ? "Paused" : "Stopped");
    toggleIconEl.textContent = state.is_running ? "pause" : "play_arrow";

    const pct = Math.max(0, Math.min(100, (elapsedMs / RHX_DEEP_READ_TARGET_MS) * 100));
    progressEl.style.width = `${pct}%`;
}

function stopDeepReadTicker() {
    if (!rhxDeepReadTicker) return;
    clearInterval(rhxDeepReadTicker);
    rhxDeepReadTicker = null;
}

function startDeepReadTicker() {
    stopDeepReadTicker();
    if (!loadDeepReadState().is_running) return;
    rhxDeepReadTicker = setInterval(() => {
        renderDeepReadTimer(loadDeepReadState());
        updateResearchRunDurationDisplay();
    }, 1000);
}

function initDeepReadSessionTimer() {
    const timerEl = document.getElementById("test-deepread-timer");
    const toggleBtn = document.getElementById("test-deepread-toggle");
    const stopBtn = document.getElementById("test-deepread-stop");
    if (!timerEl || !toggleBtn || !stopBtn) return;

    let state = loadDeepReadState();
    renderDeepReadTimer(state);
    startDeepReadTicker();

    toggleBtn.addEventListener("click", () => {
        state = loadDeepReadState();
        if (state.is_running) {
            state = pauseDeepReadSession();
            renderDeepReadTimer(state);
            stopDeepReadTicker();
            updateResearchRunDurationDisplay();
            return;
        }

        state = startDeepReadSession(false);
        renderDeepReadTimer(state);
        startDeepReadTicker();
        updateResearchRunDurationDisplay();
    });

    stopBtn.addEventListener("click", () => {
        state = stopDeepReadSession();
        renderDeepReadTimer(state);
        stopDeepReadTicker();
        updateResearchRunDurationDisplay();
    });

    document.addEventListener("visibilitychange", () => {
        renderDeepReadTimer(loadDeepReadState());
    });

    window.addEventListener("storage", (event) => {
        if (event.key !== RHX_DEEP_READ_KEY) return;
        const next = loadDeepReadState();
        renderDeepReadTimer(next);
        updateResearchRunDurationDisplay();
        if (next.is_running) startDeepReadTicker();
        else stopDeepReadTicker();
    });
}

function normalizeWorkspaces(list) {
    if (!Array.isArray(list)) return [];
    const bySession = new Map();
    for (const w of list) {
        if (!w || typeof w.session_id !== "string" || typeof w.name !== "string") continue;
        const sessionId = w.session_id.trim();
        const name = w.name.trim();
        if (!sessionId || !name) continue;
        bySession.set(sessionId, { session_id: sessionId, name });
    }

    // Also collapse exact display-name duplicates to keep dropdown options unique.
    const byName = new Map();
    for (const w of bySession.values()) {
        byName.set(w.name.toLowerCase(), w);
    }
    return Array.from(byName.values());
}

function rhxEsc(value) {
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
}

function rhxGetStoredSessionId() {
    try {
        return localStorage.getItem(RHX_ACTIVE_SESSION_KEY);
    } catch (e) {
        return null;
    }
}

function rhxSetStoredSessionId(sessionId) {
    try {
        if (sessionId) localStorage.setItem(RHX_ACTIVE_SESSION_KEY, sessionId);
    } catch (e) {
        // ignore storage issues
    }
}

async function rhxGetOrCreateSessionId() {
    const existing = rhxGetStoredSessionId();
    if (existing) {
        try {
            const probe = await fetch(`/api/workspace/${existing}`);
            if (probe.ok) return existing;
        } catch (e) {
            // fall through to create new
        }
    }

    const resp = await fetch("/api/workspace/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "ResearchHub X Workspace" }),
    });
    if (!resp.ok) throw new Error("Failed to create workspace");

    const data = await resp.json();
    const sessionId = data.session_id;
    rhxSetStoredSessionId(sessionId);
    return sessionId;
}

function getUploadUiElements() {
    return {
        statusEl: document.getElementById("test-upload-status"),
        progressWrap: document.getElementById("test-upload-progress-wrap"),
        progressBar: document.getElementById("test-upload-progress-bar"),
        progressText: document.getElementById("test-upload-progress-text"),
        progressCount: document.getElementById("test-upload-progress-count"),
        progressIcon: document.getElementById("test-upload-progress-icon"),
        uploadBtn: document.getElementById("test-upload-btn"),
    };
}

function resetUploadUi(ui) {
    if (!ui) return;
    if (ui.progressWrap) ui.progressWrap.classList.remove("hidden");
    if (ui.progressBar) {
        ui.progressBar.style.width = "0%";
        ui.progressBar.classList.remove("bg-red-500", "bg-secondary");
        ui.progressBar.classList.add("bg-primary");
    }
    if (ui.progressText) ui.progressText.textContent = "Preparing upload...";
    if (ui.progressCount) ui.progressCount.textContent = "0%";
    if (ui.progressIcon) {
        ui.progressIcon.textContent = "autorenew";
        ui.progressIcon.classList.add("animate-spin");
        ui.progressIcon.classList.remove("rhx-pop-in", "text-red-500", "text-secondary");
    }
}

function setUploadProgress(ui, percent, text) {
    if (!ui) return;
    const pct = Math.max(0, Math.min(100, percent));
    if (ui.progressBar) ui.progressBar.style.width = `${pct}%`;
    if (ui.progressCount) ui.progressCount.textContent = `${Math.round(pct)}%`;
    if (ui.progressText && text) ui.progressText.textContent = text;
}

function finalizeUploadUi(ui, successCount, totalCount) {
    if (!ui) return;
    const allSuccess = successCount === totalCount;

    if (ui.progressBar) {
        ui.progressBar.classList.remove("bg-primary");
        ui.progressBar.classList.add(allSuccess ? "bg-secondary" : "bg-red-500");
    }
    if (ui.progressIcon) {
        ui.progressIcon.classList.remove("animate-spin");
        ui.progressIcon.textContent = allSuccess ? "check_circle" : "error";
        ui.progressIcon.classList.add("rhx-pop-in");
        ui.progressIcon.classList.add(allSuccess ? "text-secondary" : "text-red-500");
    }
    if (ui.progressText) {
        ui.progressText.textContent = allSuccess
            ? "Upload complete"
            : "Upload finished with errors";
    }
}

async function rhxUploadFiles(files, ui) {
    if (!files.length) return;
    let sessionId = await rhxGetOrCreateSessionId();

    resetUploadUi(ui);
    if (ui?.uploadBtn) ui.uploadBtn.disabled = true;

    let successCount = 0;
    const totalCount = files.length;

    for (let i = 0; i < files.length; i += 1) {
        const file = files[i];
        const formData = new FormData();
        formData.append("file", file);

        if (ui?.statusEl) ui.statusEl.textContent = `Uploading ${file.name}...`;
        setUploadProgress(ui, (i / totalCount) * 100, `Uploading file ${i + 1} of ${totalCount}`);

        let resp = await fetch(`/api/workspace/${sessionId}/papers/upload`, {
            method: "POST",
            body: formData,
        });
        if (resp.status === 404) {
            const recreatedSession = await rhxGetOrCreateSessionId();
            sessionId = recreatedSession;
            resp = await fetch(`/api/workspace/${recreatedSession}/papers/upload`, {
                method: "POST",
                body: formData,
            });
        }
        if (resp.ok) successCount += 1;

        setUploadProgress(ui, ((i + 1) / totalCount) * 100, `Processed file ${i + 1} of ${totalCount}`);
    }

    if (ui?.statusEl) {
        ui.statusEl.textContent = successCount === totalCount
            ? `${successCount} file(s) uploaded`
            : `${successCount}/${totalCount} file(s) uploaded`;
    }
    finalizeUploadUi(ui, successCount, totalCount);
    if (ui?.uploadBtn) ui.uploadBtn.disabled = false;

    if (typeof window.syncTestKpis === "function") {
        window.syncTestKpis();
    }
    await initRecentPapers();
}

function initTestUpload() {
    const dropzone = document.getElementById("test-upload-dropzone");
    const input = document.getElementById("test-upload-input");
    const button = document.getElementById("test-upload-btn");
    const ui = getUploadUiElements();
    if (!dropzone || !input || !button) return;

    const pickFiles = () => input.click();
    dropzone.addEventListener("click", (e) => {
        if (e.target === input) return;
        pickFiles();
    });
    button.addEventListener("click", (e) => {
        e.stopPropagation();
        pickFiles();
    });

    input.addEventListener("change", async () => {
        const files = Array.from(input.files || []);
        if (!files.length) return;
        try {
            await rhxUploadFiles(files, ui);
        } catch (e) {
            if (ui?.statusEl) ui.statusEl.textContent = "Upload failed";
            finalizeUploadUi(ui, 0, files.length);
            if (ui?.uploadBtn) ui.uploadBtn.disabled = false;
        } finally {
            input.value = "";
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("border-primary", "bg-accent/10");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-primary", "bg-accent/10");
    });
    dropzone.addEventListener("drop", async (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-primary", "bg-accent/10");
        const files = Array.from(e.dataTransfer?.files || []);
        if (!files.length) return;
        try {
            await rhxUploadFiles(files, ui);
        } catch (err) {
            if (ui?.statusEl) ui.statusEl.textContent = "Upload failed";
            finalizeUploadUi(ui, 0, files.length);
            if (ui?.uploadBtn) ui.uploadBtn.disabled = false;
        }
    });
}

function renderUploadedPaperRow(paper) {
    const title = paper.title || "Untitled";
    const url = typeof paper.url === "string" ? paper.url : "";
    const hasExternalLink = Boolean(url && !url.startsWith("local://"));
    const published = paper.published || "";
    const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : "";
    const source = paper.source === "arxiv_search" ? "arXiv" : "Local";
    const sourceClass = paper.source === "arxiv_search"
        ? "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400"
        : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400";
    const leftMeta = authors || (paper.source === "arxiv_search" ? "arXiv" : "Local Upload");
    const authorMeta = `${leftMeta}${published ? ` • ${published}` : ""}`;
    const categories = Array.isArray(paper.categories) ? paper.categories.filter(Boolean) : [];
    const primaryCategory = (paper.primary_category || categories[0] || "").trim();
    const metadataTooltip = [
        `Title: ${title}`,
        `Author(s): ${authors || "N/A"}`,
        `Paper ID: ${paper.paper_id || "N/A"}`,
        `Categories: ${categories.length ? categories.join(", ") : "N/A"}`,
        `Primary Category: ${primaryCategory || "N/A"}`,
    ].join("\n");

    return `
<tr class="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors" data-paper-id="${paper.paper_id || ""}" title="${rhxEsc(metadataTooltip)}">
<td class="px-6 py-4">
<div class="flex items-center gap-3">
<div class="w-8 h-8 bg-red-50 dark:bg-red-900/20 rounded flex items-center justify-center text-red-600">
<span class="material-symbols-outlined text-lg">picture_as_pdf</span>
</div>
<div class="min-w-0">
<p class="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">${hasExternalLink
        ? `<a class="inline-flex items-center gap-1 hover:text-primary transition-colors" href="${rhxEsc(url)}" rel="noopener noreferrer" target="_blank"><span class="material-symbols-outlined text-sm">open_in_new</span>${rhxEsc(title)}</a>`
        : rhxEsc(title)}</p>
<p class="text-[11px] text-slate-500">${rhxEsc(authorMeta)}</p>
</div>
</div>
</td>
<td class="px-6 py-4">
<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${sourceClass}">${source}</span>
</td>
<td class="px-6 py-4">
<span class="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400">
<span class="w-1.5 h-1.5 rounded-full bg-slate-300"></span>
Pending
</span>
</td>
<td class="px-6 py-4 text-right">
<div class="inline-flex items-center gap-2">
<button class="px-3 py-1.5 text-xs font-semibold text-primary border border-primary/20 hover:bg-primary/5 rounded-lg transition-colors test-view-summary-btn" data-paper-id="${paper.paper_id || ""}" type="button">View Summary</button>
<button class="px-3 py-1.5 text-xs font-semibold text-red-600 border border-red-200 hover:bg-red-50 rounded-lg transition-colors test-remove-btn" type="button">Remove</button>
</div>
</td>
</tr>`;
}

function renderPapersEmptyStateRow() {
    return `
<tr id="test-papers-empty-row">
<td class="px-6 py-8 text-sm text-slate-500" colspan="4">
No files here, you can
<a class="text-primary font-semibold hover:underline" href="/test#test-upload-dropzone">Upload</a>
or search
<a class="text-primary font-semibold hover:underline" href="/test/search">Arxiv files</a>.
</td>
</tr>`;
}

function getStoredWorkspaceName() {
    try {
        return localStorage.getItem(RHX_WORKSPACE_NAME_KEY);
    } catch (e) {
        return null;
    }
}

function setStoredWorkspaceName(name) {
    try {
        if (name) localStorage.setItem(RHX_WORKSPACE_NAME_KEY, name);
    } catch (e) {
        // ignore storage issues
    }
}

function getStoredWorkspaces() {
    try {
        const raw = localStorage.getItem(RHX_WORKSPACES_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        const normalized = normalizeWorkspaces(parsed);
        if (JSON.stringify(parsed) !== JSON.stringify(normalized)) {
            setStoredWorkspaces(normalized);
        }
        return normalized;
    } catch (e) {
        return [];
    }
}

function setStoredWorkspaces(workspaces) {
    try {
        localStorage.setItem(RHX_WORKSPACES_KEY, JSON.stringify(normalizeWorkspaces(workspaces || [])));
    } catch (e) {
        // ignore storage issues
    }
}

function upsertWorkspaceEntry(entry) {
    const current = getStoredWorkspaces();
    const sanitized = normalizeWorkspaces([entry])[0];
    if (!sanitized) return current;

    const idx = current.findIndex((w) => w.session_id === sanitized.session_id);
    if (idx >= 0) {
        current[idx] = sanitized;
    } else {
        const sameNameIdx = current.findIndex((w) => w.name.toLowerCase() === sanitized.name.toLowerCase());
        if (sameNameIdx >= 0) current[sameNameIdx] = sanitized;
        else current.push(sanitized);
    }
    const normalized = normalizeWorkspaces(current);
    setStoredWorkspaces(normalized);
    return normalized;
}

function renderNoMatchStateRow() {
    return `
<tr id="test-papers-empty-row">
<td class="px-6 py-8 text-sm text-slate-500" colspan="4">
No matching files found for current search/filter.
</td>
</tr>`;
}

function getFilteredSortedPapers() {
    let rows = [...papersTableData];

    if (papersFilterMode === "local") {
        rows = rows.filter((p) => p.source === "local_upload");
    } else if (papersFilterMode === "arxiv") {
        rows = rows.filter((p) => p.source === "arxiv_search");
    }

    if (papersSearchQuery) {
        const q = papersSearchQuery.toLowerCase();
        rows = rows.filter((p) => {
            const title = (p.title || "").toLowerCase();
            const authors = Array.isArray(p.authors) ? p.authors.join(" ").toLowerCase() : "";
            const source = (p.source || "").toLowerCase();
            return title.includes(q) || authors.includes(q) || source.includes(q);
        });
    }

    const toTs = (p) => {
        const ts = Date.parse(p.published || "");
        return Number.isNaN(ts) ? 0 : ts;
    };

    if (papersSortMode === "newest") {
        rows.sort((a, b) => toTs(b) - toTs(a));
    } else if (papersSortMode === "oldest") {
        rows.sort((a, b) => toTs(a) - toTs(b));
    } else if (papersSortMode === "title") {
        rows.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
    }

    return rows;
}

function renderPapersTableBody() {
    const tbody = document.getElementById("test-papers-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!papersTableData.length) {
        tbody.insertAdjacentHTML("beforeend", renderPapersEmptyStateRow());
        return;
    }

    const rows = getFilteredSortedPapers();
    if (!rows.length) {
        tbody.insertAdjacentHTML("beforeend", renderNoMatchStateRow());
        return;
    }

    tbody.insertAdjacentHTML("beforeend", rows.map((paper) => renderUploadedPaperRow(paper)).join(""));
}

function updatePapersControlLabels() {
    const filterBtn = document.getElementById("test-papers-filter-btn");
    const sortBtn = document.getElementById("test-papers-sort-btn");
    if (filterBtn) {
        const filterLabel = papersFilterMode === "all" ? "All" : (papersFilterMode === "local" ? "Local" : "arXiv");
        filterBtn.innerHTML = '<span class="material-symbols-outlined text-lg">filter_list</span>Filter: ' + filterLabel;
    }
    if (sortBtn) {
        const sortLabel = papersSortMode === "newest" ? "Newest" : (papersSortMode === "oldest" ? "Oldest" : "Title A-Z");
        sortBtn.innerHTML = '<span class="material-symbols-outlined text-lg">sort</span>Sort: ' + sortLabel;
    }
}

function initPapersControls() {
    const searchInput = document.getElementById("test-papers-search-input");
    const filterBtn = document.getElementById("test-papers-filter-btn");
    const sortBtn = document.getElementById("test-papers-sort-btn");
    if (!searchInput || !filterBtn || !sortBtn) return;

    updatePapersControlLabels();

    searchInput.addEventListener("input", () => {
        papersSearchQuery = (searchInput.value || "").trim();
        renderPapersTableBody();
    });

    filterBtn.addEventListener("click", () => {
        papersFilterMode = papersFilterMode === "all" ? "local" : (papersFilterMode === "local" ? "arxiv" : "all");
        updatePapersControlLabels();
        renderPapersTableBody();
    });

    sortBtn.addEventListener("click", () => {
        papersSortMode = papersSortMode === "newest" ? "oldest" : (papersSortMode === "oldest" ? "title" : "newest");
        updatePapersControlLabels();
        renderPapersTableBody();
    });
}

function initWorkspaceDropdown() {
    const dropdownRoot = document.getElementById("test-workspace-dropdown");
    const triggerEl = document.getElementById("test-workspace-trigger");
    const menuEl = document.getElementById("test-workspace-menu");
    const optionsRoot = document.getElementById("test-workspace-options");
    const newWsBtn = document.getElementById("test-new-workspace-btn");
    const resumeFileInput = document.getElementById("test-resume-file");
    const currentEl = document.getElementById("test-workspace-current");
    if (!dropdownRoot || !triggerEl || !menuEl || !optionsRoot || !newWsBtn || !currentEl) return;

    const closeMenu = () => {
        menuEl.classList.add("hidden");
    };
    const openMenu = () => {
        menuEl.classList.remove("hidden");
    };
    const toggleMenu = () => {
        if (menuEl.classList.contains("hidden")) openMenu();
        else closeMenu();
    };

    const applySelectedWorkspace = (name) => {
        currentEl.textContent = name;
        const optionEls = optionsRoot.querySelectorAll(".test-workspace-option");
        optionEls.forEach((el) => {
            const isActive = el.getAttribute("data-workspace-name") === name;
            el.classList.toggle("bg-slate-100", isActive);
            el.classList.toggle("dark:bg-slate-800", isActive);
            el.classList.toggle("font-medium", isActive);
            if (!isActive) {
                el.classList.remove("bg-slate-100", "dark:bg-slate-800", "font-medium");
            }
        });
    };

    const renderOptions = () => {
        const workspaces = getStoredWorkspaces();
        if (!workspaces.length) {
            optionsRoot.innerHTML = '<div class="px-2 py-2 text-xs text-slate-400">No workspaces yet.</div>';
            return;
        }
        optionsRoot.innerHTML = workspaces.map((w) => (
            `<button class="w-full text-left px-2 py-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 text-sm test-workspace-option" data-workspace-name="${rhxEsc(w.name)}" data-session-id="${rhxEsc(w.session_id)}" type="button">${rhxEsc(w.name)}</button>`
        )).join("");
    };

    const refreshWorkspaceViews = async () => {
        if (typeof window.syncTestKpis === "function") {
            await window.syncTestKpis();
        }
        await initRecentPapers();
        await initTestPapersTable();
        if (typeof window.syncResearchAISynthesis === "function") {
            await window.syncResearchAISynthesis();
        }
    };

    const initializeWorkspaceState = () => {
        let activeSession = rhxGetStoredSessionId();
        let activeName = getStoredWorkspaceName();
        let workspaces = getStoredWorkspaces();

        if (activeSession) {
            const existing = workspaces.find((w) => w.session_id === activeSession);
            if (!existing) {
                workspaces = upsertWorkspaceEntry({
                    session_id: activeSession,
                    name: activeName || "Quantum Computing Lab",
                });
            } else if (activeName && existing.name !== activeName) {
                existing.name = activeName;
                setStoredWorkspaces(workspaces);
            } else {
                activeName = existing.name;
            }
        } else if (workspaces.length) {
            activeSession = workspaces[0].session_id;
            activeName = workspaces[0].name;
            rhxSetStoredSessionId(activeSession);
            setStoredWorkspaceName(activeName);
        }

        renderOptions();
        if (activeName) applySelectedWorkspace(activeName);
    };

    initializeWorkspaceState();

    triggerEl.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleMenu();
    });

    document.addEventListener("click", (event) => {
        if (!dropdownRoot.contains(event.target)) closeMenu();
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeMenu();
    });

    optionsRoot.addEventListener("click", async (event) => {
        const el = event.target.closest(".test-workspace-option");
        if (!el) return;
        event.stopPropagation();
        const selected = el.getAttribute("data-workspace-name");
        const selectedSession = el.getAttribute("data-session-id");
        if (!selected || !selectedSession) return;
        applySelectedWorkspace(selected);
        setStoredWorkspaceName(selected);
        rhxSetStoredSessionId(selectedSession);
        closeMenu();
        await refreshWorkspaceViews();
    });

    newWsBtn.addEventListener("click", async (event) => {
        event.stopPropagation();
        const name = (window.prompt("Workspace name:", "Untitled Research") || "").trim();
        if (!name) return;

        try {
            const resp = await fetch("/api/workspace/create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            if (!resp.ok) return;
            const data = await resp.json();
            const entry = { session_id: data.session_id, name: data.name || name };
            upsertWorkspaceEntry(entry);
            rhxSetStoredSessionId(entry.session_id);
            setStoredWorkspaceName(entry.name);
            renderOptions();
            applySelectedWorkspace(entry.name);
            closeMenu();
            await refreshWorkspaceViews();
        } catch (e) {
            // ignore UI errors for now
        }
    });

    if (resumeFileInput) {
        resumeFileInput.addEventListener("change", async () => {
            const file = (resumeFileInput.files || [])[0];
            if (!file) return;

            try {
                const text = await file.text();
                const sessionData = JSON.parse(text);
                const resp = await fetch("/api/workspace/resume", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_data: sessionData }),
                });
                const result = await resp.json();
                if (!resp.ok || result.error) {
                    throw new Error(result.error || "Failed to resume session");
                }

                const entry = {
                    session_id: result.session_id,
                    name: result.topic || sessionData.topic || "Resumed Workspace",
                };
                upsertWorkspaceEntry(entry);
                rhxSetStoredSessionId(entry.session_id);
                setStoredWorkspaceName(entry.name);
                renderOptions();
                applySelectedWorkspace(entry.name);
                closeMenu();
                await refreshWorkspaceViews();
            } catch (e) {
                window.alert(`Import failed: ${(e && e.message) ? e.message : "Invalid session JSON"}`);
            } finally {
                resumeFileInput.value = "";
            }
        });
    }
}

async function initTestPapersTable() {
    const tbody = document.getElementById("test-papers-tbody");
    if (!tbody) return;
    papersTableData = [];
    renderPapersTableBody();

    const sessionId = rhxGetStoredSessionId();
    if (!sessionId) return;

    try {
        const resp = await fetch(`/api/workspace/${sessionId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        papersTableData = (data.papers || []).filter((p) => p.source === "local_upload" || p.source === "arxiv_search");
        renderPapersTableBody();
    } catch (e) {
        renderPapersTableBody();
    }
}

function renderRecentPaperItem(paper) {
    const title = rhxEsc(paper.title || "Untitled");
    const published = rhxEsc(paper.published || "");
    const meta = published ? `Added ${published}` : "Added recently";
    return `
<div class="flex gap-3 min-w-0 group">
<div class="shrink-0 w-10 h-10 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center justify-center text-red-600">
<span class="material-icons-outlined text-sm">picture_as_pdf</span>
</div>
<div class="min-w-0">
<p class="text-sm font-medium truncate group-hover:text-primary transition-colors">${title}</p>
<p class="text-[10px] text-slate-400">${meta}</p>
</div>
</div>`;
}

async function initRecentPapers() {
    const container = document.getElementById("test-recent-papers-list");
    if (!container) return;

    let sessionId = rhxGetStoredSessionId();
    if (!sessionId) {
        container.innerHTML = '<div class="text-xs text-slate-400">No uploaded papers yet.</div>';
        return;
    }

    try {
        let resp = await fetch(`/api/workspace/${sessionId}`);
        if (resp.status === 404) {
            sessionId = await rhxGetOrCreateSessionId();
            resp = await fetch(`/api/workspace/${sessionId}`);
        }
        if (!resp.ok) {
            container.innerHTML = '<div class="text-xs text-slate-400">No uploaded papers yet.</div>';
            return;
        }

        const data = await resp.json();
        const uploaded = (data.papers || []).filter((p) => p.source === "local_upload" || p.source === "arxiv_search");
        const recent = uploaded.slice(-3).reverse();
        if (!recent.length) {
            container.innerHTML = '<div class="text-xs text-slate-400">No uploaded papers yet.</div>';
            return;
        }
        container.innerHTML = recent.map(renderRecentPaperItem).join("");
    } catch (e) {
        container.innerHTML = '<div class="text-xs text-slate-400">No uploaded papers yet.</div>';
    }
}

async function removePaperRow(row, button) {
    const paperId = row.dataset.paperId;
    if (!paperId) {
        renderPapersTableBody();
        return;
    }

    let sessionId = rhxGetStoredSessionId();
    if (!sessionId) {
        papersTableData = papersTableData.filter((p) => String(p.paper_id || "") !== String(paperId));
        renderPapersTableBody();
        return;
    }

    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "Removing...";
    try {
        let resp = await fetch(`/api/workspace/${sessionId}/papers/${paperId}`, {
            method: "DELETE",
        });

        if (resp.status === 404) {
            // Session may be stale after server restart; attempt one recovery path.
            sessionId = await rhxGetOrCreateSessionId();
            resp = await fetch(`/api/workspace/${sessionId}/papers/${paperId}`, {
                method: "DELETE",
            });
        }

        // Remove in UI for successful delete, and also when stale data cannot be reconciled.
        if (resp.ok || resp.status === 404) {
            papersTableData = papersTableData.filter((p) => String(p.paper_id || "") !== String(paperId));
            renderPapersTableBody();
            return;
        }

        button.disabled = false;
        button.textContent = originalText;
    } catch (e) {
        // Keep UI responsive even if backend call fails unexpectedly.
        papersTableData = papersTableData.filter((p) => String(p.paper_id || "") !== String(paperId));
        renderPapersTableBody();
    }
}

function ensurePaperSummaryModal() {
    let modal = document.getElementById("test-paper-summary-modal");
    if (modal) return modal;

    modal = document.createElement("div");
    modal.id = "test-paper-summary-modal";
    modal.className = "fixed inset-0 bg-slate-950/60 z-[80] hidden items-center justify-center p-4";
    modal.innerHTML = `
<div class="w-full max-w-3xl max-h-[85vh] overflow-hidden rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 shadow-xl">
<div class="flex items-center justify-between px-5 py-4 border-b border-slate-100 dark:border-slate-800">
<h3 id="test-paper-summary-title" class="text-base font-semibold text-slate-900 dark:text-slate-100">Paper Summary</h3>
<button id="test-paper-summary-close" type="button" class="w-8 h-8 inline-flex items-center justify-center rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
<span class="material-symbols-outlined text-lg">close</span>
</button>
</div>
<div id="test-paper-summary-body" class="px-5 py-4 text-sm text-slate-700 dark:text-slate-300 leading-relaxed overflow-y-auto max-h-[calc(85vh-64px)] whitespace-pre-wrap"></div>
</div>`;

    document.body.appendChild(modal);
    modal.addEventListener("click", (event) => {
        if (event.target === modal) closePaperSummaryModal();
    });
    const closeBtn = document.getElementById("test-paper-summary-close");
    if (closeBtn) closeBtn.addEventListener("click", closePaperSummaryModal);
    return modal;
}

async function openPaperSummaryModalById(paperId) {
    const modal = ensurePaperSummaryModal();
    const titleEl = document.getElementById("test-paper-summary-title");
    const bodyEl = document.getElementById("test-paper-summary-body");
    if (!modal || !titleEl || !bodyEl) return;

    const paper = papersTableData.find((p) => String(p.paper_id || "") === String(paperId || ""));
    if (!paper) return;

    titleEl.textContent = paper.title || "Paper Summary";
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    document.body.classList.add("overflow-hidden");

    bodyEl.textContent = "Loading summary...";
    let summary = (paper.summary || "").trim();

    if (!summary && paper.source === "arxiv_search") {
        const sessionId = rhxGetStoredSessionId();
        if (sessionId && paper.paper_id) {
            try {
                const resp = await fetch(`/api/workspace/${sessionId}/papers/${paper.paper_id}/summary/refresh`, {
                    method: "POST",
                });
                const data = await resp.json();
                if (resp.ok && data.summary) {
                    summary = String(data.summary).trim();
                    paper.summary = summary;
                }
            } catch (e) {
                // no-op
            }
        }
    }

    bodyEl.textContent = summary || "Summary not available for this paper.";
}

function closePaperSummaryModal() {
    const modal = document.getElementById("test-paper-summary-modal");
    if (!modal) return;
    modal.classList.add("hidden");
    modal.classList.remove("flex");
    document.body.classList.remove("overflow-hidden");
}

function initPaperRemoveActions() {
    const tbody = document.getElementById("test-papers-tbody");
    if (!tbody) return;

    tbody.addEventListener("click", async (event) => {
        const summaryButton = event.target.closest(".test-view-summary-btn");
        if (summaryButton) {
            await openPaperSummaryModalById(summaryButton.dataset.paperId || "");
            return;
        }

        const button = event.target.closest(".test-remove-btn");
        if (!button) return;
        const row = button.closest("tr");
        if (!row) return;
        await removePaperRow(row, button);
    });
}

function renderSearchResultCard(item, index = -1) {
    const title = rhxEsc(item.title || "Untitled");
    const url = rhxEsc(item.url || "#");
    const authors = Array.isArray(item.authors) ? item.authors.map((a) => rhxEsc(a)).join(", ") : "";
    const published = rhxEsc(item.published || "");
    const summary = rhxEsc(item.summary || "");
    const categories = Array.isArray(item.categories) ? item.categories : [];
    const isAdded = item.url && addedPaperUrls.has(item.url);
    const addBtn = isAdded
        ? '<button class="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-200 text-slate-500 cursor-not-allowed" type="button" disabled>Added</button>'
        : `<button class="px-3 py-1.5 text-xs font-semibold rounded-lg bg-primary text-white hover:bg-secondary transition-colors test-add-paper-btn" data-index="${index}" type="button">+ Add</button>`;
    const catHtml = categories.length
        ? `<div class="mt-3 flex flex-wrap gap-2">${categories.map((c) => `<span class="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">${rhxEsc(c)}</span>`).join("")}</div>`
        : "";

    return `
<article class="p-4 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-primary/40 transition-colors">
<div class="flex items-start justify-between gap-3">
<a class="inline-flex items-center gap-1 text-base font-semibold text-slate-900 dark:text-slate-100 hover:text-primary transition-colors" href="${url}" rel="noopener noreferrer" target="_blank"><span class="material-symbols-outlined text-sm">open_in_new</span>${title}</a>
${addBtn}
</div>
<p class="text-xs text-slate-500 mt-1">${authors}${authors && published ? " â€¢ " : ""}${published}</p>
<p class="text-sm text-slate-600 dark:text-slate-300 mt-3 leading-relaxed">${summary}</p>
${catHtml}
</article>`;
}

async function refreshAddedPaperUrls() {
    addedPaperUrls = new Set();
    const sessionId = rhxGetStoredSessionId();
    if (!sessionId) return;

    try {
        const resp = await fetch(`/api/workspace/${sessionId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const papers = Array.isArray(data.papers) ? data.papers : [];
        for (const p of papers) {
            if (p && typeof p.url === "string" && p.url) addedPaperUrls.add(p.url);
        }
    } catch (e) {
        // no-op
    }
}

async function addPaperFromSearch(index) {
    const paper = testSearchResults[index];
    const statusEl = document.getElementById("test-search-status");
    const resultsEl = document.getElementById("test-search-results");
    if (!paper || !paper.url) return;
    if (addedPaperUrls.has(paper.url)) return;

    let sessionId = await rhxGetOrCreateSessionId();
    if (statusEl) statusEl.textContent = "Adding paper to workspace...";
    try {
        let resp = await fetch(`/api/workspace/${sessionId}/papers/add`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, paper }),
        });

        if (resp.status === 404) {
            sessionId = await rhxGetOrCreateSessionId();
            resp = await fetch(`/api/workspace/${sessionId}/papers/add`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, paper }),
            });
        }

        if (resp.ok || resp.status === 409) {
            addedPaperUrls.add(paper.url);
            if (resultsEl) resultsEl.innerHTML = testSearchResults.map((item, idx) => renderSearchResultCard(item, idx)).join("");
            if (statusEl) statusEl.textContent = "Paper added to workspace.";
            await initRecentPapers();
            if (typeof window.syncTestKpis === "function") window.syncTestKpis();
            return;
        }

        if (statusEl) statusEl.textContent = "Failed to add paper.";
    } catch (e) {
        if (statusEl) statusEl.textContent = "Failed to add paper.";
    }
}

async function runTestSearch() {
    const queryEl = document.getElementById("test-search-query");
    const sortEl = document.getElementById("test-search-sort");
    const fromEl = document.getElementById("test-search-from");
    const toEl = document.getElementById("test-search-to");
    const maxEl = document.getElementById("test-search-max");
    const statusEl = document.getElementById("test-search-status");
    const countEl = document.getElementById("test-search-count");
    const resultsEl = document.getElementById("test-search-results");
    const btnEl = document.getElementById("test-search-btn");

    if (!queryEl || !sortEl || !fromEl || !toEl || !maxEl || !statusEl || !countEl || !resultsEl || !btnEl) return;

    const query = queryEl.value.trim();
    if (!query) {
        statusEl.textContent = "Enter a query to search arXiv.";
        return;
    }

    btnEl.disabled = true;
    statusEl.textContent = "Searching...";
    resultsEl.innerHTML = '<div class="text-sm text-slate-500">Loading results...</div>';
    await refreshAddedPaperUrls();

    try {
        const resp = await fetch("/api/arxiv/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query,
                sort_by: sortEl.value || "relevance",
                date_from: fromEl.value || null,
                date_to: toEl.value || null,
                max_results: parseInt(maxEl.value, 10) || 10,
            }),
        });

        if (!resp.ok) {
            statusEl.textContent = "Search failed.";
            resultsEl.innerHTML = '<div class="text-sm text-red-500">Failed to fetch results.</div>';
            countEl.textContent = "0 results";
            return;
        }

        const data = await resp.json();
        const results = Array.isArray(data.results) ? data.results : [];
        testSearchResults = results;
        countEl.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;

        if (!results.length) {
            statusEl.textContent = "No results found.";
            resultsEl.innerHTML = '<div class="text-sm text-slate-500">No papers matched your filters.</div>';
            return;
        }

        statusEl.textContent = "Search complete.";
        resultsEl.innerHTML = results.map((item, idx) => renderSearchResultCard(item, idx)).join("");
    } catch (e) {
        statusEl.textContent = "Search failed.";
        resultsEl.innerHTML = '<div class="text-sm text-red-500">Unexpected error while searching.</div>';
        countEl.textContent = "0 results";
    } finally {
        btnEl.disabled = false;
    }
}

function initTestSearch() {
    const queryEl = document.getElementById("test-search-query");
    const btnEl = document.getElementById("test-search-btn");
    if (!queryEl || !btnEl) return;

    btnEl.addEventListener("click", () => {
        runTestSearch();
    });
    queryEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter") runTestSearch();
    });

    const resultsEl = document.getElementById("test-search-results");
    if (resultsEl) {
        resultsEl.addEventListener("click", async (event) => {
            const addBtn = event.target.closest(".test-add-paper-btn");
            if (!addBtn) return;
            const idx = parseInt(addBtn.getAttribute("data-index") || "-1", 10);
            if (Number.isNaN(idx) || idx < 0) return;

            addBtn.disabled = true;
            const original = addBtn.textContent;
            addBtn.textContent = "Adding...";
            await addPaperFromSearch(idx);
            if (!addedPaperUrls.has((testSearchResults[idx] || {}).url)) {
                addBtn.disabled = false;
                addBtn.textContent = original || "+ Add";
            }
        });
    }
}

let testResearchEvtSource = null;
let testActivityStepCounter = 0;
let testRunDurationTicker = null;
let testResearchMissionRunning = false;
let testResearchWorkspaceData = {
    gaps: [],
    weak_gaps: [],
    ideas: [],
    evidence_assessment: {},
    insufficient_evidence_message: "",
};

const RESEARCH_AGENT_ICON = {
    Planner: "settings",
    Researcher: "travel_explore",
    Analyst: "insights",
    Critic: "gavel",
};

function resetResearchAgentCards() {
    const cards = document.querySelectorAll(".rhx-agent-card");
    if (!cards.length) return;
    cards.forEach((card) => {
        card.classList.remove("active", "done");
        const statusEl = card.querySelector(".agent-status");
        if (statusEl) statusEl.textContent = "Idle";
        const dot = card.querySelector(".agent-dot");
        if (dot) {
            dot.classList.remove("bg-primary", "bg-emerald-500", "bg-rose-500", "animate-pulse");
            dot.classList.add("bg-slate-300");
        }
    });
}

function updateResearchAgentCard(agent, status) {
    if (status === "running") {
        document.querySelectorAll(".rhx-agent-card").forEach((card) => {
            if (card.id !== `agent-${agent}`) card.classList.remove("active");
        });
    }

    const card = document.getElementById(`agent-${agent}`);
    if (!card) return;

    const statusEl = card.querySelector(".agent-status");
    const dot = card.querySelector(".agent-dot");

    if (status === "running") {
        card.classList.add("active");
        card.classList.remove("done");
        if (statusEl) statusEl.textContent = "Working...";
        if (dot) {
            dot.classList.remove("bg-slate-300", "bg-emerald-500", "bg-rose-500");
            dot.classList.add("bg-primary", "animate-pulse");
        }
    } else if (status === "complete") {
        card.classList.remove("active");
        card.classList.add("done");
        if (statusEl) statusEl.textContent = "Done";
        if (dot) {
            dot.classList.remove("bg-slate-300", "bg-primary", "bg-rose-500", "animate-pulse");
            dot.classList.add("bg-emerald-500");
        }
    } else if (status === "error") {
        card.classList.remove("active");
        card.classList.remove("done");
        if (statusEl) statusEl.textContent = "Error";
        if (dot) {
            dot.classList.remove("bg-slate-300", "bg-primary", "bg-emerald-500", "animate-pulse");
            dot.classList.add("bg-rose-500");
        }
    }
}

function closeResearchMissionStream() {
    if (testResearchEvtSource) {
        testResearchEvtSource.close();
        testResearchEvtSource = null;
    }
}

function getResearchActivityLog() {
    return document.getElementById("test-activity-log");
}

function setResearchActivityBadge(label) {
    const badge = document.getElementById("test-activity-badge");
    const dot = document.getElementById("test-activity-badge-dot");
    if (badge) badge.textContent = label;
    if (dot) {
        dot.classList.remove("bg-slate-400", "bg-emerald-500", "bg-rose-500", "animate-pulse");
        if (label === "LIVE") dot.classList.add("bg-emerald-500", "animate-pulse");
        else if (label === "ERROR") dot.classList.add("bg-rose-500");
        else dot.classList.add("bg-slate-400");
    }
}

function researchStepLabel(step) {
    return {
        generate_plan: "Planning",
        search_papers: "Searching",
        extract_limitations: "Extracting",
        cluster_limitations: "Clustering",
        identify_gaps: "Analyzing",
        generate_ideas: "Ideating",
        reflect: "Reflecting",
        save_results: "Saving",
    }[step] || step;
}

function resetResearchActivityFeed() {
    const log = getResearchActivityLog();
    if (!log) return;
    testActivityStepCounter = 0;
    log.innerHTML = '<div class="test-activity-placeholder">Agent activity will appear here when a mission is running...</div>';
    setResearchActivityBadge("IDLE");
}

function getLatestResearchActivityStream(step) {
    const steps = document.querySelectorAll(`.test-activity-step[data-step="${step}"]`);
    if (!steps.length) return null;
    return steps[steps.length - 1].querySelector(".test-activity-stream");
}

function startResearchActivityStep(step, agent) {
    const log = getResearchActivityLog();
    if (!log) return;
    const placeholder = log.querySelector(".test-activity-placeholder");
    if (placeholder) placeholder.remove();

    testActivityStepCounter += 1;
    const entryId = `test-activity-step-${step}-${testActivityStepCounter}`;
    const agentKey = String(agent || "Planner");
    const icon = RESEARCH_AGENT_ICON[agentKey] || "smart_toy";
    const tagClass = `agent-${agentKey.toLowerCase()}`;

    const stepEl = document.createElement("div");
    stepEl.className = "test-activity-step";
    stepEl.dataset.step = step;
    stepEl.id = entryId;
    stepEl.innerHTML = `
        <div class="test-activity-step-header">
            <span class="test-activity-agent ${tagClass}">
                <span class="material-icons-outlined text-[12px]">${icon}</span>
                ${rhxEsc(agentKey)}
            </span>
            <span>${rhxEsc(researchStepLabel(step))}</span>
            <span class="test-activity-spinner material-icons-outlined text-sm">autorenew</span>
        </div>
        <div class="test-activity-stream"></div>
    `;
    log.appendChild(stepEl);
    log.scrollTop = log.scrollHeight;
    setResearchActivityBadge("LIVE");
}

function appendResearchActivityToken(step, agent, token) {
    let streamEl = getLatestResearchActivityStream(step);
    if (!streamEl) {
        startResearchActivityStep(step, agent);
        streamEl = getLatestResearchActivityStream(step);
    }
    if (!streamEl) return;

    const text = String(token ?? "");
    if (text.includes("\n")) {
        const parts = text.split("\n");
        parts.forEach((part, idx) => {
            if (idx > 0) streamEl.appendChild(document.createElement("br"));
            if (part) streamEl.appendChild(document.createTextNode(part));
        });
    } else {
        streamEl.appendChild(document.createTextNode(text));
    }

    const log = getResearchActivityLog();
    if (log) log.scrollTop = log.scrollHeight;
}

function completeResearchActivityStep(step, summary) {
    const steps = document.querySelectorAll(`.test-activity-step[data-step="${step}"]`);
    if (!steps.length) return;
    const stepEl = steps[steps.length - 1];
    stepEl.classList.add("completed");
    const spinner = stepEl.querySelector(".test-activity-spinner");
    if (spinner) spinner.textContent = "check_circle";

    if (summary) {
        const streamEl = stepEl.querySelector(".test-activity-stream");
        if (streamEl) {
            const summaryEl = document.createElement("div");
            summaryEl.className = "test-activity-summary";
            summaryEl.textContent = summary;
            streamEl.appendChild(summaryEl);
        }
    }

    const log = getResearchActivityLog();
    if (log) log.scrollTop = log.scrollHeight;
}

function appendResearchPlanDetails(data) {
    if (!data || !Array.isArray(data.subtopics) || !data.subtopics.length) return;
    const streamEl = getLatestResearchActivityStream("generate_plan");
    if (!streamEl) return;

    const keywords = Array.isArray(data.keywords) ? data.keywords : [];
    const detailEl = document.createElement("div");
    detailEl.className = "test-activity-plan-detail";
    detailEl.innerHTML = `
        <div><strong>Subtopics:</strong> ${data.subtopics.map((s) => rhxEsc(s)).join(" | ")}</div>
        <div class="mt-1"><strong>Keywords:</strong> ${keywords.map((k) => `<span class="test-plan-keyword">${rhxEsc(k)}</span>`).join("")}</div>
    `;
    streamEl.appendChild(detailEl);

    const log = getResearchActivityLog();
    if (log) log.scrollTop = log.scrollHeight;
}

function appendResearchActivityError(message) {
    const log = getResearchActivityLog();
    if (!log) return;
    const placeholder = log.querySelector(".test-activity-placeholder");
    if (placeholder) placeholder.remove();

    const errEl = document.createElement("div");
    errEl.className = "test-activity-error";
    errEl.textContent = message || "Mission failed.";
    log.appendChild(errEl);
    log.scrollTop = log.scrollHeight;
    setResearchActivityBadge("ERROR");
}

function finalizeResearchActivityFeed() {
    const log = getResearchActivityLog();
    if (!log) return;
    const doneEl = document.createElement("div");
    doneEl.className = "test-activity-done";
    doneEl.textContent = "Mission completed successfully.";
    log.appendChild(doneEl);
    log.scrollTop = log.scrollHeight;
    setResearchActivityBadge("DONE");
}

function formatResearchDuration(ms) {
    const totalSec = Math.max(0, Math.floor(ms / 1000));
    const hours = String(Math.floor(totalSec / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((totalSec % 3600) / 60)).padStart(2, "0");
    const seconds = String(totalSec % 60).padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
}

function updateResearchRunDurationDisplay() {
    const el = document.getElementById("test-run-duration-value");
    if (!el) return;
    const state = loadDeepReadState();
    el.textContent = formatResearchDuration(getDeepReadElapsedMs(state));
}

function startResearchRunDurationTicker() {
    if (testRunDurationTicker) clearInterval(testRunDurationTicker);
    updateResearchRunDurationDisplay();
    testRunDurationTicker = setInterval(updateResearchRunDurationDisplay, 1000);
}

function startResearchRunTimer(reset = false) {
    startDeepReadSession(reset);
    startResearchRunDurationTicker();
    startDeepReadTicker();
}

function pauseResearchRunTimer() {
    const state = pauseDeepReadSession();
    if (!state.is_running) stopDeepReadTicker();
    updateResearchRunDurationDisplay();
}

function stopResearchRunTimer() {
    const state = pauseDeepReadSession();
    if (!state.is_running) stopDeepReadTicker();
    updateResearchRunDurationDisplay();
}

function renderResearchSynthesisTabs(activeTab = "ideas") {
    const tabs = document.querySelectorAll("[data-synthesis-tab]");
    const panels = document.querySelectorAll("[data-synthesis-panel]");
    if (!tabs.length || !panels.length) return;

    tabs.forEach((tabEl) => {
        const tab = tabEl.getAttribute("data-synthesis-tab");
        const isActive = tab === activeTab;
        tabEl.classList.toggle("bg-white", isActive);
        tabEl.classList.toggle("dark:bg-slate-700", isActive);
        tabEl.classList.toggle("text-primary", isActive);
        tabEl.classList.toggle("text-slate-500", !isActive);
        tabEl.classList.toggle("dark:text-slate-300", !isActive);
    });

    panels.forEach((panelEl) => {
        const tab = panelEl.getAttribute("data-synthesis-panel");
        panelEl.classList.toggle("hidden", tab !== activeTab);
    });
}

function initResearchSynthesisTabs() {
    const tabs = document.querySelectorAll("[data-synthesis-tab]");
    if (!tabs.length) return;
    tabs.forEach((tabEl) => {
        tabEl.addEventListener("click", () => {
            renderResearchSynthesisTabs(tabEl.getAttribute("data-synthesis-tab") || "ideas");
        });
    });
    renderResearchSynthesisTabs("ideas");
}

function researchMarkdownToHtml(md) {
    let html = rhxEsc(md || "");
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\n/g, "<br>");
    return html;
}

function renderResearchIdeasSynthesis() {
    const listEl = document.getElementById("test-synthesis-ideas-list");
    if (!listEl) return;

    const ideas = Array.isArray(testResearchWorkspaceData.ideas) ? testResearchWorkspaceData.ideas : [];
    if (!ideas.length) {
        const message = testResearchWorkspaceData.insufficient_evidence_message
            || "No ideas generated yet. Run a mission to populate ideas.";
        listEl.innerHTML = `<li class="text-sm text-slate-500 dark:text-slate-400">${rhxEsc(message)}</li>`;
        return;
    }

    listEl.innerHTML = ideas.map((idea) => {
        const title = rhxEsc(idea.title || "Untitled idea");
        const description = rhxEsc(idea.description || "");
        return `
            <li class="text-sm flex gap-2">
                <span class="material-icons-outlined text-emerald-500 text-sm">lightbulb</span>
                <span><strong>${title}</strong>${description ? `: ${description}` : ""}</span>
            </li>
        `;
    }).join("");
}

function renderResearchGapsSynthesis() {
    const listEl = document.getElementById("test-synthesis-gaps-list");
    if (!listEl) return;

    const strong = Array.isArray(testResearchWorkspaceData.gaps) ? testResearchWorkspaceData.gaps : [];
    const weak = Array.isArray(testResearchWorkspaceData.weak_gaps) ? testResearchWorkspaceData.weak_gaps : [];
    const gaps = [...strong, ...weak];
    if (!gaps.length) {
        listEl.innerHTML = '<li class="text-sm text-slate-500 dark:text-slate-400">No research gaps identified yet. Run a mission first.</li>';
        return;
    }

    listEl.innerHTML = gaps.map((gap) => {
        const text = typeof gap === "string" ? gap : (gap.gap || "");
        return `
            <li class="text-sm flex gap-2">
                <span class="material-icons-outlined text-rose-400 text-sm">warning</span>
                <span>${rhxEsc(text)}</span>
            </li>
        `;
    }).join("");
}

function renderResearchSynthesisFromWorkspace() {
    renderResearchIdeasSynthesis();
    renderResearchGapsSynthesis();
}

async function syncResearchAISynthesis() {
    const listIdeas = document.getElementById("test-synthesis-ideas-list");
    const listGaps = document.getElementById("test-synthesis-gaps-list");
    if (!listIdeas && !listGaps) return;

    const sessionId = rhxGetStoredSessionId();
    if (!sessionId) {
        testResearchWorkspaceData = {
            gaps: [],
            weak_gaps: [],
            ideas: [],
            evidence_assessment: {},
            insufficient_evidence_message: "",
        };
        renderResearchSynthesisFromWorkspace();
        return;
    }

    try {
        const resp = await fetch(`/api/workspace/${sessionId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        testResearchWorkspaceData = {
            gaps: data.gaps || [],
            weak_gaps: data.weak_gaps || [],
            ideas: data.ideas || [],
            evidence_assessment: data.evidence_assessment || {},
            insufficient_evidence_message: data.insufficient_evidence_message || "",
        };
        renderResearchSynthesisFromWorkspace();
    } catch (e) {
        // Keep current UI if refresh fails.
    }
}

function updateSynthesisFromMissionEvent(payload) {
    if (!payload || !payload.step || !payload.data) return;

    if (payload.step === "identify_gaps") {
        if (Array.isArray(payload.data.gaps)) testResearchWorkspaceData.gaps = payload.data.gaps;
        if (Array.isArray(payload.data.weak_gaps)) testResearchWorkspaceData.weak_gaps = payload.data.weak_gaps;
    }
    if (payload.step === "generate_ideas") {
        if (Array.isArray(payload.data.ideas)) testResearchWorkspaceData.ideas = payload.data.ideas;
        if (typeof payload.data.message === "string") {
            testResearchWorkspaceData.insufficient_evidence_message = payload.data.message;
        }
    }
    renderResearchSynthesisFromWorkspace();
}

function initResearchBriefGeneration() {
    const btn = document.getElementById("test-generate-brief-btn");
    const wrap = document.getElementById("test-brief-wrap");
    const content = document.getElementById("test-brief-content");
    const status = document.getElementById("test-brief-status");
    if (!btn || !wrap || !content || !status) return;

    btn.addEventListener("click", async () => {
        const sessionId = rhxGetStoredSessionId();
        if (!sessionId) {
            wrap.classList.remove("hidden");
            status.textContent = "No workspace selected";
            content.innerHTML = "<p>Create/select a workspace and run a mission first.</p>";
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="material-icons-outlined text-sm animate-spin">autorenew</span>Generating...';
        wrap.classList.remove("hidden");
        status.textContent = "Generating brief...";

        try {
            const resp = await fetch(`/api/workspace/${sessionId}/report`, { method: "POST" });
            const data = await resp.json();
            if (!resp.ok || !data.report) {
                throw new Error("Failed to generate brief");
            }
            content.innerHTML = researchMarkdownToHtml(data.report);
            status.textContent = "Brief ready";
        } catch (e) {
            content.innerHTML = "<p>Could not generate research brief.</p>";
            status.textContent = "Failed";
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<span class="material-icons-outlined text-sm">description</span>Generate Brief';
        }
    });
}

window.syncResearchAISynthesis = syncResearchAISynthesis;

function initResearchAIRun() {
    const topicEl = document.getElementById("test-research-topic");
    const modeEl = document.getElementById("test-research-mission-mode");
    const runBtn = document.getElementById("test-research-run-btn");
    const statusEl = document.getElementById("test-research-run-status");
    const clearBtn = document.getElementById("test-activity-clear-btn");
    const pauseBtn = document.getElementById("test-run-pause-btn");
    const pauseIcon = document.getElementById("test-run-pause-icon");
    const pauseLabel = document.getElementById("test-run-pause-label");
    const abortBtn = document.getElementById("test-run-abort-btn");
    if (!topicEl || !modeEl || !runBtn) return;

    if (clearBtn && !clearBtn.dataset.bound) {
        clearBtn.dataset.bound = "1";
        clearBtn.addEventListener("click", () => {
            resetResearchActivityFeed();
        });
    }

    const setRunStatus = (text, isError = false) => {
        if (!statusEl) return;
        statusEl.textContent = text || "";
        statusEl.classList.toggle("text-red-600", !!isError);
        statusEl.classList.toggle("dark:text-red-400", !!isError);
        statusEl.classList.toggle("text-slate-500", !isError);
    };

    const setPauseUi = (paused) => {
        if (!pauseIcon || !pauseLabel) return;
        pauseIcon.textContent = paused ? "play_arrow" : "pause";
        pauseLabel.textContent = paused ? "Resume" : "Pause";
    };

    const syncPauseUiFromTimer = () => {
        const state = loadDeepReadState();
        setPauseUi(!state.is_running);
    };

    const finishRun = async (msg, isError = false) => {
        testResearchMissionRunning = false;
        stopResearchRunTimer();
        setPauseUi(false);
        runBtn.disabled = false;
        runBtn.innerHTML = '<span class="material-icons-outlined text-sm">play_arrow</span>Resume Run';
        setRunStatus(msg, isError);
        await initRecentPapers();
        await initTestPapersTable();
        if (typeof window.syncTestKpis === "function") {
            await window.syncTestKpis();
        }
    };

    runBtn.addEventListener("click", async () => {
        const topic = (topicEl.value || "").trim();
        if (!topic) {
            topicEl.focus();
            setRunStatus("Enter a research topic to start.", true);
            return;
        }

        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="material-icons-outlined text-sm">autorenew</span>Starting...';
        setRunStatus("Starting mission...");
        resetResearchAgentCards();
        resetResearchActivityFeed();
        startResearchRunTimer(true);
        testResearchMissionRunning = true;
        setPauseUi(false);

        let sessionId = await rhxGetOrCreateSessionId();
        closeResearchMissionStream();

        try {
            let startResp = await fetch(`/api/workspace/${sessionId}/mission/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    topic,
                    mission_mode: modeEl.value || "autonomous_search",
                }),
            });

            if (startResp.status === 404) {
                sessionId = await rhxGetOrCreateSessionId();
                startResp = await fetch(`/api/workspace/${sessionId}/mission/start`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        topic,
                        mission_mode: modeEl.value || "autonomous_search",
                    }),
                });
            }

            if (!startResp.ok) {
                throw new Error("Failed to start mission");
            }

            runBtn.innerHTML = '<span class="material-icons-outlined text-sm animate-spin">autorenew</span>Running...';
            setRunStatus("Mission running...");

            testResearchEvtSource = new EventSource(`/api/workspace/${sessionId}/mission/stream`);
            testResearchEvtSource.onmessage = async (event) => {
                let payload = null;
                try {
                    payload = JSON.parse(event.data);
                } catch (e) {
                    return;
                }

                if (payload?.type === "token") {
                    appendResearchActivityToken(payload.step, payload.agent, payload.token);
                    return;
                }

                if (payload?.agent && payload?.status) {
                    updateResearchAgentCard(payload.agent, payload.status);
                }

                if (payload?.status === "running" && payload?.step && payload?.agent) {
                    startResearchActivityStep(payload.step, payload.agent);
                }

                if (payload?.status === "complete" && payload?.step) {
                    completeResearchActivityStep(payload.step, payload.summary || "");
                    if (payload.step === "generate_plan" && payload.data) {
                        appendResearchPlanDetails(payload.data);
                    }
                    updateSynthesisFromMissionEvent(payload);
                }

                if (payload?.status === "error") {
                    appendResearchActivityError(payload.summary || "Mission failed.");
                    await finishRun(payload.summary || "Mission failed.", true);
                    closeResearchMissionStream();
                    return;
                }

                if (payload?.step === "done") {
                    finalizeResearchActivityFeed();
                    await syncResearchAISynthesis();
                    await finishRun("Mission complete.");
                    closeResearchMissionStream();
                }
            };

            testResearchEvtSource.onerror = async () => {
                if (!testResearchEvtSource) return;
                await finishRun("Mission stream disconnected.", true);
                closeResearchMissionStream();
            };
        } catch (e) {
            await finishRun("Failed to start mission.", true);
            closeResearchMissionStream();
        }
    });

    topicEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter") runBtn.click();
    });

    if (pauseBtn && !pauseBtn.dataset.bound) {
        pauseBtn.dataset.bound = "1";
        pauseBtn.addEventListener("click", () => {
            if (!testResearchMissionRunning) return;
            const state = loadDeepReadState();
            if (state.is_running) {
                pauseResearchRunTimer();
                setPauseUi(true);
            } else {
                startResearchRunTimer(false);
                setPauseUi(false);
            }
        });
    }

    if (abortBtn && !abortBtn.dataset.bound) {
        abortBtn.dataset.bound = "1";
        abortBtn.addEventListener("click", async () => {
            if (!testResearchMissionRunning && !testResearchEvtSource) return;
            closeResearchMissionStream();
            appendResearchActivityError("Mission aborted from UI.");
            await finishRun("Mission aborted.", true);
        });
    }

    startResearchRunDurationTicker();
    syncPauseUiFromTimer();

    window.addEventListener("storage", (event) => {
        if (event.key !== RHX_DEEP_READ_KEY) return;
        updateResearchRunDurationDisplay();
        syncPauseUiFromTimer();
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initWorkspaceDropdown();
    initDeepReadSessionTimer();
    ensurePaperSummaryModal();
    initPapersControls();
    initTestUpload();
    initTestPapersTable();
    initPaperRemoveActions();
    initRecentPapers();
    initTestSearch();
    initResearchSynthesisTabs();
    initResearchBriefGeneration();
    initResearchAIRun();
    syncResearchAISynthesis();
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePaperSummaryModal();
});
