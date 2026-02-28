const RHX_ACTIVE_SESSION_KEY = "researchpilot.active_session_id";
let testSearchResults = [];
let addedPaperUrls = new Set();

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
    const published = paper.published || "";
    const authors = Array.isArray(paper.authors) ? paper.authors.join(", ") : "";
    const source = paper.source === "arxiv_search" ? "arXiv" : "Local";
    const sourceClass = paper.source === "arxiv_search"
        ? "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400"
        : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400";
    const leftMeta = authors || (paper.source === "arxiv_search" ? "arXiv" : "Local Upload");
    const authorMeta = `${leftMeta}${published ? ` • ${published}` : ""}`;

    return `
<tr class="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors" data-paper-id="${paper.paper_id || ""}">
<td class="px-6 py-4">
<div class="flex items-center gap-3">
<div class="w-8 h-8 bg-red-50 dark:bg-red-900/20 rounded flex items-center justify-center text-red-600">
<span class="material-symbols-outlined text-lg">picture_as_pdf</span>
</div>
<div class="min-w-0">
<p class="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">${rhxEsc(title)}</p>
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
<button class="px-3 py-1.5 text-xs font-semibold text-primary border border-primary/20 hover:bg-primary/5 rounded-lg transition-colors">View Summary</button>
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

function ensurePapersEmptyState() {
    const tbody = document.getElementById("test-papers-tbody");
    if (!tbody) return;
    const existingEmpty = tbody.querySelector("#test-papers-empty-row");
    const hasDataRows = tbody.querySelectorAll("tr:not(#test-papers-empty-row)").length > 0;
    if (!hasDataRows && !existingEmpty) {
        tbody.insertAdjacentHTML("beforeend", renderPapersEmptyStateRow());
    } else if (hasDataRows && existingEmpty) {
        existingEmpty.remove();
    }
}

async function initTestPapersTable() {
    const tbody = document.getElementById("test-papers-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const sessionId = rhxGetStoredSessionId();
    if (!sessionId) {
        ensurePapersEmptyState();
        return;
    }

    try {
        const resp = await fetch(`/api/workspace/${sessionId}`);
        if (!resp.ok) {
            ensurePapersEmptyState();
            return;
        }
        const data = await resp.json();
        const uploaded = (data.papers || []).filter((p) => p.source === "local_upload" || p.source === "arxiv_search");
        if (!uploaded.length) {
            ensurePapersEmptyState();
            return;
        }

        const html = uploaded.map((paper) => renderUploadedPaperRow(paper)).join("");
        tbody.insertAdjacentHTML("beforeend", html);
        ensurePapersEmptyState();
    } catch (e) {
        ensurePapersEmptyState();
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
        row.remove();
        ensurePapersEmptyState();
        return;
    }

    let sessionId = rhxGetStoredSessionId();
    if (!sessionId) {
        row.remove();
        ensurePapersEmptyState();
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
            row.remove();
            ensurePapersEmptyState();
            return;
        }

        button.disabled = false;
        button.textContent = originalText;
    } catch (e) {
        // Keep UI responsive even if backend call fails unexpectedly.
        row.remove();
        ensurePapersEmptyState();
    }
}

function initPaperRemoveActions() {
    const tbody = document.getElementById("test-papers-tbody");
    if (!tbody) return;

    tbody.addEventListener("click", async (event) => {
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
<a class="text-base font-semibold text-slate-900 dark:text-slate-100 hover:text-primary transition-colors" href="${url}" rel="noopener noreferrer" target="_blank">${title}</a>
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

document.addEventListener("DOMContentLoaded", () => {
    initTestUpload();
    initTestPapersTable();
    initPaperRemoveActions();
    initRecentPapers();
    initTestSearch();
});

