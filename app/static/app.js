const uploadForm = document.getElementById("upload-form");
const queryForm = document.getElementById("query-form");
const filesInput = document.getElementById("files");
const resetButton = document.getElementById("reset-button");
const queryInput = document.getElementById("query");
const processingList = document.getElementById("processing-list");
const processingBadge = document.getElementById("processing-badge");
const uploadResult = document.getElementById("upload-result");
const fileList = document.getElementById("file-list");
const historyList = document.getElementById("history-list");
const viewer = document.getElementById("viewer");

let indexedFiles = [];
let sessions = [];
let activeSessionId = "";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatAnswerParagraphs(value) {
  return String(value)
    .split(/\n{2,}/)
    .map((segment) => segment.trim())
    .filter(Boolean);
}

function splitAnswerSections(answer) {
  return String(answer)
    .split(/\n{2,}/)
    .map((section) => section.trim())
    .filter(Boolean);
}

function setProcessingState(state) {
  const steps = Array.from(processingList.children);
  processingBadge.classList.remove("is-busy", "is-ready");

  const activeMap = {
    idle: { badge: "Idle", active: 0, done: [] },
    selected: { badge: "Queued", active: 0, done: [] },
    uploading: { badge: "Uploading", active: 1, done: [0] },
    extracting: { badge: "Processing", active: 2, done: [0, 1] },
    indexing: { badge: "Indexing", active: 3, done: [0, 1, 2] },
    ready: { badge: "Ready", active: null, done: [0, 1, 2, 3] },
  };

  const current = activeMap[state] || activeMap.idle;
  processingBadge.textContent = current.badge;
  if (state === "ready") {
    processingBadge.classList.add("is-ready");
  } else if (state !== "idle") {
    processingBadge.classList.add("is-busy");
  }

  steps.forEach((step, index) => {
    step.classList.remove("is-active", "is-done");
    if (current.done.includes(index)) {
      step.classList.add("is-done");
    } else if (current.active === index) {
      step.classList.add("is-active");
    }
  });
}

function renderIndexedFiles() {
  if (!indexedFiles.length) {
    fileList.innerHTML = '<div class="empty-state">No files indexed yet.</div>';
    return;
  }

  fileList.innerHTML = indexedFiles
    .map(
      (file) => `
        <article class="file-card">
          <div class="file-title">${escapeHtml(file.filename)}</div>
          <div class="file-meta">Document ID ${file.document_id}</div>
          <div class="file-meta">${file.chunks_created} chunks indexed</div>
        </article>
      `
    )
    .join("");
}

function extractCitationLabels(text) {
  const matches = text.match(/\[([^\]]+)\]/g) || [];
  return matches
    .flatMap((match) =>
      match
        .slice(1, -1)
        .split(",")
        .map((part) => part.trim())
        .filter((part) => /^S\d+\s+p\.\d+(?:-\d+)?$/.test(part))
    );
}

function stripCitationLabels(text) {
  return text.replace(/\s*\[([^\]]+)\]/g, "").trim();
}

function splitAnswerIntoSegments(answer) {
  const paragraphs = formatAnswerParagraphs(answer);
  const segments = [];

  paragraphs.forEach((paragraph) => {
    const lines = paragraph
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    lines.forEach((line) => {
      segments.push({
        text: stripCitationLabels(line),
        labels: extractCitationLabels(line),
      });
    });
  });

  return segments.length ? segments : [{ text: answer || "No answer returned.", labels: [] }];
}

function inlineChipMarkup(labels) {
  const uniqueLabels = [...new Set(labels)].filter(Boolean);
  if (!uniqueLabels.length) {
    return "";
  }
  return `
    <span class="answer-inline-citations">
      ${uniqueLabels
        .map(
          (label) =>
            `<button type="button" class="inline-chip" data-labels="${escapeHtml(label)}">${escapeHtml(citationDisplayLabel(label))}</button>`
        )
        .join("")}
    </span>
  `;
}

function renderRichLine(rawText, fallbackLabels = []) {
  const labels = extractCitationLabels(rawText);
  const resolvedLabels = labels.length ? labels : fallbackLabels;
  const cleaned = escapeHtml(stripCitationLabels(rawText));
  return `
    <div class="answer-rich-line">
      ${cleaned.replace(/\n/g, "<br>")}
      ${inlineChipMarkup(resolvedLabels)}
    </div>
  `;
}

function formatAnswerAsHtml(answer, citations) {
  const sections = splitAnswerSections(answer);
  const citationLabels = citations.map((citation) => citation.label);

  if (!sections.length) {
    return '<div class="answer-block"><p>No answer returned.</p></div>';
  }

  return sections
    .map((section) => {
      const labels = citationLabels.filter((label) => section.includes(`[${label}]`));

      if (section.includes("\n- ")) {
        const lines = section
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);
        const first = lines[0].startsWith("- ") ? "" : renderRichLine(lines[0], labels);
        const bullets = lines
          .filter((line) => line.startsWith("- "))
          .map((line) => `<li>${renderRichLine(line.replace(/^- /, ""), labels)}</li>`)
          .join("");

        return `
          <div class="answer-block answer-section answer-segment" data-labels="${escapeHtml(labels.join("|"))}">
            ${first}
            <ul>${bullets}</ul>
          </div>
        `;
      }

      return `
        <div class="answer-block answer-section answer-segment" data-labels="${escapeHtml(labels.join("|"))}">
          ${renderRichLine(section, labels)}
        </div>
      `;
    })
    .join("");
}

function labelsFromNode(node) {
  return (node.dataset.labels || "")
    .split("|")
    .map((label) => label.trim())
    .filter(Boolean);
}

function bindInteractiveHighlights(container) {
  const segments = Array.from(container.querySelectorAll(".answer-segment"));
  const cards = Array.from(container.querySelectorAll(".citation-card"));
  const chips = Array.from(container.querySelectorAll(".inline-chip"));
  let lockedLabel = "";

  function activate(labels) {
    const activeLabels = Array.isArray(labels) ? labels : [labels];
    segments.forEach((segment) => {
      segment.classList.toggle(
        "is-active",
        activeLabels.some((label) => labelsFromNode(segment).includes(label))
      );
    });
    cards.forEach((card) => {
      card.classList.toggle(
        "is-active",
        activeLabels.some((label) => labelsFromNode(card).includes(label))
      );
    });
    chips.forEach((chip) => {
      chip.classList.toggle(
        "is-active",
        activeLabels.some((label) => labelsFromNode(chip).includes(label))
      );
    });
  }

  function clear() {
    if (lockedLabel) {
      activate([lockedLabel]);
      return;
    }
    segments.forEach((segment) => segment.classList.remove("is-active"));
    cards.forEach((card) => card.classList.remove("is-active"));
    chips.forEach((chip) => chip.classList.remove("is-active"));
  }

  function lock(label) {
    lockedLabel = lockedLabel === label ? "" : label;
    activate(lockedLabel ? [lockedLabel] : []);
  }

  function openCitation(card, shouldOpen) {
    if (shouldOpen) {
      card.classList.add("is-open");
    } else if (!card.classList.contains("is-pinned")) {
      card.classList.remove("is-open");
    }
  }

  segments.forEach((segment) => {
    segment.addEventListener("mouseenter", () => activate(labelsFromNode(segment)));
    segment.addEventListener("mouseleave", clear);
  });

  chips.forEach((chip) => {
    chip.addEventListener("mouseenter", () => activate(labelsFromNode(chip)));
    chip.addEventListener("mouseleave", clear);
    chip.addEventListener("click", () => lock(labelsFromNode(chip)[0] || ""));
  });

  cards.forEach((card) => {
    card.addEventListener("mouseenter", () => {
      activate(labelsFromNode(card));
      openCitation(card, true);
    });
    card.addEventListener("mouseleave", () => {
      clear();
      openCitation(card, false);
    });
    card.addEventListener("click", () => {
      const label = labelsFromNode(card)[0] || "";
      const willPin = !card.classList.contains("is-pinned");
      cards.forEach((other) => {
        other.classList.remove("is-pinned");
        if (other !== card) {
          other.classList.remove("is-open");
        }
      });
      if (willPin) {
        card.classList.add("is-pinned");
        card.classList.add("is-open");
        lock(label);
      } else {
        card.classList.remove("is-pinned");
        card.classList.remove("is-open");
        lock(label);
      }
    });
  });
}

function renderHistory() {
  if (!sessions.length) {
    historyList.innerHTML = '<div class="empty-state">Your asked questions will appear here.</div>';
    return;
  }

  historyList.innerHTML = sessions
    .map(
      (session) => `
        <article class="history-item ${session.id === activeSessionId ? "is-active" : ""}" data-session-id="${escapeHtml(session.id)}">
          <div class="history-title">${escapeHtml(session.query)}</div>
          <div class="history-meta">${escapeHtml(session.payload.intent || "-")} · ${session.payload.citations?.length || 0} citations</div>
        </article>
      `
    )
    .join("");

  historyList.querySelectorAll(".history-item").forEach((item) => {
    item.addEventListener("click", () => {
      activeSessionId = item.dataset.sessionId;
      renderHistory();
      renderViewer();
    });
  });
}

function citationDisplayLabel(label) {
  const match = label.match(/^S\d+\s+(p\.\d+(?:-\d+)?)$/);
  return match ? match[1] : label;
}

function buildViewerCard(session) {
  const payload = session.payload;
  const citations = payload.citations || [];
  const answerMarkup = formatAnswerAsHtml(payload.answer || "", citations);

  const citationMarkup = citations.length
    ? citations
        .map(
          (citation) => `
            <article class="citation-card" data-labels="${escapeHtml(citation.label)}">
              <div class="citation-title">${escapeHtml(citation.filename)} · ${escapeHtml(citationDisplayLabel(citation.label))}</div>
              <div class="citation-meta">Pages ${citation.page_start}-${citation.page_end} · score ${citation.score}</div>
              <div class="citation-preview">${escapeHtml((citation.snippet || "").slice(0, 220))}${(citation.snippet || "").length > 220 ? "..." : ""}</div>
              <div class="citation-full">${escapeHtml(citation.snippet || "")}</div>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">No citations returned for this answer.</div>';

  return `
    <article class="viewer-card">
      <div class="viewer-head">
        <div>
          <p class="panel-kicker">Current Answer</p>
          <h3>${escapeHtml(session.query)}</h3>
        </div>
        <div class="question-chip">${escapeHtml(payload.intent || "-")}</div>
      </div>
      <div class="viewer-layout">
        <section class="answer-column">
          <div class="answer-body">${answerMarkup}</div>
          <div class="meta-row">
            <span class="meta-pill">${payload.insufficient_evidence ? "Insufficient evidence" : "Evidence available"}</span>
            <span class="meta-pill">${citations.length} citations</span>
          </div>
          <div class="query-trace">
            <strong>Retrieval query</strong>
            <div>${escapeHtml(payload.rewritten_query || "No rewritten query returned.")}</div>
          </div>
        </section>
        <aside class="citation-column">
          <section class="citations-section">
            <div class="viewer-head">
              <div>
                <p class="panel-kicker">Citations</p>
                <h3>Matching evidence</h3>
              </div>
            </div>
            ${citationMarkup}
          </section>
        </aside>
      </div>
    </article>
  `;
}

function renderViewer() {
  const activeSession = sessions.find((session) => session.id === activeSessionId);
  if (!activeSession) {
    viewer.innerHTML = '<div class="results-empty">Ask a question after ingestion finishes. The answer and citations will appear here.</div>';
    return;
  }

  viewer.innerHTML = buildViewerCard(activeSession);
  bindInteractiveHighlights(viewer);
}

filesInput.addEventListener("change", () => {
  if (filesInput.files.length) {
    setProcessingState("selected");
    uploadResult.textContent = `${filesInput.files.length} file(s) selected. Click "Start ingestion" to process them.`;
  } else {
    setProcessingState("idle");
    uploadResult.textContent = "Upload a PDF to begin. The pipeline status will update here.";
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!filesInput.files.length) {
    uploadResult.textContent = "Choose at least one PDF before starting ingestion.";
    return;
  }

  const formData = new FormData();
  for (const file of filesInput.files) {
    formData.append("files", file);
  }

  setProcessingState("uploading");
  uploadResult.textContent = "Uploading files to the API...";

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      body: formData,
    });

    setProcessingState("extracting");
    uploadResult.textContent = "PDFs uploaded. Extracting text and building chunks...";

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed.");
    }

    setProcessingState("indexing");
    uploadResult.textContent = "Creating embeddings and storing the searchable index...";

    indexedFiles = payload.ingested || [];
    renderIndexedFiles();
    setProcessingState("ready");
    uploadResult.innerHTML = indexedFiles
      .map((file) => `${escapeHtml(file.filename)} is ready with ${file.chunks_created} chunks.`)
      .join("<br>");
  } catch (error) {
    setProcessingState("idle");
    uploadResult.textContent = error.message;
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }
  queryInput.value = "";

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Query failed.");
    }

    const session = {
      id: `session-${Date.now()}`,
      query,
      payload,
    };
    sessions = [session, ...sessions];
    activeSessionId = session.id;
    renderHistory();
    renderViewer();
  } catch (error) {
    const session = {
      id: `session-${Date.now()}`,
      query,
      payload: {
        answer: error.message,
        intent: "error",
        rewritten_query: "Query failed before retrieval details were returned.",
        citations: [],
        insufficient_evidence: true,
      },
    };
    sessions = [session, ...sessions];
    activeSessionId = session.id;
    renderHistory();
    renderViewer();
  }
});

resetButton.addEventListener("click", async () => {
  try {
    const response = await fetch("/reset", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Reset failed.");
    }

    indexedFiles = [];
    sessions = [];
    activeSessionId = "";
    renderIndexedFiles();
    renderHistory();
    renderViewer();
    setProcessingState("idle");
    uploadResult.textContent = "Index cleared. Start again from Upload PDFs.";
  } catch (error) {
    uploadResult.textContent = error.message;
  }
});

renderIndexedFiles();
renderHistory();
renderViewer();
