const uploadForm = document.getElementById("upload-form");
const queryForm = document.getElementById("query-form");
const uploadResult = document.getElementById("upload-result");
const chatResult = document.getElementById("chat-result");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const filesInput = document.getElementById("files");
  const formData = new FormData();
  for (const file of filesInput.files) {
    formData.append("files", file);
  }
  uploadResult.textContent = "Uploading...";
  const response = await fetch("/ingest", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  uploadResult.textContent = JSON.stringify(payload, null, 2);
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = document.getElementById("query").value.trim();
  if (!query) {
    return;
  }
  chatResult.innerHTML = "<strong>Working...</strong>";
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const payload = await response.json();
  const answer = escapeHtml(payload.answer || "");
  const rewrittenQuery = escapeHtml(payload.rewritten_query || "-");
  const citations = (payload.citations || [])
    .map((citation, index) => {
      return `<li>[${index + 1}] ${escapeHtml(citation.filename)}, pages ${citation.page_start}-${citation.page_end}, score ${citation.score}</li>`;
    })
    .join("");
  chatResult.innerHTML = `
    <strong>Answer</strong>
    <p>${answer}</p>
    <strong>Intent</strong>
    <p>${escapeHtml(payload.intent)}</p>
    <strong>Rewritten Query</strong>
    <p>${rewrittenQuery}</p>
    <strong>Citations</strong>
    <ul>${citations || "<li>No citations returned</li>"}</ul>
  `;
});
