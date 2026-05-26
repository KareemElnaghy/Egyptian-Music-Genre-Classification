const GENRE_META = {
  "Tarab": { color: "#2196F3", icon: "🎼" },
  "Egyptian Pop": { color: "#FF9800", icon: "🎤" },
  "Mahraganat": { color: "#E91E63", icon: "🎉" },
  "Shaabi": { color: "#4CAF50", icon: "🥁" },
  "Egyptian Rap": { color: "#9C27B0", icon: "🎤" },
};

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileNameEl = document.getElementById("file-name");
const classifyBtn = document.getElementById("classify-btn");
const btnText = document.getElementById("btn-text");
const spinner = document.querySelector(".spinner");
const resultsEl = document.getElementById("results");
const errorBox = document.getElementById("error-box");
const errorMsg = document.getElementById("error-msg");

let selectedFile = null;

// File Selection

function selectFile(file) {
  if (!file) return;
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["wav", "mp3"].includes(ext)) {
    showError("Unsupported format. Please select a .wav or .mp3 file.");
    return;
  }
  selectedFile = file;
  fileNameEl.textContent = file.name;
  classifyBtn.disabled = false;
  hideError();
  hideResults();
}

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  selectFile(e.dataTransfer.files[0]);
});

// Classify 

classifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();
  hideResults();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/classify", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `Server error (${res.status})`);
      return;
    }

    renderResult(data);
  } catch (err) {
    showError(
      err instanceof TypeError
        ? "Cannot reach the server. Make sure uvicorn is running on http://localhost:8000."
        : err.message
    );
  } finally {
    setLoading(false);
  }
});

// Render Result

function renderResult(data) {
  const genre = data.predicted_genre;
  const scores = data.scores;
  const meta = GENRE_META[genre] || { color: "#607D8B", icon: "🎵" };

  document.getElementById("genre-icon").textContent = meta.icon;
  document.getElementById("genre-name").textContent = genre;
  document.getElementById("genre-name").style.color = meta.color;
  document.getElementById("confidence-pct").textContent =
    (data.confidence * 100).toFixed(1) + "%";
  document.getElementById("clips-note").textContent =
    `Based on ${data.clips_analyzed} clip${data.clips_analyzed !== 1 ? "s" : ""} analyzed`;

  const chart = document.getElementById("bar-chart");
  chart.innerHTML = "";

  for (const [cls, prob] of Object.entries(scores).sort((a, b) => b[1] - a[1])) {
    const m = GENRE_META[cls] || { color: "#607D8B", icon: "🎵" };
    const pct = (prob * 100).toFixed(1);

    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label">
        <span>${cls}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:0%;background:${m.color}">
          <span class="bar-pct">${pct}%</span>
        </div>
      </div>`;
    chart.appendChild(row);
  }

  resultsEl.style.display = "block";

  // Animate bars after a tick
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      chart.querySelectorAll(".bar-fill").forEach((fill, i) => {
        const label = fill.closest(".bar-row").querySelector(".bar-label span:last-child").textContent;
        const prob = scores[label] ?? 0;
        fill.style.width = (prob * 100).toFixed(1) + "%";
        fill.querySelector(".bar-pct").style.opacity = "1";
      });
    });
  });
}

// Helper functions
function setLoading(on) {
  classifyBtn.disabled = on;
  spinner.style.display = on ? "block" : "none";
  btnText.textContent = on ? "Classifying…" : "Classify";
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBox.style.display = "block";
}

function hideError()   { errorBox.style.display = "none"; }
function hideResults() { resultsEl.style.display = "none"; }
