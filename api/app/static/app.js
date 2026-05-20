const form = document.querySelector("#job-form");
const fileInput = document.querySelector("#file");
const fileName = document.querySelector("#file-name");
const jobIdInput = document.querySelector("#job-id");
const checkStatusButton = document.querySelector("#check-status");
const statusCard = document.querySelector("#status-card");
const resultSection = document.querySelector("#results-section");
const resultGrid = document.querySelector("#result-grid");
const toast = document.querySelector("#toast");

let pollTimer = null;

function setHref(selector, href) {
  const element = document.querySelector(selector);
  if (element) {
    element.href = href;
  }
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3800);
}

function updateTimeline(status) {
  const order = ["PENDING", "RUNNING", "SUCCESS"];
  document.querySelectorAll(".timeline-item").forEach((item) => {
    const step = item.dataset.step;
    item.classList.remove("active", "done");
    if (step === status) item.classList.add("active");
    if (order.indexOf(step) < order.indexOf(status)) item.classList.add("done");
    if (status === "FAILED" && step === "RUNNING") item.classList.add("active");
  });
}

function renderStatus(payload) {
  updateTimeline(payload.status);
  statusCard.innerHTML = `
    <dl>
      <div>
        <dt>Job ID</dt>
        <dd>${payload.job_id}</dd>
      </div>
      <div>
        <dt>Status</dt>
        <dd>${payload.status}</dd>
      </div>
      <div>
        <dt>Started</dt>
        <dd>${payload.start_time || "Waiting"}</dd>
      </div>
      <div>
        <dt>Finished</dt>
        <dd>${payload.end_time || "Not finished"}</dd>
      </div>
    </dl>
  `;
}

function renderResults(jobId, payload) {
  resultSection.classList.remove("hidden");
  resultGrid.innerHTML = `
    <div class="result-card">
      <span>Best model</span>
      <strong>${payload.best_model_name}</strong>
    </div>
    <div class="result-card">
      <span>Score</span>
      <strong>${Number(payload.best_model_score).toFixed(4)}</strong>
    </div>
    <div class="result-card">
      <span>Metric</span>
      <strong>${payload.evaluation_metric}</strong>
    </div>
    <div class="result-card">
      <span>MLflow run</span>
      <strong>${payload.mlflow_run_id}</strong>
    </div>
  `;
  setHref("#summary-link", `/jobs/${jobId}/artifacts/summary_report.html`);
  setHref("#profile-link", `/jobs/${jobId}/artifacts/data_profile.html`);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with HTTP ${response.status}`);
  }
  return data;
}

async function refreshStatus() {
  const jobId = jobIdInput.value.trim();
  if (!jobId) {
    showToast("Paste or submit a job ID first.", true);
    return;
  }
  try {
    const status = await fetchJson(`/jobs/${jobId}`);
    renderStatus(status);
    if (status.status === "SUCCESS") {
      window.clearInterval(pollTimer);
      const results = await fetchJson(`/jobs/${jobId}/results`);
      renderResults(jobId, results);
      showToast("AutoML job completed.");
    } else if (status.status === "FAILED") {
      window.clearInterval(pollTimer);
      showToast("AutoML job failed. Check worker logs and output status.json.", true);
    }
  } catch (error) {
    showToast(error.message, true);
  }
}

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name || "No file selected";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  resultSection.classList.add("hidden");

  try {
    const payload = await fetchJson("/jobs", {
      method: "POST",
      body: formData,
    });
    jobIdInput.value = payload.job_id;
    renderStatus({
      job_id: payload.job_id,
      status: payload.status,
      start_time: null,
      end_time: null,
    });
    showToast("Job submitted to the Celery queue.");
    window.clearInterval(pollTimer);
    pollTimer = window.setInterval(refreshStatus, 3500);
    window.setTimeout(refreshStatus, 700);
  } catch (error) {
    showToast(error.message, true);
  }
});

checkStatusButton.addEventListener("click", refreshStatus);
