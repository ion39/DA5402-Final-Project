const inferredDockerApiBaseUrl =
  window.location.port === "8080"
    ? `${window.location.protocol}//${window.location.hostname}:8002`
    : null;
const explicitApiBaseUrl = window.CLV_API_BASE || null;
const fallbackApiCandidates = [
  `${window.location.protocol}//${window.location.hostname}:8000`,
  `${window.location.protocol}//${window.location.hostname}:8001`,
  `${window.location.protocol}//${window.location.hostname}:8002`,
  "http://127.0.0.1:8000",
  "http://127.0.0.1:8001",
  "http://127.0.0.1:8002",
  "http://localhost:8000",
  "http://localhost:8001",
  "http://localhost:8002",
];
const apiCandidates = explicitApiBaseUrl
  ? [explicitApiBaseUrl]
  : [...new Set([inferredDockerApiBaseUrl, ...fallbackApiCandidates].filter(Boolean))];

const fieldConfig = [
  ["Customer_ID", "text"],
  ["Age", "number"],
  ["Location", "select", ["Urban", "Suburban", "Rural"]],
  ["Income_Level", "select", ["Low", "Middle", "High"]],
  ["Total_Transactions", "number"],
  ["Avg_Transaction_Value", "number"],
  ["Max_Transaction_Value", "number"],
  ["Min_Transaction_Value", "number"],
  ["Total_Spent", "number"],
  ["Active_Days", "number"],
  ["Last_Transaction_Days_Ago", "number"],
  ["Loyalty_Points_Earned", "number"],
  ["Referral_Count", "number"],
  ["Cashback_Received", "number"],
  ["App_Usage_Frequency", "select", ["Daily", "Weekly", "Monthly"]],
  ["Preferred_Payment_Method", "select", ["UPI", "Credit Card", "Debit Card", "Wallet Balance"]],
  ["Support_Tickets_Raised", "number"],
  ["Issue_Resolution_Time", "number"],
  ["Customer_Satisfaction_Score", "number"],
];

const formElement = document.getElementById("predictionForm");
const submitBtn = document.getElementById("submitBtn");
const loadSampleBtn = document.getElementById("loadSampleBtn");
const resultState = document.getElementById("resultState");
const pipelineStages = document.getElementById("pipelineStages");
const heroStats = document.getElementById("heroStats");
let resolvedApiBaseUrl = null;

function titleize(value) {
  return value.replaceAll("_", " ");
}

function renderForm() {
  fieldConfig.forEach(([name, type, options]) => {
    const wrapper = document.createElement("label");
    wrapper.className = "field";

    const label = document.createElement("span");
    label.textContent = titleize(name);
    wrapper.appendChild(label);

    let input;
    if (type === "select") {
      input = document.createElement("select");
      options.forEach((option) => {
        const optionNode = document.createElement("option");
        optionNode.value = option;
        optionNode.textContent = option;
        input.appendChild(optionNode);
      });
    } else {
      input = document.createElement("input");
      input.type = type;
      if (type === "number") {
        input.step = "any";
      }
    }

    input.name = name;
    input.id = name;
    wrapper.appendChild(input);
    formElement.appendChild(wrapper);
  });
}

function getPayload() {
  const payload = {};
  fieldConfig.forEach(([name, type]) => {
    const value = document.getElementById(name).value;
    payload[name] = type === "number" ? Number(value) : value;
  });
  return payload;
}

function setPayload(payload) {
  fieldConfig.forEach(([name]) => {
    if (payload[name] !== undefined) {
      document.getElementById(name).value = payload[name];
    }
  });
}

function renderResult(data) {
  resultState.innerHTML = `
    <div class="metric-card">
      <span>Predicted CLV</span>
      <strong>${data.predicted_clv.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong>
    </div>
    <div class="metric-card">
      <span>Churn Probability</span>
      <strong>${(data.churn_probability * 100).toFixed(2)}%</strong>
    </div>
    <div class="metric-card">
      <span>Predicted Churn Label</span>
      <strong>${data.predicted_churn_label}</strong>
    </div>
    <div class="metric-card full">
      <span>Drifted Features</span>
      <strong>${data.drift_detected_features.length ? data.drift_detected_features.join(", ") : "None detected"}</strong>
    </div>
  `;
}

async function resolveApiBaseUrl() {
  if (resolvedApiBaseUrl) {
    return resolvedApiBaseUrl;
  }

  for (const baseUrl of apiCandidates) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) {
        resolvedApiBaseUrl = baseUrl;
        return resolvedApiBaseUrl;
      }
    } catch (error) {
      // Try the next candidate.
    }
  }

  throw new Error(
    "Backend API is not reachable. Start FastAPI on port 8000, 8001, or 8002."
  );
}

async function fetchJson(path, options = {}) {
  const baseUrl = await resolveApiBaseUrl();
  const response = await fetch(`${baseUrl}${path}`, options);
  const isJson = response.headers.get("content-type")?.includes("application/json");
  const data = isJson ? await response.json() : null;

  if (!response.ok) {
    throw new Error(data?.detail || `Request failed for ${path}`);
  }

  return data;
}

async function handlePredict() {
  submitBtn.disabled = true;
  submitBtn.textContent = "Estimating...";
  try {
    const data = await fetchJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getPayload()),
    });
    renderResult(data);
  } catch (error) {
    resultState.textContent = error.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Estimate CLV";
  }
}

async function loadSample() {
  loadSampleBtn.disabled = true;
  try {
    const data = await fetchJson("/sample-input");
    setPayload(data);
  } catch (error) {
    resultState.textContent = error.message;
  } finally {
    loadSampleBtn.disabled = false;
  }
}

async function loadPipelineSummary() {
  try {
    const [summary, metrics] = await Promise.all([
      fetchJson("/pipeline/summary"),
      fetchJson("/model/info"),
    ]);

    heroStats.innerHTML = `
      <div class="metric-card"><span>Rows</span><strong>${summary.throughput.records_processed}</strong></div>
      <div class="metric-card"><span>ROC-AUC</span><strong>${metrics.classifier.roc_auc.toFixed(3)}</strong></div>
      <div class="metric-card"><span>R2</span><strong>${metrics.regressor.r2.toFixed(3)}</strong></div>
    `;

    pipelineStages.innerHTML = summary.pipeline_stages
      .map(
        (stage) => `
          <article class="stage">
            <div class="stage-top">
              <h3>${stage.name}</h3>
              <span class="status">${stage.status}</span>
            </div>
            <p>${stage.description}</p>
            <div class="muted">${stage.records} records</div>
          </article>
        `
      )
      .join("");
  } catch (error) {
    heroStats.innerHTML = `<div class="result-empty">${error.message}</div>`;
    pipelineStages.innerHTML = `<div class="result-empty">${error.message}</div>`;
  }
}

renderForm();
submitBtn.addEventListener("click", handlePredict);
loadSampleBtn.addEventListener("click", loadSample);
loadPipelineSummary();
