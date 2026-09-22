"use strict";

const form = document.querySelector("#prediction-form");
const tickerInput = document.querySelector("#ticker");
const controls = document.querySelector("#run-controls");
const runStatus = document.querySelector("#run-status");
const historyStatus = document.querySelector("#history-status");
const historyBody = document.querySelector("#history-body");
let historyRequest = 0;
let running = false;

const display = (value) => value === null || value === undefined || value === "" ? "--" : String(value);
const probability = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";
const dateLabel = (value) => value ? String(value).slice(0, 10) : "--";
const ticker = () => tickerInput.value.trim().toUpperCase();

async function requestJSON(url, options) {
  const response = await fetch(url, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok || data.success === false) {
    throw new Error(data.error || `Request failed (${response.status}).`);
  }
  return data;
}

function directionClass(value) {
  const text = String(value || "").toUpperCase();
  return text.startsWith("UP") ? "up" : text.startsWith("DOWN") ? "down" : "";
}

async function loadHistory() {
  const version = ++historyRequest;
  historyBody.replaceChildren();
  document.querySelector("#history-count").textContent = "";
  historyStatus.textContent = "Loading history...";
  historyStatus.className = "muted";
  try {
    const url = new URL(form.dataset.historyUrl, window.location.origin);
    if (ticker()) url.searchParams.set("ticker", ticker());
    const data = await requestJSON(url);
    if (version !== historyRequest) return;
    if (!Array.isArray(data.history)) throw new Error("Invalid history response.");
    const fragment = document.createDocumentFragment();
    for (const row of data.history) {
      const tr = document.createElement("tr");
      const values = [display(row.timestamp), display(row.ticker), dateLabel(row.date),
        display(row.direction), probability(row.probability_up), display(row.confidence)];
      values.forEach((value, index) => {
        const td = document.createElement("td");
        td.textContent = value;
        if (index === 3) td.className = directionClass(row.direction);
        if (index === 4) td.className = "numeric";
        tr.append(td);
      });
      fragment.append(tr);
    }
    historyBody.replaceChildren(fragment);
    historyStatus.textContent = data.history.length ? "" : "No predictions recorded for this selection.";
    document.querySelector("#history-count").textContent = `${data.history.length} ${data.history.length === 1 ? "record" : "records"}`;
  } catch (error) {
    if (version !== historyRequest) return;
    historyStatus.textContent = `History unavailable: ${error.message}`;
    historyStatus.className = "error";
  }
}

function clearResult() {
  document.querySelector("#prediction-result").hidden = true;
  document.querySelector("#result-empty").hidden = false;
  document.querySelector("#result-ticker").textContent = "";
}

function showPrediction(data, selectedTicker, source = "Current session") {
  const result = data.prediction || data;
  if (!result || typeof result !== "object") throw new Error("Invalid prediction response.");
  document.querySelector("#result-direction").textContent = display(result.direction);
  document.querySelector("#result-direction").className = directionClass(result.direction);
  document.querySelector("#result-confidence").textContent = display(result.confidence);
  document.querySelector("#result-date").textContent = dateLabel(result.date || result.feature_date);
  document.querySelector("#result-probability").textContent = probability(result.probability_up);
  const atr = result.atr_pct;
  const limit = result.volatility_threshold ?? data.volatility_filter?.max_atr_pct;
  document.querySelector("#result-atr").textContent = `${display(atr)}${atr === null || atr === undefined ? "" : "%"} / ${display(limit)}${limit === null || limit === undefined ? "" : "%"}`;
  const volatilityLabels = {disabled: "Disabled", pass: "Within limit", blocked: "Triggered", unavailable: "Unavailable"};
  const risk = volatilityLabels[result.volatility_state] ||
    (result.volatility_filter_triggered ? "Triggered" : data.volatility_filter?.enabled ? "Within limit" : "Disabled");
  document.querySelector("#result-filter").textContent = risk;
  document.querySelector("#result-filter").className = risk === "Within limit" ? "up" : risk === "Triggered" || risk === "Unavailable" ? "down" : "";
  const readiness = result.readiness || data.readiness || {};
  document.querySelector("#result-target-session").textContent = display(readiness.target_session);
  document.querySelector("#result-readiness").textContent = display(result.readiness_status || readiness.status);
  document.querySelector("#result-model").textContent = display(result.model_id);
  document.querySelector("#result-ticker").textContent = data.ticker || selectedTicker;
  document.querySelector("#result-source").textContent = source;
  document.querySelector("#result-empty").hidden = true;
  document.querySelector("#prediction-result").hidden = false;
}

async function loadLatest() {
  const selectedTicker = ticker();
  if (!selectedTicker) return clearResult();
  try {
    const url = new URL(form.dataset.latestUrl, window.location.origin);
    url.searchParams.set("ticker", selectedTicker);
    const data = await requestJSON(url);
    if (ticker() !== selectedTicker) return;
    if (data.prediction) showPrediction(data.prediction, selectedTicker, "Saved latest result");
    else clearResult();
  } catch (error) {
    if (ticker() === selectedTicker) clearResult();
  }
}

async function run(kind) {
  if (running || !form.reportValidity()) return;
  const selectedTicker = ticker();
  if (!selectedTicker) { tickerInput.focus(); return; }
  const payload = {ticker: selectedTicker, fetch_latest: document.querySelector("#fetch-latest").checked};
  const volatility = document.querySelector("#volatility").value;
  if (kind === "prediction" && volatility !== "") payload.volatility = Number(volatility);
  running = true;
  controls.disabled = true;
  form.setAttribute("aria-busy", "true");
  if (kind === "prediction") clearResult();
  runStatus.className = "";
  runStatus.textContent = `${kind === "pipeline" ? "Generating features" : "Running prediction"} for ${selectedTicker}...`;
  try {
    const data = await requestJSON(kind === "pipeline" ? form.dataset.pipelineUrl : form.dataset.predictionUrl, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    if (kind === "pipeline") {
      runStatus.textContent = `${selectedTicker}: ${display(data.rows)} rows saved to ${display(data.output_path)}`;
    } else {
      showPrediction(data, selectedTicker);
      runStatus.textContent = `Prediction complete for ${selectedTicker}.${data.run_id ? ` Run ${data.run_id}.` : ""}`;
      await loadHistory();
    }
    runStatus.className = "success";
  } catch (error) {
    runStatus.textContent = `${selectedTicker}: ${error.message}`;
    runStatus.className = "error";
  } finally {
    running = false;
    controls.disabled = false;
    form.setAttribute("aria-busy", "false");
  }
}

form.addEventListener("submit", (event) => { event.preventDefault(); run("prediction"); });
document.querySelector("#pipeline-button").addEventListener("click", () => run("pipeline"));
document.querySelector("#refresh-history").addEventListener("click", loadHistory);
let tickerTimer;
tickerInput.addEventListener("input", () => {
  ++historyRequest;
  historyBody.replaceChildren();
  document.querySelector("#history-count").textContent = "";
  historyStatus.textContent = "Loading history...";
  clearResult();
  runStatus.textContent = "";
  clearTimeout(tickerTimer);
  tickerTimer = setTimeout(() => { loadHistory(); loadLatest(); }, 250);
});
loadHistory();
loadLatest();
