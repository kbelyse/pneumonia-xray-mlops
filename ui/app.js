const $ = (id) => document.getElementById(id);
const BLUE = "#2f6fbd", RED = "#c8433b";
const charts = {};

function chart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), config);
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.page).classList.add("active");
    if (btn.dataset.page === "insights") loadInsights();
    if (btn.dataset.page === "retrain") loadRetrain();
  };
});

// ---------- overview ----------
function fmtUptime(s) {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h ${m}m` : h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
}

async function loadHealth() {
  try {
    const h = await getJSON("/health");
    $("status-pill").textContent = "API online";
    $("status-pill").className = "pill ok";
    $("uptime").textContent = fmtUptime(h.uptime_seconds);
    $("started").textContent = "since " + h.started_at;
    $("version").textContent = "v" + h.model_version;
    $("served").textContent = h.predictions_served;
    $("container").textContent = "container " + h.container;
    $("latency").textContent = h.avg_latency_ms ? h.avg_latency_ms + " ms" : "-";
    const pts = h.recent_latency;
    chart("latency-chart", {
      type: "line",
      data: { labels: pts.map((p) => new Date(p.t * 1000).toLocaleTimeString()),
              datasets: [{ label: "ms per prediction", data: pts.map((p) => p.ms), borderColor: BLUE, pointRadius: 2, tension: 0.2 }] },
      options: { animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: "ms" } } } },
    });
  } catch (e) {
    $("status-pill").textContent = "API offline";
    $("status-pill").className = "pill down";
  }
}

async function loadModel() {
  const m = await getJSON("/model");
  const t = m.test_metrics;
  $("saved-at").textContent = "saved " + m.saved_at;
  $("thr").textContent = m.threshold;
  const show = [["Accuracy", t.accuracy], ["Precision", t.precision], ["Recall", t.recall],
                ["Specificity", t.specificity], ["F1", t.f1], ["ROC-AUC", t.roc_auc], ["Loss", t.loss]];
  $("metric-cards").innerHTML = show.map(([k, v]) =>
    `<div class="card"><div class="label">${k}</div><div class="value">${v.toFixed(3)}</div></div>`).join("");
  const [[tn, fp], [fn, tp]] = t.confusion_matrix;
  $("cm").innerHTML = `<tr><th></th><th>predicted NORMAL</th><th>predicted PNEUMONIA</th></tr>
    <tr><th>true NORMAL</th><td class="hit">${tn}</td><td>${fp}</td></tr>
    <tr><th>true PNEUMONIA</th><td>${fn}</td><td class="hit">${tp}</td></tr>`;
  const h = m.history || {};
  if (h.loss) {
    chart("history-chart", {
      type: "line",
      data: { labels: h.loss.map((_, i) => i + 1), datasets: [
        { label: "train loss", data: h.loss, borderColor: BLUE },
        { label: "val loss", data: h.val_loss, borderColor: BLUE, borderDash: [5, 4] },
        { label: "train accuracy", data: h.accuracy, borderColor: RED },
        { label: "val accuracy", data: h.val_accuracy, borderColor: RED, borderDash: [5, 4] },
      ] },
      options: { animation: false, scales: { x: { title: { display: true, text: "epoch" } } }, plugins: { legend: { labels: { boxWidth: 12 } } } },
    });
  }
}

// ---------- predict ----------
let chosen = null;
function choose(file) {
  chosen = file;
  $("drop-text").textContent = file.name;
  $("preview").src = URL.createObjectURL(file);
  $("preview").style.display = "block";
  $("predict-btn").disabled = false;
  $("result").className = "result empty";
  $("result").textContent = "Press Predict.";
}
$("predict-file").onchange = (e) => e.target.files[0] && choose(e.target.files[0]);
$("drop").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = (e) => { e.preventDefault(); $("drop").classList.remove("over"); e.dataTransfer.files[0] && choose(e.dataTransfer.files[0]); };

$("predict-btn").onclick = async () => {
  const fd = new FormData();
  fd.append("file", chosen);
  $("predict-btn").disabled = true;
  $("result").textContent = "Predicting...";
  try {
    const r = await getJSON("/predict", { method: "POST", body: fd });
    const pct = (r.pneumonia_probability * 100).toFixed(1);
    $("result").className = "result";
    $("result").innerHTML = `
      <div class="small">Prediction for ${r.filename}</div>
      <div class="big ${r.label}">${r.label}</div>
      <div>Probability of pneumonia: <b>${pct}%</b> (the model says PNEUMONIA above ${(r.threshold * 100).toFixed(0)}%)</div>
      <div class="bar"><div style="width:${pct}%"></div></div>
      <table class="simple">
        <tr><th>Confidence</th><td>${(r.confidence * 100).toFixed(1)}%</td></tr>
        <tr><th>Response time</th><td>${r.latency_ms} ms</td></tr>
        <tr><th>Model version</th><td>v${r.model_version}</td></tr>
        <tr><th>Answered by container</th><td>${r.container}</td></tr>
      </table>`;
    loadHealth();
  } catch (e) {
    $("result").className = "result";
    $("result").textContent = "Error: " + e.message;
  }
  $("predict-btn").disabled = false;
};

// ---------- insights ----------
function histDatasets(feature) {
  return ["NORMAL", "PNEUMONIA"].map((label, i) => ({
    label, data: feature[label].counts.map((c) => c / feature[label].counts.reduce((a, b) => a + b, 0)),
    backgroundColor: i ? "rgba(200,67,59,.55)" : "rgba(47,111,189,.55)", borderWidth: 0,
    barPercentage: 1, categoryPercentage: 1,
  }));
}
function histLabels(feature) {
  const e = feature.NORMAL.edges;
  return e.slice(0, -1).map((v, i) => Math.round((v + e[i + 1]) / 2));
}
const histOpts = (xTitle) => ({ animation: false, scales: { x: { stacked: false, title: { display: true, text: xTitle } },
  y: { title: { display: true, text: "share of images" } } } });

async function loadInsights() {
  const d = await getJSON("/insights");
  const cc = d.class_counts;
  chart("balance-chart", {
    type: "bar",
    data: { labels: ["train", "test"], datasets: [
      { label: "NORMAL", data: [cc.train.NORMAL, cc.test.NORMAL], backgroundColor: BLUE },
      { label: "PNEUMONIA", data: [cc.train.PNEUMONIA, cc.test.PNEUMONIA], backgroundColor: RED },
    ] },
    options: { animation: false, scales: { y: { title: { display: true, text: "images" } } } },
  });
  const f = d.features;
  chart("brightness-chart", { type: "bar", data: { labels: histLabels(f.centre_brightness), datasets: histDatasets(f.centre_brightness) },
    options: histOpts("mean pixel value of the centre (0 = black, 255 = white)") });
  chart("contrast-chart", { type: "bar", data: { labels: histLabels(f.contrast), datasets: histDatasets(f.contrast) },
    options: histOpts("standard deviation of pixel values") });
  chart("size-chart", { type: "bar", data: { labels: histLabels(f.width), datasets: histDatasets(f.width) },
    options: histOpts("original width in pixels") });
  const u = await getJSON("/uploads");
  const labels = ["NORMAL", "PNEUMONIA"];
  chart("uploads-chart", {
    type: "bar",
    data: { labels, datasets: [
      { label: "waiting", data: labels.map((l) => (u.by_label[l] || {}).pending || 0), backgroundColor: BLUE },
      { label: "already used", data: labels.map((l) => ((u.by_label[l] || {}).total || 0) - ((u.by_label[l] || {}).pending || 0)), backgroundColor: "#98a2b3" },
    ] },
    options: { indexAxis: "y", animation: false, scales: { x: { stacked: true }, y: { stacked: true } } },
  });
}

// ---------- upload and retrain ----------
async function loadUploads() {
  const u = await getJSON("/uploads");
  const rows = ["NORMAL", "PNEUMONIA"].map((l) => {
    const x = u.by_label[l] || { total: 0, pending: 0 };
    return `<tr><td>${l}</td><td>${x.total}</td><td>${x.pending}</td></tr>`;
  });
  $("upload-table").innerHTML = `<tr><th>Label</th><th>Uploaded</th><th>Waiting for retraining</th></tr>${rows.join("")}`;
  $("auto-note").textContent = u.auto_retrain
    ? `Retraining also starts on its own once ${u.auto_retrain_threshold} new images are waiting.` : "";
}

const pct = (v) => (v == null ? "-" : (v * 100).toFixed(1) + "%");
let polling = null;

async function loadRuns() {
  const { runs } = await getJSON("/retrain/status");
  $("runs-table").innerHTML = `<tr><th>#</th><th>Started</th><th>Trigger</th><th>Status</th><th>New / old images</th>
    <th>Accuracy before / after</th><th>F1 before / after</th><th>Result</th></tr>` +
    runs.map((r) => `<tr><td>${r.id}</td><td>${r.started_at}</td><td>${r.trigger || ""}</td>
      <td class="status-${r.status}">${r.status}${r.status === "running" ? ": " + r.step : ""}</td>
      <td>${r.n_new ?? "-"} / ${r.n_replay ?? "-"}</td>
      <td>${pct(r.old_metrics?.accuracy)} / ${pct(r.new_metrics?.accuracy)}</td>
      <td>${pct(r.old_metrics?.f1)} / ${pct(r.new_metrics?.f1)}</td>
      <td>${r.message || ""}</td></tr>`).join("");
  const running = runs.find((r) => r.status === "running");
  if (running) {
    $("run-live").innerHTML = `<div class="live"><span class="status-running">Run ${running.id} is running</span><br>Step: ${running.step}</div>`;
    $("retrain-btn").disabled = true;
    if (!polling) polling = setInterval(loadRetrain, 3000);
  } else {
    $("retrain-btn").disabled = false;
    if (polling) {
      clearInterval(polling);
      polling = null;
      const last = runs[0];
      if (last) $("run-live").innerHTML = `<div class="live"><span class="status-${last.status}">Run ${last.id} ${last.status}</span><br>${last.message || ""}</div>`;
      loadModel();
    }
  }
}

async function loadRetrain() {
  await Promise.all([loadUploads(), loadRuns()]);
}

$("upload-btn").onclick = async () => {
  const files = $("upload-files").files;
  if (!files.length) { $("upload-result").textContent = "Choose some files first."; return; }
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("label", $("upload-label").value);
  $("upload-btn").disabled = true;
  $("upload-result").textContent = `Uploading ${files.length} file(s)...`;
  try {
    const r = await getJSON("/upload", { method: "POST", body: fd });
    $("upload-result").textContent = `Saved ${r.saved.NORMAL} NORMAL and ${r.saved.PNEUMONIA} PNEUMONIA images` +
      (r.skipped ? `, skipped ${r.skipped} (duplicates or not images)` : "") +
      (r.auto_retrain_started ? ". Automatic retraining started." : ".");
    $("upload-files").value = "";
  } catch (e) {
    $("upload-result").textContent = "Error: " + e.message;
  }
  $("upload-btn").disabled = false;
  loadRetrain();
};

$("retrain-btn").onclick = async () => {
  $("retrain-msg").textContent = "";
  try {
    const r = await getJSON("/retrain", { method: "POST" });
    $("retrain-msg").textContent = `Started run ${r.run_id}.`;
  } catch (e) {
    $("retrain-msg").textContent = e.message;
  }
  loadRetrain();
};

// ---------- start ----------
loadHealth();
loadModel();
setInterval(loadHealth, 5000);
