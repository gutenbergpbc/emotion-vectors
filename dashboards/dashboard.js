"use strict";
// A snapshot reader, not a workflow engine. All displayed strings are text.
const $ = id => document.getElementById(id);
const source = new URL(new URLSearchParams(location.search).get("data") || "progress.json", location.href);
const states = new Set(["pending", "running", "waiting", "complete", "failed", "cancelled"]);
let last = null;
let rateKey = null;
let rateSamples = [];
const text = (tag, value, cls) => {
  const node = document.createElement(tag);
  node.textContent = value;
  if (cls) node.className = cls;
  return node;
};
const duration = seconds => seconds < 60 ? `${Math.floor(seconds)}s` : seconds < 3600 ? `${Math.floor(seconds / 60)}m` : `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
const age = time => duration(Math.max(0, (Date.now() - Date.parse(time)) / 1000));
const count = p => `${p.completed.toLocaleString()}${p.total == null ? "" : ` / ${p.total.toLocaleString()}`} ${p.unit}`;
function observeRate(s) {
  const stage = s.stages?.find(x => x.id === s.active_stage);
  const p = stage?.progress || s.progress;
  const key = `${s.run_id}:${stage?.id || "progress"}`;
  if (key !== rateKey || !p || !["running", "waiting"].includes(stage?.state || s.state)) {
    rateKey = key; rateSamples = [];
  }
  if (!p || !["running", "waiting"].includes(stage?.state || s.state)) return;
  const now = performance.now();
  const prev = rateSamples.at(-1);
  if (!prev || prev.completed !== p.completed) {
    if (prev && p.completed < prev.completed) rateSamples = [];
    rateSamples.push({completed: p.completed, at: now});
    if (rateSamples.length > 6) rateSamples.shift();
  }
}
function estimatedEta(p, state, s) {
  const stale = Date.now() - Date.parse(s.updated_at) > (s.stale_after_seconds || 60) * 1000;
  if (state !== "running" || stale || p.total == null || p.completed >= p.total || rateSamples.length < 2) return null;
  const first = rateSamples[0], lastSample = rateSamples.at(-1);
  const elapsed = (lastSample.at - first.at) / 1000;
  const completed = lastSample.completed - first.completed;
  if (elapsed < 10 || completed <= 0) return null;
  const rate = completed / elapsed;
  return {rate, seconds: (p.total - p.completed) / rate};
}
function validate(s) {
  if (s.schema_version !== 1 || !s.run_id || !s.title || !states.has(s.state) || !Number.isFinite(Date.parse(s.updated_at))) throw Error("Invalid snapshot identity, state, or timestamp");
  if (s.progress && s.stages) throw Error("Use progress or stages, not both");
  if (!s.progress && !s.stages) throw Error("Snapshot needs progress or stages");
  if (s.stages && !Array.isArray(s.stages)) throw Error("stages must be a list");
  if (s.links != null && (!Array.isArray(s.links) || s.links.some(x => !x || typeof x.label !== "string" || typeof x.href !== "string"))) throw Error("Invalid artifact links");
  const stages = s.stages || [];
  if (new Set(stages.map(x => x.id)).size !== stages.length || stages.some(x => !x.id || !x.label || !states.has(x.state))) throw Error("Invalid stage IDs, labels, or states");
  if (s.active_stage != null && !stages.some(x => x.id === s.active_stage)) throw Error("Unknown active_stage");
  for (const p of [s.progress, ...stages.map(x => x.progress)].filter(Boolean)) {
    if (!Number.isInteger(p.completed) || p.completed < 0 || !p.unit || (p.total != null && (!Number.isInteger(p.total) || p.total < p.completed))) throw Error("Invalid progress counter");
    for (const k of ["rate_per_second", "eta_seconds"]) if (p[k] != null && (!Number.isFinite(p[k]) || p[k] < 0)) throw Error(`Invalid ${k}`);
    if (p.last_progress_at != null && !Number.isFinite(Date.parse(p.last_progress_at))) throw Error("Invalid last_progress_at");
  }
  if (s.stale_after_seconds != null && (!Number.isFinite(s.stale_after_seconds) || s.stale_after_seconds <= 0)) throw Error("Invalid heartbeat threshold");
}
function bar(p, state, label) {
  const node = document.createElement("progress");
  node.setAttribute("aria-label", label);
  node.max = p.total > 0 ? p.total : 1;
  if (p.total != null) node.value = p.total === 0 ? (state === "complete" ? 1 : 0) : p.completed;
  return node;
}
function freshness() {
  if (!last || last.example) return;
  const stale = ["running", "waiting"].includes(last.state) && Date.now() - Date.parse(last.updated_at) > (last.stale_after_seconds || 60) * 1000;
  $("connection").textContent = `${last.state}${stale ? " · Heartbeat stale; worker status unknown" : ""} · Refreshes every 5 seconds`;
  $("connection").className = stale ? "notice" : "";
  const stage = last.stages?.find(x => x.id === last.active_stage);
  const p = stage?.progress || last.progress;
  $("freshness").textContent = `Snapshot ${age(last.updated_at)} ago${p?.last_progress_at ? ` · Last progress ${age(p.last_progress_at)} ago` : ""}`;
}
function render(s) {
  $("title").textContent = s.title;
  document.title = s.title;
  $("description").textContent = s.description || "";
  $("description").hidden = !s.description;
  $("example").hidden = !s.example;
  const stage = s.stages?.find(x => x.id === s.active_stage);
  const p = stage?.progress || s.progress;
  const state = stage?.state || s.state;
  const estimate = p ? estimatedEta(p, state, s) : null;
  const summary = $("summary"); summary.replaceChildren(); summary.hidden = !p;
  if (p) {
    const row = text("div", "", "counter");
    row.append(text("span", stage?.label || "Progress"), text("strong", count(p)));
    summary.append(row, bar(p, state, stage?.label || "Progress"));
    const metrics = [];
    if (estimate) metrics.push(`${estimate.rate.toFixed(2)} ${p.unit}/s`);
    else if (p.rate_per_second != null && state === "running") metrics.push(`${p.rate_per_second.toLocaleString()} ${p.unit}/s`);
    if (p.eta_seconds != null && p.total != null && state === "running") metrics.push(`ETA ${duration(p.eta_seconds)}`);
    else if (estimate) metrics.push(`ETA ~${duration(estimate.seconds)}`);
    if (metrics.length) summary.append(text("p", metrics.join(" · "), "muted"));
  }
  $("activity").textContent = stage?.message || s.message || "";
  $("run-error").textContent = s.error || (s.state === "failed" ? "Run failed; inspect the logs." : "");
  $("run-error").hidden = !$("run-error").textContent;
  $("stages").replaceChildren(); $("stages").hidden = !s.stages?.length;
  for (const st of s.stages || []) {
    const card = text("article", "", "stage"); card.dataset.state = st.state;
    card.append(text("h2", st.label), text("p", st.state));
    if (st.progress) {
      card.append(text("p", count(st.progress)), bar(st.progress, st.state, st.label));
      if (st.id === s.active_stage && estimate) card.append(text("p", `Estimated ${duration(estimate.seconds)} remaining · ${estimate.rate.toFixed(2)} ${st.progress.unit}/s`, "muted"));
    }
    if (st.message) card.append(text("p", st.message));
    if (st.error) card.append(text("p", st.error, "error"));
    $("stages").append(card);
  }
  $("links").replaceChildren();
  for (const link of [...(s.links || []), {label: "Progress JSON", href: source.href}]) {
    try {
      const url = new URL(link.href, source);
      if (!["http:", "https:"].includes(url.protocol)) continue;
      const a = text("a", link.label); a.href = url.href; $("links").append(a);
    } catch { /* Invalid artifact URLs are not actions. */ }
  }
  $("raw").textContent = JSON.stringify(s, null, 2);
  $("connection").className = "";
  $("connection").textContent = s.example ? `${s.state} · Frozen example` : s.state;
  $("freshness").textContent = s.example ? `Snapshot recorded ${s.updated_at}` : "";
  freshness();
}
async function refresh() {
  try {
    if (source.origin !== location.origin) throw Error("Progress JSON must be on the same origin");
    const response = await fetch(source, {cache: "no-store", signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    const snapshot = await response.json(); validate(snapshot); observeRate(snapshot);
    render(snapshot); last = snapshot; freshness();
    $("read-error").hidden = true;
  } catch (error) {
    $("read-error").textContent = `Cannot read progress: ${error.message}. ${last ? "Showing last received snapshot." : "No snapshot received."} Retrying in 5 seconds.`;
    $("read-error").hidden = false;
    if (!last) $("connection").textContent = "Progress unavailable";
  } finally { if (!last?.example) setTimeout(refresh, 5000); }
}
setInterval(freshness, 1000);
refresh();
