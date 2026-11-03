"use strict";
const $ = (id) => document.getElementById(id),
  emotions = [
    "hostile",
    "frustrated",
    "excited",
    "appreciative",
    "sad",
    "afraid",
    "surprised",
    "anxious",
    "guilty",
    "ashamed",
    "desperate",
    "calm",
  ],
  recipes = ["neutral", "other11_neutral", "other11"],
  colors = ["#416fa4", "#d88443", "#3c9284"];
const keys = emotions.flatMap((e) => recipes.map((r) => e + "__" + r)),
  pct = (n) => (n == null ? "—" : (100 * n).toFixed(1) + "%"),
  esc = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
let state = null,
  spec = null,
  calibration = {},
  results = {},
  cv = {},
  previous = null,
  paused = false,
  changes = [],
  offset = 0,
  total = 0,
  currentTab = "overview";
async function get(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) {
    const error = Error(path + ": " + r.status);
    error.status = r.status;
    throw error;
  }
  return r.json();
}
function tab(name) {
  currentTab = name;
  document
    .querySelectorAll(".tab")
    .forEach((el) => (el.hidden = el.id !== name));
  document
    .querySelectorAll("nav button")
    .forEach((el) => el.classList.toggle("active", el.dataset.tab === name));
  if (name === "examples") examples();
  history.replaceState(null, "", name === "status" ? "/status" : "/#" + name);
}
document
  .querySelectorAll("nav button")
  .forEach((el) => (el.onclick = () => tab(el.dataset.tab)));
for (const id of ["layerProbe", "exampleProbe"])
  $(id).innerHTML = keys
    .map((k) => `<option value="${k}">${k.replace("__", " / ")}</option>`)
    .join("");
$("exampleLabel").innerHTML += [...emotions, "neutral"]
  .map((e) => `<option>${e}</option>`)
  .join("");
function flatten(x, p = "", out = {}) {
  if (x && typeof x === "object" && !Array.isArray(x))
    Object.entries(x).forEach(([k, v]) => flatten(v, p ? p + "." + k : k, out));
  else out[p] = JSON.stringify(x);
  return out;
}
function raw(s) {
  if (paused) return;
  const a = flatten(previous),
    b = flatten(s),
    diff = [];
  if (previous)
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)]))
      if (a[k] !== b[k]) diff.push([k, a[k] ?? "∅", b[k] ?? "∅"]);
  if (diff.length) {
    changes.unshift({ time: new Date().toLocaleTimeString(), diff });
    changes = changes.slice(0, 20);
    $("changes").innerHTML = changes
      .map(
        (c) =>
          `<li><time>${esc(c.time)}</time>${c.diff.map((d) => `<b>${esc(d[0])}</b><span>${esc(d[1])} → ${esc(d[2])}</span>`).join("<br>")}</li>`,
      )
      .join("");
  }
  $("rawStatus").textContent = JSON.stringify(s, null, 2);
  $("statusMeta").textContent =
    "Snapshot #" +
    s.sequence +
    " · " +
    new Date(s.updated_at).toLocaleTimeString();
  previous = s;
}
function showStatus(s) {
  state = s;
  raw(s);
  $("identity").textContent =
    (s.model || "GPT-OSS 120B") + " · " + (s.revision || "Revision pending");
  $("footerRun").textContent = s.run_id;
  $("demo").hidden = s.mode !== "synthetic";
  $("failure").hidden = !s.error;
  if (s.error)
    $("failure").textContent =
      s.error + " — See Status & logs for the traceback and resume details.";
  const age = (Date.now() - Date.parse(s.updated_at)) / 1000;
  $("connection").textContent =
    age > 35 && s.state === "running"
      ? "Worker heartbeat stale"
      : s.state === "running"
        ? "● Worker running"
        : "● Connected";
  $("lastSeen").textContent =
    "Updated " + new Date(s.updated_at).toLocaleTimeString();
  $("stages").innerHTML = Object.entries(s.stages)
    .map(([name, r]) => {
      const p = r.total
        ? Math.min(100, (100 * r.completed) / r.total)
        : r.state === "complete"
          ? 100
          : 0;
      return `<div class="stage ${esc(r.state)}"><div class="title">${esc(name)}<span class="badge">${esc(r.state)}</span></div><small>${r.total ? `${r.completed.toLocaleString()} / ${r.total.toLocaleString()}` : "Awaiting stage"}${r.phase ? " · " + esc(r.phase) : ""}</small><div class="progress"><div style="width:${p}%"></div></div><small>${esc(r.current_case || r.current_probe || (r.current_layer != null ? "Layer " + r.current_layer : ""))}</small></div>`;
    })
    .join("");
  const extract = s.stages.extract;
  const info = {
    "Run ID": s.run_id,
    "Layers × width": spec ? spec.layers + " × " + spec.width : "—",
    "Runtime dtype": spec?.dtype || "—",
    "Training SHA-256": spec?.input_hashes?.["training_cases.json"] || "—",
    "Split SHA-256": spec?.input_hashes?.["split_manifest.json"] || "—",
    Revision: s.revision || "pending",
    "Extraction phase": extract.phase || "pending",
    "Verified cache reuse": extract.verified_cache_reuse || 0,
    "Current shard": extract.current_shard || "—",
    "Prefix tokens": extract.prefix_tokens || "—",
    "Elapsed extraction":
      extract.elapsed_seconds != null
        ? Math.round(extract.elapsed_seconds) + " seconds"
        : "—",
  };
  $("runDetails").innerHTML = Object.entries(info)
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`)
    .join("");
  if (
    s.provider?.job_url &&
    s.provider.job_url.startsWith("https://modal.com/")
  )
    $("runDetails").innerHTML +=
      `<dt>Provider job</dt><dd><a href="${esc(s.provider.job_url)}" target="_blank" rel="noopener">Open Modal job ↗</a></dd>`;
}
function chart() {
  const available = Object.keys(results).length;
  $("probeCount").textContent = available + " / 36";
  $("resultState").textContent =
    available === 36
      ? "Held-out results"
      : available
        ? "Partial held-out results"
        : "Awaiting evaluation";
  $("resultRows").innerHTML = keys
    .map((k) => {
      const r = results[k];
      return `<tr><td>${k.replace("__", " / ")}</td>${r ? `<td>${r.layer}</td><td>${pct(r.calibration.balanced_accuracy)}</td><td>${pct(r.heldout.balanced_accuracy)}</td><td>${r.heldout_bootstrap.balanced_accuracy_95_interval.map(pct).join(" – ")}</td><td>${pct(r.heldout.recall)}</td><td>${pct(r.heldout.mean_negative_condition_fpr)}</td><td>${r.threshold.toFixed(4)}</td>` : calibration[k] ? `<td>—</td><td>${pct(calibration[k].balanced_accuracy)}</td><td colspan="5">Calibration only · held-out pending</td>` : '<td colspan="7">Pending</td>'}</tr>`;
    })
    .join("");
  if (!available) {
    $("chart").classList.add("empty");
    $("chart").textContent =
      "Results appear after fitting and calibration. Empty bars are not zero scores.";
    return;
  }
  $("chart").classList.remove("empty");
  let svg =
    '<svg viewBox="0 0 1200 350" role="img" aria-label="36 probe balanced accuracies">';
  const y = (v) => 285 - v * 240;
  for (const v of [0, 0.25, 0.5, 0.75, 1])
    svg += `<line x1="50" x2="1185" y1="${y(v)}" y2="${y(v)}" stroke="#dce3ec" ${v === 0.5 ? 'stroke-dasharray="5 5"' : ""}/><text x="5" y="${y(v) + 4}" font-size="11" fill="#64748b">${v * 100}%</text>`;
  emotions.forEach((e, i) => {
    svg += `<text x="${96 + i * 94}" y="315" text-anchor="middle" font-size="11" fill="#64748b">${e}</text>`;
    recipes.forEach((recipe, j) => {
      const r = results[e + "__" + recipe],
        x = 62 + i * 94 + j * 22;
      if (!r) {
        svg += `<rect x="${x}" y="282" width="19" height="3" fill="#ddd"/>`;
        return;
      }
      const v = r.heldout.balanced_accuracy,
        [lo, hi] = r.heldout_bootstrap.balanced_accuracy_95_interval;
      svg += `<rect x="${x}" y="${y(v)}" width="19" height="${v * 240}" fill="${colors[j]}"><title>${e} / ${recipe}: ${pct(v)}, 95% interval ${pct(lo)}–${pct(hi)}</title></rect><path d="M${x + 9.5},${y(lo)}V${y(hi)}M${x + 5},${y(lo)}h9M${x + 5},${y(hi)}h9" fill="none" stroke="#334155"/>`;
    });
  });
  $("chart").innerHTML = svg + "</svg>";
}
function layers() {
  const key = $("layerProbe").value,
    r = cv[key];
  if (!r) {
    $("layerChart").classList.add("empty");
    $("layerChart").textContent =
      "Cross-validation results will appear after fitting.";
    $("layerSummary").textContent = "";
    return;
  }
  $("layerChart").classList.remove("empty");
  const n = r.mean_auroc.length,
    x = (i) => 50 + (i * 1080) / Math.max(1, n - 1),
    y = (v) => 290 - v * 240;
  let svg =
    '<svg viewBox="0 0 1200 340" role="img" aria-label="Training cross-validation AUROC by layer">';
  for (const v of [0, 0.25, 0.5, 0.75, 1])
    svg += `<line x1="50" x2="1130" y1="${y(v)}" y2="${y(v)}" stroke="#e4eaf1"/><text x="5" y="${y(v) + 4}" font-size="11">${pct(v)}</text>`;
  for (const fold of r.fold_aurocs)
    svg += `<polyline points="${fold.map((v, i) => x(i) + "," + y(v)).join(" ")}" fill="none" stroke="#cad6e5" stroke-width="1"/>`;
  svg += `<polyline points="${r.mean_auroc.map((v, i) => x(i) + "," + y(v)).join(" ")}" fill="none" stroke="#416fa4" stroke-width="3"/><line x1="${x(r.selected_layer)}" x2="${x(r.selected_layer)}" y1="35" y2="290" stroke="#d88443" stroke-dasharray="5 5"/>`;
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 12)))
    svg += `<text x="${x(i)}" y="315" font-size="11" text-anchor="middle">${i}</text>`;
  $("layerChart").innerHTML = svg + "</svg>";
  $("layerSummary").textContent =
    "Selected layer " +
    r.selected_layer +
    " · Mean CV AUROC " +
    pct(r.mean_auroc[r.selected_layer]) +
    " · Pale lines: individual folds · Blue: fold mean";
}
$("layerProbe").onchange = layers;
function spanStyle(r) {
  if (r.score == null) return "";
  const red = r.score >= r.threshold;
  return `color:${red ? "#8b1525" : "#154e8b"};background:rgba(${red ? "210,55,70" : "45,110,210"},${0.08 + 0.3 * r.color_distance})`;
}
let examplesRetry = null,
  examplesRequest = 0;
async function examples() {
  clearTimeout(examplesRetry);
  const request = ++examplesRequest;
  try {
    const q = new URLSearchParams({
      probe: $("exampleProbe").value,
      split: $("exampleSplit").value,
      outcome: $("exampleOutcome").value,
      label: $("exampleLabel").value,
      offset,
    });
    const data = await get("/api/examples?" + q);
    if (request !== examplesRequest) return;
    total = data.total;
    $("exampleOutcome").disabled = !data.ready;
    if (!data.ready) examplesRetry = setTimeout(() => { if (currentTab === "examples") examples(); }, 3000);
    $("exampleCount").textContent = total
      ? `${offset + 1}–${Math.min(offset + 20, total)} of ${total} examples`
      : state && state.stages.evaluate.state !== "complete"
        ? "Example scores will appear after evaluation. Waiting for the run…"
        : "No examples match these filters.";
    $("prev").disabled = offset === 0;
    $("next").disabled = offset + 20 >= total;
    $("exampleRows").innerHTML = data.rows
      .map(
        (r) =>
          `<article class="example"><div class="meta">${esc(r.id)} · ${esc(r.label)} · ${esc(r.split)} · ${esc(r.outcome)} · ${r.score == null ? "Scores pending" : `score ${r.score.toFixed(4)} / cutoff ${r.threshold.toFixed(4)}`}</div><blockquote style="${spanStyle(r)}">${esc(r.target)}</blockquote><details data-id="${esc(r.id)}" data-color="${spanStyle(r)}"><summary>Inspect conversation and marked target</summary><div class="context">Loading…</div></details></article>`,
      )
      .join("");
    document.querySelectorAll("details[data-id]").forEach(
      (el) =>
        (el.ontoggle = async () => {
          if (!el.open || el.dataset.loaded) return;
          try {
            const c = await get(
              "/api/example?id=" + encodeURIComponent(el.dataset.id),
            );
            const container = el.querySelector(".context");
            container.innerHTML = c.messages
              .map((m, i) => {
                let body = "";
                for (const stream of ["reasoning", "content"]) {
                  if (!m[stream]) continue;
                  let text =
                    c.render_mode === "sales_with_reasoning"
                      ? m[stream].trim()
                      : m[stream];
                  const t = c.target;
                  const chars = Array.from(text);
                  body +=
                    `<div class="message"><b>${esc(m.role)} / ${stream}</b><br>` +
                    (i === t.message_index && stream === t.stream
                      ? esc(chars.slice(0, t.char_start).join("")) +
                        `<mark style="${el.dataset.color}">` +
                        esc(chars.slice(t.char_start, t.char_end).join("")) +
                        "</mark>" +
                        esc(chars.slice(t.char_end).join(""))
                      : esc(text)) +
                    "</div>";
                }
                return body;
              })
              .join("");
            el.dataset.loaded = "true";
          } catch (e) {
            el.querySelector(".context").textContent = e.message;
          }
        }),
    );
  } catch (e) {
    if (request !== examplesRequest) return;
    $("exampleRows").replaceChildren();
    $("prev").disabled = true;
    $("next").disabled = true;
    $("exampleCount").textContent = e.status === 404
      ? "Example files are downloading from the worker. Retrying automatically…"
      : "Could not load examples. Retrying automatically…";
    examplesRetry = setTimeout(() => {
      if (currentTab === "examples") examples();
    }, 2000);
  }
}
for (const id of [
  "exampleProbe",
  "exampleSplit",
  "exampleOutcome",
  "exampleLabel",
])
  $(id).onchange = () => {
    offset = 0;
    examples();
  };
$("prev").onclick = () => {
  offset = Math.max(0, offset - 20);
  examples();
};
$("next").onclick = () => {
  offset += 20;
  examples();
};
$("pause").onclick = () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume viewer" : "Pause viewer";
  if (!paused && state) raw(state);
};
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("rawStatus").textContent);
    $("copy").textContent = "Copied";
    setTimeout(() => ($("copy").textContent = "Copy JSON"), 1500);
  } catch {
    $("copy").textContent = "Select JSON to copy";
  }
};
let lastData = "";
async function poll() {
  try {
    if (!spec) spec = await get("/api/spec");
    const s = await get("/api/status");
    showStatus(s);
    const dataKey = JSON.stringify([
      s.stages.fit.state,
      s.stages.evaluate.state,
      s.stages.evaluate.completed,
      s.stages.evaluate.phase,
      s.stages.report.state,
    ]);
    if (dataKey !== lastData) {
      const [r, c, v, cal] = await Promise.all([
        get("/api/results"),
        get("/api/cv"),
        get("/api/verification"),
        get("/api/calibration"),
      ]);
      results = r.results || {};
      calibration = cal.results || {};
      cv = c;
      chart();
      layers();
      $("verifyState").textContent =
        v.status === "passed" ? "Passed" : "Pending";
      $("gates").innerHTML = (
        v.checks || [
          "Frozen inputs and extraction identity",
          "Training CV and 36 unit vectors",
          "Calibration optimum and exported predictions",
          "Report assets and link checks",
        ]
      )
        .map(
          (t) =>
            "<li>" + (v.status === "passed" ? "✓ " : "○ ") + esc(t) + "</li>",
        )
        .join("");
      $("artifacts").innerHTML = (
        v.status === "passed"
          ? [
              "report.html",
              "all_emotions_unified.svg",
              "all_emotions_unified.png",
              "all_emotions_unified.pdf",
              "calibrated_models.json",
              "verification.json",
            ]
          : ["run_spec.json"]
      )
        .map((n) => `<a href="/files/${n}">${n}</a>`)
        .join("");
      lastData = dataKey;
      if (currentTab === "examples") examples();
    }
    const events = await get("/api/events");
    $("eventLog").textContent = events.map((e) => JSON.stringify(e)).join("\n");
  } catch (e) {
    $("connection").textContent = "Disconnected — retrying";
    $("lastSeen").textContent = e.message;
  } finally {
    setTimeout(poll, 2000);
  }
}
chart();
tab(
  location.pathname === "/status"
    ? "status"
    : ["overview", "layers", "examples", "status"].includes(
          location.hash.slice(1),
        )
      ? location.hash.slice(1)
      : "overview",
);
poll();
