// Image Compare — frontend logic. Vanilla JS, no build step.

const $ = (sel) => document.querySelector(sel);

const state = {
  mode: "hash",
  browseTarget: null,
  browseCurrentPath: null,
  currentJobId: null,
  pollTimer: null,
  logPollTimer: null,
  matches: [],
  logExpanded: false,
  knownLogCount: 0,
};

// ---------------------------------------------------------------
// Settings sliders
// ---------------------------------------------------------------
function bindSlider(inputId, labelId, suffix = "%") {
  const input = $("#" + inputId);
  const label = $("#" + labelId);
  input.addEventListener("input", () => {
    label.textContent = input.value + suffix;
  });
}
bindSlider("visualThreshold", "visualThresholdVal");
bindSlider("filenameThreshold", "filenameThresholdVal");

// ---------------------------------------------------------------
// Mode toggle
// ---------------------------------------------------------------
$("#modeToggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;
  state.mode = btn.dataset.mode;
  document.querySelectorAll("#modeToggle button").forEach((b) => b.classList.toggle("active", b === btn));
  if (state.mode === "clip" || state.mode === "dino") {
    checkModelStatus(state.mode);
  } else {
    $("#clipStatus").textContent = "";
    $("#clipStatus").className = "clip-status";
  }
});

async function checkModelStatus(mode) {
  const el = $("#clipStatus");
  const label = mode === "dino" ? "DINOv2" : "CLIP";
  el.textContent = `checking ${label} availability…`;
  el.className = "clip-status";
  try {
    const res = await fetch(`/api/${mode}-status`);
    const data = await res.json();
    if (!data.dependencies_installed) {
      el.className = "clip-status warn";
      el.innerHTML = `⚠ ${label} dependencies not installed — run <span class="hint-code">${data.install_hint}</span>, then restart the server.`;
    } else if (data.weights_cache_dir_size_mb < 50) {
      el.className = "clip-status warn";
      el.textContent = `✓ ${label} ready — first run will download model weights.`;
    } else {
      el.className = "clip-status ok";
      el.textContent = `✓ ${label} ready (weights cache: ${Math.round(data.weights_cache_dir_size_mb)}MB)`;
    }
  } catch {
    el.className = "clip-status warn";
    el.textContent = `Could not check ${label} status.`;
  }
}

// ---------------------------------------------------------------
// Live scan-count preview (debounced) for both folder inputs
// ---------------------------------------------------------------
function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

async function refreshScanCount(folderInputId, countElId) {
  const folder = $("#" + folderInputId).value.trim();
  const countEl = $("#" + countElId);
  if (!folder) {
    countEl.textContent = "";
    return;
  }
  try {
    const res = await fetch("/api/scan-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        folder,
        recursive: $("#recursive").checked,
        include_patterns: $("#includePatterns").value,
        exclude_patterns: $("#excludePatterns").value,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      countEl.textContent = err.detail || "Folder not found";
      return;
    }
    const data = await res.json();
    countEl.innerHTML = `<span class="num">${data.count}</span> image${data.count === 1 ? "" : "s"} match current filters`;
  } catch {
    countEl.textContent = "";
  }
}

const debouncedRefresh1 = debounce(() => refreshScanCount("folder1", "count1"), 400);
const debouncedRefresh2 = debounce(() => refreshScanCount("folder2", "count2"), 400);

["folder1"].forEach(() => $("#folder1").addEventListener("input", debouncedRefresh1));
["folder2"].forEach(() => $("#folder2").addEventListener("input", debouncedRefresh2));
["recursive", "includePatterns", "excludePatterns"].forEach((id) => {
  $("#" + id).addEventListener("input", () => {
    debouncedRefresh1();
    debouncedRefresh2();
  });
});

// ---------------------------------------------------------------
// Drag & drop folder upload
//
// Browsers deliberately hide the real filesystem path of dropped
// files/folders, so this walks the dropped directory tree client-side
// (via the de-facto-standard webkitGetAsEntry API) and uploads the
// files to the server, which reconstructs the folder structure and
// hands back a real server-side path -- same as if the user had typed
// or browsed to it.
// ---------------------------------------------------------------
function setupDropZone(zoneId, targetInputId) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add("drag-active");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove("drag-active");
    })
  );
  zone.addEventListener("drop", (e) => handleFolderDrop(e, targetInputId));
}

function setDropStatus(targetInputId, text, isError = false) {
  const el = document.getElementById(targetInputId + "DropStatus");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function walkEntry(entry, prefix, collected) {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file(
        (file) => {
          collected.push({ file, relativePath: prefix + entry.name });
          resolve();
        },
        () => resolve() // unreadable file — skip rather than fail the whole drop
      );
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const readAllEntries = () => {
        reader.readEntries((entries) => {
          if (entries.length === 0) {
            resolve();
            return;
          }
          Promise.all(entries.map((child) => walkEntry(child, prefix + entry.name + "/", collected))).then(
            readAllEntries // directory readers may need multiple reads to exhaust large folders
          );
        }, () => resolve());
      };
      readAllEntries();
    } else {
      resolve();
    }
  });
}

async function handleFolderDrop(e, targetInputId) {
  const items = e.dataTransfer.items;
  if (!items || items.length === 0) return;

  const topLevelEntries = [];
  for (let i = 0; i < items.length; i++) {
    const getEntry = items[i].webkitGetAsEntry || items[i].getAsEntry;
    const entry = getEntry ? getEntry.call(items[i]) : null;
    if (entry) topLevelEntries.push(entry);
  }

  if (topLevelEntries.length === 0) {
    setDropStatus(targetInputId, "Couldn't read the dropped item — try Browse instead.", true);
    return;
  }

  setDropStatus(targetInputId, "Reading folder…");
  const collected = [];
  await Promise.all(topLevelEntries.map((entry) => walkEntry(entry, "", collected)));

  if (collected.length === 0) {
    setDropStatus(targetInputId, "No files found in the dropped item.", true);
    return;
  }

  if (collected.length > 3000) {
    const proceed = confirm(
      `This folder contains ${collected.length} files. Uploading that many may take a while — continue?`
    );
    if (!proceed) {
      setDropStatus(targetInputId, "");
      return;
    }
  }

  uploadDroppedFiles(collected, targetInputId);
}

function uploadDroppedFiles(collected, targetInputId) {
  const formData = new FormData();
  for (const { file, relativePath } of collected) {
    formData.append("files", file, relativePath);
  }

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload-folder");

  xhr.upload.addEventListener("progress", (evt) => {
    if (evt.lengthComputable) {
      const pct = Math.round((evt.loaded / evt.total) * 100);
      setDropStatus(targetInputId, `Uploading ${collected.length} file(s)… ${pct}%`);
    } else {
      setDropStatus(targetInputId, `Uploading ${collected.length} file(s)…`);
    }
  });

  xhr.onload = () => {
    if (xhr.status === 200) {
      const data = JSON.parse(xhr.responseText);
      $("#" + targetInputId).value = data.path;
      $("#" + targetInputId).dispatchEvent(new Event("input"));
      setDropStatus(targetInputId, `✓ Uploaded ${data.file_count} file(s)`);
    } else {
      let msg = "Upload failed.";
      try {
        msg = JSON.parse(xhr.responseText).detail || msg;
      } catch {
        /* keep default message */
      }
      setDropStatus(targetInputId, msg, true);
    }
  };
  xhr.onerror = () => setDropStatus(targetInputId, "Upload failed (network error).", true);

  xhr.send(formData);
  setDropStatus(targetInputId, `Uploading ${collected.length} file(s)… 0%`);
}

setupDropZone("folder1Dropzone", "folder1");
setupDropZone("folder2Dropzone", "folder2");

// prevent the browser from navigating away / opening the file if a drop
// lands just outside a dropzone
["dragover", "drop"].forEach((evt) =>
  window.addEventListener(evt, (e) => {
    if (!e.target.closest(".dropzone")) e.preventDefault();
  })
);

// ---------------------------------------------------------------
// Browse modal (server-side folder picker)
// ---------------------------------------------------------------
document.querySelectorAll("[data-browse-target]").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.browseTarget = btn.dataset.browseTarget;
    const existing = $("#" + state.browseTarget).value.trim();
    openBrowse(existing || null);
  });
});

async function openBrowse(path) {
  $("#browseBackdrop").classList.add("visible");
  await loadBrowseDir(path);
}

async function loadBrowseDir(path) {
  const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#browsePath").textContent = err.detail || "Unable to open directory";
    return;
  }
  const data = await res.json();
  state.browseCurrentPath = data.current_path;
  $("#browsePath").textContent = data.current_path;

  const list = $("#browseList");
  list.innerHTML = "";
  data.entries
    .filter((e) => e.is_dir)
    .forEach((entry) => {
      const row = document.createElement("div");
      row.className = "browse-item";
      row.innerHTML = `<span class="icon">▸</span> ${escapeHtml(entry.name)}`;
      row.addEventListener("click", () => loadBrowseDir(entry.path));
      list.appendChild(row);
    });

  if (data.entries.filter((e) => e.is_dir).length === 0) {
    list.innerHTML = `<div class="browse-item" style="cursor:default; color: var(--muted-dim);">No subfolders here</div>`;
  }
}

$("#browseUp").addEventListener("click", async () => {
  const res = await fetch(`/api/browse?path=${encodeURIComponent(state.browseCurrentPath)}`);
  const data = await res.json();
  if (data.parent_path) await loadBrowseDir(data.parent_path);
});

$("#browseComputer").addEventListener("click", () => loadBrowseDir("__DRIVES__"));

$("#browseChoose").addEventListener("click", () => {
  if (state.browseCurrentPath === "This PC") return; // not a real, usable folder
  if (state.browseTarget && state.browseCurrentPath) {
    $("#" + state.browseTarget).value = state.browseCurrentPath;
    $("#" + state.browseTarget).dispatchEvent(new Event("input"));
  }
  $("#browseBackdrop").classList.remove("visible");
});

$("#browseClose").addEventListener("click", () => $("#browseBackdrop").classList.remove("visible"));
$("#browseBackdrop").addEventListener("click", (e) => {
  if (e.target === $("#browseBackdrop")) $("#browseBackdrop").classList.remove("visible");
});

// ---------------------------------------------------------------
// Run comparison
// ---------------------------------------------------------------
function showError(msg) {
  const banner = $("#errorBanner");
  banner.textContent = msg;
  banner.classList.add("visible");
}
function clearError() {
  $("#errorBanner").classList.remove("visible");
}

$("#runBtn").addEventListener("click", startComparison);

async function startComparison() {
  clearError();
  const folder1 = $("#folder1").value.trim();
  const folder2 = $("#folder2").value.trim();
  if (!folder1 || !folder2) {
    showError("Please provide both folder paths.");
    return;
  }

  const payload = {
    folder1,
    folder2,
    mode: state.mode,
    recursive: $("#recursive").checked,
    include_patterns: $("#includePatterns").value,
    exclude_patterns: $("#excludePatterns").value,
    visual_threshold: parseFloat($("#visualThreshold").value),
    filename_threshold: parseFloat($("#filenameThreshold").value),
    deep_rotation: $("#deepRotation").checked,
    hash_size: 16,
  };

  $("#runBtn").disabled = true;
  $("#runHint").textContent = "starting…";
  $("#resultsSection").style.display = "none";
  $("#summaryPanel").innerHTML = "";
  resetPauseButton();
  resetLogPanel();

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || "Could not start comparison.");
      $("#runBtn").disabled = false;
      $("#runHint").textContent = "";
      return;
    }
    const data = await res.json();
    state.currentJobId = data.job_id;
    $("#progressPanel").classList.add("visible");
    pollJob();
  } catch (e) {
    showError("Network error starting comparison.");
    $("#runBtn").disabled = false;
    $("#runHint").textContent = "";
  }
}

function formatEta(seconds) {
  if (seconds == null) return "—";
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function resetPauseButton() {
  const btn = $("#pauseBtn");
  btn.dataset.paused = "0";
  btn.textContent = "Pause";
  btn.disabled = false;
  $("#stopBtn").disabled = false;
  $("#progressFill").classList.remove("paused");
}

$("#pauseBtn").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  const btn = $("#pauseBtn");
  const isPaused = btn.dataset.paused === "1";
  const endpoint = isPaused ? "resume" : "pause";
  btn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${state.currentJobId}/${endpoint}`, { method: "POST" });
    if (res.ok) {
      btn.dataset.paused = isPaused ? "0" : "1";
      btn.textContent = isPaused ? "Pause" : "Resume";
      $("#progressFill").classList.toggle("paused", !isPaused);
    }
  } finally {
    btn.disabled = false;
  }
});

$("#stopBtn").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  const btn = $("#stopBtn");
  btn.disabled = true;
  btn.textContent = "Stopping…";
  await fetch(`/api/jobs/${state.currentJobId}/stop`, { method: "POST" });
});

async function pollJob() {
  if (!state.currentJobId) return;
  try {
    await fetchAndRenderLogs(state.currentJobId);

    const res = await fetch(`/api/jobs/${state.currentJobId}`);
    const progress = await res.json();

    $("#progressPhase").textContent =
      progress.status === "paused" ? "paused" : progress.phase || progress.status;
    $("#progressStats").textContent =
      `${progress.processed_items} / ${progress.total_items} · ${progress.percent}% · ETA ${formatEta(progress.eta_seconds)}`;
    $("#progressFill").style.width = `${progress.percent}%`;

    if (progress.status === "done") {
      $("#runBtn").disabled = false;
      $("#runHint").textContent = "";
      await loadResults();
      $("#progressPanel").classList.remove("visible");
      return;
    }
    if (progress.status === "error") {
      $("#runBtn").disabled = false;
      $("#runHint").textContent = "";
      $("#progressPanel").classList.remove("visible");
      showError(progress.error_message || "Comparison failed.");
      return;
    }
    if (progress.status === "cancelled") {
      $("#runBtn").disabled = false;
      $("#runHint").textContent = "";
      $("#progressPanel").classList.remove("visible");
      showError("Comparison stopped.");
      return;
    }
    state.pollTimer = setTimeout(pollJob, 700);
  } catch (e) {
    state.pollTimer = setTimeout(pollJob, 1500);
  }
}

// ---------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------
async function loadResults() {
  const res = await fetch(`/api/jobs/${state.currentJobId}/results`);
  const data = await res.json();
  state.matches = data.matches;
  renderSummary(data.summary);
  renderResults();
}

$("#exportXlsxBtn").addEventListener("click", () => {
  if (!state.currentJobId) return;
  // a plain navigation (not fetch) so the browser handles the
  // Content-Disposition: attachment header as a normal file download
  const link = document.createElement("a");
  link.href = `/api/jobs/${state.currentJobId}/export.xlsx`;
  link.download = "";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

function renderSummary(summary) {
  const panel = $("#summaryPanel");
  if (!summary) {
    panel.innerHTML = "";
    return;
  }

  const modeLabel = { hash: "Hash", clip: "CLIP", dino: "DINOv2" }[summary.mode] || summary.mode;
  const fmtPct = (v) => (v == null ? "—" : `${v.toFixed(1)}%`);
  const fmtTime = (s) => (s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

  const stats = [
    ["Mode", modeLabel],
    ["Deep Rotation", summary.deep_rotation_used ? "on" : "off"],
    ["Images (A / B)", `${summary.total_images_a} / ${summary.total_images_b}`],
    ["Pairs compared", summary.total_pairs_compared.toLocaleString()],
    ["Matches found", summary.total_matches.toLocaleString(), true],
    ["Exact duplicates (MD5)", summary.exact_duplicates.toLocaleString()],
    ["Highest similarity", fmtPct(summary.highest_similarity)],
    ["Average similarity", fmtPct(summary.average_similarity)],
    ["Lowest similarity", fmtPct(summary.lowest_similarity)],
    ["Threshold", fmtPct(summary.visual_threshold)],
    ["Elapsed", fmtTime(summary.elapsed_seconds)],
  ];

  panel.innerHTML = `
    <h3 class="summary-title">Summary</h3>
    <div class="summary-grid">
      ${stats
        .map(
          ([label, value, accent]) => `
        <div class="summary-stat">
          <span class="stat-label">${label}</span>
          <span class="stat-value${accent ? " accent" : ""}">${value}</span>
        </div>`
        )
        .join("")}
    </div>`;
}

function meterColor(score) {
  if (score >= 90) return "var(--meter-high)";
  if (score >= 70) return "var(--meter-mid)";
  return "var(--meter-low)";
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderResults() {
  $("#resultsSection").style.display = "block";
  $("#resultsCount").textContent = `${state.matches.length} match${state.matches.length === 1 ? "" : "es"}`;

  const body = $("#resultsBody");
  if (state.matches.length === 0) {
    body.innerHTML = `<div class="empty-state">No matches at or above the current threshold. Try lowering the visual threshold or enabling Deep Rotation.</div>`;
    return;
  }

  const sprockets = `<div class="sprocket-row">${Array.from({ length: 10 }).map(() => "<span></span>").join("")}</div>`;

  body.innerHTML = `<div class="match-grid">${state.matches
    .map((m, idx) => {
      const color = meterColor(m.visual_similarity);
      return `
      <div class="match-card" data-idx="${idx}">
        ${sprockets}
        <div class="frame-pair">
          <div class="frame" data-preview="${idx}">
            <img src="${m.thumb1_url}" loading="lazy" alt="${escapeHtml(m.image1_name)}" />
            <span class="preview-btn">Preview</span>
          </div>
          <div class="frame" data-preview="${idx}">
            <img src="${m.thumb2_url}" loading="lazy" alt="${escapeHtml(m.image2_name)}" />
            <span class="preview-btn">Preview</span>
          </div>
        </div>
        ${sprockets}
        <div class="card-body">
          <div class="filenames">
            <span title="${escapeHtml(m.image1_relative_path)}">${escapeHtml(m.image1_name)}</span>
            <span title="${escapeHtml(m.image2_relative_path)}">${escapeHtml(m.image2_name)}</span>
          </div>
          <div class="meter">
            <div class="meter-track">
              <div class="meter-fill" style="width:${m.visual_similarity}%; background:${color};"></div>
              <div class="meter-needle" style="left:${m.visual_similarity}%;"></div>
            </div>
            <div class="meter-score" style="color:${color};">${m.visual_similarity.toFixed(1)}%</div>
          </div>
          <div class="badge-row">
            ${m.is_exact_duplicate ? `<span class="badge dup">EXACT DUPLICATE · MD5</span>` : ""}
            ${m.best_rotation_angle ? `<span class="badge rotation">best @ ${m.best_rotation_angle}°</span>` : ""}
            <span class="badge fname">filename ${m.filename_similarity.toFixed(0)}%</span>
          </div>
        </div>
      </div>`;
    })
    .join("")}</div>`;

  document.querySelectorAll("[data-preview]").forEach((el) => {
    el.addEventListener("click", () => openPreview(parseInt(el.dataset.preview, 10)));
  });
}

// ---------------------------------------------------------------
// Activity log
// ---------------------------------------------------------------
$("#logToggle").addEventListener("click", () => {
  state.logExpanded = !state.logExpanded;
  $("#logToggle").setAttribute("aria-expanded", String(state.logExpanded));
  $("#logBody").hidden = !state.logExpanded;
});

function resetLogPanel() {
  state.knownLogCount = 0;
  $("#logBody").innerHTML = "";
  $("#logCount").textContent = "";
}

function formatLogTimestamp(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour12: false });
  } catch {
    return iso;
  }
}

async function fetchAndRenderLogs(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}/logs`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.entries.length === state.knownLogCount) return; // nothing new

    const body = $("#logBody");
    const wasAtBottom = body.scrollTop + body.clientHeight >= body.scrollHeight - 8;

    const newEntries = data.entries.slice(state.knownLogCount);
    for (const entry of newEntries) {
      const line = document.createElement("div");
      line.className = `log-line level-${entry.level}`;
      line.innerHTML = `<span class="log-dot"></span><span class="log-ts">${formatLogTimestamp(entry.timestamp)}</span><span class="log-msg"></span>`;
      line.querySelector(".log-msg").textContent = entry.message;
      body.appendChild(line);
    }
    state.knownLogCount = data.entries.length;
    $("#logCount").textContent = `(${data.entries.length})`;

    if (wasAtBottom || newEntries.length === data.entries.length) {
      body.scrollTop = body.scrollHeight;
    }
  } catch {
    // logs are best-effort; a transient fetch failure shouldn't disrupt the run
  }
}

// ---------------------------------------------------------------
// Preview modal (side-by-side, full resolution)
// ---------------------------------------------------------------
function openPreview(idx) {
  const m = state.matches[idx];
  const url1 = `/api/image?job_id=${state.currentJobId}&path=${encodeURIComponent(m.image1_path)}`;
  const url2 = `/api/image?job_id=${state.currentJobId}&path=${encodeURIComponent(m.image2_path)}`;
  $("#previewImg1").src = url1;
  $("#previewImg2").src = url2;
  $("#previewMeta1").innerHTML = `<strong>${escapeHtml(m.image1_name)}</strong>${escapeHtml(m.image1_relative_path)}`;
  $("#previewMeta2").innerHTML = `<strong>${escapeHtml(m.image2_name)}</strong>${escapeHtml(m.image2_relative_path)}`;
  $("#previewBackdrop").classList.add("visible");
  state.previewIdx = idx;
}

$("#previewClose").addEventListener("click", () => $("#previewBackdrop").classList.remove("visible"));
$("#previewBackdrop").addEventListener("click", (e) => {
  if (e.target === $("#previewBackdrop")) $("#previewBackdrop").classList.remove("visible");
});

document.addEventListener("keydown", (e) => {
  if (!$("#previewBackdrop").classList.contains("visible")) return;
  if (e.key === "Escape") $("#previewBackdrop").classList.remove("visible");
  if (e.key === "ArrowRight" && state.previewIdx < state.matches.length - 1) openPreview(state.previewIdx + 1);
  if (e.key === "ArrowLeft" && state.previewIdx > 0) openPreview(state.previewIdx - 1);
});
