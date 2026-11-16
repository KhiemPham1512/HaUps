import { createPreview } from "./preview.js";

const $ = (selector) => document.querySelector(selector);
let selected = [];
let lastJobs = "";
let uploading = false;
let capabilities = null;
const viewer = createPreview();
let previewJobId = null;
let previewJobKey = null;
let previewMode = "upscale";
let previewHadWallpaper = false;
let dimensions = null;
let viewedScale = null;

let focusedFile = null;
let localPreviewUrl = null;
let currentJobs = [];
let focusVersion = 0;
const selectedJobs = new Set();
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const MAX_BATCH_BYTES = 1024 * 1024 * 1024;
const fileDimensions = new WeakMap();
const scaleValue = () => $('input[name="scale"]:checked').value;
const jobHasResult = (job) =>
  Boolean(job.available_scales?.length || job.status === "done");

function formatBytes(bytes) {
  if (bytes == null || !(bytes > 0)) return "";
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1).replace(".", ",")} MB`;
}

function updateBulkBar() {
  const completed = currentJobs.filter(jobHasResult);
  const validIds = new Set(completed.map((job) => job.id));
  for (const id of [...selectedJobs]) {
    if (!validIds.has(id)) selectedJobs.delete(id);
  }
  const selectAll = $("#select-all-jobs");
  selectAll.disabled = completed.length === 0;
  selectAll.checked =
    completed.length > 0 && selectedJobs.size === completed.length;
  selectAll.indeterminate =
    selectedJobs.size > 0 && selectedJobs.size < completed.length;
  const includeWallpaper = $("#include-wallpaper").checked;
  let totalBytes = 0;
  for (const job of completed) {
    if (!selectedJobs.has(job.id)) continue;
    totalBytes += job.result_bytes?.[String(job.scale)] || 0;
    if (includeWallpaper && job.wallpaper_available) {
      totalBytes += job.wallpaper_bytes || 0;
    }
  }
  const totalSize = formatBytes(totalBytes);
  $("#bulk-selected-count").textContent = totalSize
    ? `Đã chọn ${selectedJobs.size} ảnh · ${totalSize}`
    : `Đã chọn ${selectedJobs.size} ảnh`;
  $("#download-selected").disabled = selectedJobs.size === 0;
  $("#download-all").disabled = completed.length === 0;
}

function triggerDownload(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  a.hidden = true;
  document.body.append(a);
  a.click();
  a.remove();
}

async function bulkDownload(jobsToDownload) {
  const includeWallpaper = $("#include-wallpaper").checked;
  const urls = [];
  for (const job of jobsToDownload) {
    const downloadUrl =
      job.download_url ||
      `${job.source_url.replace(/source$/, "download")}?scale=${job.scale}`;
    urls.push(downloadUrl);
    if (includeWallpaper && job.wallpaper_available) {
      urls.push(`/api/jobs/${job.id}/wallpaper`);
    }
  }
  for (let i = 0; i < urls.length; i++) {
    if (i > 0) await new Promise((resolve) => setTimeout(resolve, 300));
    triggerDownload(urls[i]);
  }
}

$("#select-all-jobs").addEventListener("change", (event) => {
  const completed = currentJobs.filter(jobHasResult);
  if (event.target.checked) {
    for (const job of completed) selectedJobs.add(job.id);
  } else {
    selectedJobs.clear();
  }
  for (const checkbox of document.querySelectorAll(".job-select")) {
    checkbox.checked = selectedJobs.has(checkbox.dataset.jobId);
  }
  updateBulkBar();
});

$("#include-wallpaper").addEventListener("change", updateBulkBar);

$("#download-selected").addEventListener("click", () => {
  bulkDownload(
    currentJobs.filter((job) => jobHasResult(job) && selectedJobs.has(job.id)),
  );
});

$("#download-all").addEventListener("click", () => {
  bulkDownload(currentJobs.filter(jobHasResult));
});

function updateScaleDetails() {
  $("#original-size").textContent = dimensions
    ? `${dimensions.width} × ${dimensions.height} px`
    : "Chọn ảnh để xem";
  $("#size-reference").textContent = dimensions?.filename || "";
  for (const factor of [2, 4, 8]) {
    // backend rejects 8x above 10 MP (queue.MAX_PIXELS_8X); say so before upload
    $(`[data-scale-size="${factor}"]`).textContent = !dimensions
      ? "Chưa có ảnh"
      : factor === 8 && dimensions.width * dimensions.height > 10_000_000
        ? "Ảnh > 10 MP: chỉ 2×/4×"
        : `${dimensions.width * factor} × ${dimensions.height * factor} px`;
  }
  $("#preset-help").textContent =
    $("#preset").value === "photo"
      ? "Ảnh chụp: cảnh quan, vật thể, người thật từ máy ảnh hoặc điện thoại."
      : "Anime / Minh họa: ảnh vẽ, truyện tranh, tranh kỹ thuật số.";
  $("#scale-hint").textContent =
    scaleValue() === "2"
      ? ($("#preset").value === "photo" ? "Nên bắt đầu ở 2× · " : "") +
        "2×: xử lý bằng model 4× rồi thu nhỏ."
      : scaleValue() === "8"
        ? "8× gấp 16 lần 2×: AI 4× rồi phóng thêm, không thêm bước phục hồi."
        : "4×: xử lý bằng model 4×, giữ đúng tỷ lệ gốc.";
}

async function readDimensions(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    return {
      width: image.naturalWidth,
      height: image.naturalHeight,
      filename: file.name,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function notice(message) {
  $("#notice").textContent = message;
  $("#notice").hidden = !message;
}
function updateSelection() {
  $("#selection").hidden = selected.length === 0;
  $("#selected-count").textContent = `${selected.length} ảnh đã chọn`;
  $("#selected-files").replaceChildren(
    ...selected.map((file) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "file-choice";
      button.textContent = file.name;
      button.setAttribute("aria-pressed", String(file === focusedFile));
      button.addEventListener("click", () => focusFile(file));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "text-button remove-file";
      remove.textContent = "Bỏ chọn";
      remove.setAttribute("aria-label", `Bỏ chọn ${file.name}`);
      remove.disabled = uploading;
      remove.addEventListener("click", () => {
        if (uploading) return;
        const index = selected.indexOf(file);
        selected.splice(index, 1);
        $("#files").value = "";
        if (!selected.length) {
          clearSelection();
          $("#files").focus();
          return;
        }
        if (focusedFile === file)
          focusFile(selected[Math.min(index, selected.length - 1)]);
        updateSelection();
        document.querySelectorAll(".remove-file")[
          Math.min(index, selected.length - 1)
        ].focus();
      });
      li.append(button, remove);
      return li;
    }),
  );
  $("#submit").disabled = !selected.length || uploading || !capabilities;
  for (const selector of [
    "#files",
    "#preset",
    "#clear-selection",
    "#scale-options",
  ])
    $(selector).disabled = uploading;
}
function syncPreviewFooter() {
  const footer = $("#preview-footer");
  if (!footer) return;
  footer.hidden =
    Boolean($("#result-group")?.hidden) &&
    Boolean($("#wallpaper-form")?.hidden);
}

function hideWallpaper() {
  previewMode = "upscale";
  previewHadWallpaper = false;
  if ($("#preview-mode")) {
    $("#preview-mode").hidden = true;
    $("#mode-upscale").setAttribute("aria-pressed", "true");
    $("#mode-wallpaper").setAttribute("aria-pressed", "false");
  }
  if ($("#wallpaper-prompt")) delete $("#wallpaper-prompt").dataset.jobId;
  $("#wallpaper-form").hidden = true;
  syncPreviewFooter();
}

async function focusFile(file) {
  hideWallpaper();
  focusedFile = file;
  previewJobId = null;
  previewJobKey = null;
  viewer.clear("Đang đọc ảnh để xem trước…", file.name);
  if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
  localPreviewUrl = null;
  const version = ++focusVersion;
  dimensions = null;
  updateScaleDetails();
  document
    .querySelectorAll(".file-choice")
    .forEach((button, index) =>
      button.setAttribute(
        "aria-pressed",
        String(index === selected.indexOf(file)),
      ),
    );
  try {
    const info = fileDimensions.get(file) || (await readDimensions(file));
    fileDimensions.set(file, info);
    if (version !== focusVersion || !selected.includes(file)) return;
    dimensions = info;
    updateScaleDetails();
    localPreviewUrl = URL.createObjectURL(file);
    viewer.show({
      name: file.name,
      sourceUrl: localPreviewUrl,
      message: `${info.width} × ${info.height} px · Ảnh gốc, chưa xử lý`,
    });
  } catch (_) {
    if (version === focusVersion) {
      viewer.clear(
        "Không đọc được ảnh. Hãy chọn PNG, JPG hoặc WebP hợp lệ.",
        file.name,
      );
      notice(
        "Không đọc được kích thước ảnh. Hãy chọn ảnh PNG, JPG hoặc WebP hợp lệ.",
      );
    }
  }
}
function choose(files) {
  if (uploading) return;
  if (!files.length) {
    clearSelection();
    return;
  }
  if (files.length > 20) {
    notice("Mỗi lần chọn tối đa 20 ảnh.");
    return;
  }
  const list = [...files];
  const oversized = list.find((file) => file.size > MAX_FILE_BYTES);
  if (oversized) {
    notice(`${oversized.name}: Vượt quá giới hạn 50 MB/ảnh.`);
    return;
  }
  const totalBytes = list.reduce((sum, file) => sum + file.size, 0);
  if (totalBytes > MAX_BATCH_BYTES) {
    notice("Tổng dung lượng vượt quá 1 GB mỗi lần tải.");
    return;
  }
  selected = list;

  if (selected[0]) focusFile(selected[0]);
  notice("");
  updateSelection();
}
$("#files").addEventListener("change", (event) => choose(event.target.files));
function clearSelection() {
  hideWallpaper();
  selected = [];
  focusVersion++;
  focusedFile = null;
  dimensions = null;
  previewJobId = null;
  previewJobKey = null;
  viewer.clear();
  if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
  localPreviewUrl = null;
  $("#files").value = "";
  updateScaleDetails();
  updateSelection();
  const latest = [...currentJobs]
    .reverse()
    .find((job) => job.status === "done");
  if (latest) compare(latest);
}
$("#clear-selection").addEventListener("click", clearSelection);
for (const event of ["dragenter", "dragover"])
  $("#dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("#dropzone").classList.add("drag");
  });
$("#dropzone").addEventListener("dragleave", () =>
  $("#dropzone").classList.remove("drag"),
);
$("#dropzone").addEventListener("drop", (e) => {
  e.preventDefault();
  $("#dropzone").classList.remove("drag");
  choose(e.dataTransfer.files);
});
$("#preset").addEventListener("change", updateScaleDetails);
for (const input of document.querySelectorAll('input[name="scale"]'))
  input.addEventListener("change", async () => {
    updateScaleDetails();
    if (!previewJobId || selected.length) return;
    const job = currentJobs.find((item) => item.id === previewJobId);
    if (!job) return;
    previewMode = "upscale";
    viewedScale = Number(scaleValue());
    const available =
      job.available_scales || (job.status === "done" ? [job.scale] : []);
    if (available.includes(viewedScale)) return compare(job, true);
    try {
      const response = await fetch(`/api/jobs/${job.id}/scale`, {
        method: "POST",
        body: new URLSearchParams({ scale: String(viewedScale) }),
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.detail || "Không tạo được kích thước này.");
      notice("");
      lastJobs = "";
      render(data.jobs);
    } catch (error) {
      notice(error.message);
      if (previewJobId !== job.id) return;
      viewedScale = job.scale;
      compare(job, true);
    }
  });
$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (uploading || !selected.length) return;
  const form = new FormData();
  selected.forEach((file) => form.append("files", file));
  form.append("scale", scaleValue());
  form.append("preset", $("#preset").value);
  uploading = true;
  updateSelection();
  notice("Đang tải ảnh lên ứng dụng local…");
  try {
    const response = await fetch("/api/batches", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail
            .map(
              (item) =>
                `${item.filename || ""}: ${item.error || "Lựa chọn không hợp lệ."}`,
            )
            .join("\n")
        : data.detail;
      throw new Error(detail || "Không thêm được ảnh vào hàng đợi.");
    }
    selected = [];
    focusedFile = null;
    focusVersion++;
    $("#files").value = "";
    if (data.jobs[0]) compare(data.jobs[0]);
    notice(
      data.rejected.length
        ? data.rejected
            .map((item) => `${item.filename}: ${item.error}`)
            .join("\n")
        : "",
    );
    await refresh(false);
  } catch (error) {
    notice(
      `${error.message}\nKiểm tra hàng đợi trước khi gửi lại để tránh thêm trùng.`,
    );
  } finally {
    uploading = false;
    updateSelection();
  }
});

function updateWallpaper(job) {
  const form = $("#wallpaper-form");
  const available =
    job.available_scales || (job.status === "done" ? [job.scale] : []);
  const hasWallpaper = Boolean(job.wallpaper_available);
  const modeToggle = $("#preview-mode");
  if (modeToggle) {
    modeToggle.hidden = !hasWallpaper;
    const showingWallpaper = previewMode === "wallpaper" && hasWallpaper;
    $("#mode-upscale").setAttribute("aria-pressed", String(!showingWallpaper));
    $("#mode-wallpaper").setAttribute("aria-pressed", String(showingWallpaper));
  }
  form.hidden =
    (!capabilities?.outpainting && !hasWallpaper) || !available.length;
  const promptInput = $("#wallpaper-prompt");
  if (promptInput.dataset.jobId !== job.id) {
    promptInput.dataset.jobId = job.id;
    promptInput.value = job.wallpaper_prompt ?? "";
  }
  const busy = ["queued", "running"].includes(job.status);
  const wallpaperBusy = job.task === "wallpaper" && busy;
  promptInput.disabled = wallpaperBusy || !capabilities?.outpainting;
  const completedPrompt = job.wallpaper_completed_prompt ?? "";
  const sceneEl = $("#wallpaper-scene");
  if (sceneEl) {
    const showScene =
      hasWallpaper &&
      completedPrompt === "" &&
      typeof job.wallpaper_scene === "string";
    const sceneText = showScene
      ? job.wallpaper_scene
        ? `AI nhận ra: ${job.wallpaper_scene}`
        : "AI không nhận ra bối cảnh — nên nhập mô tả rồi bấm Tạo lại."
      : "";
    sceneEl.textContent = sceneText;
    sceneEl.hidden = !showScene;
    if (sceneText) sceneEl.title = sceneText;
    else sceneEl.removeAttribute("title");
  }
  const promptUnchanged =
    hasWallpaper && promptInput.value.trim() === completedPrompt;
  $("#wallpaper-create").hidden = false;
  $("#wallpaper-create").textContent = hasWallpaper
    ? "Tạo lại"
    : "Tạo hình nền AI";
  $("#wallpaper-create").disabled =
    busy || !capabilities?.outpainting || promptUnchanged;
  const statusText =
    job.task === "wallpaper"
      ? {
          queued: "Đang chờ tạo nền AI…",
          running: "GPU đang tạo nền AI…",
          failed: job.error,
        }[job.status] || ""
      : "";
  $("#wallpaper-status").textContent = statusText;
  $("#wallpaper-status").hidden = !statusText;
  if (statusText) $("#wallpaper-status").title = statusText;
  else $("#wallpaper-status").removeAttribute("title");
  $("#wallpaper-result").hidden = !hasWallpaper;
  if (hasWallpaper) {
    const url = `/api/jobs/${job.id}/wallpaper`;
    const wallpaperSize = formatBytes(job.wallpaper_bytes);
    $("#wallpaper-download").href = url;
    $("#wallpaper-download").textContent = wallpaperSize
      ? `Tải hình nền · ${wallpaperSize}`
      : "Tải hình nền";
    $("#wallpaper-open").href = `${url}?inline=1`;
  }
  syncPreviewFooter();
}
$("#wallpaper-prompt").addEventListener("input", () => {
  const job = currentJobs.find((item) => item.id === previewJobId);
  if (job) updateWallpaper(job);
});
$("#mode-upscale").addEventListener("click", () => {
  if (previewMode === "upscale") return;
  previewMode = "upscale";
  const job = currentJobs.find((item) => item.id === previewJobId);
  if (job) compare(job, true);
});
$("#mode-wallpaper").addEventListener("click", () => {
  if (previewMode === "wallpaper") return;
  previewMode = "wallpaper";
  const job = currentJobs.find((item) => item.id === previewJobId);
  if (job?.wallpaper_available) compare(job, true);
});
$("#wallpaper-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const jobId = previewJobId;
  if (!jobId || $("#wallpaper-create").disabled) return;
  const prompt = $("#wallpaper-prompt").value.trim();
  $("#wallpaper-create").disabled = true;
  $("#wallpaper-prompt").disabled = true;
  try {
    const response = await fetch(`/api/jobs/${jobId}/wallpaper`, {
      method: "POST",
      body: new URLSearchParams({ prompt }),
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "Không tạo được nền AI.",
      );
    notice("");
    lastJobs = "";
    render(data.jobs);
  } catch (error) {
    notice(error.message);
    if (previewJobId === jobId) {
      const job = currentJobs.find((item) => item.id === jobId);
      if (job) updateWallpaper(job);
      else {
        $("#wallpaper-create").disabled = false;
        $("#wallpaper-prompt").disabled = false;
      }
    }
  }
});

function compare(job, force = false) {
  const isNewJob = previewJobId !== job.id;
  const wallpaperReady =
    Boolean(job.wallpaper_available) &&
    !(job.task === "wallpaper" && job.status !== "done");
  if (isNewJob) {
    previewMode = "upscale";
    previewHadWallpaper = wallpaperReady;
  } else if (!wallpaperReady) {
    previewHadWallpaper = false;
    if (!job.wallpaper_available) previewMode = "upscale";
  } else if (!previewHadWallpaper) {
    previewHadWallpaper = true;
    previewMode = "wallpaper";
  }
  updateWallpaper(job);
  const factor = !isNewJob && viewedScale ? viewedScale : job.scale;
  viewedScale = factor;
  const available =
    job.available_scales || (job.status === "done" ? [job.scale] : []);
  const showingWallpaper =
    previewMode === "wallpaper" && Boolean(job.wallpaper_available);
  const resultBytes = job.result_bytes?.[String(factor)] ?? "";
  const wallpaperBytes = job.wallpaper_bytes ?? "";
  const key = `${job.id}:${job.status}:${factor}:${available.join(",")}:${job.download_url}:${Boolean(job.wallpaper_available)}:${showingWallpaper ? "wallpaper" : "upscale"}:${resultBytes}:${wallpaperBytes}`;
  if (localPreviewUrl) {
    URL.revokeObjectURL(localPreviewUrl);
    localPreviewUrl = null;
  }
  if (previewJobKey === key && !force) return;
  focusVersion++;
  previewJobId = job.id;
  previewJobKey = key;
  dimensions = { width: job.width, height: job.height, filename: job.filename };
  focusedFile = null;
  for (const button of document.querySelectorAll(".file-choice"))
    button.setAttribute("aria-pressed", "false");
  if (!selected.length)
    $(`input[name="scale"][value="${factor}"]`).checked = true;
  updateScaleDetails();
  const completed = available.includes(factor);
  const downloadUrl = `${job.source_url.replace(/source$/, "download")}?scale=${factor}`;
  const message = showingWallpaper
    ? "Hình nền AI · 3840 × 2160"
    : completed
      ? `Kết quả ${factor}× · ${job.width * factor} × ${job.height * factor} px`
      : {
          queued: "Đang chờ xử lý",
          running: "GPU đang xử lý ảnh này",
          failed: job.error,
        }[job.status];
  viewer.show({
    name: job.filename,
    sourceUrl: job.source_url,
    resultUrl: completed ? `${downloadUrl}&inline=1` : null,
    wallpaperUrl: showingWallpaper
      ? `/api/jobs/${job.id}/wallpaper?inline=1`
      : null,
    downloadUrl: completed ? downloadUrl : null,
    message,
  });
  const resultSize = completed
    ? formatBytes(job.result_bytes?.[String(factor)])
    : "";
  $("#preview-download").textContent = resultSize
    ? `Tải PNG · ${resultSize}`
    : "Tải PNG";
}
function render(jobs) {
  currentJobs = jobs;
  if (previewJobId) {
    const current = jobs.find((job) => job.id === previewJobId);
    if (current) compare(current);
    else {
      previewJobId = null;
      previewJobKey = null;
      dimensions = null;
      hideWallpaper();
      viewer.clear();
      updateScaleDetails();
    }
  } else if (!selected.length && !dimensions) {
    const latest = [...jobs].reverse().find((job) => job.status === "done");
    if (latest) compare(latest);
  }
  $("#count").textContent = `${jobs.length} ảnh`;
  const finished = jobs.filter((job) =>
    ["done", "failed"].includes(job.status),
  ).length;
  const running = jobs.filter((job) => job.status === "running").length;
  const waiting = jobs.filter((job) => job.status === "queued").length;
  const failed = jobs.filter((job) => job.status === "failed").length;
  $("#progress").max = jobs.length || 1;
  $("#progress").value = finished;
  $("#summary").textContent = jobs.length
    ? `${finished}/${jobs.length} ảnh hoàn tất · ${running} đang xử lý · ${waiting} đang chờ${failed ? ` · ${failed} ảnh lỗi` : ""}`
    : "";
  $("#empty").hidden = jobs.length > 0;
  const completedIds = new Set(jobs.filter(jobHasResult).map((job) => job.id));
  for (const id of [...selectedJobs]) {
    if (!completedIds.has(id)) selectedJobs.delete(id);
  }
  $("#jobs").replaceChildren(
    ...jobs.map((job) => {
      const row = $("#job-template").content.firstElementChild.cloneNode(true);
      row.dataset.status = job.status;
      const hasResult = jobHasResult(job);
      const selectBox = row.querySelector(".job-select");
      if (hasResult) {
        selectBox.dataset.jobId = job.id;
        selectBox.checked = selectedJobs.has(job.id);
        selectBox.setAttribute("aria-label", `Chọn ${job.filename}`);
        selectBox.addEventListener("change", () => {
          if (selectBox.checked) selectedJobs.add(job.id);
          else selectedJobs.delete(job.id);
          updateBulkBar();
        });
      } else {
        selectBox.remove();
      }
      row.querySelector(".thumbnail").src = job.source_url;
      row.querySelector(".filename").textContent = job.filename;
      row.querySelector(".job-meta").textContent =
        `${job.width} × ${job.height} → ${job.width * job.scale} × ${job.height * job.scale} · ${job.preset === "photo" ? "Ảnh chụp" : "Anime"}`;
      row.querySelector(".job-status").textContent = {
        queued: "Đang chờ",
        running: "Đang xử lý",
        done: `Hoàn tất · ${job.seconds} giây`,
        failed: job.error,
      }[job.status];
      const actions = row.querySelector(".job-actions");
      const remove = row.querySelector(".remove-job");
      remove.disabled = job.status === "running";
      remove.title = remove.disabled ? "Chờ GPU xử lý xong để bỏ ảnh" : "";
      remove.setAttribute("aria-label", `Bỏ khỏi hàng đợi ${job.filename}`);
      remove.addEventListener("click", async () => {
        remove.disabled = true;
        try {
          const response = await fetch(`/api/jobs/${job.id}/remove`, {
            method: "POST",
            body: "",
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || "Không bỏ được ảnh.");
          notice("");
          await refresh(false);
          $("#queue-title").focus();
        } catch (error) {
          notice(error.message);
          remove.disabled = job.status === "running";
        }
      });
      actions.hidden = !hasResult;
      const link = row.querySelector(".download");
      if (job.download_url) link.href = job.download_url;
      for (const button of row.querySelectorAll(".open-job")) {
        button.setAttribute("aria-label", `Xem ảnh ${job.filename}`);
        button.addEventListener("click", () => {
          viewedScale = job.scale;
          compare(job, true);
          if (window.innerWidth <= 800) {
            $("#preview-panel").scrollIntoView({ block: "nearest" });
          }
        });
      }
      return row;
    }),
  );
  updateBulkBar();
}
async function refresh(schedule = true) {
  try {
    const response = await fetch("/api/jobs");
    if (!response.ok) throw new Error("Không đọc được hàng đợi.");
    const data = await response.json();
    const serial = JSON.stringify(data.jobs);
    if (serial !== lastJobs) {
      lastJobs = serial;
      render(data.jobs);
    }
    if ($("#summary").dataset.offline) {
      delete $("#summary").dataset.offline;
      render(data.jobs);
    }
  } catch (_) {
    $("#summary").textContent =
      "Mất kết nối. Kiểm tra cửa sổ ứng dụng; không gửi lại ảnh khi chưa rõ trạng thái.";
    $("#summary").dataset.offline = "true";
  } finally {
    if (schedule) setTimeout(refresh, 1200);
  }
}
async function init() {
  try {
    const response = await fetch("/api/capabilities");
    if (!response.ok) throw new Error();
    capabilities = await response.json();
    for (const option of [...$("#preset").options])
      if (!capabilities.presets.includes(option.value)) option.remove();
    if (!capabilities.presets.length) {
      capabilities = null;
      notice("Thiếu engine hoặc model. Xem hướng dẫn cài đặt trong README.");
    }
  } catch (_) {
    notice("Không kết nối được ứng dụng local. Hãy mở lại bằng start.bat.");
  }
  updateScaleDetails();
  updateSelection();
  refresh();
}
init();
