const $ = (selector) => document.querySelector(selector);

export function createPreview() {
  let generation = 0;
  let nativeWidth = 0,
    nativeHeight = 0;
  let currentZoom = 100;
  let exactScrollLeft = 0,
    exactScrollTop = 0,
    appliedScrollLeft = 0,
    appliedScrollTop = 0;
  let wheelHintTimer = null;
  let lastViewportSize = "";
  const viewport = $("#image-viewport");
  const zoomValue = $("#zoom-value");
  const zoomFit = $("#zoom-fit");
  const zoomActual = $("#zoom-actual");
  const wheelHint = $("#wheel-hint");
  const stage = $(".comparison");
  const control = $("#compare-range");

  function setZoomDisabled(disabled) {
    zoomFit.disabled = disabled;
    zoomActual.disabled = disabled;
  }

  function fitRatio() {
    return Math.min(
      viewport.clientWidth / nativeWidth,
      viewport.clientHeight / nativeHeight,
    );
  }

  function zoom(value, anchor = null, resetCenter = false) {
    const raw = Math.max(100, Math.min(800, Number(value)));
    const percent = Math.round(raw);
    currentZoom = raw;
    zoomValue.textContent = `${percent}%`;
    if (!nativeWidth || !nativeHeight) return;
    const oldW = parseFloat(stage.style.width) || viewport.clientWidth;
    const oldH = parseFloat(stage.style.height) || viewport.clientHeight;
    const anchorX = anchor ? anchor.x : viewport.clientWidth / 2;
    const anchorY = anchor ? anchor.y : viewport.clientHeight / 2;
    const oldOffsetX = Math.max(0, (viewport.clientWidth - oldW) / 2);
    const oldOffsetY = Math.max(0, (viewport.clientHeight - oldH) / 2);
    const scrollLeft =
      viewport.scrollLeft === appliedScrollLeft
        ? exactScrollLeft
        : viewport.scrollLeft;
    const scrollTop =
      viewport.scrollTop === appliedScrollTop
        ? exactScrollTop
        : viewport.scrollTop;
    const relX =
      resetCenter || !oldW
        ? 0.5
        : Math.max(
            0,
            Math.min(1, (scrollLeft + anchorX - oldOffsetX) / oldW),
          );
    const relY =
      resetCenter || !oldH
        ? 0.5
        : Math.max(
            0,
            Math.min(1, (scrollTop + anchorY - oldOffsetY) / oldH),
          );
    const fit = fitRatio();
    const newW = (nativeWidth * fit * raw) / 100;
    const newH = (nativeHeight * fit * raw) / 100;
    stage.style.width = `${newW}px`;
    stage.style.height = `${newH}px`;
    const newOffsetX = Math.max(0, (viewport.clientWidth - newW) / 2);
    const newOffsetY = Math.max(0, (viewport.clientHeight - newH) / 2);
    exactScrollLeft = Math.max(0, relX * newW + newOffsetX - anchorX);
    exactScrollTop = Math.max(0, relY * newH + newOffsetY - anchorY);
    viewport.scrollLeft = Math.round(exactScrollLeft);
    viewport.scrollTop = Math.round(exactScrollTop);
    appliedScrollLeft = viewport.scrollLeft;
    appliedScrollTop = viewport.scrollTop;
  }

  function stepZoom(delta, anchor = null) {
    const base =
      delta > 0
        ? Math.floor(currentZoom / 25) * 25
        : Math.ceil(currentZoom / 25) * 25;
    zoom(base + delta, anchor);
  }

  function layout(resetCenter = false) {
    if (!nativeWidth || !nativeHeight) return;
    zoom(currentZoom, null, resetCenter);
  }

  zoomFit.addEventListener("click", () => zoom(100, null, true));
  zoomActual.addEventListener("click", () => {
    if (!nativeWidth || !nativeHeight) return;
    zoom(100 / fitRatio());
  });
  new ResizeObserver(() => {
    const size = `${viewport.clientWidth}x${viewport.clientHeight}`;
    if (size === lastViewportSize) return;
    lastViewportSize = size;
    layout(false);
  }).observe(viewport);

  viewport.addEventListener(
    "wheel",
    (event) => {
      if (stage.hidden || !nativeWidth) return;
      if (window.innerWidth <= 800 && !event.ctrlKey) {
        if (wheelHint) {
          wheelHint.hidden = false;
          clearTimeout(wheelHintTimer);
          wheelHintTimer = setTimeout(() => {
            wheelHint.hidden = true;
          }, 1500);
        }
        return;
      }
      event.preventDefault();
      if (!event.deltaY) return;
      if (wheelHint) wheelHint.hidden = true;
      const rect = viewport.getBoundingClientRect();
      stepZoom(event.deltaY < 0 ? 25 : -25, {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      });
    },
    { passive: false },
  );

  viewport.addEventListener("dblclick", (event) => {
    if (stage.hidden || !nativeWidth) return;
    const rect = viewport.getBoundingClientRect();
    const next = Math.round(currentZoom) === 100 ? 200 : 100;
    zoom(next, {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  });

  viewport.addEventListener("keydown", (event) => {
    if (stage.hidden || !nativeWidth) return;
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      stepZoom(25);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      stepZoom(-25);
    } else if (event.key === "0") {
      event.preventDefault();
      zoom(100, null, true);
    } else if (event.target === viewport) {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        viewport.scrollLeft -= Math.round(viewport.clientWidth * 0.1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        viewport.scrollLeft += Math.round(viewport.clientWidth * 0.1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        viewport.scrollTop -= Math.round(viewport.clientHeight * 0.1);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        viewport.scrollTop += Math.round(viewport.clientHeight * 0.1);
      }
      exactScrollLeft = appliedScrollLeft = viewport.scrollLeft;
      exactScrollTop = appliedScrollTop = viewport.scrollTop;
    }
  });

  function split(value) {
    const percent = Math.max(0, Math.min(100, Math.round(Number(value))));
    control.value = percent;
    stage.style.setProperty("--split", `${percent}%`);
    $("#before-mask").style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
    control.setAttribute(
      "aria-valuetext",
      `Vạch chia ở ${percent}% chiều ngang, ảnh gốc bên trái và ảnh upscale bên phải`,
    );
  }

  function moveTo(event) {
    const bounds = stage.getBoundingClientRect();
    split(((event.clientX - bounds.left) / bounds.width) * 100);
  }
  control.addEventListener("input", () => split(control.value));
  let drag = null;
  viewport.addEventListener("contextmenu", (event) => event.preventDefault());
  viewport.addEventListener("pointerdown", (event) => {
    if (stage.hidden || !event.isPrimary || ![0, 2].includes(event.button))
      return;
    event.preventDefault();
    if (event.button === 2) {
      if (!control.disabled) moveTo(event);
      return;
    }
    const handle = $("#split-divider span").getBoundingClientRect();
    const onHandle =
      !control.disabled &&
      event.clientX >= handle.left - 10 &&
      event.clientX <= handle.right + 10 &&
      event.clientY >= stage.getBoundingClientRect().top &&
      event.clientY <= stage.getBoundingClientRect().bottom;
    const comparing =
      onHandle || (event.pointerType === "touch" && !control.disabled);
    drag = {
      id: event.pointerId,
      comparing,
      x: event.clientX,
      y: event.clientY,
      left: viewport.scrollLeft,
      top: viewport.scrollTop,
    };
    viewport.setPointerCapture(event.pointerId);
    if (comparing) {
      control.focus({ preventScroll: true });
      moveTo(event);
    }
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!drag || drag.id !== event.pointerId) return;
    if (drag.comparing) moveTo(event);
    else {
      viewport.scrollLeft = drag.left + drag.x - event.clientX;
      viewport.scrollTop = drag.top + drag.y - event.clientY;
      exactScrollLeft = appliedScrollLeft = viewport.scrollLeft;
      exactScrollTop = appliedScrollTop = viewport.scrollTop;
    }
  });
  function stopDrag(event) {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    if (viewport.hasPointerCapture(event.pointerId))
      viewport.releasePointerCapture(event.pointerId);
  }
  viewport.addEventListener("pointerup", stopDrag);
  viewport.addEventListener("pointercancel", stopDrag);
  viewport.addEventListener("lostpointercapture", stopDrag);

  function syncFooter() {
    const footer = $("#preview-footer");
    if (!footer) return;
    footer.hidden =
      Boolean($("#result-group")?.hidden) &&
      Boolean($("#wallpaper-form")?.hidden);
  }

  function clear(
    message = "Chọn ảnh để xem trước hoặc chọn một kết quả trong hàng đợi.",
    name = "Xem ảnh",
  ) {
    generation++;
    nativeWidth = nativeHeight = 0;
    currentZoom = 100;
    exactScrollLeft = exactScrollTop = appliedScrollLeft = appliedScrollTop = 0;
    clearTimeout(wheelHintTimer);
    if (wheelHint) wheelHint.hidden = true;
    stage.style.width = "";
    stage.style.height = "";
    stage.hidden = true;
    control.disabled = true;
    setZoomDisabled(true);
    zoomValue.textContent = message.startsWith("Đang ") ? "" : "100%";
    $("#preview-placeholder").hidden = false;
    $("#preview-placeholder").textContent = message;
    $("#preview-name").textContent = name;
    $("#preview-name").title = name;
    $("#preview-meta").textContent = "";
    $("#preview-meta").removeAttribute("title");
    $("#comparison-help").textContent = "";
    if ($("#result-group")) $("#result-group").hidden = true;
    $("#preview-links").hidden = true;
    $("#preview-download").hidden = true;
    syncFooter();
    $("#before").removeAttribute("src");
    $("#after").removeAttribute("src");
  }

  async function show({
    name,
    sourceUrl,
    resultUrl,
    wallpaperUrl,
    downloadUrl,
    message,
  }) {
    clear("Đang tải ảnh để xem…");
    const current = generation;
    $("#preview-name").textContent = name;
    $("#preview-name").title = name;
    $("#preview-meta").textContent = message || "";
    if (message) $("#preview-meta").title = message;
    else $("#preview-meta").removeAttribute("title");
    try {
      const primaryUrl = wallpaperUrl || sourceUrl;
      const beforeLoader = new Image();
      beforeLoader.src = primaryUrl;
      const afterLoader = !wallpaperUrl && resultUrl ? new Image() : null;
      if (afterLoader) afterLoader.src = resultUrl;
      await Promise.all([
        beforeLoader.decode(),
        afterLoader ? afterLoader.decode() : Promise.resolve(),
      ]);
      if (current !== generation) return;
      $("#before").src = primaryUrl;
      $("#before").alt = wallpaperUrl ? "Hình nền AI" : "Ảnh gốc";
      const paired = Boolean(!wallpaperUrl && resultUrl);
      $("#after").hidden = !paired;
      if (paired) $("#after").src = resultUrl;
      else $("#after").removeAttribute("src");
      const reference = paired ? afterLoader : beforeLoader;
      stage.style.setProperty(
        "--image-ratio",
        reference.naturalWidth / reference.naturalHeight,
      );
      nativeWidth = reference.naturalWidth;
      nativeHeight = reference.naturalHeight;
      currentZoom = 100;
      const hasResultActions = Boolean(resultUrl || downloadUrl);
      if ($("#result-group")) $("#result-group").hidden = !hasResultActions;
      $("#preview-links").hidden = !hasResultActions;
      if (sourceUrl) $("#source-link").href = sourceUrl;
      $("#result-link").hidden = !resultUrl;
      $("#preview-download").hidden = !downloadUrl;
      if (resultUrl) $("#result-link").href = resultUrl;
      if (downloadUrl) $("#preview-download").href = downloadUrl;
      syncFooter();
      $("#preview-placeholder").hidden = true;
      stage.hidden = false;
      layout(true);
      setZoomDisabled(false);
      control.disabled = !paired;
      control.hidden = !paired;
      $("#split-divider").hidden = !paired;
      $("#comparison-help").textContent = "";
      split(paired ? 50 : 100);
    } catch (_) {
      if (current === generation) {
        clear(
          "Không tải được ảnh để xem. Hãy chọn lại ảnh hoặc kiểm tra kết quả còn được lưu trong ứng dụng.",
        );
        $("#preview-name").textContent = name;
        $("#preview-name").title = name;
      }
    }
  }
  return { show, clear };
}
