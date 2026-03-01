(() => {
  const inputA = document.getElementById("input-a");
  const inputB = document.getElementById("input-b");
  const previewA = document.getElementById("preview-a");
  const previewB = document.getElementById("preview-b");
  const dropA = document.getElementById("drop-a");
  const dropB = document.getElementById("drop-b");
  const clearA = document.getElementById("clear-a");
  const clearB = document.getElementById("clear-b");
  const cardA = document.getElementById("card-a");
  const cardB = document.getElementById("card-b");
  const analyzeBtn = document.getElementById("analyze-btn");
  const progressWrap = document.getElementById("progress-wrap");
  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");
  const errorBox = document.getElementById("error-box");
  const errorMsg = document.getElementById("error-msg");
  const results = document.getElementById("results");
  const metricsRow = document.getElementById("metrics-row");
  const instructionList = document.getElementById("instruction-list");
  const resultImg = document.getElementById("result-img");
  const downloadBtn = document.getElementById("download-btn");

  let fileA = null;
  let fileB = null;

  // ---- File handling helpers ----
  function setImage(side, file) {
    if (!file || !file.type.startsWith("image/")) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (side === "a") {
        fileA = file;
        previewA.src = e.target.result;
        dropA.classList.add("has-image");
        cardA.classList.add("has-image");
      } else {
        fileB = file;
        previewB.src = e.target.result;
        dropB.classList.add("has-image");
        cardB.classList.add("has-image");
      }
      updateAnalyzeBtn();
    };
    reader.readAsDataURL(file);
  }

  function clearImage(side) {
    if (side === "a") {
      fileA = null;
      previewA.src = "";
      inputA.value = "";
      dropA.classList.remove("has-image");
      cardA.classList.remove("has-image");
    } else {
      fileB = null;
      previewB.src = "";
      inputB.value = "";
      dropB.classList.remove("has-image");
      cardB.classList.remove("has-image");
    }
    updateAnalyzeBtn();
  }

  function updateAnalyzeBtn() {
    analyzeBtn.disabled = !(fileA && fileB);
  }

  // ---- Input change events ----
  inputA.addEventListener("change", () => { if (inputA.files[0]) setImage("a", inputA.files[0]); });
  inputB.addEventListener("change", () => { if (inputB.files[0]) setImage("b", inputB.files[0]); });

  // ---- Clear buttons ----
  clearA.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); clearImage("a"); });
  clearB.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); clearImage("b"); });

  // ---- Drag and drop ----
  function setupDrop(dropZone, side) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file) setImage(side, file);
    });
  }
  setupDrop(dropA, "a");
  setupDrop(dropB, "b");

  // ---- Progress animation ----
  const progressSteps = [
    { pct: 15, label: "Detecting keypoints…" },
    { pct: 40, label: "Matching features…" },
    { pct: 65, label: "Computing homography…" },
    { pct: 85, label: "Generating visualization…" },
  ];
  let progressTimer = null;
  let stepIdx = 0;

  function startProgress() {
    progressWrap.hidden = false;
    progressFill.style.width = "5%";
    progressLabel.textContent = "Starting analysis…";
    stepIdx = 0;
    progressTimer = setInterval(() => {
      if (stepIdx < progressSteps.length) {
        const s = progressSteps[stepIdx++];
        progressFill.style.width = s.pct + "%";
        progressLabel.textContent = s.label;
      }
    }, 600);
  }

  function finishProgress(success) {
    clearInterval(progressTimer);
    progressFill.style.width = "100%";
    progressLabel.textContent = success ? "Done!" : "Analysis complete.";
    setTimeout(() => { progressWrap.hidden = true; }, 800);
  }

  // ---- Error display ----
  function showError(msg) {
    errorMsg.textContent = msg;
    errorBox.hidden = false;
  }
  function hideError() { errorBox.hidden = true; }

  // ---- Metric card builder ----
  function buildMetrics(metrics) {
    const defs = [
      { key: "translation_x", label: "Shift X", fmt: (v) => `${v > 0 ? "+" : ""}${v}px` },
      { key: "translation_y", label: "Shift Y", fmt: (v) => `${v > 0 ? "+" : ""}${v}px` },
      { key: "rotation_deg", label: "Rotation", fmt: (v) => `${v > 0 ? "+" : ""}${v}°` },
      { key: "scale", label: "Scale", fmt: (v) => `${v}×` },
      { key: "inlier_matches", label: "Inlier Matches", fmt: (v) => `${v}` },
      { key: "total_matches", label: "Total Matches", fmt: (v) => `${v}` },
    ];
    metricsRow.innerHTML = defs
      .map(
        ({ key, label, fmt }) => `
          <div class="metric-card">
            <div class="metric-label">${label}</div>
            <div class="metric-value">${fmt(metrics[key])}</div>
          </div>`
      )
      .join("");
  }

  // ---- Analyze ----
  analyzeBtn.addEventListener("click", async () => {
    hideError();
    results.hidden = true;
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    startProgress();

    const formData = new FormData();
    formData.append("image_a", fileA);
    formData.append("image_b", fileB);

    try {
      const res = await fetch("/analyze", { method: "POST", body: formData });
      const data = await res.json();

      finishProgress(res.ok);

      if (!res.ok) {
        showError(data.error || "An unknown error occurred.");
        return;
      }

      // Populate results
      buildMetrics(data.metrics);

      instructionList.innerHTML = data.instructions
        .map((instr) => `<li>${escHtml(instr)}</li>`)
        .join("");

      resultImg.src = data.result_image;
      downloadBtn.href = data.result_image;
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      finishProgress(false);
      showError("Network error — is the server running? " + err.message);
    } finally {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
      updateAnalyzeBtn();
    }
  });

  function escHtml(str) {
    return str.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }
})();
