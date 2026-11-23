// Sample page (templates/sample_detail.html): shows this sample's chromatogram plots (files named `<sample_id>_*`).
(function () {
  const { studyId, sampleId } = pageData();

  async function loadPlots() {
    const container = document.getElementById("sample-plots-container");
    const res = await fetch(`/studies/${encodeURIComponent(studyId)}/results/plots`);
    const plots = await res.json();
    const mine = plots.filter((p) => p.category === "chromatograms" && p.filename.startsWith(`${sampleId}_`));
    if (mine.length === 0) {
      container.textContent = "No chromatogram plots for this sample yet -- run the study first.";
      return;
    }
    container.innerHTML = `<div class="plot-grid">${
      mine.map((p) => `<a href="${esc(safeUrl(p.url))}" target="_blank" rel="noopener"><img src="${esc(safeUrl(p.url))}" alt="${esc(p.filename)}" title="${esc(p.filename)}"></a>`).join("")
    }</div>`;
  }

  loadPlots();
})();
