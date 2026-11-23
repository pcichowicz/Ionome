// Study page (templates/study_detail.html): readiness banner, sample table, result tabs and the Run button.
// Tab contents are fetched the first time a tab is opened and again after a run finishes (`loaded` is cleared).
// Running a study: POST /run, then poll the per-stage status every 3 s until the study leaves RUNNING.
(function () {
  const { studyId } = pageData();
  const studyUrl = `/studies/${encodeURIComponent(studyId)}`;

  // -- tabs, lazy-load each panel's data the first time it's shown --
  const loaders = { plots: loadPlots, table: loadTablePreview, qc: loadQcReport, runs: loadRuns };
  const loaded = new Set();

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.getElementById(`tab-${tab}`).classList.add("active");
      if (!loaded.has(tab) && loaders[tab]) {
        loaded.add(tab);
        loaders[tab]();
      }
    });
  });

  // "N of M expected samples present" banner
    const res = await fetch(`${studyUrl}/sample-readiness`);
    const data = await res.json();
    const line = document.getElementById("readiness-line");
    line.textContent = data.missing.length === 0
      ? `${data.present.length} of ${data.expected.length} expected samples present.`
      : `${data.present.length} of ${data.expected.length} expected samples present -- missing: ${data.missing.join(", ")}`;  // textContent: no escaping needed
  }

  async function loadSamples() {
    const res = await fetch(`${studyUrl}/samples`);
    const samples = await res.json();
    document.getElementById("samples-tbody").innerHTML = samples.map((s) => `
      <tr data-sample-id="${esc(s.sample_id)}">
        <td>${esc(s.sample_id)}</td>
        <td>${esc(s.description) || "\u2014"}</td>
        <td><span class="badge ${cssClass(s.status)}">${esc(s.status)}</span></td>
      </tr>
    `).join("");
    document.querySelectorAll("#samples-tbody tr").forEach((row) => {
      row.style.cursor = "pointer";
      row.addEventListener("click", () => {
        window.location.href = `${studyUrl}/samples/${encodeURIComponent(row.dataset.sampleId)}`;
      });
    });
  }

  async function loadPlots() {
    const container = document.getElementById("plots-container");
    const res = await fetch(`${studyUrl}/results/plots`);
    const plots = await res.json();
    if (plots.length === 0) {
      container.textContent = "No plots yet -- run the study first.";
      return;
    }
    const byCategory = {};
    plots.forEach((p) => { (byCategory[p.category] ||= []).push(p); });
    const categories = Object.keys(byCategory);

    const subtabsHtml = `<nav class="subtabs">${
      categories.map((c, i) => `<button class="subtab-btn${i === 0 ? " active" : ""}" data-category="${esc(c)}">${esc(c.replace(/_/g, " "))}</button>`).join("")
    }</nav>`;
    const panelsHtml = categories.map((c, i) => `
      <div class="plot-grid subtab-panel${i === 0 ? " active" : ""}" data-category="${esc(c)}">
        ${byCategory[c].map((p) => `<a href="${esc(safeUrl(p.url))}" target="_blank" rel="noopener"><img src="${esc(safeUrl(p.url))}" alt="${esc(p.filename)}" title="${esc(p.filename)}"></a>`).join("")}
      </div>
    `).join("");
    container.innerHTML = subtabsHtml + panelsHtml;

    container.querySelectorAll(".subtab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        container.querySelectorAll(".subtab-btn").forEach((b) => b.classList.remove("active"));
        container.querySelectorAll(".subtab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        Array.from(container.querySelectorAll(".subtab-panel"))
          .find((p) => p.dataset.category === btn.dataset.category).classList.add("active");
      });
    });
  }

  async function loadTablePreview() {
    const container = document.getElementById("table-container");
    const res = await fetch(`${studyUrl}/results/table/preview`);
    if (!res.ok) {
      container.textContent = "No feature table / library yet -- run the study first.";
      return;
    }
    const data = await res.json();
    const headerRow = data.columns.map((c) => `<th>${esc(c)}</th>`).join("");
    const bodyRows = data.rows.map((row) =>
      `<tr>${data.columns.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`
    ).join("");
    container.innerHTML = `
      <p class="hint">Showing ${data.rows.length} of ${data.total_rows} rows.
        <a href="${esc(safeUrl(data.download_url))}">Download full CSV</a></p>
      <div class="table-scroll"><table><thead><tr>${headerRow}</tr></thead><tbody>${bodyRows}</tbody></table></div>
    `;
  }

  async function loadQcReport() {
    const container = document.getElementById("qc-report-container");
    const res = await fetch(`${studyUrl}/results/qc-report`);
    if (!res.ok) {
      container.textContent = "No QC report yet -- run the study first.";
      return;
    }
    container.textContent = JSON.stringify(await res.json(), null, 2);
  }

  async function loadRuns() {
    const res = await fetch(`${studyUrl}/results/runs`);
    const runs = await res.json();
    document.getElementById("runs-tbody").innerHTML = runs.map((r) => `
      <tr>
        <td>${esc(r.run_id)}</td>
        <td><span class="badge ${cssClass(r.status)}">${esc(r.status)}</span></td>
        <td>${esc(r.started_at) || "\u2014"}</td>
        <td>${esc(r.finished_at) || "\u2014"}</td>
        <td>${esc(r.n_steps)}</td>
      </tr>
    `).join("");
  }

  const runBtn = document.getElementById("run-btn");
  const runLine = document.getElementById("run-status-line");

  // follow one run until it ends, show the current stage in the status line, then re-enable the button and refresh the data
    // poll the run's steps until none is pending or running
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      let steps;
      try {
        const res = await fetch(`${studyUrl}/runs/${encodeURIComponent(runId)}`);
        if (!res.ok) {
          // no step rows (yet): fine while the run is starting, a dead end once the study is no longer RUNNING
          const st = await (await fetch(`${studyUrl}/status`)).json();
          if (st.status !== "RUNNING") { runLine.textContent = `Run ${st.status.toLowerCase()}.`; break; }
          continue;
        }
        steps = await res.json();
      } catch (err) {
        continue;
      }
      const failed = steps.find((st) => st.status === "FAILED");
      const active = steps.find((st) => st.status === "RUNNING" || st.status === "PENDING");
      if (failed) {
        runLine.textContent = `Run failed at ${failed.step_name}: ${failed.error_msg || "no message"}`;
        break;
      }
      if (active) {
        runLine.textContent = `Running: ${active.step_name} (${steps.filter((st) => st.status === "SUCCESS").length} steps done)`;
        continue;
      }
      // the last step finished; the experiment row says whether the whole run is over
      const status = await (await fetch(`${studyUrl}/status`)).json();
      if (status.status !== "RUNNING") {
        runLine.textContent = status.status === "SUCCESS" ? "Run finished." : `Run ${status.status.toLowerCase()}.`;
        break;
      }
    }
    runBtn.disabled = false;
    loadSamples();
    loaded.clear();           // plots, table, QC report and run history reload on next view
  }

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    runLine.textContent = "Starting\u2026";
    let res;
    try {
      res = await fetch(`${studyUrl}/run`, { method: "POST" });
    } catch (err) {
      runLine.textContent = "Couldn't reach the server.";
      runBtn.disabled = false;
      return;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      runLine.textContent = res.status === 409 ? "This study is already running." : `Couldn't start the run: ${data.detail || res.status}`;
      if (res.status === 409) watchLatestRun(); else runBtn.disabled = false;
      return;
    }
    runLine.textContent = `Run started (run_id: ${data.run_id}).`;
    watchRun(data.run_id);
  });

  // after a 409 (already running): attach to the newest run instead of starting another
    const runs = await (await fetch(`${studyUrl}/results/runs`)).json();
    if (runs.length) watchRun(runs[0].run_id); else runBtn.disabled = false;
  }

  loadReadiness();
  loadSamples();
})();
