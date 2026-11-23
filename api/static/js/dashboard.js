// Dashboard (templates/dashboard.html): renders the study table from the studies embedded in the page,
// then refreshes it from GET /studies (every 5 s while a study is running, every 20 s otherwise).
(function () {
  const tbody = document.getElementById("studies-tbody");
  if (!tbody) return; // empty-state, nothing to render/poll

  const POLL_IDLE_MS = 20000;
  const POLL_ACTIVE_MS = 5000;

  // ISO timestamp -> local date and time, or a dash
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // "9 total · 1 failed · 8 success ..." (the failed part only when there are failures)
    const failedPart = counts.failed
      ? `<span class="failed-count">${esc(counts.failed)} failed</span> · `
      : "";
    return `${esc(counts.total)} total · ${failedPart}${esc(counts.success)} success · ${esc(counts.running)} running · ${esc(counts.pending)} pending`;
  }

  // draw the rows and make each clickable; returns true if any study is running (-> poll faster)
    tbody.innerHTML = studies.map((s) => `
      <tr data-study-id="${esc(s.study_id)}">
        <td>${esc(s.study_id)}</td>
        <td>${esc(s.dataset_profile) || "—"}</td>
        <td><span class="badge ${cssClass(s.status)}">${esc(s.status)}</span></td>
        <td class="sample-counts">${sampleCountsText(s.sample_counts)}</td>
        <td>${esc(fmtDate(s.created_at))}</td>
        <td>${esc(fmtDate(s.last_run_at))}</td>
      </tr>
    `).join("");

    tbody.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", () => {
        window.location.href = `/studies/${encodeURIComponent(row.dataset.studyId)}`;
      });
    });

    return studies.some((s) => s.status === "RUNNING");
  }

  // refresh loop: re-schedules itself; a failed fetch is logged and retried at the idle interval
    let anyRunning = false;
    try {
      const res = await fetch("/studies");
      const studies = await res.json();
      anyRunning = render(studies);
    } catch (err) {
      console.error("failed to refresh studies", err);
    }
    setTimeout(poll, anyRunning ? POLL_ACTIVE_MS : POLL_IDLE_MS);
  }

  const initial = JSON.parse(document.getElementById("initial-studies").textContent);
  render(initial);
  setTimeout(poll, POLL_IDLE_MS);
})();
