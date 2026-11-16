// Small bits of behaviour that HTMX doesn't cover.
(function () {
  "use strict";

  const csrf = document.querySelector('meta[name="csrf-token"]')?.content;

  // Send the CSRF token with every HTMX request.
  document.body.addEventListener("htmx:configRequest", (e) => {
    if (csrf) e.detail.headers["X-CSRF-Token"] = csrf;
  });

  // Flash messages: close button, and success messages fade after a while.
  function wireFlashes(root) {
    root.querySelectorAll(".flash").forEach((el) => {
      el.querySelector("button")?.addEventListener("click", () => el.remove());
      if (el.classList.contains("flash-success")) setTimeout(() => el.remove(), 6000);
    });
  }
  wireFlashes(document);

  // Press "/" anywhere to jump to the page's search box.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "/" || e.target.closest("input, textarea, select")) return;
    const box = document.querySelector("[data-search]");
    if (box) { e.preventDefault(); box.focus(); box.select(); }
  });

  // Login page: clicking a demo account fills the form.
  document.querySelectorAll("[data-demo-user]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("username").value = btn.dataset.demoUser;
      document.getElementById("password").value = btn.dataset.demoPass;
      document.getElementById("password").focus();
    });
  });

  // Charts. Each <canvas data-chart="..."> carries its data as JSON.
  const palette = {
    ink: "#3d454d", grid: "#ece9e3", accent: "#1d4f91",
    high: "#b42318", medium: "#d38a1c", low: "#1f7a4d",
  };

  function drawCharts(root) {
    if (!window.Chart) return;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = "#6b737b";

    root.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
      if (canvas._chart) return;
      const data = JSON.parse(canvas.dataset.values);
      canvas._chart = new Chart(canvas, builders[canvas.dataset.chart](data));
    });
  }

  const axis = { grid: { color: palette.grid }, border: { display: false } };

  const builders = {
    hourly(data) {
      const now = new Date().getHours();
      return {
        type: "bar",
        data: {
          labels: data.map((_, h) => String(h).padStart(2, "0")),
          datasets: [{
            data,
            backgroundColor: data.map((_, h) => (h === now ? palette.accent : "#b9c8dd")),
            borderRadius: 3,
            maxBarThickness: 18,
          }],
        },
        options: {
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { title: (i) => `${i[0].label}:00` } } },
          scales: {
            x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
            y: { ...axis, beginAtZero: true, ticks: { precision: 0 } },
          },
        },
      };
    },

    vitals(data) {
      const labels = data.labels.map((t) =>
        new Date(t).toLocaleDateString(undefined, { day: "numeric", month: "short" })
      );
      const line = (label, values, color, axisId) => ({
        label, data: values, borderColor: color, backgroundColor: color,
        tension: 0.25, pointRadius: 3, borderWidth: 2, yAxisID: axisId, spanGaps: true,
      });
      return {
        type: "line",
        data: {
          labels,
          datasets: [
            line("Heart rate", data.heart_rate, palette.high, "y"),
            line("Systolic BP", data.systolic_bp, palette.accent, "y"),
            line("Resp. rate", data.resp_rate, "#7a5af5", "y"),
            line("SpO₂ %", data.spo2, palette.low, "y2"),
          ],
        },
        options: {
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: { legend: { position: "bottom", labels: { boxWidth: 10, boxHeight: 10 } } },
          scales: {
            x: { grid: { display: false } },
            y: { ...axis, title: { display: true, text: "bpm / mmHg / per min" } },
            y2: { ...axis, position: "right", min: 70, max: 100, grid: { display: false },
                  title: { display: true, text: "SpO₂ %" } },
          },
        },
      };
    },

    leaderboard(data) {
      return {
        type: "bar",
        data: {
          labels: data.map((d) => d.name),
          datasets: [{
            label: "CV ROC-AUC",
            data: data.map((d) => d.cv_roc_auc),
            backgroundColor: data.map((_, i) => (i === 0 ? palette.accent : "#b9c8dd")),
            borderRadius: 3,
            maxBarThickness: 26,
          }],
        },
        options: {
          indexAxis: "y",
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { x: { ...axis, min: 0.5, max: 1 }, y: { grid: { display: false } } },
        },
      };
    },
  };

  drawCharts(document);
  document.body.addEventListener("htmx:afterSwap", (e) => {
    drawCharts(e.target);
    wireFlashes(e.target);
  });
})();
