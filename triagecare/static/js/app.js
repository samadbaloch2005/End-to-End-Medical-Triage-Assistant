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
      document.querySelectorAll("[data-demo-user]").forEach((b) => b.classList.toggle("picked", b === btn));
      document.getElementById("password").focus();
    });
  });

  // Charts. Each <canvas data-chart="..."> carries its data as JSON.
  const palette = {
    ink: "#334155", grid: "#eef1f7", accent: "#4f46e5",
    high: "#e11d48", medium: "#f59e0b", low: "#10b981",
  };

  function drawCharts(root) {
    if (!window.Chart) return;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = "#64748b";

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
            backgroundColor: data.map((_, h) => (h === now ? palette.accent : "#c7d2fe")),
            hoverBackgroundColor: palette.accent,
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

    risk(data) {
      const total = data.high + data.medium + data.low;
      return {
        type: "doughnut",
        data: {
          labels: ["High", "Medium", "Low"],
          datasets: [{
            data: total ? [data.high, data.medium, data.low] : [0, 0, 1],
            backgroundColor: total ? [palette.high, palette.medium, palette.low] : ["#eef1f7"],
            borderWidth: 3, borderColor: "#fff", hoverOffset: 6,
          }],
        },
        options: {
          maintainAspectRatio: false, cutout: "68%",
          plugins: { legend: { display: false }, tooltip: { enabled: total > 0 } },
        },
        plugins: [{
          id: "centre",
          afterDraw(chart) {
            const { ctx, chartArea: a } = chart;
            ctx.save();
            ctx.textAlign = "center";
            ctx.fillStyle = "#0f172a";
            ctx.font = "700 26px " + Chart.defaults.font.family;
            ctx.fillText(total, (a.left + a.right) / 2, (a.top + a.bottom) / 2 + 4);
            ctx.fillStyle = "#64748b";
            ctx.font = "500 11px " + Chart.defaults.font.family;
            ctx.fillText("triaged", (a.left + a.right) / 2, (a.top + a.bottom) / 2 + 20);
            ctx.restore();
          },
        }],
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
            backgroundColor: data.map((_, i) => (i === 0 ? palette.accent : "#c7d2fe")),
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

  // Pop-up alerts when a patient with critical vitals is waiting.
  // Each alert shows once per browser; the dashboard lists all of them.
  const shell = document.querySelector("[data-alerts]");
  const toasts = document.getElementById("toasts");
  const seenKey = "triagecare.seenAlerts";
  const seen = new Set(JSON.parse((() => { try { return localStorage.getItem(seenKey); } catch { return null; } })() || "[]"));

  function beep() {
    try {
      const ac = new (window.AudioContext || window.webkitAudioContext)();
      [0, 0.22].forEach((t) => {
        const o = ac.createOscillator(), g = ac.createGain();
        o.frequency.value = 880; o.connect(g); g.connect(ac.destination);
        g.gain.setValueAtTime(0.08, ac.currentTime + t);
        g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + t + 0.18);
        o.start(ac.currentTime + t); o.stop(ac.currentTime + t + 0.2);
      });
    } catch { /* sound is optional */ }
  }

  function showToast(a) {
    const el = document.createElement("div");
    el.className = "toast";
    if (a.level !== "critical") el.style.borderLeftColor = "var(--medium-bright)";
    const body = document.createElement("div");
    const title = document.createElement("b");
    title.textContent = `${a.title}: ${a.patient}`;
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = a.details.join(" · ");
    const link = document.createElement("a");
    link.href = `/visits/${a.visit_id}`;
    link.textContent = "Open patient";
    link.className = "small";
    body.append(title, sub, link);
    const close = document.createElement("button");
    close.type = "button"; close.setAttribute("aria-label", "Dismiss"); close.textContent = "×";
    close.addEventListener("click", () => el.remove());
    el.append(body, close);
    toasts.append(el);
    setTimeout(() => el.remove(), 20000);
  }

  async function pollAlerts() {
    if (!shell || !toasts || document.visibilityState !== "visible") return;
    try {
      const res = await fetch(shell.dataset.alerts, { headers: { Accept: "application/json" } });
      if (!res.ok) return;
      const alerts = await res.json();
      const fresh = alerts.filter((a) => !seen.has(`${a.visit_id}:${a.level}`));
      fresh.slice(0, 3).forEach(showToast);
      if (fresh.some((a) => a.level === "critical")) beep();
      fresh.forEach((a) => seen.add(`${a.visit_id}:${a.level}`));
      try { localStorage.setItem(seenKey, JSON.stringify([...seen].slice(-200))); } catch { /* private mode */ }
    } catch { /* offline: try again next time */ }
  }
  if (shell) { pollAlerts(); setInterval(pollAlerts, 20000); }

  drawCharts(document);
  document.body.addEventListener("htmx:afterSwap", (e) => {
    drawCharts(e.target);
    wireFlashes(e.target);
  });
})();
