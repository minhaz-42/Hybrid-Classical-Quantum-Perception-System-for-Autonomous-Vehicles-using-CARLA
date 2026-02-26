/* ============================================================
   QuantumDrive \u2014 App JavaScript  (Green Claymorphism Theme)
   ============================================================ */

// \u2500\u2500\u2500 Green-tinted colour palette \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
const COLORS = {
  green:  '#16a34a', greenBg:  'rgba(22,163,74,.12)',
  blue:   '#3b82f6', blueBg:   'rgba(59,130,246,.12)',
  purple: '#8b5cf6', purpleBg: 'rgba(139,92,246,.12)',
  orange: '#f59e0b', orangeBg: 'rgba(245,158,11,.12)',
  red:    '#ef4444', redBg:    'rgba(239,68,68,.12)',
  teal:   '#14b8a6', tealBg:   'rgba(20,184,166,.12)',
  grid:   'rgba(0,0,0,.04)'
};

// Weather emoji map
const WEATHER_ICONS = {
  'Clear': '\u2600\ufe0f', 'Rain': '\U0001f327\ufe0f', 'Fog': '\U0001f32b\ufe0f', 'Snow': '\u2744\ufe0f',
  'Storm': '\u26c8\ufe0f', 'Night': '\U0001f319', 'Dusk': '\U0001f305', 'Dawn': '\U0001f304',
  'Overcast': '\u2601\ufe0f', 'Cloudy': '\u2601\ufe0f', 'Hail': '\U0001f328\ufe0f',
  'clear': '\u2600\ufe0f', 'rain': '\U0001f327\ufe0f', 'fog': '\U0001f32b\ufe0f', 'snow': '\u2744\ufe0f',
  'storm': '\u26c8\ufe0f', 'night': '\U0001f319', 'dusk': '\U0001f305', 'dawn': '\U0001f304'
};

// Chart.js defaults
if (typeof Chart !== 'undefined') {
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size   = 12;
  Chart.defaults.color        = '#4b5563';
  Chart.defaults.plugins.legend.labels.usePointStyle  = true;
  Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.plugins.tooltip.backgroundColor = '#14532d';
  Chart.defaults.plugins.tooltip.cornerRadius    = 10;
  Chart.defaults.plugins.tooltip.padding          = 12;
  Chart.defaults.plugins.tooltip.titleFont        = { weight: '600' };
}

// Utility
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function getCSRF() {
  if (window.CSRF_TOKEN) return window.CSRF_TOKEN;
  const el = $('input[name="csrfmiddlewaretoken"]') || $('[name=csrf-token]');
  if (el) return el.value || el.content;
  const cookie = document.cookie
    .split(';')
    .map(part => part.trim())
    .find(part => part.startsWith('csrftoken='));
  return cookie ? decodeURIComponent(cookie.split('=')[1]) : '';
}

async function apiFetch(url, opts = {}) {
  const csrf = getCSRF();
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(csrf ? { 'X-CSRFToken': csrf } : {}),
      ...opts.headers
    },
    ...opts
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

// Settings (localStorage)
const SETTINGS = {
  _d: {
    pollInterval: 3000, maxPoints: 30, autoStart: false,
    quantumEnabled: true, noiseEnabled: false, noiseStrength: 0.05,
    animations: true, compact: false
  },
  get(k)    { try { const v = localStorage.getItem('qd_'+k); return v !== null ? JSON.parse(v) : this._d[k]; } catch { return this._d[k]; } },
  set(k, v) { localStorage.setItem('qd_'+k, JSON.stringify(v)); },
  all()     { const o = {}; for (const k of Object.keys(this._d)) o[k] = this.get(k); return o; }
};

// Charts registry
const charts = {};

function makeLineDataset(label, color, bgColor) {
  return {
    label, data: [], borderColor: color, backgroundColor: bgColor,
    borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 5,
    tension: .4, fill: true
  };
}

function pushPoint(chart, label, datasets) {
  const max = SETTINGS.get('maxPoints');
  chart.data.labels.push(label);
  if (chart.data.labels.length > max) chart.data.labels.shift();
  datasets.forEach((val, i) => {
    chart.data.datasets[i].data.push(val);
    if (chart.data.datasets[i].data.length > max) chart.data.datasets[i].data.shift();
  });
  chart.update('none');
}

// Helpers
function setText(sel, val)    { const el = $(sel); if (el) el.textContent = val; }
function setInputVal(sel, v)  { const el = $(sel); if (el) el.value = v; }
function getInputVal(sel, d)  { const el = $(sel); return el ? (parseFloat(el.value) || d) : d; }
function setChecked(sel, v)   { const el = $(sel); if (el) el.checked = !!v; }
function getChecked(sel)      { const el = $(sel); return el ? el.checked : false; }
function setBar(sel, value) {
  const el = $(sel);
  if (!el) return;
  const width = Math.max(0, Math.min(100, value));
  el.style.width = `${width.toFixed(1)}%`;
}

function setConfidenceRing(value) {
  const el = $('#confidenceCircle');
  if (!el) return;
  const pct = Math.max(0, Math.min(100, value * 100));
  el.setAttribute('stroke-dasharray', `${pct}, 100`);
}

let uptimeTimer = null;
let uptimeStart = null;

function startUptime() {
  if (uptimeTimer) return;
  uptimeStart = new Date();
  const tick = () => {
    const sec = Math.max(0, Math.floor((Date.now() - uptimeStart.getTime()) / 1000));
    const hh = String(Math.floor(sec / 3600)).padStart(2, '0');
    const mm = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
    const ss = String(sec % 60).padStart(2, '0');
    setText('#sessionUptime', `${hh}:${mm}:${ss}`);
  };
  tick();
  uptimeTimer = setInterval(tick, 1000);
}

function stopUptime() {
  if (uptimeTimer) clearInterval(uptimeTimer);
  uptimeTimer = null;
  uptimeStart = null;
  setText('#sessionUptime', '00:00:00');
}

// Topbar badges update
function updateTopbar(simStatus) {
  if (!simStatus) return;
  const running = !!simStatus.running;

  // Status badge
  const statusDot = $('#simStatusBadge .status-dot');
  if (statusDot) {
    statusDot.classList.remove('online', 'offline');
    statusDot.classList.add(running ? 'online' : 'offline');
  }
  setText('#simStatusText', running ? 'Running' : 'Offline');

  // Weather badge
  const weatherName = simStatus.weather || 'Clear';
  const weatherIcon = WEATHER_ICONS[weatherName] || '\u2600\ufe0f';
  const weatherIconEl = $('#weatherBadge .weather-icon');
  if (weatherIconEl) weatherIconEl.textContent = weatherIcon;
  setText('#weatherText', weatherName);

  // Quantum badge
  const quantumOn = simStatus.quantum_enabled !== false;
  setText('#quantumText', quantumOn ? 'Quantum ON' : 'Quantum OFF');
}

function updateSimulationStatus(s) {
  if (!s || typeof s !== 'object') return;
  const running = !!s.running;

  // Update topbar
  updateTopbar(s);

  // Update simulation page controls
  updateSimStatus(running ? 'Running' : 'Stopped', running ? 'green' : 'muted');
  if (s.session_id != null) setText('#sessionId', String(s.session_id));
  if (s.weather) {
    setText('#sessionWeather', s.weather);
    setText('#weatherLargeText', s.weather);
    const icon = WEATHER_ICONS[s.weather] || '\u2600\ufe0f';
    setText('#weatherLargeIcon', icon);
  }
  if (typeof s.quantum_enabled === 'boolean') {
    setText('#sessionQuantum', s.quantum_enabled ? 'ON' : 'OFF');
    setText('#quantumToggleLabel', s.quantum_enabled ? 'Enabled' : 'Disabled');
    setChecked('#quantumToggle', s.quantum_enabled);
  }
  const startBtn = $('#btnStart');
  const stopBtn = $('#btnStop');
  if (startBtn) startBtn.disabled = running;
  if (stopBtn) stopBtn.disabled = !running;
}

function downloadFile(name, content, type) {
  const blob = new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// DASHBOARD PAGE
function initDashboard() {
  const steerCtx = $('#chartSteering');
  if (steerCtx) {
    charts.steering = new Chart(steerCtx, {
      type: 'line',
      data: { labels: [], datasets: [
        makeLineDataset('Steering', COLORS.green, COLORS.greenBg),
        makeLineDataset('Throttle', COLORS.blue,  COLORS.blueBg),
        makeLineDataset('Brake',    COLORS.red,   COLORS.redBg)
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { min: -1, max: 1, grid: { color: COLORS.grid } },
          x: { grid: { display: false } }
        },
        plugins: { legend: { position: 'top' } },
        interaction: { mode: 'index', intersect: false }
      }
    });
  }

  const confCtx = $('#chartConfidence');
  if (confCtx) {
    charts.confidence = new Chart(confCtx, {
      type: 'line',
      data: { labels: [], datasets: [
        makeLineDataset('Classical', COLORS.blue,   COLORS.blueBg),
        makeLineDataset('Quantum',   COLORS.purple, COLORS.purpleBg)
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { min: 0, max: 1, grid: { color: COLORS.grid } },
          x: { grid: { display: false } }
        },
        plugins: { legend: { position: 'top' } },
        interaction: { mode: 'index', intersect: false }
      }
    });
  }

  startPolling();
}

// SIMULATION PAGE
const drivingState = {
  speed: 0, steering: 0, decision: '\u2014', sign: 'None',
  confidence: 0, laneOffset: 0, weather: 'Clear',
  roadOffset: 0, animFrame: null
};

function initSimulation() {
  const confCtx = $('#chartConfidence');
  if (confCtx && !charts.confidence) {
    charts.confidence = new Chart(confCtx, {
      type: 'line',
      data: { labels: [], datasets: [
        makeLineDataset('Classical', COLORS.blue,   COLORS.blueBg),
        makeLineDataset('Quantum',   COLORS.purple, COLORS.purpleBg)
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { min: 0, max: 1, grid: { color: COLORS.grid } },
          x: { grid: { display: false } }
        },
        interaction: { mode: 'index', intersect: false }
      }
    });
  }

  initDrivingViz();

  apiFetch('/api/simulation-status/')
    .then(s => {
      updateSimulationStatus(s);
      if (s.running) {
        simRunning = true;
        startUptime();
        startPolling();
        startDrivingAnimation();
      }
    })
    .catch((e) => console.warn('Simulation status:', e));
}

// Driving Visualization
let drivingCanvas = null;
let drivingCtx = null;

function initDrivingViz() {
  drivingCanvas = $('#drivingCanvas');
  if (!drivingCanvas) return;
  drivingCtx = drivingCanvas.getContext('2d');

  function resizeCanvas() {
    const parent = drivingCanvas.parentElement;
    if (parent) {
      drivingCanvas.width = parent.clientWidth;
      drivingCanvas.height = Math.max(300, parent.clientHeight);
    }
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  drawDrivingFrame();
}

function startDrivingAnimation() {
  if (drivingState.animFrame) return;
  function loop() {
    drawDrivingFrame();
    drivingState.animFrame = requestAnimationFrame(loop);
  }
  drivingState.animFrame = requestAnimationFrame(loop);
}

function stopDrivingAnimation() {
  if (drivingState.animFrame) {
    cancelAnimationFrame(drivingState.animFrame);
    drivingState.animFrame = null;
  }
}

function updateDrivingViz(vehicleState, decisionData) {
  if (vehicleState) {
    drivingState.speed = Number(vehicleState.speed_kmh || 0);
    drivingState.steering = Number(vehicleState.steering_angle || 0);
    drivingState.laneOffset = Number(vehicleState.lane_offset || 0);
    drivingState.sign = vehicleState.detected_sign || 'None';
  }
  if (decisionData) {
    drivingState.decision = decisionData.decision || decisionData.action || '\u2014';
    drivingState.confidence = Number(decisionData.confidence || decisionData.quantum_confidence || decisionData.classical_confidence || 0);
  }

  setText('#vizSpeed', `${drivingState.speed.toFixed(0)}`);
  setText('#vizDecision', drivingState.decision);
  setText('#vizSteering', `${drivingState.steering.toFixed(2)}\u00b0`);
  setText('#vizSign', drivingState.sign);
}

function drawDrivingFrame() {
  if (!drivingCtx || !drivingCanvas) return;
  const W = drivingCanvas.width;
  const H = drivingCanvas.height;
  const ctx = drivingCtx;
  const weather = drivingState.weather;

  drivingState.roadOffset = (drivingState.roadOffset + Math.max(1, drivingState.speed * 0.08)) % 40;

  // Sky
  let skyColor = '#87CEEB';
  let groundColor = '#4a4a4a';
  if (weather === 'Night') { skyColor = '#0a0a2e'; groundColor = '#1a1a1a'; }
  else if (weather === 'Dusk' || weather === 'Dawn') { skyColor = '#ff7e5f'; groundColor = '#3a3a3a'; }
  else if (weather === 'Fog') { skyColor = '#c0c0c0'; groundColor = '#6a6a6a'; }
  else if (weather === 'Rain' || weather === 'Storm') { skyColor = '#5a6a7a'; groundColor = '#3a3a3a'; }
  else if (weather === 'Snow') { skyColor = '#d0d8e0'; groundColor = '#e0e0e0'; }

  const skyGrad = ctx.createLinearGradient(0, 0, 0, H * 0.45);
  skyGrad.addColorStop(0, skyColor);
  skyGrad.addColorStop(1, '#e0e8f0');
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, W, H * 0.45);

  ctx.fillStyle = groundColor;
  ctx.fillRect(0, H * 0.45, W, H * 0.55);

  // Perspective road
  const roadTopLeft = W * 0.38;
  const roadTopRight = W * 0.62;
  const roadBottomLeft = W * 0.05;
  const roadBottomRight = W * 0.95;
  const horizonY = H * 0.45;
  const steerShift = drivingState.steering * 30;

  ctx.fillStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(roadTopLeft + steerShift, horizonY);
  ctx.lineTo(roadTopRight + steerShift, horizonY);
  ctx.lineTo(roadBottomRight, H);
  ctx.lineTo(roadBottomLeft, H);
  ctx.closePath();
  ctx.fill();

  // White edge lines
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(roadTopLeft + steerShift, horizonY);
  ctx.lineTo(roadBottomLeft, H);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(roadTopRight + steerShift, horizonY);
  ctx.lineTo(roadBottomRight, H);
  ctx.stroke();

  // Center dashed line (animated)
  const centerTopX = (roadTopLeft + roadTopRight) / 2 + steerShift;
  const centerBottomX = (roadBottomLeft + roadBottomRight) / 2;
  const numDashes = 12;
  for (let i = 0; i < numDashes; i++) {
    const t1 = (i / numDashes + drivingState.roadOffset / 400) % 1;
    const t2 = ((i + 0.4) / numDashes + drivingState.roadOffset / 400) % 1;
    if (t1 >= 1 || t2 >= 1) continue;
    const x1 = centerTopX + (centerBottomX - centerTopX) * t1;
    const y1 = horizonY + (H - horizonY) * t1;
    const x2 = centerTopX + (centerBottomX - centerTopX) * t2;
    const y2 = horizonY + (H - horizonY) * t2;
    const dashWidth = 1 + t1 * 4;
    ctx.strokeStyle = '#ffcc00';
    ctx.lineWidth = dashWidth;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // Car
  const carW = 50;
  const carH = 80;
  const carX = W / 2 - carW / 2 + drivingState.laneOffset * 60;
  const carY = H - carH - 30;

  const carGrad = ctx.createLinearGradient(carX, carY, carX, carY + carH);
  carGrad.addColorStop(0, '#16a34a');
  carGrad.addColorStop(1, '#0d7a35');
  ctx.fillStyle = carGrad;
  roundRect(ctx, carX, carY, carW, carH, 8);
  ctx.fill();

  // Windshield
  ctx.fillStyle = 'rgba(135,206,235,0.6)';
  roundRect(ctx, carX + 6, carY + 8, carW - 12, 20, 4);
  ctx.fill();

  // Headlights
  ctx.fillStyle = '#ffeb3b';
  ctx.beginPath(); ctx.arc(carX + 8, carY + 4, 4, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(carX + carW - 8, carY + 4, 4, 0, Math.PI * 2); ctx.fill();

  // Taillights
  ctx.fillStyle = '#ef4444';
  ctx.beginPath(); ctx.arc(carX + 8, carY + carH - 4, 3, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(carX + carW - 8, carY + carH - 4, 3, 0, Math.PI * 2); ctx.fill();

  // Lane offset indicator
  if (Math.abs(drivingState.laneOffset) > 0.15) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(W / 2, carY + carH / 2);
    ctx.lineTo(carX + carW / 2, carY + carH / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Traffic sign
  if (drivingState.sign && drivingState.sign !== 'None') {
    const signX = W * 0.82;
    const signY = H * 0.35;
    ctx.fillStyle = '#777';
    ctx.fillRect(signX + 12, signY + 24, 4, 30);
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    roundRect(ctx, signX, signY, 28, 24, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#333';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(drivingState.sign.substring(0, 6), signX + 14, signY + 16);
    ctx.textAlign = 'start';
  }

  // Decision arrow
  const dec = drivingState.decision;
  const arrowX = W / 2;
  const arrowY = carY - 20;
  ctx.fillStyle = '#16a34a';
  ctx.font = 'bold 18px Inter, sans-serif';
  ctx.textAlign = 'center';
  if (dec === 'Left' || dec === 'Lane Change Left' || dec === 'left') {
    ctx.fillText('\u25c0', arrowX - 25, arrowY);
  } else if (dec === 'Right' || dec === 'Lane Change Right' || dec === 'right') {
    ctx.fillText('\u25b6', arrowX + 25, arrowY);
  } else if (dec === 'Straight' || dec === 'Forward' || dec === 'Accelerate' || dec === 'forward') {
    ctx.fillText('\u25b2', arrowX, arrowY - 5);
  } else if (dec === 'Emergency Stop' || dec === 'Slow Down' || dec === 'Brake' || dec === 'brake') {
    ctx.fillStyle = '#ef4444';
    ctx.fillText('\u25a0', arrowX, arrowY);
  }
  ctx.textAlign = 'start';

  // Confidence bar
  const barW = 120;
  const barH = 8;
  const barX = W / 2 - barW / 2;
  const barY = H - 15;
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  roundRect(ctx, barX, barY, barW, barH, 4);
  ctx.fill();
  const confWidth = barW * drivingState.confidence;
  if (confWidth > 0) {
    ctx.fillStyle = drivingState.confidence > 0.7 ? '#16a34a' : drivingState.confidence > 0.4 ? '#f59e0b' : '#ef4444';
    roundRect(ctx, barX, barY, confWidth, barH, 4);
    ctx.fill();
  }

  // Weather effects
  if (weather === 'Rain' || weather === 'Storm') {
    for (let i = 0; i < 60; i++) {
      const rx = Math.random() * W;
      const ry = Math.random() * H;
      ctx.strokeStyle = 'rgba(100,149,237,0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(rx - 1, ry + 8); ctx.stroke();
    }
  }
  if (weather === 'Snow') {
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.beginPath(); ctx.arc(Math.random() * W, Math.random() * H, 2, 0, Math.PI * 2); ctx.fill();
    }
  }
  if (weather === 'Fog') {
    ctx.fillStyle = 'rgba(200,200,200,0.25)';
    ctx.fillRect(0, 0, W, H);
  }
  if (weather === 'Night') {
    ctx.fillStyle = 'rgba(0,0,30,0.3)';
    ctx.fillRect(0, 0, W, H);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// Simulation actions
let simRunning = false, pollTimer = null;

async function startSimulation() {
  let result = null;
  try {
    result = await apiFetch('/api/start-simulation/', { method: 'POST', body: JSON.stringify({}) });
  } catch (e) {
    console.warn('Start sim:', e);
    updateSimStatus('Failed', 'red');
    return;
  }
  simRunning = true;
  updateSimStatus('Running', 'green');
  const startBtn = $('#btnStart');
  const stopBtn  = $('#btnStop');
  if (startBtn) startBtn.disabled = true;
  if (stopBtn)  stopBtn.disabled  = false;
  if (result?.session_id) setText('#sessionId', result.session_id);
  if (result?.weather) {
    setText('#sessionWeather', result.weather);
    setText('#weatherLargeText', result.weather);
    const icon = WEATHER_ICONS[result.weather] || '\u2600\ufe0f';
    setText('#weatherLargeIcon', icon);
  }
  if (typeof result?.quantum_enabled === 'boolean') {
    const q = result.quantum_enabled ? 'ON' : 'OFF';
    setText('#sessionQuantum', q);
    setText('#quantumToggleLabel', result.quantum_enabled ? 'Enabled' : 'Disabled');
    setChecked('#quantumToggle', result.quantum_enabled);
  }
  updateTopbar({
    running: true,
    weather: result?.weather || 'Clear',
    quantum_enabled: result?.quantum_enabled !== false
  });
  startUptime();
  startPolling();
  startDrivingAnimation();
}

async function stopSimulation() {
  try {
    await apiFetch('/api/stop-simulation/', { method: 'POST', body: JSON.stringify({}) });
  } catch {}
  simRunning = false;
  updateSimStatus('Stopped', 'orange');
  const startBtn = $('#btnStart');
  const stopBtn  = $('#btnStop');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn)  stopBtn.disabled  = true;
  setText('#sessionQuantum', getChecked('#quantumToggle') ? 'ON' : 'OFF');
  updateTopbar({ running: false, weather: drivingState.weather, quantum_enabled: getChecked('#quantumToggle') });
  stopUptime();
  stopPolling();
  stopDrivingAnimation();
  const dpEl = $('#sessionDataPoints');
  if (dpEl) dpEl.textContent = '0';
}

function updateSimStatus(text, color) {
  const badge = $('#simControlBadge');
  if (badge) {
    badge.textContent = text;
    badge.className = 'clay-badge badge-' + color;
  }
  setText('#sessionStatus', text);
}

async function changeWeather(preset) {
  if (!preset) {
    const sel = $('#weatherSelect');
    preset = sel ? sel.value : 'Clear';
  }
  try {
    const res = await apiFetch('/api/change-weather/', { method: 'POST', body: JSON.stringify({ weather: preset }) });
    if (res?.weather) {
      setText('#sessionWeather', res.weather);
      setText('#weatherLargeText', res.weather);
      const icon = WEATHER_ICONS[res.weather] || '\u2600\ufe0f';
      setText('#weatherLargeIcon', icon);
      drivingState.weather = res.weather;
    }
  } catch (e) { console.warn('Weather:', e); }
}

async function toggleQuantum(enabled) {
  SETTINGS.set('quantumEnabled', !!enabled);
  setText('#quantumToggleLabel', enabled ? 'Enabled' : 'Disabled');
  setText('#sessionQuantum', enabled ? 'ON' : 'OFF');
  try {
    await apiFetch('/api/toggle-quantum/', { method: 'POST', body: JSON.stringify({ enabled: !!enabled }) });
  } catch (e) { console.warn('Toggle quantum:', e); }
}

// Polling
function startPolling() {
  if (pollTimer) return;
  poll();
  pollTimer = setInterval(poll, SETTINGS.get('pollInterval'));
}
function stopPolling() { clearInterval(pollTimer); pollTimer = null; }

async function poll() {
  try {
    const [state, logs, simStatus] = await Promise.all([
      apiFetch('/api/vehicle-state/'),
      apiFetch('/api/decision-logs/'),
      apiFetch('/api/simulation-status/')
    ]);
    updateState(state);
    const latestLog = Array.isArray(logs) ? logs[0] : logs;
    updateDecision(latestLog);
    updateSimulationStatus(simStatus);
    updateDrivingViz(state, latestLog);
    if (simStatus.weather) drivingState.weather = simStatus.weather;
  } catch (e) { console.warn('Poll error:', e); }
}

function updateState(s) {
  const steering = Number(s.steering_angle || 0);
  const throttle = Number(s.throttle || 0);
  const brake = Number(s.brake || 0);
  const speed = Number(s.speed_kmh || 0);
  const lane = Number(s.lane_offset || 0);

  setText('#metricSteering', `${steering.toFixed(2)}\u00b0`);
  setText('#metricThrottle', throttle.toFixed(2));
  setText('#metricBrake', brake.toFixed(2));
  setText('#metricSpeed', `${speed.toFixed(1)} km/h`);
  setText('#metricLaneOffset', `${lane.toFixed(2)} m`);
  setText('#metricSign', s.detected_sign || 'None');
  setText('#simSign', s.detected_sign || 'None');
  const dpEl = $('#sessionDataPoints');
  if (dpEl) {
    const current = Number(dpEl.textContent || 0);
    dpEl.textContent = Number.isFinite(current) ? String(current + 1) : '1';
  }

  setBar('#barSteering', ((steering + 1) / 2) * 100);
  setBar('#barThrottle', throttle * 100);
  setBar('#barBrake', brake * 100);
  setBar('#barSpeed', Math.min(100, (speed / 120) * 100));

  // Update weather icon on dashboard
  if (s.weather) {
    const icon = WEATHER_ICONS[s.weather] || '\u2600\ufe0f';
    setText('#weatherLargeIcon', icon);
    setText('#weatherLargeText', s.weather);
  }

  // Update model type labels
  if (s.model_type) {
    setText('#modelTypeLabel', s.model_type === 'quantum' ? 'Quantum Model' : 'Classical Model');
    setText('#modelTypeSub', s.model_type === 'quantum' ? 'Hybrid Neural Network' : 'Standard Neural Network');
  }

  const now = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  if (charts.steering) {
    pushPoint(charts.steering, now, [steering, throttle, brake]);
  }
}

function updateDecision(d) {
  if (!d || typeof d !== 'object') return;
  const now = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const decision = d.decision || d.action || '\u2014';
  const confidence = Number((d.confidence ?? d.quantum_confidence ?? d.classical_confidence) || 0);

  setText('#metricDecision', decision);
  setText('#simDecision', decision);
  setText('#metricConfidence', (confidence * 100).toFixed(2) + '%');
  setText('#simConfidence', (confidence * 100).toFixed(2) + '%');
  setConfidenceRing(confidence);

  if (charts.confidence) {
    pushPoint(charts.confidence, now, [
      Number(d.classical_confidence || d.confidence || 0),
      Number(d.quantum_confidence || d.confidence || 0)
    ]);
  }

  addLogRow(d, now);
}

// Log table
function addLogRow(d, time) {
  const tbody = $('#logsBody');
  if (!tbody) return;

  const actionColors = {
    accelerate: 'green', brake: 'red', steer: 'blue',
    lane_change: 'purple', emergency_stop: 'orange',
    left: 'blue', right: 'blue', straight: 'green', forward: 'green',
    slow_down: 'orange',
    'Accelerate': 'green', 'Brake': 'red', 'Steer': 'blue',
    'Left': 'blue', 'Right': 'blue', 'Straight': 'green', 'Forward': 'green',
    'Emergency Stop': 'orange', 'Slow Down': 'orange',
    'Lane Change Left': 'purple', 'Lane Change Right': 'purple'
  };
  const action = d.decision || d.action || '\u2014';
  const color = actionColors[action] || 'green';
  const conf = Number(d.confidence ?? d.quantum_confidence ?? d.classical_confidence ?? 0);
  const steerErr = Number(d.steering_error ?? 0);
  const laneDev = Number(d.lane_deviation ?? 0);
  const collision = !!d.collision_flag;
  const model = d.model_type || (d.quantum_confidence ? 'quantum' : 'classical');

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${d.id || '\u2014'}</td>
    <td><span class="clay-badge badge-${model === 'quantum' ? 'purple' : 'blue'}">${model}</span></td>
    <td><span class="clay-badge badge-${color}">${action}</span></td>
    <td>${(conf * 100).toFixed(2)}%</td>
    <td>${steerErr.toFixed(4)}</td>
    <td>${laneDev.toFixed(4)}</td>
    <td>${collision ? 'Yes' : 'No'}</td>
    <td>${d.weather || '\u2014'}</td>
    <td>${time}</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);
  while (tbody.children.length > 50) tbody.removeChild(tbody.lastChild);
}

// LOGS PAGE
function initLogs() {
  loadLogs();
}

async function loadLogs() {
  try {
    const modelType = $('#logFilterModel')?.value || '';
    const url = modelType ? `/api/decision-logs/?model_type=${encodeURIComponent(modelType)}` : '/api/decision-logs/';
    const data = await apiFetch(url);
    const tbody = $('#logsBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    const rows = Array.isArray(data) ? data : [];
    rows.slice(0, 100).forEach(item => {
      const t = item.timestamp ? new Date(item.timestamp).toLocaleTimeString('en-GB') : new Date().toLocaleTimeString('en-GB');
      addLogRow(item, t);
    });
    updateLogSummary(rows);
  } catch (e) { console.warn('Fetch logs:', e); }
}

function exportCSV() {
  window.open('/api/export-csv/', '_blank');
}

function updateLogSummary(rows) {
  const total = rows.length;
  const avgConfidence = total ? rows.reduce((a, r) => a + Number(r.confidence || 0), 0) / total : 0;
  const collisions = rows.reduce((a, r) => a + (r.collision_flag ? 1 : 0), 0);
  const avgDeviation = total ? rows.reduce((a, r) => a + Number(r.lane_deviation || 0), 0) / total : 0;

  setText('#logTotalEntries', String(total));
  setText('#logAvgConfidence', (avgConfidence * 100).toFixed(1) + '%');
  setText('#logCollisions', String(collisions));
  setText('#logAvgDeviation', avgDeviation.toFixed(4));
}

// COMPARISON PAGE
function initComparison() {
  loadComparison();
}

async function loadComparison() {
  try {
    const data = await apiFetch('/api/comparison-data/');

    // Update summary cards - correct element IDs from comparison.html
    setText('#classicalConfidence', ((data.classical?.avg_confidence || 0) * 100).toFixed(1) + '%');
    setText('#classicalDeviation', (data.classical?.avg_lane_deviation || 0).toFixed(4));
    setText('#classicalSteeringErr', (data.classical?.avg_steering_error || 0).toFixed(4));
    setText('#classicalCollisions', data.classical?.collision_count || 0);
    setText('#classicalTotal', data.classical?.total || 0);

    setText('#quantumConfidence', ((data.quantum?.avg_confidence || 0) * 100).toFixed(1) + '%');
    setText('#quantumDeviation', (data.quantum?.avg_lane_deviation || 0).toFixed(4));
    setText('#quantumSteeringErr', (data.quantum?.avg_steering_error || 0).toFixed(4));
    setText('#quantumCollisions', data.quantum?.collision_count || 0);
    setText('#quantumTotal', data.quantum?.total || 0);

    // Lane Deviation chart
    const laneCtx = $('#chartLaneDeviation');
    if (laneCtx && !charts.laneDeviation) {
      charts.laneDeviation = new Chart(laneCtx, {
        type: 'bar',
        data: {
          labels: ['Classical', 'Quantum'],
          datasets: [{
            data: [
              data.classical?.avg_lane_deviation || 0,
              data.quantum?.avg_lane_deviation || 0
            ],
            backgroundColor: [COLORS.blue, COLORS.purple],
            borderRadius: 10, borderWidth: 0, barThickness: 48
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { min: 0, grid: { color: COLORS.grid }, title: { display: true, text: 'Lane Deviation (m)' } },
            x: { grid: { display: false } }
          }
        }
      });
    } else if (charts.laneDeviation) {
      charts.laneDeviation.data.datasets[0].data = [
        data.classical?.avg_lane_deviation || 0,
        data.quantum?.avg_lane_deviation || 0
      ];
      charts.laneDeviation.update();
    }

    // Confidence Comparison chart
    const confCtx = $('#chartConfidenceComparison');
    if (confCtx && !charts.confComparison) {
      charts.confComparison = new Chart(confCtx, {
        type: 'bar',
        data: {
          labels: ['Classical', 'Quantum'],
          datasets: [{
            data: [
              (data.classical?.avg_confidence || 0) * 100,
              (data.quantum?.avg_confidence || 0) * 100
            ],
            backgroundColor: [COLORS.blue, COLORS.purple],
            borderRadius: 10, borderWidth: 0, barThickness: 48
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { min: 0, max: 100, grid: { color: COLORS.grid }, title: { display: true, text: 'Confidence (%)' } },
            x: { grid: { display: false } }
          }
        }
      });
    } else if (charts.confComparison) {
      charts.confComparison.data.datasets[0].data = [
        (data.classical?.avg_confidence || 0) * 100,
        (data.quantum?.avg_confidence || 0) * 100
      ];
      charts.confComparison.update();
    }

    // Collisions chart
    const collCtx = $('#chartCollisions');
    if (collCtx && !charts.collisions) {
      charts.collisions = new Chart(collCtx, {
        type: 'bar',
        data: {
          labels: ['Classical', 'Quantum'],
          datasets: [{
            data: [
              data.classical?.collision_count || 0,
              data.quantum?.collision_count || 0
            ],
            backgroundColor: [COLORS.red, COLORS.green],
            borderRadius: 10, borderWidth: 0, barThickness: 48
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { min: 0, grid: { color: COLORS.grid }, title: { display: true, text: 'Collisions' } },
            x: { grid: { display: false } }
          }
        }
      });
    } else if (charts.collisions) {
      charts.collisions.data.datasets[0].data = [
        data.classical?.collision_count || 0,
        data.quantum?.collision_count || 0
      ];
      charts.collisions.update();
    }

  } catch (e) { console.warn('Comparison:', e); }
}

// ANALYTICS PAGE
async function initAnalytics() {
  try {
    const data = await apiFetch('/api/analytics-data/');

    // Populate stat cards
    setText('#analyticsSessions', data.total_sessions || 0);
    setText('#analyticsDecisions', data.total_decisions || 0);
    setText('#analyticsQuantum', data.quantum_decisions || 0);
    setText('#analyticsCollisions', data.total_collisions || 0);

    // Decision distribution doughnut
    const distCtx = $('#chartDecisionDist');
    if (distCtx) {
      const dist = data.decision_distribution || {};
      const labels = Object.keys(dist);
      const values = Object.values(dist);
      const bgColors = [COLORS.green, COLORS.blue, COLORS.purple, COLORS.orange, COLORS.red, COLORS.teal];

      if (labels.length > 0) {
        charts.decisionDist = new Chart(distCtx, {
          type: 'doughnut',
          data: {
            labels: labels,
            datasets: [{
              data: values,
              backgroundColor: bgColors.slice(0, Math.max(labels.length, bgColors.length)),
              borderWidth: 0
            }]
          },
          options: {
            responsive: true, maintainAspectRatio: false, cutout: '65%',
            plugins: { legend: { position: 'bottom' } }
          }
        });
      } else {
        initAnalyticsFallbackChart(distCtx, 'doughnut');
      }
    }

    // Confidence trend line
    const trendCtx = $('#chartConfidenceTrend');
    if (trendCtx) {
      const trend = data.confidence_trend || {};
      const classicalTrend = trend.classical || {};
      const quantumTrend = trend.quantum || {};
      const labels = classicalTrend.labels || quantumTrend.labels || [];
      const classicalValues = classicalTrend.values || [];
      const quantumValues = quantumTrend.values || [];

      if (labels.length > 0) {
        charts.confTrend = new Chart(trendCtx, {
          type: 'line',
          data: { labels, datasets: [
            { label: 'Classical', data: classicalValues, borderColor: COLORS.blue, backgroundColor: COLORS.blueBg, borderWidth: 2.5, pointRadius: 3, tension: .4, fill: true },
            { label: 'Quantum', data: quantumValues, borderColor: COLORS.purple, backgroundColor: COLORS.purpleBg, borderWidth: 2.5, pointRadius: 3, tension: .4, fill: true }
          ]},
          options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
              y: { min: 0, max: 1, grid: { color: COLORS.grid } },
              x: { grid: { display: false } }
            },
            interaction: { mode: 'index', intersect: false }
          }
        });
      } else {
        initAnalyticsFallbackChart(trendCtx, 'line');
      }
    }

    // Sessions table
    const sessionsBody = $('#sessionsBody');
    if (sessionsBody && data.sessions) {
      sessionsBody.innerHTML = '';
      data.sessions.forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${s.id}</td>
          <td><span class="clay-badge badge-${s.status === 'running' ? 'green' : 'muted'}">${s.status}</span></td>
          <td><span class="clay-badge badge-${s.quantum_enabled ? 'purple' : 'blue'}">${s.quantum_enabled ? 'Quantum' : 'Classical'}</span></td>
          <td>${s.started_at || '\u2014'}</td>
          <td>${s.stopped_at || '\u2014'}</td>
        `;
        sessionsBody.appendChild(tr);
      });
    }

    // Weather accuracy bar chart
    const weatherCtx = $('#chartWeatherAccuracy');
    if (weatherCtx) {
      const wa = data.weather_accuracy || {};
      const weatherLabels = Object.keys(wa);
      const classicalAccs = weatherLabels.map(w => wa[w]?.classical || 0);
      const quantumAccs = weatherLabels.map(w => wa[w]?.quantum || 0);

      if (weatherLabels.length > 0) {
        charts.weatherAcc = new Chart(weatherCtx, {
          type: 'bar',
          data: {
            labels: weatherLabels,
            datasets: [
              { label: 'Classical', data: classicalAccs, backgroundColor: COLORS.blue, borderRadius: 8, borderWidth: 0 },
              { label: 'Quantum', data: quantumAccs, backgroundColor: COLORS.purple, borderRadius: 8, borderWidth: 0 }
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
              y: { min: 0, max: 100, grid: { color: COLORS.grid } },
              x: { grid: { display: false } }
            },
            plugins: { legend: { position: 'top' } }
          }
        });
      } else {
        initAnalyticsFallbackChart(weatherCtx, 'bar');
      }
    }

  } catch (e) {
    console.warn('Analytics:', e);
    initAnalyticsFallback();
  }
}

function initAnalyticsFallbackChart(ctx, type) {
  if (type === 'doughnut') {
    new Chart(ctx, {
      type: 'doughnut',
      data: { labels: ['No Data'], datasets: [{ data: [1], backgroundColor: ['#e5e7eb'], borderWidth: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { display: false } } }
    });
  } else if (type === 'line') {
    new Chart(ctx, {
      type: 'line',
      data: { labels: ['\u2014'], datasets: [{ label: 'No Data', data: [0], borderColor: '#e5e7eb', borderWidth: 2 }] },
      options: { responsive: true, maintainAspectRatio: false }
    });
  } else {
    new Chart(ctx, {
      type: 'bar',
      data: { labels: ['No Data'], datasets: [{ data: [0], backgroundColor: '#e5e7eb', borderRadius: 8 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });
  }
}

function initAnalyticsFallback() {
  const distCtx = $('#chartDecisionDist');
  const trendCtx = $('#chartConfidenceTrend');
  const weatherCtx = $('#chartWeatherAccuracy');
  if (distCtx) initAnalyticsFallbackChart(distCtx, 'doughnut');
  if (trendCtx) initAnalyticsFallbackChart(trendCtx, 'line');
  if (weatherCtx) initAnalyticsFallbackChart(weatherCtx, 'bar');
}

// SETTINGS PAGE
function initSettings() {
  const settings = SETTINGS.all();

  setInputVal('#settingPollInterval',  settings.pollInterval);
  setInputVal('#settingMaxPoints',     settings.maxPoints);
  setChecked('#settingAutoStart',      settings.autoStart);
  setChecked('#settingQuantum',        settings.quantumEnabled);
  setChecked('#settingNoise',          settings.noiseEnabled);
  setInputVal('#settingNoiseStrength', settings.noiseStrength);
  setChecked('#settingAnimations',     settings.animations);
  setChecked('#settingCompact',        settings.compact);

  $$('.setting-control input, .setting-control select').forEach(el => {
    el.addEventListener('change', () => {
      SETTINGS.set('pollInterval',   getInputVal('#settingPollInterval', 3000));
      SETTINGS.set('maxPoints',      getInputVal('#settingMaxPoints', 30));
      SETTINGS.set('autoStart',      getChecked('#settingAutoStart'));
      SETTINGS.set('quantumEnabled', getChecked('#settingQuantum'));
      SETTINGS.set('noiseEnabled',   getChecked('#settingNoise'));
      SETTINGS.set('noiseStrength',  getInputVal('#settingNoiseStrength', 0.05));
      SETTINGS.set('animations',     getChecked('#settingAnimations'));
      SETTINGS.set('compact',        getChecked('#settingCompact'));
    });
  });

  const exportBtn  = $('#btnExportData');
  const refreshBtn = $('#btnRefreshData');
  const clearBtn   = $('#btnClearData');

  if (exportBtn) exportBtn.addEventListener('click', () => {
    const data = JSON.stringify(SETTINGS.all(), null, 2);
    downloadFile('quantumdrive_settings.json', data, 'application/json');
  });
  if (refreshBtn) refreshBtn.addEventListener('click', () => location.reload());
  if (clearBtn) clearBtn.addEventListener('click', () => {
    if (confirm('Clear all stored data?')) { localStorage.clear(); location.reload(); }
  });
}

// LANDING PAGE
function initLanding() {
  const menuBtn  = $('#landingMenuBtn');
  const navLinks = $('.landing-nav-links');
  if (menuBtn && navLinks) {
    menuBtn.addEventListener('click', () => navLinks.classList.toggle('open'));
  }
}

// QD GLOBAL NAMESPACE - used by onclick="" handlers in templates
window.QD = {
  startSimulation,
  stopSimulation,
  changeWeather,
  toggleQuantum,
  exportCSV,
  loadLogs,
  loadComparison
};

// SIDEBAR TOGGLE (mobile)
function initSidebar() {
  const toggle  = $('#menuToggle');
  const sidebar = $('.sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }
}

// Page router
document.addEventListener('DOMContentLoaded', () => {
  const body = document.body;
  const path = window.location.pathname;

  if (body.classList.contains('landing-body') || path === '/') {
    initLanding();
    return;
  }

  if (body.classList.contains('auth-body')) return;

  initSidebar();

  $$('.nav-item').forEach(item => {
    if (item.getAttribute('href') === path) item.classList.add('active');
    else item.classList.remove('active');
  });

  if (SETTINGS.get('animations')) {
    $$('.clay-card, .clay-card-mini').forEach((card, i) => {
      card.classList.add('animate-in');
      card.style.animationDelay = `${i * 0.06}s`;
    });
  }

  // Fetch topbar status on every authenticated page load
  apiFetch('/api/simulation-status/')
    .then(updateTopbar)
    .catch(() => {});

  if (path.includes('dashboard'))  initDashboard();
  if (path.includes('simulation')) initSimulation();
  if (path.includes('logs'))       initLogs();
  if (path.includes('comparison')) initComparison();
  if (path.includes('analytics'))  initAnalytics();
  if (path.includes('settings'))   initSettings();
});
