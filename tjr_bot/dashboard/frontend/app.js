/**
 * app.js — TJR Bot Dashboard Real-time Frontend
 * WebSocket connection + Chart.js rendering + live UI updates
 */

'use strict';

// ============================================================
// State
// ============================================================
let ws = null;
let liveData = {};
let stats = {};
let tradeHistory = [];
let wsReconnectDelay = 1000;
let equityChart = null;
let winLossChart = null;
let rDistChart = null;
let sessionChart = null;
let lastPhase = '';

// ============================================================
// Utility helpers
// ============================================================
const $ = id => document.getElementById(id);
const fmt = (n, d = 2) => (typeof n === 'number' ? n.toFixed(d) : '--');
const fmtCcy = n => typeof n === 'number'
  ? '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  : '$--';
const fmtLevel = (n, digits = 5) => typeof n === 'number' && n > 0
  ? n.toFixed(digits)
  : '--';
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function pnlClass(v) {
  if (v > 0)  return 'profit';
  if (v < 0)  return 'loss';
  return 'neutral';
}

// ============================================================
// Clock
// ============================================================
function updateClock() {
  const now = new Date();
  // EST = UTC-5
  const est = new Date(now.getTime() - 5 * 3600000);
  const hh = String(est.getUTCHours()).padStart(2, '0');
  const mm = String(est.getUTCMinutes()).padStart(2, '0');
  const ss = String(est.getUTCSeconds()).padStart(2, '0');
  $('clockDisplay').textContent = `${hh}:${mm}:${ss} EST`;
}
setInterval(updateClock, 1000);
updateClock();

// ============================================================
// WebSocket Connection
// ============================================================
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws`;

  ws = new WebSocket(url);

  ws.onopen = () => {
    wsReconnectDelay = 1000;
    setLiveStatus(true);
    console.log('WS connected');
    // Also fetch history on connect
    fetchHistory();
  };

  ws.onmessage = evt => {
    try {
      liveData = JSON.parse(evt.data);
      if (liveData === 'pong') return;
      updateUI(liveData);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    setLiveStatus(false);
    console.log(`WS closed — reconnecting in ${wsReconnectDelay}ms`);
    setTimeout(connectWS, wsReconnectDelay);
    wsReconnectDelay = Math.min(wsReconnectDelay * 2, 30000);
  };

  ws.onerror = err => {
    console.error('WS error:', err);
    ws.close();
  };
}

function setLiveStatus(live) {
  const dot   = $('liveDot');
  const label = $('liveLabel');
  if (live) {
    dot.className   = 'dot live';
    label.textContent = 'LIVE';
    label.style.color = 'var(--green)';
  } else {
    dot.className   = 'dot dead';
    label.textContent = 'DISCONNECTED';
    label.style.color = 'var(--red)';
  }
}

// Keep WS alive
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send('ping');
}, 20000);

// ============================================================
// Fetch history (REST)
// ============================================================
async function fetchHistory() {
  try {
    const res = await fetch('/api/history');
    const json = await res.json();
    tradeHistory = json.trades || [];
    stats = json.statistics || {};
    updateTradesTable(tradeHistory);
    updateStatCards(null, stats);
    updateMiniCharts(stats);
  } catch (e) {
    console.error('History fetch error:', e);
  }
}

// Poll history every 30 s
setInterval(fetchHistory, 30000);

// ============================================================
// Main UI Update
// ============================================================
function updateUI(d) {
  if (!d) return;

  // Symbol display
  if (d.symbol) $('symDisplay').textContent = d.symbol;

  // Stat cards
  updateStatCards(d, stats);

  // Bot status
  updateBotStatus(d);

  // Checklist
  updateChecklist(d);

  // Position
  updatePosition(d);

  // Session levels
  updateLevels(d);

  // Risk monitor
  updateRiskMonitor(d);

  // Equity chart
  if (d.equity_history && d.equity_history.length > 0) {
    updateEquityChart(d.equity_history);
  }
}

// ============================================================
// Stat Cards
// ============================================================
function updateStatCards(d, s) {
  if (d) {
    const bal  = d.account_balance;
    const eq   = d.account_equity;
    const pnl  = d.daily_pnl_pct;
    $('statBalance').textContent = fmtCcy(bal);
    $('statEquity').textContent  = fmtCcy(eq);
    $('statDailyPnL').textContent = (pnl >= 0 ? '+' : '') + fmt(pnl, 2) + '%';
    $('statDailyPnL').className = 'card-value ' + pnlClass(pnl);
  }
  if (s && s.total_trades > 0) {
    $('statWinRate').textContent = fmt(s.win_rate, 1) + '%';
    $('statWinRate').className = 'card-value ' + pnlClass(s.win_rate - 50);
    $('statProfitFactor').textContent = fmt(s.profit_factor, 2);
    $('statProfitFactor').className = 'card-value ' + pnlClass(s.profit_factor - 1);
  }
}

// ============================================================
// Bot Status Panel
// ============================================================
function updateBotStatus(d) {
  const phase = d.phase || 'IDLE';

  // Animate on phase change
  if (phase !== lastPhase) {
    const el = $('stPhase');
    el.classList.add('fade-in');
    el.addEventListener('animationend', () => el.classList.remove('fade-in'), { once: true });
    lastPhase = phase;
  }

  $('stPhase').textContent = phase;
  $('stPhase').style.borderColor = phaseColor(phase);
  $('stPhase').style.color = phaseColor(phase);

  const sessionMap = {
    'NY_KILLZONE': 'NY Kill Zone',
    'LONDON_KILLZONE': 'London Kill Zone',
    'ASIA': 'Asia Session',
    'OFF_SESSION': 'Off Session',
  };
  $('stSession').textContent = sessionMap[d.session] || d.session || '--';

  const biasEl = $('stBias');
  if (d.htf_bias === 'BULLISH') {
    biasEl.textContent = '↑ BULLISH';
    biasEl.style.color = 'var(--green)';
  } else if (d.htf_bias === 'BEARISH') {
    biasEl.textContent = '↓ BEARISH';
    biasEl.style.color = 'var(--red)';
  } else {
    biasEl.textContent = '— NEUTRAL';
    biasEl.style.color = 'var(--text-dim)';
  }

  if (d.sweep_detected) {
    $('stSweep').textContent = `✓ @ ${fmtLevel(d.sweep_level)} (${d.sweep_direction})`;
    $('stSweep').style.color = 'var(--green)';
  } else {
    $('stSweep').textContent = 'Watching...';
    $('stSweep').style.color = 'var(--text-dim)';
  }

  if (d.m5_bos_confirmed) {
    $('stBOS').textContent = '✓ Confirmed';
    $('stBOS').style.color = 'var(--green)';
  } else {
    $('stBOS').textContent = 'Pending...';
    $('stBOS').style.color = 'var(--text-dim)';
  }

  $('stChecklist').textContent = `${d.checklist_score || 0}/8`;
  $('stTrades').textContent = `${d.trades_today || 0}/3`;
}

function phaseColor(phase) {
  const map = {
    'IDLE':             'var(--text-dim)',
    'AWAIT_SWEEP':      'var(--amber)',
    'SWEEP_CONFIRMED':  'var(--amber)',
    'AWAIT_M5_BOS':     'var(--blue)',
    'AWAIT_M5_RETRACE': 'var(--blue)',
    'AWAIT_M1_ENTRY':   'var(--purple)',
    'TRADE_OPEN':       'var(--green)',
    'DAY_HALTED':       'var(--red)',
  };
  return map[phase] || 'var(--text-dim)';
}

// ============================================================
// 8-Factor Checklist
// ============================================================
const CHK_DETAILS = [
  d => d.session ? sessionMap2(d.session) : 'Not active',
  d => d.htf_bias !== 'NEUTRAL' ? d.htf_bias : 'Indeterminate',
  d => d.sweep_detected ? `@ ${fmtLevel(d.sweep_level)}` : 'Watching...',
  d => d.m5_bos_confirmed ? 'Confirmed' : 'Waiting...',
  d => d.fvg_active ? `${fmtLevel(d.fvg_low)} – ${fmtLevel(d.fvg_high)}` : 'Not yet',
  d => d.ob_active ? `${fmtLevel(d.ob_low)} – ${fmtLevel(d.ob_high)}` : 'Not yet',
  () => 'Optional',
  d => d.checklist ? (d.checklist[7] ? 'Confirmed' : 'Not yet') : 'Not yet',
];

function sessionMap2(s) {
  const m = { 'NY_KILLZONE': 'NY KZ', 'LONDON_KILLZONE': 'London KZ', 'ASIA': 'Asia' };
  return m[s] || s;
}

function updateChecklist(d) {
  const checks = Array.isArray(d.checklist) ? d.checklist : new Array(8).fill(false);
  const score = d.checklist_score || 0;

  for (let i = 0; i < 8; i++) {
    const li = $(`chk${i}`);
    const statusEl = li.querySelector('.chk-status');
    const detailEl = $(`chkD${i}`);

    const confirmed = checks[i];
    const isOptional = i === 6;

    if (isOptional) {
      li.className = 'chk-item disabled';
      statusEl.textContent = '○';
    } else if (confirmed) {
      if (!li.classList.contains('confirmed')) {
        li.classList.add('fade-in');
        li.addEventListener('animationend', () => li.classList.remove('fade-in'), { once: true });
      }
      li.className = 'chk-item confirmed';
      statusEl.textContent = '✓';
    } else {
      li.className = 'chk-item pending';
      statusEl.textContent = '⏳';
    }

    if (detailEl && CHK_DETAILS[i]) {
      try { detailEl.textContent = CHK_DETAILS[i](d); } catch (_) {}
    }
  }

  const pct = (score / 8) * 100;
  $('checklistBarFill').style.width = pct + '%';
  $('checklistBarFill').style.background = score >= 6 ? 'var(--green)' :
                                           score >= 4 ? 'var(--amber)' : 'var(--red)';
  $('checklistScoreLbl').textContent = `Score: ${'█'.repeat(score)}${'░'.repeat(8 - score)} ${score}/8`;
}

// ============================================================
// Open Position Panel
// ============================================================
function updatePosition(d) {
  const panel = $('positionContent');

  if (!d.trade_active) {
    panel.innerHTML = '<p class="no-position">No active position</p>';
    return;
  }

  const dir = d.trade_direction || 'BUY';
  const pnlUSD = d.open_pnl_usd || 0;
  const pnlR   = d.open_pnl_r   || 0;
  const isProfit = pnlUSD >= 0;

  // Progress to TP1
  const sl    = d.stop_loss  || 0;
  const tp1   = d.tp1        || 0;
  const entry = d.entry_price || 0;
  let pct = 0;
  if (tp1 !== entry && sl !== entry) {
    const range = Math.abs(tp1 - entry);
    const moved = dir === 'BUY'
      ? (d.account_equity - entry)   // approximation
      : (entry - d.account_equity);
    pct = clamp(Math.abs(pnlR) / 1.0 * 100, 0, 100);
  }

  panel.innerHTML = `
    <div class="position-content">
      <div class="pos-header ${dir.toLowerCase()}">${dir} ${d.symbol || ''}</div>
      <div class="pos-info-grid">
        <div class="pos-kv"><span class="k">Entry</span><span class="v">${fmtLevel(entry)}</span></div>
        <div class="pos-kv"><span class="k">SL</span><span class="v" style="color:var(--red)">${fmtLevel(sl)}</span></div>
        <div class="pos-kv"><span class="k">TP1</span><span class="v" style="color:var(--green)">${fmtLevel(tp1)}</span></div>
        <div class="pos-kv"><span class="k">TP2</span><span class="v" style="color:var(--green-dim)">${fmtLevel(d.tp2)}</span></div>
      </div>
      <div class="pos-pnl ${isProfit ? 'profit' : 'loss'}">
        ${isProfit ? '+' : ''}${fmtCcy(pnlUSD)} (${isProfit ? '+' : ''}${fmt(pnlR, 2)}R)
      </div>
      <div class="pos-progress-wrap">
        <div class="pos-progress-label">${fmt(pct, 0)}% to TP1</div>
        <div class="pos-progress-bar">
          <div class="pos-progress-fill" style="width:${pct}%;background:${isProfit ? 'var(--green)' : 'var(--red)'}"></div>
        </div>
      </div>
    </div>
  `;
}

// ============================================================
// Session Levels
// ============================================================
function updateLevels(d) {
  const digs = 5;
  $('lvPDH').textContent  = fmtLevel(d.prev_day_high, digs);
  $('lvPDL').textContent  = fmtLevel(d.prev_day_low,  digs);
  $('lvAsiaH').textContent = fmtLevel(d.asia_high, digs);
  $('lvAsiaL').textContent = fmtLevel(d.asia_low,  digs);
  $('lvLonH').textContent  = fmtLevel(d.london_high, digs);
  $('lvLonL').textContent  = fmtLevel(d.london_low,  digs);
  $('lvH4H').textContent   = fmtLevel(d.h4_swing_high, digs);
  $('lvH4L').textContent   = fmtLevel(d.h4_swing_low,  digs);
}

// ============================================================
// Risk Monitor
// ============================================================
function updateRiskMonitor(d) {
  // Daily loss: cap is 1.5R
  const dailyR = Math.abs(Math.min(d.daily_pnl_r || 0, 0));
  const dailyPct = clamp(dailyR / 1.5 * 100, 0, 100);
  $('riskDailyFill').style.width = dailyPct + '%';
  $('riskDailyVal').textContent  = `${fmt(dailyR, 2)} / 1.5R`;

  // Trades today (max 3)
  const trades = d.trades_today || 0;
  $('riskTradesFill').style.width = clamp(trades / 3 * 100, 0, 100) + '%';
  $('riskTradesVal').textContent  = `${trades} / 3`;

  // Consecutive losses (max 2)
  const consec = d.consec_losses || 0;
  $('riskConsecFill').style.width = clamp(consec / 2 * 100, 0, 100) + '%';
  $('riskConsecVal').textContent  = `${consec} / 2`;

  // MT5 status
  const mt5El = $('riskMT5Status');
  if (d.mt5_connected === false) {
    mt5El.textContent = 'DISCONNECTED';
    mt5El.style.color = 'var(--red)';
  } else if (d._demo_mode) {
    mt5El.textContent = 'DEMO MODE';
    mt5El.style.color = 'var(--amber)';
  } else {
    mt5El.textContent = 'CONNECTED';
    mt5El.style.color = 'var(--green)';
  }

  // Max drawdown from stats
  if (stats.max_drawdown_r !== undefined) {
    $('riskMaxDD').textContent = `-${fmt(stats.max_drawdown_r, 2)}R`;
  }
}

// ============================================================
// Trades Table
// ============================================================
function updateTradesTable(trades) {
  const tbody = $('tradesBody');
  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="no-data">No trade history</td></tr>';
    return;
  }
  // Show last 15 trades, most recent first
  const recent = [...trades].reverse().slice(0, 15);
  tbody.innerHTML = recent.map(t => {
    const r = parseFloat(t.result_r || 0);
    const usd = parseFloat(t.result_usd || 0);
    const isWin = r > 0;
    const rClass = isWin ? 'td-win' : 'td-loss';
    const dirClass = t.direction === 'BUY' ? 'td-buy' : 'td-sell';
    const dt = t.datetime ? t.datetime.substring(0, 16) : '--';
    return `<tr>
      <td>${dt}</td>
      <td>${t.symbol || '--'}</td>
      <td class="${dirClass}">${t.direction || '--'}</td>
      <td>${parseFloat(t.entry || 0).toFixed(5)}</td>
      <td>${t.exit && t.exit !== '0' ? parseFloat(t.exit).toFixed(5) : '--'}</td>
      <td class="${rClass}">${r > 0 ? '+' : ''}${fmt(r, 2)}R</td>
      <td class="${rClass}">${usd > 0 ? '+' : ''}$${fmt(Math.abs(usd), 2)}</td>
    </tr>`;
  }).join('');
}

// ============================================================
// Charts
// ============================================================
const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 300 },
  plugins: { legend: { labels: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 10 } } } },
};

// Equity Curve
function initEquityChart() {
  const ctx = $('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Equity',
        data: [],
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0,255,136,0.06)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 10 } },
          grid:  { color: '#1a1a1a' },
        },
      },
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => '$' + ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 }),
          },
        },
      },
    },
  });
}

function updateEquityChart(history) {
  if (!equityChart) return;
  const labels = history.map(h => h.t ? h.t.substring(11, 16) : '');
  const data   = history.map(h => h.e);
  equityChart.data.labels   = labels;
  equityChart.data.datasets[0].data = data;
  equityChart.update('none');
}

// Win/Loss Donut
function initWinLossChart() {
  const ctx = $('winLossChart').getContext('2d');
  winLossChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Wins', 'Losses'],
      datasets: [{
        data: [0, 0],
        backgroundColor: ['#00ff88', '#ff3355'],
        borderColor: '#111',
        borderWidth: 2,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 10 } } },
        tooltip: {
          callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed} trades` },
        },
      },
    },
  });
}

// R Distribution Bar
function initRDistChart() {
  const ctx = $('rDistChart').getContext('2d');
  rDistChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Trades',
        data: [],
        backgroundColor: [],
        borderRadius: 2,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: {
          ticks: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 9 } },
          grid:  { display: false },
        },
        y: {
          ticks: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 9 } },
          grid:  { color: '#1a1a1a' },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

// Session Performance Bar
function initSessionChart() {
  const ctx = $('sessionChart').getContext('2d');
  sessionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['London KZ', 'NY KZ'],
      datasets: [
        {
          label: 'Win Rate %',
          data: [0, 0],
          backgroundColor: 'rgba(0,255,136,0.6)',
          borderRadius: 2,
        },
        {
          label: 'Avg R',
          data: [0, 0],
          backgroundColor: 'rgba(0,170,255,0.6)',
          borderRadius: 2,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: {
          ticks: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 9 } },
          grid:  { display: false },
        },
        y: {
          ticks: { color: '#6a7280', font: { family: 'JetBrains Mono', size: 9 } },
          grid:  { color: '#1a1a1a' },
        },
      },
    },
  });
}

function updateMiniCharts(s) {
  if (!s) return;

  // Win/Loss
  if (winLossChart) {
    winLossChart.data.datasets[0].data = [s.wins || 0, s.losses || 0];
    winLossChart.update();
  }

  // R Distribution
  if (rDistChart && s.r_distribution) {
    const buckets = Object.keys(s.r_distribution).sort((a, b) => +a - +b);
    const colors  = buckets.map(k => +k >= 0 ? 'rgba(0,255,136,0.7)' : 'rgba(255,51,85,0.7)');
    rDistChart.data.labels = buckets.map(k => `${+k > 0 ? '+' : ''}${k}R`);
    rDistChart.data.datasets[0].data = buckets.map(k => s.r_distribution[k]);
    rDistChart.data.datasets[0].backgroundColor = colors;
    rDistChart.update();
  }

  // Session chart
  if (sessionChart && s.session_stats) {
    const sessKeys = ['LONDON_KILLZONE', 'NY_KILLZONE'];
    const labels   = ['London KZ', 'NY KZ'];
    const winRates = sessKeys.map(k => {
      const ss = s.session_stats[k];
      if (!ss || !ss.trades) return 0;
      return (ss.wins / ss.trades) * 100;
    });
    const avgRs = sessKeys.map(k => {
      const ss = s.session_stats[k];
      if (!ss || !ss.trades) return 0;
      return ss.total_r / ss.trades;
    });
    sessionChart.data.labels = labels;
    sessionChart.data.datasets[0].data = winRates;
    sessionChart.data.datasets[1].data = avgRs;
    sessionChart.update();
  }
}

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initEquityChart();
  initWinLossChart();
  initRDistChart();
  initSessionChart();
  connectWS();
});
