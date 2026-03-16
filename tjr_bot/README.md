# TJR Strategy — MetaTrader 5 Expert Advisor + Local Dashboard

A complete automated trading system implementing Tyler J. Riches' (TJR) ICT/SMC strategy:
**Liquidity Sweeps → Break of Structure → Fair Value Gap / Order Block → Entry**

---

## System Overview

| Component | Tech | Purpose |
|-----------|------|---------|
| `MT5/TJR_EA.mq5` | MQL5 | Executes the TJR strategy automatically |
| `dashboard/server.py` | FastAPI + WebSocket | Real-time data server |
| `dashboard/mt5_bridge.py` | Python + MT5 API | Reads live MT5 data |
| `dashboard/frontend/` | HTML / CSS / JS | Live trading dashboard at localhost:8000 |

---

## Requirements

### Software
- **MetaTrader 5** (any broker version)
- **Python 3.11+** — https://python.org/downloads
- Windows 10/11 (MT5 is Windows-native; dashboard runs on any OS)

### Broker
- ECN or RAW Spread account recommended (low latency, tight spreads)
- Broker must allow algorithmic trading (DLL imports may be required)
- Supported symbols: XAUUSD, NAS100/US100, EURUSD, GBPUSD, BTCUSD, etc.

---

## Installation

### Quick Install (Windows)
```
1. Unzip the tjr_bot folder
2. Double-click install.bat
3. Follow the on-screen instructions
```

### Manual Install
```bash
# Install Python dependencies
cd tjr_bot/dashboard
pip install -r requirements.txt

# Copy EA file manually
# Copy MT5/TJR_EA.mq5 to:
# %APPDATA%\MetaQuotes\Terminal\<ID>\MQL5\Experts\TJR_EA.mq5
```

---

## MT5 Expert Advisor Setup

### Step 1 — Compile the EA
1. Open MetaTrader 5
2. Press **F4** to open MetaEditor
3. In MetaEditor, navigate to: `Navigator → Expert Advisors → TJR_EA`
4. Press **F7** to compile
5. Verify: **0 errors, 0 warnings** (or only non-critical warnings)

### Step 2 — Attach EA to Chart
1. In MT5, open a new chart:
   - Primary: **XAUUSD M5** (Gold, 5-minute — highest win rate)
   - Optional: **NAS100 M5** or **EURUSD M5**
2. From the **Navigator** panel → **Expert Advisors** → drag `TJR_EA` onto the chart
3. In the EA settings dialog:
   - ✅ **Allow live trading**
   - ✅ **Allow DLL imports** (required for file writing)

### Step 3 — Configure EA Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `ServerUTC_Offset` | `0` | **CRITICAL** — Set to your broker's UTC offset. E.g., if broker server shows UTC+3, set to `3` |
| `RiskPct` | `0.5` | Risk per trade as % of balance |
| `PropFirmMode` | `false` | Set `true` for prop firm challenges (0.25% risk, 1 trade/day) |
| `MaxTradesPerDay` | `3` | Maximum trades per session day |
| `DailyLossCap_R` | `1.5` | Halt trading after this many R lost in a day |
| `MinRR` | `2.0` | Skip trade if R:R is below this |
| `SL_BufferPips` | `3.0` | Extra pips beyond sweep wick for stop loss |
| `Use_SMT` | `false` | Enable SMT divergence filter |
| `SMT_Symbol` | `""` | Correlated symbol (e.g., `XAGUSD` for Gold) |
| `TradeLondonKZ` | `true` | Trade London Kill Zone (2-5 AM EST) |
| `TradeNYKZ` | `true` | Trade NY Kill Zone (7-11 AM EST) — primary window |

### Step 4 — Finding Your Broker's UTC Offset
1. In MT5, look at the bottom bar — it shows **server time**
2. Check what UTC offset that corresponds to
3. Common examples:
   - Broker shows 15:00 when it's 12:00 UTC → offset = +3
   - Broker shows 12:00 when it's 12:00 UTC → offset = 0
   - Broker shows 10:00 when it's 12:00 UTC → offset = -2

---

## Dashboard Setup

### Start the Dashboard
```bash
cd tjr_bot/dashboard
python server.py
```
Then open: **http://localhost:8000**

### Alternative (with auto-reload for development)
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Top bar** | Live indicator, symbol, real-time EST clock |
| **Stat cards** | Balance, equity, daily P&L, win rate, profit factor |
| **Equity curve** | Live line chart updating every minute |
| **Bot status** | Current EA phase, session, HTF bias, sweep/BOS status |
| **8-Factor checklist** | Live checklist showing which of 8 conditions are met |
| **Open position** | Active trade P&L, entry/SL/TP levels, progress bar |
| **Win/Loss donut** | Overall win rate pie chart |
| **R Distribution** | Histogram of trade results by R-multiple |
| **Session performance** | London KZ vs NY KZ win rates and avg R |
| **Session levels** | Asia/London/NY/Previous Day highs and lows |
| **Risk monitor** | Daily loss used, trades today, consecutive losses |
| **Recent trades** | Last 15 closed trades with results |

---

## How the Strategy Works (TJR's 5 Tools)

### 1. Liquidity Sweeps
Price wicks **beyond** a key level (session high/low, PDH/PDL) then **closes back inside**.
- Bullish sweep: Low swept → price closes above → expect rally
- Bearish sweep: High swept → price closes below → expect drop

### 2. Break of Structure (BOS)
After a sweep, an impulse candle breaks a recent swing:
- Body > 50% of candle's total range
- Closes past the 79% Fibonacci level of its range

### 3. Fair Value Gap (FVG)
3-candle imbalance where candle 1 wick and candle 3 wick don't overlap candle 2 body.
- FVGs are for **retracements, not reversals** (TJR rule)
- Entry: wait for price to retrace back into the FVG

### 4. Order Block (OB)
The **last candle of opposite color before the BOS impulse**.
- Bullish OB: last bearish candle before a bull move
- Bearish OB: last bullish candle before a bear move

### 5. Equilibrium (EQ)
The 50% Fibonacci level of any range. Used as entry when no FVG is present.

### The Full Entry Chain
```
a) Price hits key level (PDH/PDL/Asia H-L/London H-L)
b) 5M: BOS candle (body > 50%, closes past 79% fib)
c) 5M: Wait for retrace into OB / FVG / EQ zone
d) 1M: Final BOS/iFVG confirmation
e) ENTER trade
f) SL: Beyond sweep wick + buffer
g) TP: Key levels in opposite direction (1R → 2R → 3R)
```

### Session Schedule (EST)
| Session | Time (EST) | Role |
|---------|-----------|------|
| Asia | 8 PM – 12 AM | Accumulation — mark the range |
| London Kill Zone | 2 AM – 5 AM | Manipulation — sweeps Asia levels |
| NY Kill Zone | 7 AM – 11 AM | Distribution — PRIMARY trading window |
| Hard cutoff | 11 AM | No new trades after this |

---

## Risk Management Rules

| Rule | Value |
|------|-------|
| Default risk | 0.5% per trade |
| Prop firm mode | 0.25% per trade, 1 trade/day |
| Daily loss cap | 1.5R or 2% of balance (whichever hits first) |
| Max trades/day | 3 |
| Consecutive losses | Stop after 2 |
| Partial TP | 50% closed at 1R, runner to 3R |
| Break-even | Moved to entry only after 1R + new M5 BOS in direction |

---

## Data Files (written by EA)

| File | Location | Contents |
|------|----------|---------|
| `tjr_live_data.json` | MT5 Common Files folder | Live EA state, updated every tick |
| `tjr_trade_history.csv` | MT5 Common Files folder | All trade records |

The dashboard reads these files via the MT5 Python API.

---

## Frequently Asked Questions

### "EA compiled with errors"
- Check that you're using MetaTrader 5 (not MT4 — MQL5 ≠ MQL4)
- Ensure MetaEditor is updated (Help → Check for Updates)
- The `#include <Trade\Trade.mqh>` files come with MT5 by default

### "EA is running but not taking trades"
- Check the EA inputs — `ServerUTC_Offset` must be set correctly
- Verify the current time is within London KZ (2-5 AM EST) or NY KZ (7-11 AM EST)
- Check the on-chart info panel (top-right) — it shows current phase and checklist score
- Minimum checklist score of 5/8 required before entry fires
- If `PropFirmMode=true`, only 1 trade per day is allowed

### "Dashboard shows MT5 DISCONNECTED"
- MT5 must be running for the Python API to connect
- Try: `pip install MetaTrader5 --upgrade`
- On Windows, MT5 must be started before the dashboard
- The dashboard will retry every 5 seconds automatically
- In demo mode, the dashboard still works with the EA's JSON file

### "Wrong session times / trades not firing"
- The most common issue is incorrect `ServerUTC_Offset`
- Formula: `server_hour = EST_hour + 5 + ServerUTC_Offset`
- Test: At 9 AM EST (NY KZ), what hour does MT5 server show?
  - If server shows 14:00, then 14 = 9+5+0 → offset = 0
  - If server shows 17:00, then 17 = 9+5+3 → offset = 3

### "Lots are 0 / trade not placed"
- SL might be too tight (< 3 pips) — increase `SL_BufferPips`
- Balance may be too low for minimum lot size
- Check broker's minimum lot and pip value

### "JSON file not found by dashboard"
- Enable "Allow DLL imports" in MT5 EA settings
- The EA writes to the **Common** files folder — check `FILE_COMMON` flag is working
- Manually check: `%APPDATA%\MetaQuotes\Terminal\Common\Files\tjr_live_data.json`

---

## Recommended Broker Settings
- Account type: **ECN or RAW Spread** (not market maker)
- Leverage: 100:1 for forex, 200:1 for indices
- Minimum deposit: $1,000+ (for proper lot sizing at 0.5% risk)
- Execution: Under 50ms latency recommended
- Symbols must support: XAUUSD, NAS100 (or US100), EURUSD

---

## Architecture Notes

```
MT5 Terminal (Windows)
  └── TJR_EA.mq5
        ├── Reads: H4, H1, M5, M1 price data
        ├── Writes: tjr_live_data.json (every tick)
        └── Writes: tjr_trade_history.csv (each trade)

Python Dashboard (localhost:8000)
  └── server.py (FastAPI)
        ├── mt5_bridge.py
        │     ├── Reads: tjr_live_data.json (EA output)
        │     └── Reads: MT5 Python API (account, positions)
        ├── WebSocket → broadcasts every 500ms to browser
        └── Serves: frontend/ (HTML + CSS + JS)

Browser (http://localhost:8000)
  └── app.js
        ├── WebSocket client → receives data every 500ms
        ├── Chart.js → equity curve, win/loss, R-dist, session charts
        └── Updates all panels in real-time
```

---

## Disclaimer

This software is for educational purposes. Trading involves significant financial risk.
Past performance does not guarantee future results. Always test on a demo account first.
The TJR strategy concepts are based on publicly available educational content.
