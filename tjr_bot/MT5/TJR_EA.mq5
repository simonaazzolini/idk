//+------------------------------------------------------------------+
//|                                                     TJR_EA.mq5  |
//|                         TJR Strategy Expert Advisor v3.0        |
//|         ICT/SMC: Liquidity Sweeps + BOS + FVG + OB + EQ         |
//+------------------------------------------------------------------+
#property copyright "TJR Strategy EA"
#property link      ""
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

//--- Input Parameters
// SESSION TIMES (EST — adjust ServerUTC_Offset for broker server time)
input int    AsiaStart_H       = 20;    // Asia session open (EST hour)
input int    AsiaEnd_H         = 0;     // Asia session close (EST hour)
input int    LondonStart_H     = 2;     // London KZ start (EST)
input int    LondonEnd_H       = 5;     // London KZ end (EST)
input int    NYStart_H         = 7;     // NY KZ start (EST)
input int    NYEnd_H           = 10;    // NY KZ end (EST)
input int    HardCutoff_H      = 11;    // No trades after this hour (EST)
input int    ServerUTC_Offset  = 0;     // Broker server UTC offset (e.g. 2 for UTC+2)

// RISK
input double RiskPct           = 0.5;   // % of balance per trade
input double MinRR             = 2.0;   // Minimum reward:risk ratio
input double DailyLossCap_R    = 1.5;   // Daily loss in R → halt
input double DailyLossCap_Pct  = 2.0;   // Daily loss % → halt
input int    MaxTradesPerDay   = 3;     // Max trades per day
input int    ConsecLossLimit   = 2;     // Halt after N consecutive losses
input bool   PropFirmMode      = false; // Ultra-conservative: 0.25% risk, 1 trade/day
input bool   AutoBreakEven     = true;  // Auto move SL to break even
input bool   UsePartialTP      = true;  // Partial TP at 1R, runner to 3R

// STRATEGY
input int    SwingLookback     = 10;    // Bars to detect swing highs/lows
input int    BOS_MinBodyPct    = 50;    // BOS candle body % of range
input double Fib79_Min         = 0.79;  // 79% fib threshold for BOS confirmation
input int    FVG_MinPips       = 2;     // Minimum FVG gap in pips
input double SL_BufferPips     = 3.0;   // Buffer beyond sweep wick for SL
input double PartialTP_R       = 1.0;   // Take 50% at X R
input double FinalTP_R         = 3.0;   // Final TP at X R
input bool   Use_SMT           = false; // Enable SMT divergence filter
input string SMT_Symbol        = "";    // Correlated symbol for SMT

// SESSION FILTERS
input bool   TradeLondonKZ     = true;  // Trade London kill zone
input bool   TradeNYKZ         = true;  // Trade NY kill zone (primary)
input bool   SkipNewsBuffer    = true;  // Skip ±15 min around high-impact news
input int    NewsBufferMins    = 15;    // News buffer in minutes

// DISPLAY
input bool   DrawSessionLevels = true;  // Draw session H/L lines on chart
input bool   DrawFVGBoxes      = true;  // Draw FVG zones
input bool   DrawOBBoxes       = true;  // Draw OB zones
input bool   ShowInfoPanel     = true;  // Show on-chart info panel
input bool   AlertOnSignal     = true;  // Alert when all 8 factors align
input bool   AlertOnSweep      = true;  // Alert on sweep detection

//--- Enums
enum EA_PHASE {
   PHASE_IDLE,
   PHASE_AWAIT_SWEEP,
   PHASE_SWEEP_CONFIRMED,
   PHASE_AWAIT_M5_BOS,
   PHASE_AWAIT_M5_RETRACE,
   PHASE_AWAIT_M1_ENTRY,
   PHASE_TRADE_OPEN,
   PHASE_DAY_HALTED
};

enum HTF_BIAS {
   BIAS_BULLISH,
   BIAS_BEARISH,
   BIAS_NEUTRAL
};

enum SESSION_TYPE {
   SESSION_NONE,
   SESSION_ASIA,
   SESSION_LONDON_KZ,
   SESSION_NY_KZ
};

//--- Global State
EA_PHASE    g_Phase             = PHASE_IDLE;
HTF_BIAS    g_HTFBias           = BIAS_NEUTRAL;
SESSION_TYPE g_CurrentSession   = SESSION_NONE;

// Session levels
double g_AsiaHigh      = 0, g_AsiaLow      = 0;
double g_LondonHigh    = 0, g_LondonLow    = 0;
double g_PrevDayHigh   = 0, g_PrevDayLow   = 0;
double g_H4SwingHigh   = 0, g_H4SwingLow   = 0;
double g_H1SwingHigh   = 0, g_H1SwingLow   = 0;

// Sweep info
bool   g_SweepDetected    = false;
bool   g_SweepBullish     = false;  // true = low swept → expect rally
double g_SweepLevel       = 0;
datetime g_SweepTime      = 0;

// BOS info
bool   g_M5_BOS_Confirmed = false;
bool   g_BOSBullish       = false;
double g_BOSLevel         = 0;
int    g_BOSBarIndex      = 0;
int    g_BOSTimeout       = 20;  // candles
int    g_BOSWaitCount     = 0;

// FVG zone
bool   g_FVGActive        = false;
double g_FVGHigh          = 0;
double g_FVGLow           = 0;
bool   g_FVGBullish       = false;

// OB zone
bool   g_OBActive         = false;
double g_OBHigh           = 0;
double g_OBLow            = 0;
bool   g_OBBullish        = false;

// EQ level
double g_EQLevel          = 0;

// Checklist
int    g_ChecklistScore   = 0;
bool   g_Check[8];  // 8-factor checklist

// Trade state
bool   g_TradeActive      = false;
ulong  g_TicketTP1        = 0;
ulong  g_TicketTP2        = 0;
double g_EntryPrice       = 0;
double g_StopLoss         = 0;
double g_TP1              = 0;
double g_TP2              = 0;
double g_TP3              = 0;
bool   g_PartialClosed    = false;
bool   g_BESet            = false;
bool   g_TradeIsBuy       = true;
double g_InitialRisk_USD  = 0;
double g_InitialSL_Pips   = 0;

// Daily tracking
int    g_TradesToday      = 0;
int    g_WinsToday        = 0;
int    g_LossesToday      = 0;
double g_DailyPnL_Pct    = 0;
double g_DailyPnL_R      = 0;
int    g_ConsecLosses     = 0;
datetime g_LastResetDay   = 0;

// Candle tracking
datetime g_LastM5Candle   = 0;
datetime g_LastH4Candle   = 0;
datetime g_LastH1Candle   = 0;

// Objects
CTrade  g_Trade;
CPositionInfo g_Position;

// ATR handle for FVG filter
int g_ATRHandle = INVALID_HANDLE;

// Pip size
double g_PipSize  = 0.0001;
double g_TickSize = 0.00001;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit() {
   // Determine pip size for this symbol
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   g_TickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(digits == 3 || digits == 5) {
      g_PipSize = g_TickSize * 10;
   } else {
      g_PipSize = g_TickSize;
   }

   // ATR indicator for FVG size filter
   g_ATRHandle = iATR(_Symbol, PERIOD_M5, 14);
   if(g_ATRHandle == INVALID_HANDLE) {
      Print("TJR EA: Failed to create ATR indicator handle");
      return INIT_FAILED;
   }

   // Trade object setup
   g_Trade.SetExpertMagicNumber(202603);
   g_Trade.SetDeviationInPoints(20);
   g_Trade.SetTypeFilling(ORDER_FILLING_FOK);

   // Reset daily tracking
   ResetDailyTracking();

   // Initial HTF bias assessment
   UpdateHTFBias();

   // Draw initial session levels
   UpdateSessionLevels();

   Print("TJR EA v3.0 initialized. Symbol: ", _Symbol,
         " | Digits: ", digits,
         " | PipSize: ", g_PipSize,
         " | Risk: ", (PropFirmMode ? 0.25 : RiskPct), "%");

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
   DeleteAllTJRObjects();
   if(g_ATRHandle != INVALID_HANDLE)
      IndicatorRelease(g_ATRHandle);
}

//+------------------------------------------------------------------+
//| Expert tick handler                                              |
//+------------------------------------------------------------------+
void OnTick() {
   // Check daily reset
   CheckDailyReset();

   // Update session levels on new H4/H1 candle
   datetime curH4 = iTime(_Symbol, PERIOD_H4, 0);
   datetime curH1 = iTime(_Symbol, PERIOD_H1, 0);
   if(curH4 != g_LastH4Candle) {
      g_LastH4Candle = curH4;
      UpdateH4SwingLevels();
      UpdateHTFBias();
   }
   if(curH1 != g_LastH1Candle) {
      g_LastH1Candle = curH1;
      UpdateSessionLevels();
   }

   // Determine current session
   g_CurrentSession = GetCurrentSession();

   // Update checklist
   UpdateChecklist();

   // Main state machine — only process on new M5 candle for most logic
   datetime curM5 = iTime(_Symbol, PERIOD_M5, 0);
   bool newM5Candle = (curM5 != g_LastM5Candle);
   if(newM5Candle) g_LastM5Candle = curM5;

   switch(g_Phase) {
      case PHASE_IDLE:
         RunPhaseIdle(newM5Candle);
         break;
      case PHASE_AWAIT_SWEEP:
         RunPhaseAwaitSweep(newM5Candle);
         break;
      case PHASE_SWEEP_CONFIRMED:
         RunPhaseSweepConfirmed(newM5Candle);
         break;
      case PHASE_AWAIT_M5_BOS:
         RunPhaseAwaitM5BOS(newM5Candle);
         break;
      case PHASE_AWAIT_M5_RETRACE:
         RunPhaseAwaitM5Retrace();
         break;
      case PHASE_AWAIT_M1_ENTRY:
         RunPhaseAwaitM1Entry();
         break;
      case PHASE_TRADE_OPEN:
         RunPhaseTradeOpen();
         break;
      case PHASE_DAY_HALTED:
         // Nothing — wait for daily reset
         break;
   }

   // Draw visuals and write JSON
   if(ShowInfoPanel) DrawInfoPanel();
   if(DrawSessionLevels) DrawSessionLines();
   WriteJSONFile();
}

//+------------------------------------------------------------------+
//| Timer — write JSON once per second                               |
//+------------------------------------------------------------------+
void OnTimer() {
   WriteJSONFile();
}

//+------------------------------------------------------------------+
//| PHASE IDLE — watch for session and key level approach            |
//+------------------------------------------------------------------+
void RunPhaseIdle(bool newM5Candle) {
   if(!IsSessionActive()) return;
   if(IsHardCutoffPassed()) return;
   if(g_Phase == PHASE_DAY_HALTED) return;
   if(g_TradesToday >= (PropFirmMode ? 1 : MaxTradesPerDay)) return;

   g_Phase = PHASE_AWAIT_SWEEP;
}

//+------------------------------------------------------------------+
//| PHASE AWAIT SWEEP — monitor for liquidity sweep                  |
//+------------------------------------------------------------------+
void RunPhaseAwaitSweep(bool newM5Candle) {
   if(!IsSessionActive() || IsHardCutoffPassed()) {
      g_Phase = PHASE_IDLE;
      return;
   }
   if(!newM5Candle) return;

   // Check sweep on each key level
   if(CheckSweepOnLevel(g_PrevDayHigh, false) ||
      CheckSweepOnLevel(g_PrevDayLow,  true)  ||
      CheckSweepOnLevel(g_AsiaHigh,    false)  ||
      CheckSweepOnLevel(g_AsiaLow,     true)   ||
      CheckSweepOnLevel(g_LondonHigh,  false)  ||
      CheckSweepOnLevel(g_LondonLow,   true)   ||
      CheckSweepOnLevel(g_H4SwingHigh, false)  ||
      CheckSweepOnLevel(g_H4SwingLow,  true)   ||
      CheckSweepOnLevel(g_H1SwingHigh, false)  ||
      CheckSweepOnLevel(g_H1SwingLow,  true)) {
      // Sweep confirmed by CheckSweepOnLevel
   }
}

//+------------------------------------------------------------------+
//| Check if a sweep occurred on a given level                       |
//| isBullish: true = low was swept (low below level, closed above)  |
//+------------------------------------------------------------------+
bool CheckSweepOnLevel(double level, bool expectBullish) {
   if(level == 0) return false;
   double lastLow   = iLow(_Symbol,  PERIOD_M5, 1);
   double lastHigh  = iHigh(_Symbol, PERIOD_M5, 1);
   double lastClose = iClose(_Symbol, PERIOD_M5, 1);
   double lastOpen  = iOpen(_Symbol,  PERIOD_M5, 1);

   bool swept = false;
   if(expectBullish) {
      // Bullish sweep: wick below level, candle closed ABOVE level
      swept = (lastLow < level) && (lastClose > level);
   } else {
      // Bearish sweep: wick above level, candle closed BELOW level
      swept = (lastHigh > level) && (lastClose < level);
   }

   if(swept) {
      g_SweepDetected  = true;
      g_SweepBullish   = expectBullish;
      g_SweepLevel     = level;
      g_SweepTime      = iTime(_Symbol, PERIOD_M5, 1);

      // Draw sweep arrow
      string lbl = "TJR_Sweep_" + TimeToString(g_SweepTime);
      if(!ObjectFind(0, lbl)) {
         ObjectCreate(0, lbl, OBJ_ARROW, 0, g_SweepTime,
                      expectBullish ? lastLow - g_PipSize * 5 : lastHigh + g_PipSize * 5);
         ObjectSetInteger(0, lbl, OBJPROP_ARROWCODE, expectBullish ? 233 : 234);
         ObjectSetInteger(0, lbl, OBJPROP_COLOR, expectBullish ? clrLime : clrRed);
         ObjectSetInteger(0, lbl, OBJPROP_WIDTH, 2);
      }
      // Label
      string txtLbl = "TJR_SweepTxt_" + TimeToString(g_SweepTime);
      ObjectCreate(0, txtLbl, OBJ_TEXT, 0, g_SweepTime,
                   expectBullish ? lastLow - g_PipSize * 10 : lastHigh + g_PipSize * 10);
      ObjectSetString(0, txtLbl, OBJPROP_TEXT, expectBullish ? "SWEEP ↑" : "SWEEP ↓");
      ObjectSetInteger(0, txtLbl, OBJPROP_COLOR, expectBullish ? clrLime : clrRed);
      ObjectSetInteger(0, txtLbl, OBJPROP_FONTSIZE, 9);

      if(AlertOnSweep)
         Alert("TJR EA: Sweep detected on ", _Symbol,
               " | Level: ", DoubleToString(level, _Digits),
               " | Direction: ", expectBullish ? "BULLISH" : "BEARISH");

      g_Phase = PHASE_SWEEP_CONFIRMED;
      g_BOSWaitCount = 0;
      g_M5_BOS_Confirmed = false;
      g_Check[2] = true;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| PHASE SWEEP CONFIRMED — drop to M5 for BOS detection            |
//+------------------------------------------------------------------+
void RunPhaseSweepConfirmed(bool newM5Candle) {
   // Transition immediately to BOS waiting
   g_Phase = PHASE_AWAIT_M5_BOS;
}

//+------------------------------------------------------------------+
//| PHASE AWAIT M5 BOS — watch for impulse candle after sweep        |
//+------------------------------------------------------------------+
void RunPhaseAwaitM5BOS(bool newM5Candle) {
   if(!newM5Candle) return;

   g_BOSWaitCount++;
   if(g_BOSWaitCount > g_BOSTimeout) {
      // Timeout — reset
      Print("TJR EA: BOS timeout, resetting to IDLE");
      ResetSetup();
      return;
   }

   // Check M5 candle 1 (just closed) for BOS
   if(DetectBOS_M5(1)) {
      g_M5_BOS_Confirmed = true;
      g_Check[3] = true;
      // Scan for FVG around the BOS bar
      DetectFVG_M5(1);
      // Detect order block
      DetectOrderBlock_M5(1);
      // Calculate equilibrium
      if(g_SweepLevel > 0) {
         double rangeHigh = g_BOSBullish ? iHigh(_Symbol, PERIOD_M5, 1) : g_SweepLevel;
         double rangeLow  = g_BOSBullish ? g_SweepLevel : iLow(_Symbol, PERIOD_M5, 1);
         g_EQLevel = rangeLow + (rangeHigh - rangeLow) * 0.5;
      }

      if(AlertOnSignal && g_ChecklistScore >= 5)
         Alert("TJR EA: M5 BOS confirmed on ", _Symbol,
               " | Checklist: ", g_ChecklistScore, "/8",
               " | Direction: ", g_BOSBullish ? "BULLISH" : "BEARISH");

      g_Phase = PHASE_AWAIT_M5_RETRACE;
   }
}

//+------------------------------------------------------------------+
//| Detect BOS on M5 — returns true if valid BOS candle              |
//+------------------------------------------------------------------+
bool DetectBOS_M5(int shift) {
   double o = iOpen(_Symbol,  PERIOD_M5, shift);
   double h = iHigh(_Symbol,  PERIOD_M5, shift);
   double l = iLow(_Symbol,   PERIOD_M5, shift);
   double c = iClose(_Symbol, PERIOD_M5, shift);
   double range = h - l;
   if(range < g_PipSize * 2) return false;

   double bodySize = MathAbs(c - o);
   double bodyPct  = bodySize / range;
   if(bodyPct * 100 < BOS_MinBodyPct) return false;

   bool bullBOS = (c > o); // bull candle
   bool bearBOS = (c < o); // bear candle

   // After bullish sweep, we want a bull BOS moving UP
   // After bearish sweep, we want a bear BOS moving DOWN
   if(g_SweepBullish && !bullBOS) return false;
   if(!g_SweepBullish && !bearBOS) return false;

   // 79% Fibonacci extension check
   double fib79;
   if(bullBOS) {
      fib79 = l + range * Fib79_Min;
      if(c < fib79) return false;
   } else {
      fib79 = h - range * Fib79_Min;
      if(c > fib79) return false;
   }

   // Must break a prior swing
   double priorSwingH = 0, priorSwingL = 999999;
   for(int i = shift + 1; i <= shift + SwingLookback && i < 500; i++) {
      double ph = iHigh(_Symbol, PERIOD_M5, i);
      double pl = iLow(_Symbol,  PERIOD_M5, i);
      if(ph > priorSwingH) priorSwingH = ph;
      if(pl < priorSwingL) priorSwingL = pl;
   }
   if(bullBOS && c <= priorSwingH) return false;
   if(bearBOS && c >= priorSwingL) return false;

   g_BOSBullish   = bullBOS;
   g_BOSLevel     = bullBOS ? h : l;
   g_BOSBarIndex  = shift;
   g_Check[7] = true; // 79% fib confirmed
   return true;
}

//+------------------------------------------------------------------+
//| Detect Fair Value Gap on M5 around BOS bar                       |
//+------------------------------------------------------------------+
void DetectFVG_M5(int bosShift) {
   // Scan 3-candle windows near the BOS
   for(int i = bosShift; i <= bosShift + 5 && i < 500; i++) {
      double h1 = iHigh(_Symbol,  PERIOD_M5, i + 2);
      double l1 = iLow(_Symbol,   PERIOD_M5, i + 2);
      double h2 = iHigh(_Symbol,  PERIOD_M5, i + 1);
      double l2 = iLow(_Symbol,   PERIOD_M5, i + 1);
      double h3 = iHigh(_Symbol,  PERIOD_M5, i);
      double l3 = iLow(_Symbol,   PERIOD_M5, i);

      // ATR filter
      double atrBuf[];
      ArraySetAsSeries(atrBuf, true);
      double atrVal = 0;
      if(CopyBuffer(g_ATRHandle, 0, i, 1, atrBuf) > 0)
         atrVal = atrBuf[0];

      // Bullish FVG: candle1_high < candle3_low
      if(g_BOSBullish && l3 > h1) {
         double gap = l3 - h1;
         if(gap >= FVG_MinPips * g_PipSize &&
            (atrVal == 0 || gap >= atrVal * 0.3)) {
            g_FVGActive  = true;
            g_FVGBullish = true;
            g_FVGLow     = h1;
            g_FVGHigh    = l3;
            g_Check[4]   = true;
            if(DrawFVGBoxes) DrawFVGBox();
            break;
         }
      }
      // Bearish FVG: candle1_low > candle3_high
      if(!g_BOSBullish && h3 < l1) {
         double gap = l1 - h3;
         if(gap >= FVG_MinPips * g_PipSize &&
            (atrVal == 0 || gap >= atrVal * 0.3)) {
            g_FVGActive  = true;
            g_FVGBullish = false;
            g_FVGHigh    = l1;
            g_FVGLow     = h3;
            g_Check[4]   = true;
            if(DrawFVGBoxes) DrawFVGBox();
            break;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Detect Order Block on M5 — first candle prior to expansionary    |
//+------------------------------------------------------------------+
void DetectOrderBlock_M5(int bosShift) {
   // OB = last candle of opposite color BEFORE the BOS impulse
   for(int i = bosShift + 1; i <= bosShift + 10 && i < 500; i++) {
      double o = iOpen(_Symbol,  PERIOD_M5, i);
      double h = iHigh(_Symbol,  PERIOD_M5, i);
      double l = iLow(_Symbol,   PERIOD_M5, i);
      double c = iClose(_Symbol, PERIOD_M5, i);

      bool isBearCandle = (c < o);
      bool isBullCandle = (c > o);

      // For bullish move, OB is last bearish candle before impulse
      if(g_BOSBullish && isBearCandle) {
         g_OBActive  = true;
         g_OBBullish = true;
         g_OBHigh    = h;
         g_OBLow     = l;
         g_Check[5]  = true;
         if(DrawOBBoxes) DrawOBBox(i);
         break;
      }
      // For bearish move, OB is last bullish candle before impulse
      if(!g_BOSBullish && isBullCandle) {
         g_OBActive  = true;
         g_OBBullish = false;
         g_OBHigh    = h;
         g_OBLow     = l;
         g_Check[5]  = true;
         if(DrawOBBoxes) DrawOBBox(i);
         break;
      }
   }
}

//+------------------------------------------------------------------+
//| PHASE AWAIT M5 RETRACE — wait for price to enter FVG/OB zone     |
//+------------------------------------------------------------------+
void RunPhaseAwaitM5Retrace() {
   if(IsHardCutoffPassed()) { ResetSetup(); return; }

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price = g_BOSBullish ? ask : bid;

   // Check if price re-entered the entry zone (FVG, OB, or EQ)
   bool inFVG = g_FVGActive && (price >= g_FVGLow) && (price <= g_FVGHigh);
   bool inOB  = g_OBActive  && (price >= g_OBLow)  && (price <= g_OBHigh);
   bool inEQ  = (g_EQLevel > 0) &&
                MathAbs(price - g_EQLevel) < g_PipSize * 5;

   if(inFVG || inOB || inEQ) {
      g_Phase = PHASE_AWAIT_M1_ENTRY;
   }

   // If price blows through the zone without entering, still valid until invalidation
   // Invalidation: price closes BEYOND the sweep level
   double lastClose = iClose(_Symbol, PERIOD_M5, 1);
   if(g_SweepBullish && lastClose < g_SweepLevel - g_PipSize * 5) {
      Print("TJR EA: Setup invalidated — price accepted beyond sweep low");
      ResetSetup();
   }
   if(!g_SweepBullish && lastClose > g_SweepLevel + g_PipSize * 5) {
      Print("TJR EA: Setup invalidated — price accepted beyond sweep high");
      ResetSetup();
   }
}

//+------------------------------------------------------------------+
//| PHASE AWAIT M1 ENTRY — final M1 confirmation before entry        |
//+------------------------------------------------------------------+
void RunPhaseAwaitM1Entry() {
   if(IsHardCutoffPassed()) { ResetSetup(); return; }

   // Check M1 BOS for final confirmation
   if(DetectM1BOS()) {
      // All conditions met — fire the trade if checklist score is sufficient
      UpdateChecklist();
      if(g_ChecklistScore >= 5) {  // Minimum 5/8 required
         PlaceTrade();
      } else {
         Print("TJR EA: Checklist insufficient (", g_ChecklistScore, "/8) — skipping");
         ResetSetup();
      }
   }

   // Check if price exits the zone
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price = g_BOSBullish ? ask : bid;
   bool inFVG = g_FVGActive && (price >= g_FVGLow - g_PipSize) && (price <= g_FVGHigh + g_PipSize);
   bool inOB  = g_OBActive  && (price >= g_OBLow  - g_PipSize) && (price <= g_OBHigh  + g_PipSize);
   bool inEQ  = (g_EQLevel > 0) && MathAbs(price - g_EQLevel) < g_PipSize * 8;

   if(!inFVG && !inOB && !inEQ) {
      // Price exited zone — go back to retrace phase
      g_Phase = PHASE_AWAIT_M5_RETRACE;
   }
}

//+------------------------------------------------------------------+
//| Detect BOS on M1 for final entry confirmation                    |
//+------------------------------------------------------------------+
bool DetectM1BOS() {
   double o = iOpen(_Symbol,  PERIOD_M1, 1);
   double h = iHigh(_Symbol,  PERIOD_M1, 1);
   double l = iLow(_Symbol,   PERIOD_M1, 1);
   double c = iClose(_Symbol, PERIOD_M1, 1);
   double range = h - l;
   if(range < g_PipSize) return false;

   double bodySize = MathAbs(c - o);
   if(bodySize / range < 0.4) return false; // slightly relaxed for M1

   if(g_BOSBullish && c > o) {
      // Bullish M1 candle in bullish setup
      double fib79 = l + range * 0.70;
      return c >= fib79;
   }
   if(!g_BOSBullish && c < o) {
      // Bearish M1 candle in bearish setup
      double fib79 = h - range * 0.70;
      return c <= fib79;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Place trade with full risk management                            |
//+------------------------------------------------------------------+
void PlaceTrade() {
   if(g_TradeActive) return;
   if(g_TradesToday >= (PropFirmMode ? 1 : MaxTradesPerDay)) return;

   double actualRisk = PropFirmMode ? 0.25 : RiskPct;
   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double ask        = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid        = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double entryPrice, slPrice, tp1Price, tp2Price, tp3Price;
   g_TradeIsBuy = g_BOSBullish;

   if(g_TradeIsBuy) {
      entryPrice = ask;
      // SL: below sweep wick + buffer
      slPrice  = g_SweepLevel - SL_BufferPips * g_PipSize;
      // Use OB bottom if tighter
      if(g_OBActive && g_OBLow - SL_BufferPips * g_PipSize > slPrice)
         slPrice = g_OBLow - SL_BufferPips * g_PipSize;
   } else {
      entryPrice = bid;
      // SL: above sweep wick + buffer
      slPrice  = g_SweepLevel + SL_BufferPips * g_PipSize;
      // Use OB top if tighter
      if(g_OBActive && g_OBHigh + SL_BufferPips * g_PipSize < slPrice)
         slPrice = g_OBHigh + SL_BufferPips * g_PipSize;
   }

   double slPips = MathAbs(entryPrice - slPrice) / g_PipSize;
   if(slPips < 3) { Print("TJR EA: SL too tight, skipping"); return; }

   // RR check
   double rr = (g_TradeIsBuy)
      ? (MathAbs(g_LondonHigh > 0 ? g_LondonHigh : g_H4SwingHigh) - entryPrice) / (entryPrice - slPrice)
      : (entryPrice - MathAbs(g_LondonLow > 0 ? g_LondonLow : g_H4SwingLow)) / (slPrice - entryPrice);
   // Calculate TPs based on R
   double riskPips = MathAbs(entryPrice - slPrice);
   tp1Price = g_TradeIsBuy ? entryPrice + riskPips * PartialTP_R  : entryPrice - riskPips * PartialTP_R;
   tp2Price = g_TradeIsBuy ? entryPrice + riskPips * 2.0           : entryPrice - riskPips * 2.0;
   tp3Price = g_TradeIsBuy ? entryPrice + riskPips * FinalTP_R    : entryPrice - riskPips * FinalTP_R;

   // Lot sizing
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double pipValue  = tickValue * (g_PipSize / g_TickSize);
   double riskAmount = balance * (actualRisk / 100.0);
   double lots = riskAmount / (slPips * pipValue);
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lots = MathFloor(lots / stepLot) * stepLot;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   if(lots <= 0) { Print("TJR EA: Lot size zero, skipping"); return; }

   double tp1Lots = NormalizeDouble(lots * 0.5, 2);
   double tp2Lots = NormalizeDouble(lots * 0.3, 2);
   double tp3Lots = NormalizeDouble(lots - tp1Lots - tp2Lots, 2);
   if(tp1Lots < minLot) tp1Lots = minLot;

   string comment = StringFormat("TJR|%s|%s|%d/8",
                                 g_TradeIsBuy ? "BUY" : "SELL",
                                 GetSessionName(g_CurrentSession),
                                 g_ChecklistScore);

   bool result = false;
   if(UsePartialTP) {
      result = g_Trade.PositionOpen(_Symbol,
                                    g_TradeIsBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL,
                                    lots, entryPrice, slPrice, tp3Price, comment);
   } else {
      result = g_Trade.PositionOpen(_Symbol,
                                    g_TradeIsBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL,
                                    lots, entryPrice, slPrice, tp3Price, comment);
   }

   if(result) {
      g_TradeActive   = true;
      g_EntryPrice    = entryPrice;
      g_StopLoss      = slPrice;
      g_TP1           = tp1Price;
      g_TP2           = tp2Price;
      g_TP3           = tp3Price;
      g_PartialClosed = false;
      g_BESet         = false;
      g_InitialRisk_USD  = riskAmount;
      g_InitialSL_Pips   = slPips;
      g_TradesToday++;
      g_Phase         = PHASE_TRADE_OPEN;

      Print("TJR EA: Trade placed | ", comment,
            " | Entry: ", DoubleToString(entryPrice, _Digits),
            " | SL: ", DoubleToString(slPrice, _Digits),
            " | Lots: ", DoubleToString(lots, 2),
            " | Checklist: ", g_ChecklistScore, "/8");

      WriteTradeHistoryCSV(entryPrice, slPrice, tp1Price, tp3Price, "OPEN");
   } else {
      Print("TJR EA: Trade failed! Error: ", GetLastError(),
            " | ", g_Trade.ResultComment());
   }
}

//+------------------------------------------------------------------+
//| PHASE TRADE OPEN — manage active position                        |
//+------------------------------------------------------------------+
void RunPhaseTradeOpen() {
   if(!PositionSelect(_Symbol)) {
      // Position no longer open — check result
      HandleTradeClosed();
      return;
   }

   double currentPrice = g_TradeIsBuy
      ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
      : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double riskPips = g_InitialSL_Pips;
   double movedPips = g_TradeIsBuy
      ? (currentPrice - g_EntryPrice) / g_PipSize
      : (g_EntryPrice - currentPrice) / g_PipSize;
   double currentR = movedPips / riskPips;

   // Partial TP at 1R
   if(UsePartialTP && !g_PartialClosed && currentR >= PartialTP_R) {
      // Modify position — close half
      double posLots = PositionGetDouble(POSITION_VOLUME);
      double closeLots = NormalizeDouble(posLots * 0.5, 2);
      double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      if(closeLots >= minLot) {
         if(g_Trade.PositionClosePartial(_Symbol, closeLots)) {
            g_PartialClosed = true;
            // Move TP2 target
            Print("TJR EA: Partial TP taken at 1R | R: ", DoubleToString(currentR, 2));
         }
      }
   }

   // Break-even: only after 1R AND new M5 BOS in direction
   if(AutoBreakEven && !g_BESet && currentR >= 1.0) {
      if(DetectBOS_M5(1) && g_BOSBullish == g_TradeIsBuy) {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         double newSL = g_EntryPrice + (g_TradeIsBuy ? g_PipSize : -g_PipSize);
         if(g_Trade.PositionModify(ticket, newSL, g_TP3)) {
            g_StopLoss = newSL;
            g_BESet    = true;
            Print("TJR EA: Break-even set after structure confirmation");
         }
      }
   }

   // Draw trade lines
   DrawTradeLevels();

   // Check daily loss cap
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double dailyPnLPct = (equity - balance) / balance * 100.0;
   if(dailyPnLPct < -DailyLossCap_Pct) {
      Print("TJR EA: Daily loss cap (", DailyLossCap_Pct, "%) hit — halting");
      g_Trade.PositionClose(_Symbol);
      g_Phase = PHASE_DAY_HALTED;
   }
}

//+------------------------------------------------------------------+
//| Handle trade close — log result                                  |
//+------------------------------------------------------------------+
void HandleTradeClosed() {
   // Find the last deal result
   HistorySelect(TimeCurrent() - 86400, TimeCurrent());
   double lastProfit = 0;
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--) {
      ulong ticket = HistoryDealGetTicket(i);
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT ||
         HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_INOUT) {
         lastProfit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         break;
      }
   }

   double resultR = g_InitialRisk_USD > 0 ? lastProfit / g_InitialRisk_USD : 0;
   if(lastProfit > 0) {
      g_WinsToday++;
      g_ConsecLosses = 0;
   } else {
      g_LossesToday++;
      g_ConsecLosses++;
      g_DailyPnL_R += resultR;
   }
   g_DailyPnL_R += resultR;

   WriteTradeHistoryCSV(g_EntryPrice, g_StopLoss, g_TP1, g_TP3,
                        StringFormat("CLOSED|R:%.2f|USD:%.2f", resultR, lastProfit));

   Print("TJR EA: Trade closed | R: ", DoubleToString(resultR, 2),
         " | USD: ", DoubleToString(lastProfit, 2),
         " | Consec losses: ", g_ConsecLosses);

   // Check consecutive loss limit
   if(g_ConsecLosses >= ConsecLossLimit) {
      Print("TJR EA: Consecutive loss limit hit — halting for the day");
      g_Phase = PHASE_DAY_HALTED;
   } else if(g_DailyPnL_R < -DailyLossCap_R) {
      Print("TJR EA: Daily R loss cap hit — halting");
      g_Phase = PHASE_DAY_HALTED;
   } else {
      g_TradeActive = false;
      ResetSetup();
   }
}

//+------------------------------------------------------------------+
//| UPDATE HTF BIAS based on H4 swing structure                      |
//+------------------------------------------------------------------+
void UpdateHTFBias() {
   // Scan last SwingLookback*2 H4 candles for HH/HL or LH/LL
   int lookback = SwingLookback * 2;
   double highs[], lows[];
   ArrayResize(highs, lookback);
   ArrayResize(lows,  lookback);

   for(int i = 0; i < lookback; i++) {
      highs[i] = iHigh(_Symbol, PERIOD_H4, i + 1);
      lows[i]  = iLow(_Symbol,  PERIOD_H4, i + 1);
   }

   // Simple structure: compare last two swing highs and lows
   double sh1 = 0, sh2 = 0, sl1 = 999999, sl2 = 999999;
   int swingCount = 0;
   for(int i = 1; i < lookback - 1; i++) {
      if(highs[i] > highs[i-1] && highs[i] > highs[i+1]) {
         if(swingCount == 0) sh1 = highs[i];
         else if(swingCount == 1) { sh2 = highs[i]; break; }
         swingCount++;
      }
   }
   swingCount = 0;
   for(int i = 1; i < lookback - 1; i++) {
      if(lows[i] < lows[i-1] && lows[i] < lows[i+1]) {
         if(swingCount == 0) sl1 = lows[i];
         else if(swingCount == 1) { sl2 = lows[i]; break; }
         swingCount++;
      }
   }

   // HH + HL = bullish, LH + LL = bearish
   bool hhDetected = (sh1 > sh2 && sh2 > 0);
   bool hlDetected = (sl1 > sl2 && sl2 < 999999);
   bool lhDetected = (sh1 < sh2 && sh2 > 0);
   bool llDetected = (sl1 < sl2 && sl2 < 999999);

   if(hhDetected && hlDetected) g_HTFBias = BIAS_BULLISH;
   else if(lhDetected && llDetected) g_HTFBias = BIAS_BEARISH;
   else g_HTFBias = BIAS_NEUTRAL;

   // Update swing levels
   g_H4SwingHigh = sh1;
   g_H4SwingLow  = sl1;
   g_Check[1] = (g_HTFBias != BIAS_NEUTRAL);
}

//+------------------------------------------------------------------+
//| Update session H/L levels                                        |
//+------------------------------------------------------------------+
void UpdateSessionLevels() {
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);

   // Previous day H/L (daily bar 1)
   g_PrevDayHigh = iHigh(_Symbol, PERIOD_D1, 1);
   g_PrevDayLow  = iLow(_Symbol,  PERIOD_D1, 1);

   // Convert EST session times to server time
   int estOffset = 5 + ServerUTC_Offset; // EST = UTC-5, server = UTC+offset

   // Asia session (20:00 - 00:00 EST prev day, or find bars)
   // Find Asia bars in current session context
   FindSessionHL(AsiaStart_H + estOffset, AsiaEnd_H + estOffset,
                 g_AsiaHigh, g_AsiaLow);

   // London KZ
   FindSessionHL(LondonStart_H + estOffset, LondonEnd_H + estOffset,
                 g_LondonHigh, g_LondonLow);

   // H1 swings
   double h1Hi = 0, h1Lo = 999999;
   for(int i = 1; i <= SwingLookback; i++) {
      double hh = iHigh(_Symbol, PERIOD_H1, i);
      double hl = iLow(_Symbol,  PERIOD_H1, i);
      if(hh > h1Hi) h1Hi = hh;
      if(hl < h1Lo) h1Lo = hl;
   }
   g_H1SwingHigh = h1Hi;
   g_H1SwingLow  = h1Lo;
}

//+------------------------------------------------------------------+
//| Find session high/low for a given hour range                     |
//+------------------------------------------------------------------+
void FindSessionHL(int startH, int endH, double &outHigh, double &outLow) {
   // Normalize hours
   startH = startH % 24;
   endH   = endH % 24;

   outHigh = 0;
   outLow  = 999999;

   // Scan M5 bars for today's session
   for(int i = 1; i <= 500; i++) {
      datetime barTime = iTime(_Symbol, PERIOD_M5, i);
      MqlDateTime bt;
      TimeToStruct(barTime, bt);

      bool inSession;
      if(startH < endH) {
         inSession = (bt.hour >= startH && bt.hour < endH);
      } else {
         // Crosses midnight
         inSession = (bt.hour >= startH || bt.hour < endH);
      }

      // Only look at today and yesterday
      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);
      if(bt.day_of_year < now.day_of_year - 1) break;

      if(inSession) {
         double h = iHigh(_Symbol, PERIOD_M5, i);
         double l = iLow(_Symbol,  PERIOD_M5, i);
         if(h > outHigh) outHigh = h;
         if(l < outLow)  outLow  = l;
      }
   }
   if(outLow == 999999) outLow = 0;
}

//+------------------------------------------------------------------+
//| Update H4 swing levels                                           |
//+------------------------------------------------------------------+
void UpdateH4SwingLevels() {
   double hi = 0, lo = 999999;
   for(int i = 1; i <= SwingLookback; i++) {
      double h = iHigh(_Symbol, PERIOD_H4, i);
      double l = iLow(_Symbol,  PERIOD_H4, i);
      if(h > hi) hi = h;
      if(l < lo) lo = l;
   }
   g_H4SwingHigh = hi;
   g_H4SwingLow  = (lo == 999999) ? 0 : lo;
}

//+------------------------------------------------------------------+
//| Get current session                                              |
//+------------------------------------------------------------------+
SESSION_TYPE GetCurrentSession() {
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   int serverHour = dt.hour;
   int estOffset  = 5 + ServerUTC_Offset;

   // Convert server time back to EST
   int estHour = (serverHour - estOffset + 24) % 24;

   // Asia: 20:00 - 00:00 EST
   bool inAsia = (estHour >= AsiaStart_H || estHour < AsiaEnd_H);
   // London KZ: 2:00 - 5:00 EST
   bool inLondon = (estHour >= LondonStart_H && estHour < LondonEnd_H);
   // NY KZ: 7:00 - 11:00 EST
   bool inNY = (estHour >= NYStart_H && estHour < NYEnd_H);

   if(inNY && TradeNYKZ) return SESSION_NY_KZ;
   if(inLondon && TradeLondonKZ) return SESSION_LONDON_KZ;
   if(inAsia) return SESSION_ASIA;
   return SESSION_NONE;
}

//+------------------------------------------------------------------+
//| Check if a tradeable session is active                           |
//+------------------------------------------------------------------+
bool IsSessionActive() {
   SESSION_TYPE s = GetCurrentSession();
   return (s == SESSION_LONDON_KZ || s == SESSION_NY_KZ);
}

//+------------------------------------------------------------------+
//| Check if hard cutoff passed (11 AM EST)                          |
//+------------------------------------------------------------------+
bool IsHardCutoffPassed() {
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   int estOffset = 5 + ServerUTC_Offset;
   int estHour   = (dt.hour - estOffset + 24) % 24;
   return (estHour >= HardCutoff_H);
}

//+------------------------------------------------------------------+
//| Update the 8-factor checklist                                    |
//+------------------------------------------------------------------+
void UpdateChecklist() {
   // ① Session active
   g_Check[0] = IsSessionActive();
   // ② HTF Bias
   g_Check[1] = (g_HTFBias != BIAS_NEUTRAL);
   // ③ Liquidity sweep — set in CheckSweepOnLevel
   // g_Check[2] already managed
   // ④ BOS — set in DetectBOS_M5
   // g_Check[3] already managed
   // ⑤ FVG — set in DetectFVG_M5
   // g_Check[4] already managed
   // ⑥ OB — set in DetectOrderBlock_M5
   // g_Check[5] already managed
   // ⑦ SMT Divergence (optional)
   if(Use_SMT && SMT_Symbol != "")
      g_Check[6] = CheckSMT();
   else
      g_Check[6] = !Use_SMT; // disabled = not blocking
   // ⑧ 79% Fib — set in DetectBOS_M5
   // g_Check[7] already managed

   // Also check HTF bias alignment
   if(g_SweepDetected) {
      if(g_SweepBullish && g_HTFBias == BIAS_BEARISH) g_Check[1] = false;
      if(!g_SweepBullish && g_HTFBias == BIAS_BULLISH) g_Check[1] = false;
   }

   g_ChecklistScore = 0;
   for(int i = 0; i < 8; i++)
      if(g_Check[i]) g_ChecklistScore++;
}

//+------------------------------------------------------------------+
//| SMT Divergence check                                             |
//+------------------------------------------------------------------+
bool CheckSMT() {
   if(SMT_Symbol == "") return false;

   // Compare last swing high/low between main and correlated symbol
   double mainSwH = iHigh(_Symbol, PERIOD_M5, 1);
   double mainSwL = iLow(_Symbol,  PERIOD_M5, 1);

   // Previous swing on main
   double mainPrevH = iHigh(_Symbol, PERIOD_M5, 2);
   double mainPrevL = iLow(_Symbol,  PERIOD_M5, 2);

   double corrSwH = iHigh(SMT_Symbol, PERIOD_M5, 1);
   double corrSwL = iLow(SMT_Symbol,  PERIOD_M5, 1);
   double corrPrevH = iHigh(SMT_Symbol, PERIOD_M5, 2);
   double corrPrevL = iLow(SMT_Symbol,  PERIOD_M5, 2);

   // Bearish divergence: main makes new high, corr does NOT
   if(g_SweepBullish) {
      bool mainNewHigh = mainSwH > mainPrevH;
      bool corrNewHigh = corrSwH > corrPrevH;
      return mainNewHigh && !corrNewHigh;
   } else {
      bool mainNewLow = mainSwL < mainPrevL;
      bool corrNewLow = corrSwL < corrPrevL;
      return mainNewLow && !corrNewLow;
   }
}

//+------------------------------------------------------------------+
//| Daily reset check                                                |
//+------------------------------------------------------------------+
void CheckDailyReset() {
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   MqlDateTime lastDt;
   TimeToStruct(g_LastResetDay, lastDt);

   if(dt.day != lastDt.day) {
      ResetDailyTracking();
   }
}

void ResetDailyTracking() {
   g_TradesToday  = 0;
   g_WinsToday    = 0;
   g_LossesToday  = 0;
   g_DailyPnL_Pct = 0;
   g_DailyPnL_R   = 0;
   g_ConsecLosses = 0;
   g_LastResetDay = TimeCurrent();
   if(g_Phase == PHASE_DAY_HALTED) {
      g_Phase = PHASE_IDLE;
      ResetSetup();
      Print("TJR EA: Daily reset — resuming trading");
   }
}

//+------------------------------------------------------------------+
//| Reset current setup back to waiting                              |
//+------------------------------------------------------------------+
void ResetSetup() {
   g_SweepDetected    = false;
   g_SweepBullish     = false;
   g_SweepLevel       = 0;
   g_M5_BOS_Confirmed = false;
   g_FVGActive        = false;
   g_FVGHigh          = 0;
   g_FVGLow           = 0;
   g_OBActive         = false;
   g_OBHigh           = 0;
   g_OBLow            = 0;
   g_EQLevel          = 0;
   g_BOSWaitCount     = 0;
   g_TradeActive      = false;
   g_PartialClosed    = false;
   g_BESet            = false;
   for(int i = 2; i < 8; i++) g_Check[i] = false; // Keep session + bias
   g_ChecklistScore   = 0;
   if(g_Phase != PHASE_DAY_HALTED)
      g_Phase = IsSessionActive() ? PHASE_AWAIT_SWEEP : PHASE_IDLE;
}

//+------------------------------------------------------------------+
//| Get session name string                                          |
//+------------------------------------------------------------------+
string GetSessionName(SESSION_TYPE s) {
   switch(s) {
      case SESSION_ASIA:      return "ASIA";
      case SESSION_LONDON_KZ: return "LONDON_KILLZONE";
      case SESSION_NY_KZ:     return "NY_KILLZONE";
      default:                return "OFF_SESSION";
   }
}

//+------------------------------------------------------------------+
//| Get phase name string                                            |
//+------------------------------------------------------------------+
string GetPhaseName(EA_PHASE p) {
   switch(p) {
      case PHASE_IDLE:             return "IDLE";
      case PHASE_AWAIT_SWEEP:      return "AWAIT_SWEEP";
      case PHASE_SWEEP_CONFIRMED:  return "SWEEP_CONFIRMED";
      case PHASE_AWAIT_M5_BOS:     return "AWAIT_M5_BOS";
      case PHASE_AWAIT_M5_RETRACE: return "AWAIT_M5_RETRACE";
      case PHASE_AWAIT_M1_ENTRY:   return "AWAIT_M1_ENTRY";
      case PHASE_TRADE_OPEN:       return "TRADE_OPEN";
      case PHASE_DAY_HALTED:       return "DAY_HALTED";
      default:                     return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| VISUAL — Draw info panel (top-right corner)                      |
//+------------------------------------------------------------------+
void DrawInfoPanel() {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double pnlPct  = balance > 0 ? (equity - balance) / balance * 100.0 : 0;

   string panel = "TJR_Panel";
   string txt = StringFormat(
      "TJR EA v3.0\n"
      "Phase: %s\n"
      "HTF Bias: %s\n"
      "Session: %s\n"
      "Sweep: %s\n"
      "M5 BOS: %s\n"
      "Checklist: %d/8\n"
      "Trades Today: %d/%d\n"
      "Daily P&L: %+.2f%%\n"
      "Risk Used: %.2f%%",
      GetPhaseName(g_Phase),
      g_HTFBias == BIAS_BULLISH ? "BULLISH ↑" : g_HTFBias == BIAS_BEARISH ? "BEARISH ↓" : "NEUTRAL",
      GetSessionName(g_CurrentSession),
      g_SweepDetected ? StringFormat("@ %.5f ✓", g_SweepLevel) : "Watching...",
      g_M5_BOS_Confirmed ? "Confirmed ✓" : "Pending...",
      g_ChecklistScore,
      g_TradesToday, PropFirmMode ? 1 : MaxTradesPerDay,
      pnlPct,
      PropFirmMode ? 0.25 : RiskPct
   );

   if(ObjectFind(0, panel) < 0)
      ObjectCreate(0, panel, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, panel, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, panel, OBJPROP_XDISTANCE, 200);
   ObjectSetInteger(0, panel, OBJPROP_YDISTANCE, 20);
   ObjectSetString(0, panel, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, panel, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, panel, OBJPROP_FONTSIZE, 9);
   ObjectSetString(0, panel, OBJPROP_FONT, "Courier New");
   ObjectSetInteger(0, panel, OBJPROP_BACK, false);
}

//+------------------------------------------------------------------+
//| VISUAL — Draw session H/L lines                                  |
//+------------------------------------------------------------------+
void DrawSessionLines() {
   datetime t1 = iTime(_Symbol, PERIOD_M5, 200);
   datetime t2 = iTime(_Symbol, PERIOD_M5, 0) + 3600;

   struct SLine { string name; double price; color clr; string lbl; };
   SLine lines[] = {
      {"TJR_AsiaH",  g_AsiaHigh,    clrGold,       "Asia H"},
      {"TJR_AsiaL",  g_AsiaLow,     clrGold,       "Asia L"},
      {"TJR_LonH",   g_LondonHigh,  clrDodgerBlue, "London H"},
      {"TJR_LonL",   g_LondonLow,   clrDodgerBlue, "London L"},
      {"TJR_PDH",    g_PrevDayHigh, clrSilver,     "PDH"},
      {"TJR_PDL",    g_PrevDayLow,  clrSilver,     "PDL"}
   };

   for(int i = 0; i < ArraySize(lines); i++) {
      if(lines[i].price <= 0) continue;
      string nm = lines[i].name;
      if(ObjectFind(0, nm) < 0)
         ObjectCreate(0, nm, OBJ_HLINE, 0, 0, lines[i].price);
      ObjectSetDouble(0, nm, OBJPROP_PRICE, lines[i].price);
      ObjectSetInteger(0, nm, OBJPROP_COLOR, lines[i].clr);
      ObjectSetInteger(0, nm, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, nm, OBJPROP_WIDTH, 1);
   }
}

//+------------------------------------------------------------------+
//| VISUAL — Draw FVG box                                            |
//+------------------------------------------------------------------+
void DrawFVGBox() {
   if(!g_FVGActive) return;
   string nm = "TJR_FVG_" + TimeToString(TimeCurrent());
   datetime t1 = iTime(_Symbol, PERIOD_M5, 5);
   datetime t2 = iTime(_Symbol, PERIOD_M5, 0) + 86400; // extends right
   if(ObjectFind(0, nm) < 0)
      ObjectCreate(0, nm, OBJ_RECTANGLE, 0, t1, g_FVGHigh, t2, g_FVGLow);
   ObjectSetInteger(0, nm, OBJPROP_COLOR,
                    g_FVGBullish ? clrLimeGreen : clrRed);
   ObjectSetInteger(0, nm, OBJPROP_FILL, true);
   ObjectSetInteger(0, nm, OBJPROP_BACK, true);
   ObjectSetInteger(0, nm, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, nm, OBJPROP_WIDTH, 1);
}

//+------------------------------------------------------------------+
//| VISUAL — Draw Order Block box                                    |
//+------------------------------------------------------------------+
void DrawOBBox(int shift) {
   if(!g_OBActive) return;
   datetime t1 = iTime(_Symbol, PERIOD_M5, shift);
   datetime t2 = iTime(_Symbol, PERIOD_M5, 0) + 86400;
   string nm = "TJR_OB_" + TimeToString(t1);
   if(ObjectFind(0, nm) < 0)
      ObjectCreate(0, nm, OBJ_RECTANGLE, 0, t1, g_OBHigh, t2, g_OBLow);
   ObjectSetInteger(0, nm, OBJPROP_COLOR,
                    g_OBBullish ? clrDodgerBlue : clrOrange);
   ObjectSetInteger(0, nm, OBJPROP_FILL, false);
   ObjectSetInteger(0, nm, OBJPROP_BACK, true);
   ObjectSetInteger(0, nm, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, nm, OBJPROP_WIDTH, 2);
}

//+------------------------------------------------------------------+
//| VISUAL — Draw active trade levels                                |
//+------------------------------------------------------------------+
void DrawTradeLevels() {
   if(!g_TradeActive) return;
   struct TLine { string nm; double price; color clr; string txt; };
   TLine tlines[] = {
      {"TJR_TradeEntry", g_EntryPrice, clrYellow,    "Entry"},
      {"TJR_TradeSL",    g_StopLoss,   clrRed,       "SL"},
      {"TJR_TradeTP1",   g_TP1,        clrLime,      "TP1"},
      {"TJR_TradeTP2",   g_TP2,        clrLimeGreen, "TP2"},
      {"TJR_TradeTP3",   g_TP3,        clrGreen,     "TP3"}
   };
   for(int i = 0; i < ArraySize(tlines); i++) {
      string nm = tlines[i].nm;
      if(ObjectFind(0, nm) < 0)
         ObjectCreate(0, nm, OBJ_HLINE, 0, 0, tlines[i].price);
      ObjectSetDouble(0, nm,  OBJPROP_PRICE, tlines[i].price);
      ObjectSetInteger(0, nm, OBJPROP_COLOR, tlines[i].clr);
      ObjectSetInteger(0, nm, OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, nm, OBJPROP_WIDTH, 1);
   }
}

//+------------------------------------------------------------------+
//| Delete all TJR objects from chart                                |
//+------------------------------------------------------------------+
void DeleteAllTJRObjects() {
   for(int i = ObjectsTotal(0) - 1; i >= 0; i--) {
      string nm = ObjectName(0, i);
      if(StringFind(nm, "TJR_") == 0)
         ObjectDelete(0, nm);
   }
}

//+------------------------------------------------------------------+
//| Write live data JSON file for dashboard                          |
//+------------------------------------------------------------------+
void WriteJSONFile() {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double pnlPct  = balance > 0 ? (equity - balance) / balance * 100.0 : 0;
   double openPnlUSD = equity - balance;
   double openPnlR   = g_InitialRisk_USD > 0 ? openPnlUSD / g_InitialRisk_USD : 0;

   string json = StringFormat(
      "{\n"
      "  \"timestamp\": \"%s\",\n"
      "  \"symbol\": \"%s\",\n"
      "  \"phase\": \"%s\",\n"
      "  \"htf_bias\": \"%s\",\n"
      "  \"session\": \"%s\",\n"
      "  \"sweep_detected\": %s,\n"
      "  \"sweep_direction\": \"%s\",\n"
      "  \"sweep_level\": %.5f,\n"
      "  \"m5_bos_confirmed\": %s,\n"
      "  \"entry_zone_active\": %s,\n"
      "  \"checklist_score\": %d,\n"
      "  \"checklist\": [%s,%s,%s,%s,%s,%s,%s,%s],\n"
      "  \"trade_active\": %s,\n"
      "  \"trade_direction\": \"%s\",\n"
      "  \"entry_price\": %.5f,\n"
      "  \"stop_loss\": %.5f,\n"
      "  \"tp1\": %.5f,\n"
      "  \"tp2\": %.5f,\n"
      "  \"open_pnl_r\": %.4f,\n"
      "  \"open_pnl_usd\": %.2f,\n"
      "  \"trades_today\": %d,\n"
      "  \"wins_today\": %d,\n"
      "  \"losses_today\": %d,\n"
      "  \"daily_pnl_pct\": %.4f,\n"
      "  \"daily_pnl_r\": %.4f,\n"
      "  \"consec_losses\": %d,\n"
      "  \"account_balance\": %.2f,\n"
      "  \"account_equity\": %.2f,\n"
      "  \"asia_high\": %.5f,\n"
      "  \"asia_low\": %.5f,\n"
      "  \"london_high\": %.5f,\n"
      "  \"london_low\": %.5f,\n"
      "  \"prev_day_high\": %.5f,\n"
      "  \"prev_day_low\": %.5f,\n"
      "  \"h4_swing_high\": %.5f,\n"
      "  \"h4_swing_low\": %.5f,\n"
      "  \"fvg_active\": %s,\n"
      "  \"fvg_high\": %.5f,\n"
      "  \"fvg_low\": %.5f,\n"
      "  \"ob_active\": %s,\n"
      "  \"ob_high\": %.5f,\n"
      "  \"ob_low\": %.5f,\n"
      "  \"eq_level\": %.5f\n"
      "}",
      TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
      _Symbol,
      GetPhaseName(g_Phase),
      g_HTFBias == BIAS_BULLISH ? "BULLISH" : g_HTFBias == BIAS_BEARISH ? "BEARISH" : "NEUTRAL",
      GetSessionName(g_CurrentSession),
      g_SweepDetected ? "true" : "false",
      g_SweepBullish ? "BULLISH" : "BEARISH",
      g_SweepLevel,
      g_M5_BOS_Confirmed ? "true" : "false",
      (g_Phase == PHASE_AWAIT_M5_RETRACE || g_Phase == PHASE_AWAIT_M1_ENTRY) ? "true" : "false",
      g_ChecklistScore,
      g_Check[0]?"true":"false", g_Check[1]?"true":"false",
      g_Check[2]?"true":"false", g_Check[3]?"true":"false",
      g_Check[4]?"true":"false", g_Check[5]?"true":"false",
      g_Check[6]?"true":"false", g_Check[7]?"true":"false",
      g_TradeActive ? "true" : "false",
      g_TradeActive ? (g_TradeIsBuy ? "BUY" : "SELL") : "NONE",
      g_EntryPrice,
      g_StopLoss,
      g_TP1, g_TP2,
      openPnlR, openPnlUSD,
      g_TradesToday,
      g_WinsToday,
      g_LossesToday,
      pnlPct,
      g_DailyPnL_R,
      g_ConsecLosses,
      balance, equity,
      g_AsiaHigh, g_AsiaLow,
      g_LondonHigh, g_LondonLow,
      g_PrevDayHigh, g_PrevDayLow,
      g_H4SwingHigh, g_H4SwingLow,
      g_FVGActive ? "true" : "false", g_FVGHigh, g_FVGLow,
      g_OBActive ? "true" : "false",  g_OBHigh,  g_OBLow,
      g_EQLevel
   );

   int fh = FileOpen("tjr_live_data.json",
                     FILE_WRITE | FILE_TXT | FILE_COMMON);
   if(fh != INVALID_HANDLE) {
      FileWriteString(fh, json);
      FileClose(fh);
   }
}

//+------------------------------------------------------------------+
//| Write trade history CSV                                          |
//+------------------------------------------------------------------+
void WriteTradeHistoryCSV(double entry, double sl, double tp1,
                          double tp3, string notes) {
   // Write header if file doesn't exist
   bool fileExists = FileIsExist("tjr_trade_history.csv", FILE_COMMON);

   int fh = FileOpen("tjr_trade_history.csv",
                     FILE_WRITE | FILE_READ | FILE_CSV | FILE_COMMON);
   if(fh == INVALID_HANDLE) return;

   if(!fileExists) {
      FileWriteString(fh,
         "datetime,symbol,direction,entry,sl,tp1,tp2,"
         "exit,result_r,result_usd,session,phase_at_entry,"
         "checklist_score,notes\n");
   }

   // Seek to end
   FileSeek(fh, 0, SEEK_END);

   string row = StringFormat("%s,%s,%s,%.5f,%.5f,%.5f,%.5f,"
                              "0,0,0,%s,%s,%d,%s\n",
      TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
      _Symbol,
      g_TradeIsBuy ? "BUY" : "SELL",
      entry, sl, tp1, tp3,
      GetSessionName(g_CurrentSession),
      GetPhaseName(g_Phase),
      g_ChecklistScore,
      notes
   );

   FileWriteString(fh, row);
   FileClose(fh);
}
//+------------------------------------------------------------------+
