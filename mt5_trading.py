#!/usr/bin/env python3
"""
MT5 Ensemble Trader — Swing (D1) + Scalp (M5)

Credentials are loaded from .env (copy .env.example → .env and fill in values).
CLI flags override .env values when provided.

Usage:
  Paper backtest — swing (D1):
    python mt5_trading.py --mode paper --style swing

  Paper backtest — scalp (M5):
    python mt5_trading.py --mode paper --style scalp

  Live trading — swing:
    python mt5_trading.py --mode live --style swing

  Live trading — scalp:
    python mt5_trading.py --mode live --style scalp

Options:
  --mode      paper | live                       (default: paper)
  --style     swing | scalp                      (default: swing)
  --csv       CSV data file                      (default: auto by style)
  --split     train/test split 0-1               (default: 0.8, paper only)
  --balance   starting balance USD               (default: MT5_BALANCE env or 1000)
  --ensemble  path to models directory           (default: auto by style)
  --symbol    MT5 symbol name                    (default: MT5_SYMBOL env or XAUUSD)
  --risk      fraction of balance per trade      (default: MT5_RISK env or 0.02)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from mt5_connector import MT5Connector, add_indicators
from ensemble_trader import EnsembleTrader
from market_regime_detector import MarketRegimeDetector
from news_filter import NewsFilter


# ── Session filter ────────────────────────────────────────────────────────────

class ScalpSessionFilter:
    """
    Block scalp ENTRIES outside active XAUUSD liquidity windows.
    Existing positions are always managed — only new entries are blocked.

    Gold liquidity windows (UTC):
      07:00–09:00  London open      — directional energy, tight spreads
      09:00–12:00  London session   — steady flow
      12:00–13:00  Pre-NY buildup   — momentum pickup
      13:00–17:00  London/NY overlap — highest volume, best for scalping
      17:00–21:00  NY afternoon     — active until NY close

    Blocked:  21:00–07:00 UTC (Asian session + overnight — thin, wide spreads)
    """

    # UTC hour ranges where new scalp entries are permitted [start, end)
    ALLOWED_HOURS: list = [(7, 21)]   # 07:00–21:00 UTC covers all active sessions

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def is_allowed(self, now: datetime = None) -> tuple:
        """Returns (allowed: bool, reason: str)."""
        if not self.enabled:
            return True, ''
        now  = now or datetime.now(timezone.utc)
        hour = now.hour
        for start, end in self.ALLOWED_HOURS:
            if start <= hour < end:
                return True, ''
        session = self._session_label(hour)
        return False, f"Outside active session — {session} ({hour:02d}:{now.minute:02d} UTC)"

    @staticmethod
    def _session_label(hour: int) -> str:
        if 0 <= hour < 7:
            return "Asian session"
        if 21 <= hour < 24:
            return "NY close / overnight"
        return "closed"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler('mt5_trading.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Style configs ─────────────────────────────────────────────────────────────

STYLE_CONFIG = {
    'swing': {
        'csv':           'xauusd_data.csv',
        'ensemble_path': './ensemble_models/',
        'config_file':   'ensemble_config.json',
        'min_bars':      30,
        'lookback':      10,
        'stop_loss_pct': 0.02,
        'tp_multiplier': 4.0,
        'min_signal':    0.20,
        'mt5_timeframe': 'D1',
        'live_interval': 3600,
        'mt5_bars':      100,
    },
    'scalp': {
        'csv':           'xauusd_m5_data.csv',
        'ensemble_path': './scalp_models/',
        'config_file':   'ensemble_config.json',
        'min_bars':      30,
        'lookback':      20,
        'stop_loss_pct': 0.0015,    # 0.15% SL
        'rr_targets':    [0.5, 1.0, 2.0],
        'tp_fractions':  [1/3, 1/3, 1/3],
        'min_signal':    0.15,
        'mt5_timeframe': 'M5',
        'live_interval': 300,       # poll every 5 min
        'mt5_bars':      100,
    },
    # High-frequency scalp on M1 bars — tighter SL/TP, more signals per day
    'hf_scalp': {
        'csv':           'xauusd_m1_data.csv',
        'ensemble_path': './hf_scalp_models/',
        'config_file':   'ensemble_config.json',
        'min_bars':      30,
        'lookback':      20,
        'stop_loss_pct': 0.0008,    # 0.08% SL (tighter for M1 micro-moves)
        'rr_targets':    [0.5, 1.0, 2.0],
        'tp_fractions':  [1/3, 1/3, 1/3],
        'min_signal':    0.05,      # lower threshold → more entries per session
        'mt5_timeframe': 'M1',
        'live_interval': 60,        # poll every 1 min
        'mt5_bars':      100,
    },
}


# ── MTF indicator helpers ─────────────────────────────────────────────────────

def _compute_htf_features(htf_df: pd.DataFrame) -> dict:
    """Compute M15/M30 indicators from a fetched HTF OHLCV dataframe.
    Returns dict with keys: ema_cross, rsi, trend, bos, fvg (latest bar values).
    """
    if htf_df is None or len(htf_df) < 10:
        return {'ema_cross': 0.0, 'rsi': 0.5, 'trend': 0.0, 'bos': 0.0, 'fvg': 0.0}

    close = htf_df['Close']
    ema8  = close.ewm(span=8,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(7).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rsi7  = 100 - (100 / (1 + gain / (loss + 1e-9)))
    trend = np.sign(ema21.diff())

    swing_high = htf_df['High'].shift(1).rolling(5).max()
    swing_low  = htf_df['Low'].shift(1).rolling(5).min()
    bos = np.where(close > swing_high, 1.0,
          np.where(close < swing_low, -1.0, 0.0))
    fvg = np.where(htf_df['Low']  > htf_df['High'].shift(2),  1.0,
          np.where(htf_df['High'] < htf_df['Low'].shift(2),  -1.0, 0.0))

    last_close = float(close.iloc[-1])
    return {
        'ema_cross': float((ema8.iloc[-1] - ema21.iloc[-1]) / (last_close + 1e-9)),
        'rsi':       float(rsi7.iloc[-1]) / 100.0,
        'trend':     float(trend.iloc[-1]),
        'bos':       float(bos[-1]),
        'fvg':       float(fvg[-1]),
    }


def _mtf_features_from_df_row(row) -> np.ndarray:
    """Extract precomputed MTF columns from a df row (paper backtest mode)."""
    return np.array([
        float(row.get('m30_ema_cross', 0.0)),
        float(row.get('m30_rsi',       0.5)),
        float(row.get('m30_trend',     0.0)),
        float(row.get('m15_ema_cross', 0.0)),
        float(row.get('m15_rsi',       0.5)),
        float(row.get('m15_bos',       0.0)),
        float(row.get('m15_fvg',       0.0)),
    ], dtype=np.float32)


def _add_scalp_mtf_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and forward-fill M15/M30 MTF columns into base-TF df.
    Used for paper backtests so MTF obs are available without fetching from MT5.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse from 'date' column
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return df   # can't resample — return as-is

    def _resample_agg(rule: str) -> pd.DataFrame:
        r = df[['Open', 'High', 'Low', 'Close', 'Volume']].resample(
            rule, label='right', closed='right'
        )
        htf = r.agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                     'Close': 'last', 'Volume': 'sum'})
        return htf.dropna(subset=['Close'])

    def _indicators(htf: pd.DataFrame):
        ema8  = htf['Close'].ewm(span=8,  adjust=False).mean()
        ema21 = htf['Close'].ewm(span=21, adjust=False).mean()
        delta = htf['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(7).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rsi7  = 100 - (100 / (1 + gain / (loss + 1e-9)))
        trend = np.sign(ema21.diff())
        swing_high = htf['High'].shift(1).rolling(5).max()
        swing_low  = htf['Low'].shift(1).rolling(5).min()
        bos = np.where(htf['Close'] > swing_high, 1.0,
              np.where(htf['Close'] < swing_low, -1.0, 0.0))
        fvg = np.where(htf['Low']  > htf['High'].shift(2),  1.0,
              np.where(htf['High'] < htf['Low'].shift(2),  -1.0, 0.0))
        return pd.DataFrame({
            'ema_cross': (ema8 - ema21) / (htf['Close'] + 1e-9),
            'rsi':       rsi7 / 100.0,
            'trend':     trend,
            'bos':       bos,
            'fvg':       fvg,
        }, index=htf.index)

    m30 = _indicators(_resample_agg('30min'))
    m15 = _indicators(_resample_agg('15min'))

    df['m30_ema_cross'] = m30['ema_cross'].reindex(df.index, method='ffill')
    df['m30_rsi']       = m30['rsi'].reindex(df.index, method='ffill')
    df['m30_trend']     = m30['trend'].reindex(df.index, method='ffill')
    df['m15_ema_cross'] = m15['ema_cross'].reindex(df.index, method='ffill')
    df['m15_rsi']       = m15['rsi'].reindex(df.index, method='ffill')
    df['m15_bos']       = m15['bos'].reindex(df.index, method='ffill')
    df['m15_fvg']       = m15['fvg'].reindex(df.index, method='ffill')

    return df.fillna(0)


# ── Scalp observation builder ─────────────────────────────────────────────────

def _build_scalp_obs(df: pd.DataFrame, position_sign: float = 0.0,
                     mtf_live: dict = None) -> np.ndarray:
    """28-feature (base) or 35-feature (MTF) observation matching ScalpTradingEnv.

    position_sign: np.sign(position) — +1.0 long, -1.0 short, 0.0 flat.
    mtf_live: dict from _compute_htf_features() for live M30 bars + M15 bars,
              or None to read precomputed MTF columns from df (paper mode).
    When mtf_live or MTF columns present → 35 features; otherwise 28.
    """
    lookback = 20
    close = float(df['Close'].iloc[-1])
    prices = (df['Close'].iloc[-lookback:].values.astype(float) / close) - 1.0

    row = df.iloc[-1]
    ema8  = float(row.get('ema8',  close))
    ema21 = float(row.get('ema21', close))
    indicators = np.array([
        (ema8 - ema21) / (close + 1e-9),
        float(row.get('rsi7',     50.0)) / 100.0,
        float(row.get('atr14',    0.0))  / (close + 1e-9),
        float(row.get('macd',     0.0))  / (close + 1e-9),
        float(row.get('macd_sig', 0.0))  / (close + 1e-9),
        float(row.get('stoch_k',  50.0)) / 100.0,
        float(position_sign),
        0.0,    # time-in-trade unknown in live polling loop
    ], dtype=np.float32)

    obs = np.concatenate([prices.astype(np.float32), indicators])

    # MTF extension (7 features) — only for hf_scalp (35-feature models)
    if mtf_live is not None:
        # Live mode: use freshly fetched M30/M15 bars
        m30, m15 = mtf_live
        mtf = np.array([
            m30['ema_cross'], m30['rsi'], m30['trend'],
            m15['ema_cross'], m15['rsi'], m15['bos'], m15['fvg'],
        ], dtype=np.float32)
        obs = np.concatenate([obs, mtf])
    elif 'm30_ema_cross' in row.index if hasattr(row, 'index') else 'm30_ema_cross' in row:
        # Paper mode: MTF columns precomputed in df
        obs = np.concatenate([obs, _mtf_features_from_df_row(row)])

    return obs.reshape(1, -1)


# ── Swing observation builder ─────────────────────────────────────────────────

def _build_swing_obs(df: pd.DataFrame) -> np.ndarray:
    """15-feature observation matching TradingEnv."""
    lookback = 10
    prices = df['Close'].iloc[-lookback:].values.astype(float)
    row = df.iloc[-1]
    rsi  = float(row.get('rsi',         50.0))
    macd = float(row.get('macd',        0.0))
    sig  = float(row.get('signal_line', 0.0))
    return np.concatenate([prices, [rsi, macd, sig, 0.0, 1000.0]]).reshape(1, -1)


# ── Scalp indicator calculator (for live M5 bars from MT5) ───────────────────

def _add_scalp_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema8']  = df['Close'].ewm(span=8,  adjust=False).mean()
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(7).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(7).mean()
    df['rsi7'] = 100 - (100 / (1 + gain / loss))

    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift()).abs()
    lpc = (df['Low']  - df['Close'].shift()).abs()
    df['atr14'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd']     = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()

    low5  = df['Low'].rolling(5).min()
    high5 = df['High'].rolling(5).max()
    df['stoch_k'] = 100 * (df['Close'] - low5) / (high5 - low5 + 1e-9)

    return df.fillna(0)


# ── Main trader class ─────────────────────────────────────────────────────────

class MT5EnsembleTrader:
    def __init__(
        self,
        connector: MT5Connector,
        style: str = 'swing',
        ensemble_path: str = None,
        symbol: str = 'XAUUSD',
        risk_pct: float = 0.02,
    ):
        self.connector = connector
        self.style     = style
        self.symbol    = symbol
        self.risk_pct  = risk_pct
        self.cfg       = STYLE_CONFIG[style]

        ep = ensemble_path or self.cfg['ensemble_path']
        logger.info(f"[{style.upper()}] Loading ensemble from {ep} ...")
        self.ensemble = EnsembleTrader()
        self.ensemble.load_ensemble(ep)

        self.regime_detector  = MarketRegimeDetector()
        self.stop_loss_pct    = self.cfg['stop_loss_pct']
        self.min_signal       = self.cfg['min_signal']
        self.trailing_stop_pct = self.stop_loss_pct  # updated by regime for swing

        # Swing still uses a single TP multiplier; scalp uses RR-based partials
        self.tp_multiplier = self.cfg.get('tp_multiplier', 4.0)

        # Per-ticket TP tracking for scalp partial exits
        # ticket → {tp_prices, tp_hit, initial_vol, direction}
        self._scalp_tp_info: dict = {}

        # News filter: skip entries during high-impact events (scalp styles only)
        self.news_filter = NewsFilter(
            pause_minutes_before=15,
            pause_minutes_after=30,
            enabled=self._is_scalp,
        )

        # Session filter: only open scalp entries during active London/NY windows
        self.session_filter = ScalpSessionFilter(enabled=self._is_scalp)

        # MTF cache: (m30_features_dict, m15_features_dict) — refreshed each live bar
        # Used for hf_scalp obs (35-feature) and confluence gate (both scalp styles)
        self._mtf_live_cache: tuple = None   # None until first live bar

    # ── Style helpers ────────────────────────────────────────────────────────

    @property
    def _is_scalp(self) -> bool:
        """True for both 'scalp' (M5) and 'hf_scalp' (M1) styles."""
        return self.style in ('scalp', 'hf_scalp')

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self, df: pd.DataFrame, positions: list = None) -> np.ndarray:
        if self._is_scalp:
            pos_sign = self._scalp_position_sign(positions or [])
            # hf_scalp uses 35-feature obs (MTF extended); scalp uses 28-feature obs
            mtf_live = self._mtf_live_cache if self.style == 'hf_scalp' else None
            return _build_scalp_obs(df, pos_sign, mtf_live=mtf_live)
        return _build_swing_obs(df)

    def _scalp_position_sign(self, positions: list) -> float:
        """Return np.sign(position) equivalent for the live obs — matches training env."""
        if not positions:
            return 0.0
        pos = positions[0]
        return 1.0 if pos.type == MT5Connector.ORDER_BUY else -1.0

    # ── Regime (swing only) ──────────────────────────────────────────────────

    def _update_regime(self, df: pd.DataFrame):
        if self._is_scalp:
            return
        try:
            regime, params = self.regime_detector.detect_regime(df, len(df) - 1)
            self.trailing_stop_pct = params['trailing_stop_pct']
            self.min_signal        = params['min_confidence_threshold']
            logger.debug(f"Regime={regime.value}")
        except Exception as exc:
            logger.debug(f"Regime skipped: {exc}")

    # ── ATR spike / news bar detection ──────────────────────────────────────

    # Mirrors ScalpTradingEnv._is_news_bar() — consistent between train and live
    ATR_SPIKE_MULTIPLIER = 2.5
    ATR_LOOKBACK         = 50   # bars of rolling average to compare against

    def _is_atr_spike(self, df: pd.DataFrame) -> bool:
        """Return True if the latest bar shows abnormal volatility (unscheduled news).

        Two checks (either triggers the block):
          1. Rolling ATR14 vs its 50-bar mean — mirrors training env _is_news_bar()
          2. Current bar true range vs ATR14 mean — catches single-bar spikes faster
        """
        if 'atr14' not in df.columns or len(df) < self.ATR_LOOKBACK + 1:
            return False

        atr_mean = float(df['atr14'].iloc[-(self.ATR_LOOKBACK + 1):-1].mean())
        if atr_mean == 0:
            return False

        # Check 1: rolling ATR elevated (same as training env)
        atr_now = float(df['atr14'].iloc[-1])
        if (atr_now / atr_mean) > self.ATR_SPIKE_MULTIPLIER:
            return True

        # Check 2: current bar's raw true range vs average ATR (faster single-bar detection)
        last = df.iloc[-1]
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else float(last['Open'])
        true_range = max(
            float(last['High']) - float(last['Low']),
            abs(float(last['High']) - prev_close),
            abs(float(last['Low'])  - prev_close),
        )
        if (true_range / atr_mean) > self.ATR_SPIKE_MULTIPLIER:
            return True

        return False

    # ── MTF confluence gate ──────────────────────────────────────────────────

    def _mtf_confluence_ok(self, action: float, df: pd.DataFrame) -> bool:
        """Check M30 trend bias + M15 BOS/FVG before any scalp entry.

        Logic:
          Long signal  (action > 0): M30 trend bullish (ema8 > ema21) AND
                                     M15 shows bullish BOS or bullish FVG.
          Short signal (action < 0): M30 trend bearish AND M15 bearish BOS or FVG.

        Returns True (enter allowed) if confluence conditions are met.
        For paper backtest, reads precomputed columns from df row.
        For live, uses self._mtf_live_cache.
        """
        if not self._is_scalp:
            return True     # swing style — no MTF filter

        bullish = action > 0

        # --- Live mode: use cached fetched bars ---
        if self._mtf_live_cache is not None:
            m30_feat, m15_feat = self._mtf_live_cache
            m30_bull = m30_feat['ema_cross'] > 0
            m15_setup = m15_feat['bos'] > 0 or m15_feat['fvg'] > 0
            if bullish:
                ok = m30_bull and m15_setup
            else:
                m30_bear  = m30_feat['ema_cross'] < 0
                m15_setup_bear = m15_feat['bos'] < 0 or m15_feat['fvg'] < 0
                ok = m30_bear and m15_setup_bear
            return ok

        # --- Paper mode: read precomputed MTF columns from df slice ---
        row = df.iloc[-1]
        has_mtf = hasattr(row, 'get') and 'm30_ema_cross' in (
            row.index if hasattr(row, 'index') else {}
        )
        if not has_mtf:
            return True     # MTF columns not in df — pass through (first bars / legacy)

        m30_cross = float(row.get('m30_ema_cross', 0.0))
        m15_bos   = float(row.get('m15_bos', 0.0))
        m15_fvg   = float(row.get('m15_fvg', 0.0))

        if bullish:
            return m30_cross > 0 and (m15_bos > 0 or m15_fvg > 0)
        else:
            return m30_cross < 0 and (m15_bos < 0 or m15_fvg < 0)

    # ── Position sizing ──────────────────────────────────────────────────────

    # Scalp risk scales down with each additional position to cap total exposure
    SCALP_RISK_SCALE = [1.0, 0.75, 0.5]   # 1st=100%, 2nd=75%, 3rd=50% of risk_pct

    def _calc_volume(self, price: float, pos_count: int = 0) -> float:
        account = self.connector.get_account_info()

        # Cent accounts (USC) store balance in cents — convert to USD for sizing
        balance_usd = account.balance / 100.0 if account.currency == 'USC' else account.balance

        risk_pct = self.risk_pct
        if self._is_scalp:
            base_risk = min(self.risk_pct, 0.01)
            scale     = self.SCALP_RISK_SCALE[min(pos_count, 2)]
            risk_pct  = base_risk * scale

        risk_amount   = balance_usd * risk_pct
        risk_in_price = price * self.stop_loss_pct
        volume = risk_amount / (risk_in_price * 100)
        logger.debug(
            f"[SIZING] pos#{pos_count+1} balance=${balance_usd:.2f}({account.currency}) "
            f"risk={risk_pct*100:.2f}% → ${risk_amount:.2f} → {volume:.4f} lots"
        )
        return round(max(0.01, min(volume, 10.0)), 2)

    # ── Core bar handler ─────────────────────────────────────────────────────

    # Confidence thresholds to add 2nd and 3rd scalp position
    SCALP_CONF_THRESHOLDS = [0.0, 0.60, 0.75]  # pos_count 0 → 1 → 2
    MAX_SCALP_POSITIONS   = 3

    def _own_positions(self, positions: list) -> list:
        """Filter to only positions opened by this bot instance (by comment prefix)."""
        prefix = f'{self.style}_'
        return [p for p in positions if p.comment.startswith(prefix)]

    def on_bar(self, df: pd.DataFrame):
        min_bars = self.cfg['min_bars']
        if len(df) < min_bars:
            return

        self._update_regime(df)

        tick = self.connector.get_tick(self.symbol)
        if tick is None:
            logger.warning("No tick — skipping bar")
            return

        current_price = tick.ask

        # Only count/manage positions this bot opened (identified by comment prefix)
        all_positions = self.connector.get_positions(self.symbol)
        positions     = self._own_positions(all_positions)
        n_pos         = len(positions)

        obs = self._build_obs(df, positions)
        action, confidence = self.ensemble.predict_ensemble(obs)
        action     = float(np.squeeze(action))
        confidence = float(np.squeeze(confidence))

        logger.info(
            f"[{self.style.upper()}] {self.connector.current_bar_date} | "
            f"price={current_price:.2f} action={action:+.3f} conf={confidence:.3f} "
            f"pos={n_pos} (total_mt5={len(all_positions)})"
        )

        # Scalp: manage partial TP exits before checking for new entries
        if self._is_scalp and positions:
            self._manage_scalp_tps(positions, current_price)
            all_positions = self.connector.get_positions(self.symbol)
            positions     = self._own_positions(all_positions)
            n_pos         = len(positions)

        # Manage signal flips (close opposing direction positions)
        if positions:
            self._manage_positions(positions, action)
            all_positions = self.connector.get_positions(self.symbol)
            positions     = self._own_positions(all_positions)
            n_pos         = len(positions)

        # Session gate: block new entries outside active windows (position management above is unaffected)
        session_ok, session_reason = self.session_filter.is_allowed()
        if not session_ok:
            logger.debug(f"[SESSION] No new entries — {session_reason}")
            return

        # News gate layer 1: schedule-based (NFP, FOMC, CPI etc.)
        news_blocked, news_reason = self.news_filter.is_news_window()
        if news_blocked:
            logger.info(f"[NEWS] No new entries — {news_reason}")
            return

        # News gate layer 2: ATR spike detection — catches any unscheduled volatility event
        if self._is_scalp and self._is_atr_spike(df):
            logger.info(f"[NEWS] ATR spike detected — skipping entry (unscheduled volatility)")
            return

        # MTF confluence gate: only for hf_scalp (trained with MTF in obs).
        # scalp (M5) models were trained without MTF — gate would over-filter.
        if self.style == 'hf_scalp' and not self._mtf_confluence_ok(action, df):
            logger.debug(f"[MTF] No confluence — skip entry (action={action:+.3f})")
            return

        # Entry: scalp allows up to 3 with increasing confluence, swing stays 1
        max_pos = self.MAX_SCALP_POSITIONS if self._is_scalp else 1
        if n_pos < max_pos:
            self._try_entry(action, confidence, current_price, n_pos, positions)

    def _try_entry(self, action: float, confidence: float, price: float,
                   pos_count: int = 0, existing: list = None):
        existing = existing or []

        # For 2nd and 3rd scalp trades require higher confidence
        if self._is_scalp and pos_count > 0:
            required = self.SCALP_CONF_THRESHOLDS[min(pos_count, 2)]
            if confidence < required:
                logger.debug(
                    f"[{self.style.upper()}] Skip entry #{pos_count+1}: "
                    f"conf={confidence:.3f} < {required:.2f} required"
                )
                return

        # Direction consistency: don't mix longs and shorts
        if existing:
            existing_type = existing[0].type
            if action > self.min_signal and existing_type != MT5Connector.ORDER_BUY:
                return
            if action < -self.min_signal and existing_type != MT5Connector.ORDER_SELL:
                return

        tag = f"#{pos_count+1}" if self._is_scalp else ""

        if action > self.min_signal:
            volume = self._calc_volume(price, pos_count)
            sl_dist = price * self.stop_loss_pct
            sl = round(price - sl_dist, 2)
            if self._is_scalp:
                # MT5 TP set at TP3 (RR:2) as safety net; TP1/TP2 managed via partial closes
                rr_targets = self.cfg['rr_targets']
                tp = round(price + sl_dist * rr_targets[-1], 2)
            else:
                tp = round(price + sl_dist * self.tp_multiplier, 2)
            result = self.connector.place_order(
                self.symbol, MT5Connector.ORDER_BUY, volume,
                sl=sl, tp=tp, comment=f'{self.style}_buy{tag}'
            )
            if result.success and self._is_scalp:
                self._register_scalp_tps(result.ticket, price, volume, direction=1)

        elif action < -self.min_signal:
            volume = self._calc_volume(price, pos_count)
            sl_dist = price * self.stop_loss_pct
            sl = round(price + sl_dist, 2)
            if self._is_scalp:
                rr_targets = self.cfg['rr_targets']
                tp = round(price - sl_dist * rr_targets[-1], 2)
            else:
                tp = round(price - sl_dist * self.tp_multiplier, 2)
            result = self.connector.place_order(
                self.symbol, MT5Connector.ORDER_SELL, volume,
                sl=sl, tp=tp, comment=f'{self.style}_sell{tag}'
            )
            if result.success and self._is_scalp:
                self._register_scalp_tps(result.ticket, price, volume, direction=-1)

    def _manage_positions(self, positions: list, action: float):
        """Close any position whose direction conflicts with current signal."""
        for pos in positions:
            is_long = pos.type == MT5Connector.ORDER_BUY
            flip = (is_long and action < -self.min_signal) or \
                   (not is_long and action > self.min_signal)
            if flip:
                self.connector.close_position(pos.ticket, comment='signal_flip')

    def _manage_position(self, pos, action: float):
        self._manage_positions([pos], action)

    def _register_scalp_tps(self, ticket: int, entry: float, volume: float, direction: int):
        """Store TP price levels for a newly opened scalp position."""
        sl_dist   = entry * self.stop_loss_pct
        rr_targets = self.cfg['rr_targets']
        fractions  = self.cfg['tp_fractions']
        if direction == 1:
            tp_prices = [round(entry + sl_dist * rr, 2) for rr in rr_targets]
        else:
            tp_prices = [round(entry - sl_dist * rr, 2) for rr in rr_targets]
        self._scalp_tp_info[ticket] = {
            'tp_prices':   tp_prices,
            'tp_hit':      [False, False, False],
            'initial_vol': volume,
            'fractions':   fractions,
            'direction':   direction,
            'entry':       entry,
        }
        logger.info(
            f"[SCALP] Registered TPs ticket#{ticket} | "
            f"TP1={tp_prices[0]:.2f} TP2={tp_prices[1]:.2f} TP3={tp_prices[2]:.2f}"
        )

    def _manage_scalp_tps(self, positions: list, current_price: float):
        """Check and execute partial TP exits for open scalp positions."""
        for pos in positions:
            ticket = pos.ticket
            info   = self._scalp_tp_info.get(ticket)
            if info is None:
                continue

            direction = info['direction']
            tp_prices = info['tp_prices']
            fractions = info['fractions']

            for i, (tp_price, fraction) in enumerate(zip(tp_prices, fractions)):
                if info['tp_hit'][i]:
                    continue

                tp_reached = (direction == 1 and current_price >= tp_price) or \
                             (direction == -1 and current_price <= tp_price)
                if not tp_reached:
                    continue

                info['tp_hit'][i] = True
                close_vol = round(info['initial_vol'] * fraction, 2)
                close_vol = max(close_vol, 0.01)

                result = self.connector.partial_close_position(
                    ticket, close_vol, comment=f'tp{i+1}'
                )
                if result.success:
                    rr = self.cfg['rr_targets'][i]
                    logger.info(
                        f"[SCALP] TP{i+1} RR:{rr} hit @ {current_price:.2f} | "
                        f"Closed {close_vol:.2f}lot ticket#{ticket}"
                    )
                    # TP1 hit → move SL to breakeven
                    if i == 0:
                        buf = info['entry'] * 0.0002
                        be_sl = round(
                            info['entry'] + buf if direction == 1
                            else info['entry'] - buf,
                            2
                        )
                        self.connector.modify_sl(ticket, be_sl)
                        logger.info(f"[SCALP] SL moved to breakeven {be_sl:.2f} ticket#{ticket}")

        # Clean up info for fully-closed tickets
        open_tickets = {p.ticket for p in positions}
        stale = [t for t in self._scalp_tp_info if t not in open_tickets]
        for t in stale:
            del self._scalp_tp_info[t]

    # ── Paper backtest ───────────────────────────────────────────────────────

    def run_paper_backtest(self, csv_path: str = None, split: float = 0.8) -> dict:
        csv = csv_path or self.cfg['csv']
        label = f"MT5 Paper Backtest [{self.style.upper()}]"
        logger.info("=" * 55)
        logger.info(f"  {label}")
        logger.info("=" * 55)

        df = self.connector.load_csv(csv)

        # Add scalp indicators and precompute MTF columns for paper backtest
        if self._is_scalp:
            df = _add_scalp_indicators(df)
            logger.info("Adding MTF features (M15, M30) for paper backtest ...")
            df = _add_scalp_mtf_to_df(df)

        n  = len(df)
        start = int(n * split)

        logger.info(f"Total bars: {n:,} | Test: {n-start:,}")
        logger.info(f"Test window: {df.index[start]} → {df.index[-1]}")

        self.connector.set_paper_data(df)
        self.connector._paper_idx = start

        bar_count = 0
        while True:
            bars = df.iloc[:self.connector._paper_idx + 1]
            self.on_bar(bars)
            bar_count += 1
            if not self.connector.step_paper():
                break

        self.connector.close_all_positions()
        logger.info(f"Processed {bar_count:,} test bars")

        trades  = self.connector.get_paper_trades()
        account = self.connector.get_account_info()
        results = self._summarize(trades, account)
        self._print_results(results, label)

        out_prefix = f'mt5_{self.style}_paper'
        trades.to_csv(f'{out_prefix}_trades.csv', index=False)
        with open(f'{out_prefix}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved: {out_prefix}_trades.csv | {out_prefix}_results.json")

        return results

    def _summarize(self, trades: pd.DataFrame, account) -> dict:
        init = self.connector._paper_initial_balance
        if trades.empty:
            return {'total_trades': 0, 'final_balance': round(account.balance, 2),
                    'initial_balance': init, 'return_pct': 0.0}
        wins   = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]
        return {
            'style':          self.style,
            'total_trades':   len(trades),
            'win_rate_pct':   round(len(wins) / len(trades) * 100, 2),
            'total_pnl':      round(trades['pnl'].sum(), 2),
            'avg_win':        round(wins['pnl'].mean(), 2) if len(wins) else 0,
            'avg_loss':       round(losses['pnl'].mean(), 2) if len(losses) else 0,
            'profit_factor':  round(
                wins['pnl'].sum() / abs(losses['pnl'].sum()), 2
            ) if len(losses) and losses['pnl'].sum() != 0 else float('inf'),
            'initial_balance': round(init, 2),
            'final_balance':   round(account.balance, 2),
            'return_pct':      round((account.balance - init) / init * 100, 2),
        }

    def _print_results(self, results: dict, label: str = ''):
        print("\n" + "=" * 55)
        print(f"  {label or 'Results'}")
        print("=" * 55)
        labels = {
            'total_trades':   'Total Trades',
            'win_rate_pct':   'Win Rate %',
            'total_pnl':      'Total PnL $',
            'avg_win':        'Avg Win $',
            'avg_loss':       'Avg Loss $',
            'profit_factor':  'Profit Factor',
            'initial_balance':'Initial Balance $',
            'final_balance':  'Final Balance $',
            'return_pct':     'Return %',
        }
        for key, lbl in labels.items():
            if key in results:
                print(f"  {lbl:<24} {results[key]}")
        print("=" * 55 + "\n")

    # ── Live loop ────────────────────────────────────────────────────────────

    def run_live(self):
        interval  = self.cfg['live_interval']
        timeframe = self.cfg['mt5_timeframe']
        bar_count = self.cfg['mt5_bars']

        logger.info(f"[{self.style.upper()}] Live trading started | "
                    f"Timeframe={timeframe} | Poll every {interval}s | Ctrl+C to stop")
        try:
            while True:
                df = self.connector.get_rates(self.symbol, count=bar_count,
                                              timeframe=timeframe)
                if df is not None and len(df) >= self.cfg['min_bars']:
                    if self._is_scalp:
                        df = _add_scalp_indicators(df)
                        # Fetch HTF bars for MTF confluence gate + hf_scalp obs
                        m30_raw = self.connector.get_rates(self.symbol, count=60,
                                                           timeframe='M30')
                        m15_raw = self.connector.get_rates(self.symbol, count=60,
                                                           timeframe='M15')
                        self._mtf_live_cache = (
                            _compute_htf_features(m30_raw),
                            _compute_htf_features(m15_raw),
                        )
                    self.on_bar(df)
                else:
                    logger.warning("Insufficient bars — skipping")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stop signal — closing all positions")
            self.connector.close_all_positions()
            self.connector.disconnect()
            logger.info("Shutdown complete")


# ── MT5 get_rates for M5 ─────────────────────────────────────────────────────
# Monkey-patch connector to support M5 timeframe in live mode

_original_get_rates = MT5Connector.get_rates

def _get_rates_with_tf(self, symbol='XAUUSD', count=100, timeframe='D1'):
    if self.mode == 'paper':
        return _original_get_rates(self, symbol, count)
    tf_map = {
        'D1': 'TIMEFRAME_D1',
        'M5': 'TIMEFRAME_M5',
        'M1': 'TIMEFRAME_M1',
        'H1': 'TIMEFRAME_H1',
    }
    mt5_tf = getattr(self._mt5, tf_map.get(timeframe, 'TIMEFRAME_D1'))
    rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open':'Open','high':'High','low':'Low',
                        'close':'Close','tick_volume':'Volume'}, inplace=True)
    return add_indicators(df)

MT5Connector.get_rates = _get_rates_with_tf


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MT5 Ensemble Trader — Swing & Scalp',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--mode',  choices=['paper', 'live'], default='paper')
    parser.add_argument('--style', choices=['swing', 'scalp', 'hf_scalp'], default='swing')
    parser.add_argument('--csv',      default=None, help='Override CSV data file')
    parser.add_argument('--split',    type=float, default=0.8)
    parser.add_argument('--ensemble', default=None, help='Override models directory')
    parser.add_argument('--balance',  type=float, default=None)
    parser.add_argument('--symbol',   default=None)
    parser.add_argument('--risk',     type=float, default=None)
    parser.add_argument('--login',    type=int, default=None)
    parser.add_argument('--password', default=None)
    parser.add_argument('--server',   default=None)
    args = parser.parse_args()

    login    = args.login    or int(os.getenv('MT5_LOGIN', 0))
    password = args.password or os.getenv('MT5_PASSWORD', '')
    server   = args.server   or os.getenv('MT5_SERVER', '')
    symbol   = args.symbol   or os.getenv('MT5_SYMBOL', 'XAUUSD')
    balance  = args.balance  or float(os.getenv('MT5_BALANCE', 1000.0))
    risk     = args.risk     or float(os.getenv('MT5_RISK', 0.02))

    csv_path = args.csv or STYLE_CONFIG[args.style]['csv']

    if args.mode == 'live':
        missing = [k for k, v in [('MT5_LOGIN', login), ('MT5_PASSWORD', password),
                                   ('MT5_SERVER', server)] if not v]
        if missing:
            parser.error(
                f"Live mode requires: {', '.join(missing)}\n"
                "Set them in .env or pass as --login / --password / --server."
            )

    connector = MT5Connector(
        mode=args.mode,
        csv_path=csv_path,
        initial_balance=balance,
        login=login, password=password, server=server,
    )

    trader = MT5EnsembleTrader(
        connector=connector,
        style=args.style,
        ensemble_path=args.ensemble,
        symbol=symbol,
        risk_pct=risk,
    )

    if args.mode == 'paper':
        trader.run_paper_backtest(csv_path=csv_path, split=args.split)
    else:
        trader.run_live()


if __name__ == '__main__':
    main()
