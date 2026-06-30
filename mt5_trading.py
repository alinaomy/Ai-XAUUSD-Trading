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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from mt5_connector import MT5Connector, add_indicators
from ensemble_trader import EnsembleTrader
from market_regime_detector import MarketRegimeDetector

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
        'live_interval': 3600,     # poll every hour (D1 bar)
        'mt5_bars':      100,
    },
    'scalp': {
        'csv':           'xauusd_m5_data.csv',
        'ensemble_path': './scalp_models/',
        'config_file':   'scalp_config.json',
        'min_bars':      30,
        'lookback':      20,
        'stop_loss_pct': 0.0015,
        'tp_multiplier': 3.0,
        'min_signal':    0.15,
        'mt5_timeframe': 'M5',
        'live_interval': 300,      # poll every 5 minutes (M5 bar)
        'mt5_bars':      100,
    },
}


# ── Scalp observation builder ─────────────────────────────────────────────────

def _build_scalp_obs(df: pd.DataFrame) -> np.ndarray:
    """28-feature observation matching ScalpTradingEnv."""
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
        0.0,   # position placeholder
        0.0,   # time-in-trade placeholder
    ], dtype=np.float32)
    return np.concatenate([prices.astype(np.float32), indicators]).reshape(1, -1)


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
        self.tp_multiplier    = self.cfg['tp_multiplier']
        self.min_signal       = self.cfg['min_signal']
        self.trailing_stop_pct = self.stop_loss_pct  # updated by regime for swing

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self, df: pd.DataFrame) -> np.ndarray:
        if self.style == 'scalp':
            return _build_scalp_obs(df)
        return _build_swing_obs(df)

    # ── Regime (swing only) ──────────────────────────────────────────────────

    def _update_regime(self, df: pd.DataFrame):
        if self.style != 'swing':
            return
        try:
            regime, params = self.regime_detector.detect_regime(df, len(df) - 1)
            self.trailing_stop_pct = params['trailing_stop_pct']
            self.min_signal        = params['min_confidence_threshold']
            logger.debug(f"Regime={regime.value}")
        except Exception as exc:
            logger.debug(f"Regime skipped: {exc}")

    # ── Position sizing ──────────────────────────────────────────────────────

    # Scalp risk scales down with each additional position to cap total exposure
    SCALP_RISK_SCALE = [1.0, 0.75, 0.5]   # 1st=100%, 2nd=75%, 3rd=50% of risk_pct

    def _calc_volume(self, price: float, pos_count: int = 0) -> float:
        account = self.connector.get_account_info()
        risk_pct = self.risk_pct
        if self.style == 'scalp':
            # Cap base risk at 1% for scalp, then scale down per additional position
            base_risk = min(self.risk_pct, 0.01)
            scale     = self.SCALP_RISK_SCALE[min(pos_count, 2)]
            risk_pct  = base_risk * scale
        risk_amount   = account.balance * risk_pct
        risk_in_price = price * self.stop_loss_pct
        volume = risk_amount / (risk_in_price * 100)
        logger.debug(
            f"[SIZING] pos#{pos_count+1} risk={risk_pct*100:.2f}% "
            f"→ ${risk_amount:.2f} risk → {volume:.2f} lots"
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

        obs = self._build_obs(df)
        action, confidence = self.ensemble.predict_ensemble(obs)
        action     = float(np.squeeze(action))
        confidence = float(np.squeeze(confidence))

        tick = self.connector.get_tick(self.symbol)
        if tick is None:
            logger.warning("No tick — skipping bar")
            return

        current_price = tick.ask

        # Only count/manage positions this bot opened (identified by comment prefix)
        # This prevents interfering with manual trades or other strategies
        all_positions = self.connector.get_positions(self.symbol)
        positions     = self._own_positions(all_positions)
        n_pos         = len(positions)

        logger.info(
            f"[{self.style.upper()}] {self.connector.current_bar_date} | "
            f"price={current_price:.2f} action={action:+.3f} conf={confidence:.3f} "
            f"pos={n_pos} (total_mt5={len(all_positions)})"
        )

        # Manage only bot-owned positions — close any that get a signal flip
        if positions:
            self._manage_positions(positions, action)
            all_positions = self.connector.get_positions(self.symbol)
            positions     = self._own_positions(all_positions)
            n_pos         = len(positions)

        # Entry: scalp allows up to 3 with increasing confluence, swing stays 1
        max_pos = self.MAX_SCALP_POSITIONS if self.style == 'scalp' else 1
        if n_pos < max_pos:
            self._try_entry(action, confidence, current_price, n_pos, positions)

    def _try_entry(self, action: float, confidence: float, price: float,
                   pos_count: int = 0, existing: list = None):
        existing = existing or []

        # For 2nd and 3rd scalp trades require higher confidence
        if self.style == 'scalp' and pos_count > 0:
            required = self.SCALP_CONF_THRESHOLDS[min(pos_count, 2)]
            if confidence < required:
                logger.debug(
                    f"[SCALP] Skip entry #{pos_count+1}: "
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

        tag = f"#{pos_count+1}" if self.style == 'scalp' else ""

        if action > self.min_signal:
            volume = self._calc_volume(price, pos_count)
            sl = round(price * (1 - self.stop_loss_pct), 2)
            tp = round(price * (1 + self.stop_loss_pct * self.tp_multiplier), 2)
            self.connector.place_order(
                self.symbol, MT5Connector.ORDER_BUY, volume,
                sl=sl, tp=tp, comment=f'{self.style}_buy{tag}'
            )
        elif action < -self.min_signal:
            volume = self._calc_volume(price, pos_count)
            sl = round(price * (1 + self.stop_loss_pct), 2)
            tp = round(price * (1 - self.stop_loss_pct * self.tp_multiplier), 2)
            self.connector.place_order(
                self.symbol, MT5Connector.ORDER_SELL, volume,
                sl=sl, tp=tp, comment=f'{self.style}_sell{tag}'
            )

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

    # ── Paper backtest ───────────────────────────────────────────────────────

    def run_paper_backtest(self, csv_path: str = None, split: float = 0.8) -> dict:
        csv = csv_path or self.cfg['csv']
        label = f"MT5 Paper Backtest [{self.style.upper()}]"
        logger.info("=" * 55)
        logger.info(f"  {label}")
        logger.info("=" * 55)

        df = self.connector.load_csv(csv)
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
        tf_label  = self.cfg['mt5_timeframe']
        bar_count = self.cfg['mt5_bars']

        logger.info(f"[{self.style.upper()}] Live trading started | "
                    f"Timeframe={tf_label} | Poll every {interval}s | Ctrl+C to stop")
        try:
            while True:
                df = self.connector.get_rates(self.symbol, count=bar_count)
                if df is not None and len(df) >= self.cfg['min_bars']:
                    # Add scalp indicators if needed (swing indicators added in get_rates)
                    if self.style == 'scalp':
                        df = _add_scalp_indicators(df)
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
    parser.add_argument('--style', choices=['swing', 'scalp'], default='swing')
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
