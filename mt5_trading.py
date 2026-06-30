#!/usr/bin/env python3
"""
MT5 Ensemble Trader
Combines the RL ensemble with a MetaTrader 5 connector.

Credentials are loaded from .env (copy .env.example → .env and fill in values).
CLI flags override .env values when provided.

Usage:
  Paper backtest (CSV data, no MT5 needed):
    python mt5_trading.py --mode paper

  Live trading (Windows + MT5 terminal):
    python mt5_trading.py --mode live

Options:
  --mode      paper | live          (default: paper)
  --csv       path to xauusd_data.csv
  --split     train/test split 0-1  (default: 0.8, paper only)
  --balance   starting balance USD  (default: MT5_BALANCE env or 1000)
  --ensemble  path to ensemble_models/ directory
  --symbol    MT5 symbol name       (default: MT5_SYMBOL env or XAUUSD)
  --risk      fraction of balance risked per trade (default: MT5_RISK env or 0.02)
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env from project root

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

MIN_BARS = 30      # bars needed before trading begins
LOOKBACK = 10      # price bars fed into ensemble observation


class MT5EnsembleTrader:
    def __init__(
        self,
        connector: MT5Connector,
        ensemble_path: str = './ensemble_models/',
        symbol: str = 'XAUUSD',
        risk_pct: float = 0.02,
        stop_loss_pct: float = 0.02,
    ):
        self.connector = connector
        self.symbol = symbol
        self.risk_pct = risk_pct
        self.stop_loss_pct = stop_loss_pct

        logger.info(f"Loading ensemble from {ensemble_path} ...")
        self.ensemble = EnsembleTrader()
        self.ensemble.load_ensemble(ensemble_path)

        self.regime_detector = MarketRegimeDetector()

        # Risk params — overwritten each bar by regime detector
        self.trailing_stop_pct = 0.025
        self.breakeven_trigger_pct = 0.015
        self.min_signal = 0.20
        self.tp_multiplier = 4.0   # TP = entry ± stop_loss_pct * tp_multiplier

    # ── Observation ─────────────────────────────────────────────────────────

    def _build_obs(self, df: pd.DataFrame) -> np.ndarray:
        """Build the 15-feature vector matching TradingEnv observation space."""
        prices = df['Close'].iloc[-LOOKBACK:].values.astype(float)
        row = df.iloc[-1]
        rsi = float(row.get('rsi', 50.0))
        macd = float(row.get('macd', 0.0))
        sig = float(row.get('signal_line', 0.0))
        # position=0 and balance=1000 are placeholders (ensemble was trained this way)
        return np.concatenate([prices, [rsi, macd, sig, 0.0, 1000.0]]).reshape(1, -1)

    # ── Regime ──────────────────────────────────────────────────────────────

    def _update_regime(self, df: pd.DataFrame):
        try:
            regime, params = self.regime_detector.detect_regime(df, len(df) - 1)
            self.trailing_stop_pct = params['trailing_stop_pct']
            self.breakeven_trigger_pct = params['breakeven_trigger']
            self.min_signal = params['min_confidence_threshold']
            logger.debug(
                f"Regime={regime.value} trail={self.trailing_stop_pct:.3f} "
                f"min_sig={self.min_signal:.2f}"
            )
        except Exception as exc:
            logger.warning(f"Regime detection skipped ({exc}), using defaults")

    # ── Position sizing ──────────────────────────────────────────────────────

    def _calc_volume(self, price: float) -> float:
        """Lots based on balance × risk_pct, floored at 0.01."""
        account = self.connector.get_account_info()
        risk_amount = account.balance * self.risk_pct
        # approx: 1 lot XAUUSD moves ~$1 per 0.01 price move; risk in price = price * stop_pct
        risk_in_price = price * self.stop_loss_pct
        volume = risk_amount / (risk_in_price * 100)   # 100 oz per lot
        return round(max(0.01, min(volume, 10.0)), 2)

    # ── Core bar handler ─────────────────────────────────────────────────────

    def on_bar(self, df: pd.DataFrame):
        """Called on every new completed bar with all bars up to now."""
        if len(df) < MIN_BARS:
            return

        self._update_regime(df)

        obs = self._build_obs(df)
        action, confidence = self.ensemble.predict_ensemble(obs)
        action = float(np.squeeze(action))
        confidence = float(np.squeeze(confidence))

        tick = self.connector.get_tick(self.symbol)
        if tick is None:
            logger.warning("No tick — skipping bar")
            return

        current_price = tick.ask
        positions = self.connector.get_positions(self.symbol)
        has_pos = len(positions) > 0

        logger.info(
            f"{self.connector.current_bar_date or 'live'} | "
            f"price={current_price:.2f} action={action:+.3f} conf={confidence:.3f} "
            f"pos={'YES' if has_pos else 'NO'}"
        )

        if not has_pos:
            self._try_entry(action, confidence, current_price)
        else:
            self._manage_position(positions[0], action, current_price)

    def _try_entry(self, action: float, confidence: float, price: float):
        if action > self.min_signal:
            volume = self._calc_volume(price)
            sl = round(price * (1 - self.stop_loss_pct), 2)
            tp = round(price * (1 + self.stop_loss_pct * self.tp_multiplier), 2)
            self.connector.place_order(
                self.symbol, MT5Connector.ORDER_BUY, volume,
                sl=sl, tp=tp, comment=f'ens_buy_c{confidence:.2f}'
            )

        elif action < -self.min_signal:
            volume = self._calc_volume(price)
            sl = round(price * (1 + self.stop_loss_pct), 2)
            tp = round(price * (1 - self.stop_loss_pct * self.tp_multiplier), 2)
            self.connector.place_order(
                self.symbol, MT5Connector.ORDER_SELL, volume,
                sl=sl, tp=tp, comment=f'ens_sell_c{confidence:.2f}'
            )

    def _manage_position(self, pos, action: float, price: float):
        """Close on opposite signal."""
        is_long = pos.type == MT5Connector.ORDER_BUY
        flip = (is_long and action < -self.min_signal) or \
               (not is_long and action > self.min_signal)
        if flip:
            self.connector.close_position(pos.ticket, comment='signal_flip')

    # ── Paper backtest ───────────────────────────────────────────────────────

    def run_paper_backtest(self, csv_path: str = 'xauusd_data.csv', split: float = 0.8) -> dict:
        logger.info("=" * 55)
        logger.info("  MT5 Paper Backtest")
        logger.info("=" * 55)

        df = self.connector.load_csv(csv_path)
        n = len(df)
        start = int(n * split)

        logger.info(f"Total bars: {n} | Training: {start} | Test: {n - start}")
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
        logger.info(f"Processed {bar_count} test bars")

        trades = self.connector.get_paper_trades()
        account = self.connector.get_account_info()
        results = self._summarize(trades, account)
        self._print_results(results)

        trades.to_csv('mt5_paper_trades.csv', index=False)
        with open('mt5_paper_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved: mt5_paper_trades.csv | mt5_paper_results.json")

        return results

    def _summarize(self, trades: pd.DataFrame, account) -> dict:
        init = self.connector._paper_initial_balance
        if trades.empty:
            return {
                'total_trades': 0,
                'final_balance': round(account.balance, 2),
                'initial_balance': init,
                'return_pct': 0.0,
            }

        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]

        return {
            'total_trades': len(trades),
            'win_rate_pct': round(len(wins) / len(trades) * 100, 2),
            'total_pnl': round(trades['pnl'].sum(), 2),
            'avg_win': round(wins['pnl'].mean(), 2) if len(wins) else 0,
            'avg_loss': round(losses['pnl'].mean(), 2) if len(losses) else 0,
            'profit_factor': round(
                wins['pnl'].sum() / abs(losses['pnl'].sum()), 2
            ) if len(losses) and losses['pnl'].sum() != 0 else float('inf'),
            'initial_balance': round(init, 2),
            'final_balance': round(account.balance, 2),
            'return_pct': round((account.balance - init) / init * 100, 2),
        }

    def _print_results(self, results: dict):
        print("\n" + "=" * 50)
        print("  MT5 Paper Backtest Results")
        print("=" * 50)
        labels = {
            'total_trades': 'Total Trades',
            'win_rate_pct': 'Win Rate %',
            'total_pnl': 'Total PnL $',
            'avg_win': 'Avg Win $',
            'avg_loss': 'Avg Loss $',
            'profit_factor': 'Profit Factor',
            'initial_balance': 'Initial Balance $',
            'final_balance': 'Final Balance $',
            'return_pct': 'Return %',
        }
        for key, label in labels.items():
            if key in results:
                print(f"  {label:<22} {results[key]}")
        print("=" * 50 + "\n")

    # ── Live loop ────────────────────────────────────────────────────────────

    def run_live(self, interval_seconds: int = 3600):
        """Live trading loop — polls MT5 every `interval_seconds` (default 1 h for daily bars)."""
        logger.info("Live MT5 trading started. Ctrl+C to stop.")
        try:
            while True:
                df = self.connector.get_rates(self.symbol, count=100)
                if df is not None and len(df) >= MIN_BARS:
                    self.on_bar(df)
                else:
                    logger.warning("Insufficient bars for decision")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stop signal received — closing all positions")
            self.connector.close_all_positions()
            self.connector.disconnect()
            logger.info("Shutdown complete")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MT5 Ensemble Trader',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--csv', default='xauusd_data.csv')
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--ensemble', default='./ensemble_models/')
    # These default to env vars; CLI flags override when explicitly passed
    parser.add_argument('--balance', type=float, default=None)
    parser.add_argument('--symbol', default=None)
    parser.add_argument('--risk', type=float, default=None)
    parser.add_argument('--login', type=int, default=None)
    parser.add_argument('--password', default=None)
    parser.add_argument('--server', default=None)
    args = parser.parse_args()

    # Resolve: CLI flag > .env variable > hard default
    login    = args.login    or int(os.getenv('MT5_LOGIN', 0))
    password = args.password or os.getenv('MT5_PASSWORD', '')
    server   = args.server   or os.getenv('MT5_SERVER', '')
    symbol   = args.symbol   or os.getenv('MT5_SYMBOL', 'XAUUSD')
    balance  = args.balance  or float(os.getenv('MT5_BALANCE', 1000.0))
    risk     = args.risk     or float(os.getenv('MT5_RISK', 0.02))

    if args.mode == 'live':
        missing = [k for k, v in [('MT5_LOGIN', login), ('MT5_PASSWORD', password), ('MT5_SERVER', server)] if not v]
        if missing:
            parser.error(
                f"Live mode requires: {', '.join(missing)}\n"
                "Set them in .env or pass as CLI flags (--login, --password, --server)."
            )

    connector = MT5Connector(
        mode=args.mode,
        csv_path=args.csv,
        initial_balance=balance,
        login=login,
        password=password,
        server=server,
    )

    trader = MT5EnsembleTrader(
        connector=connector,
        ensemble_path=args.ensemble,
        symbol=symbol,
        risk_pct=risk,
    )

    if args.mode == 'paper':
        trader.run_paper_backtest(csv_path=args.csv, split=args.split)
    else:
        trader.run_live()


if __name__ == '__main__':
    main()
