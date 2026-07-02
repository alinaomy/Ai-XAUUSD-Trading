#!/usr/bin/env python3
"""
MetaTrader 5 connector.

mode='paper' — simulates MT5 execution using local CSV data (works on any OS)
mode='live'  — connects to a real MT5 terminal (Windows only, requires MetaTrader5 package)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float
    currency: str = 'USD'


@dataclass
class SymbolTick:
    symbol: str
    bid: float
    ask: float
    time: datetime


@dataclass
class Position:
    ticket: int
    symbol: str
    type: int        # 0 = buy, 1 = sell
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    comment: str = ""


@dataclass
class TradeResult:
    success: bool
    ticket: int = 0
    comment: str = ""


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger Bands, Stochastic on OHLCV dataframe."""
    df = df.copy()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['sma20'] = df['Close'].rolling(20).mean()
    df['std20'] = df['Close'].rolling(20).std()
    df['upper_band'] = df['sma20'] + df['std20'] * 2
    df['lower_band'] = df['sma20'] - df['std20'] * 2

    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    return df.dropna()


class MT5Connector:
    ORDER_BUY = 0
    ORDER_SELL = 1

    def __init__(
        self,
        mode: str = 'paper',
        csv_path: str = 'xauusd_data.csv',
        initial_balance: float = 1000.0,
        login: int = 0,
        password: str = '',
        server: str = '',
    ):
        if mode not in ('paper', 'live'):
            raise ValueError("mode must be 'paper' or 'live'")

        self.mode = mode
        self._positions: Dict[int, Position] = {}
        self._ticket_counter = 1

        if mode == 'live':
            self._init_live(login, password, server)
        else:
            self._paper_balance = initial_balance
            self._paper_initial_balance = initial_balance
            self._paper_data: Optional[pd.DataFrame] = None
            self._paper_idx: int = 0
            self._paper_csv_path = csv_path
            self._paper_trades: List[dict] = []
            logger.info(f"Paper mode | balance=${initial_balance:.2f}")

    # ── Connection ───────────────────────────────────────────────────────────

    def _init_live(self, login: int, password: str, server: str):
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise ImportError(
                "MetaTrader5 package not found.\n"
                "Install on Windows: pip install MetaTrader5\n"
                "Note: live mode only works on Windows with MT5 terminal running."
            )
        self._mt5 = mt5
        if not mt5.initialize(login=login, password=password, server=server):
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
        logger.info(f"Connected to MT5 terminal: {mt5.terminal_info().name}")

    def connect(self) -> bool:
        if self.mode == 'paper':
            return True
        return self._mt5.initialize()

    def disconnect(self):
        if self.mode == 'live':
            self._mt5.shutdown()
            logger.info("Disconnected from MT5")

    # ── Data ─────────────────────────────────────────────────────────────────

    def load_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load and prepare CSV data (adds indicators). Used for paper mode."""
        path = csv_path or self._paper_csv_path
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        return add_indicators(df)

    def get_rates(self, symbol: str = 'XAUUSD', count: int = 100) -> Optional[pd.DataFrame]:
        """Return last `count` bars.

        Paper mode: slices from loaded CSV up to the current bar index.
        Live mode:  fetches D1 bars from MT5 terminal.
        """
        if self.mode == 'paper':
            if self._paper_data is None:
                raise RuntimeError("Call set_paper_data() before get_rates().")
            start = max(0, self._paper_idx - count + 1)
            return self._paper_data.iloc[start:self._paper_idx + 1].copy()

        rates = self._mt5.copy_rates_from_pos(symbol, self._mt5.TIMEFRAME_D1, 0, count)
        if rates is None:
            logger.error(f"MT5 get_rates failed: {self._mt5.last_error()}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        return add_indicators(df)

    def get_tick(self, symbol: str = 'XAUUSD') -> Optional[SymbolTick]:
        if self.mode == 'paper':
            if self._paper_data is None:
                raise RuntimeError("Paper data not loaded.")
            price = float(self._paper_data.iloc[self._paper_idx]['Close'])
            spread = price * 0.0001
            return SymbolTick(symbol=symbol, bid=price - spread, ask=price + spread, time=datetime.now())

        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return SymbolTick(symbol=symbol, bid=tick.bid, ask=tick.ask,
                          time=datetime.fromtimestamp(tick.time))

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        if self.mode == 'paper':
            unrealized = 0.0
            if self._paper_data is not None:
                price = float(self._paper_data.iloc[self._paper_idx]['Close'])
                for pos in self._positions.values():
                    if pos.type == self.ORDER_BUY:
                        unrealized += (price - pos.price_open) * pos.volume
                    else:
                        unrealized += (pos.price_open - price) * pos.volume
            equity = self._paper_balance + unrealized
            return AccountInfo(balance=self._paper_balance, equity=equity,
                               margin=0.0, free_margin=equity, profit=unrealized)

        info = self._mt5.account_info()
        return AccountInfo(balance=info.balance, equity=info.equity,
                           margin=info.margin, free_margin=info.margin_free,
                           profit=info.profit, currency=info.currency)

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        order_type: int,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = '',
    ) -> TradeResult:
        if self.mode == 'paper':
            return self._paper_place(symbol, order_type, volume, sl, tp, comment)

        tick = self._mt5.symbol_info_tick(symbol)
        price = tick.ask if order_type == self.ORDER_BUY else tick.bid
        request = {
            'action': self._mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(volume),
            'type': order_type,
            'price': price,
            'sl': sl,
            'tp': tp,
            'comment': comment,
            'type_time': self._mt5.ORDER_TIME_GTC,
            'type_filling': self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        if result.retcode == self._mt5.TRADE_RETCODE_DONE:
            direction = 'BUY' if order_type == self.ORDER_BUY else 'SELL'
            logger.info(f"[MT5] {direction} {volume} {symbol} @ {price:.2f} ticket#{result.order}")
            return TradeResult(success=True, ticket=result.order, comment=result.comment)
        logger.error(f"[MT5] Order failed: retcode={result.retcode} {result.comment}")
        return TradeResult(success=False, comment=result.comment)

    def _paper_place(self, symbol, order_type, volume, sl, tp, comment) -> TradeResult:
        tick = self.get_tick(symbol)
        price = tick.ask if order_type == self.ORDER_BUY else tick.bid
        ticket = self._ticket_counter
        self._ticket_counter += 1
        self._positions[ticket] = Position(
            ticket=ticket, symbol=symbol, type=order_type,
            volume=volume, price_open=price, sl=sl, tp=tp, profit=0.0, comment=comment,
        )
        direction = 'BUY' if order_type == self.ORDER_BUY else 'SELL'
        logger.info(f"[PAPER] {direction} {volume:.4f} {symbol} @ {price:.2f} (ticket #{ticket})")
        return TradeResult(success=True, ticket=ticket)

    def close_position(self, ticket: int, comment: str = '') -> TradeResult:
        if ticket not in self._positions:
            return TradeResult(success=False, comment=f"Ticket {ticket} not found")

        pos = self._positions[ticket]

        if self.mode == 'paper':
            return self._paper_close(ticket, pos, comment)

        tick = self._mt5.symbol_info_tick(pos.symbol)
        close_type = self.ORDER_SELL if pos.type == self.ORDER_BUY else self.ORDER_BUY
        price = tick.bid if pos.type == self.ORDER_BUY else tick.ask
        request = {
            'action': self._mt5.TRADE_ACTION_DEAL,
            'symbol': pos.symbol,
            'volume': pos.volume,
            'type': close_type,
            'position': ticket,
            'price': price,
            'comment': comment,
            'type_time': self._mt5.ORDER_TIME_GTC,
            'type_filling': self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        if result.retcode == self._mt5.TRADE_RETCODE_DONE:
            del self._positions[ticket]
            return TradeResult(success=True, ticket=ticket)
        logger.error(f"[MT5] Close failed: {result.retcode} {result.comment}")
        return TradeResult(success=False, comment=result.comment)

    def _paper_close(self, ticket: int, pos: Position, comment: str) -> TradeResult:
        tick = self.get_tick(pos.symbol)
        close_price = tick.bid if pos.type == self.ORDER_BUY else tick.ask

        if pos.type == self.ORDER_BUY:
            pnl = (close_price - pos.price_open) * pos.volume
        else:
            pnl = (pos.price_open - close_price) * pos.volume

        self._paper_balance += pnl
        bar_date = str(self._paper_data.index[self._paper_idx]) if self._paper_data is not None else ''
        self._paper_trades.append({
            'ticket': ticket,
            'symbol': pos.symbol,
            'type': 'BUY' if pos.type == self.ORDER_BUY else 'SELL',
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_close': close_price,
            'pnl': round(pnl, 2),
            'comment': comment,
            'date_close': bar_date,
        })
        del self._positions[ticket]
        logger.info(
            f"[PAPER] CLOSE #{ticket} {'BUY' if pos.type==0 else 'SELL'} "
            f"@ {close_price:.2f} | PnL=${pnl:+.2f} | Balance=${self._paper_balance:.2f}"
        )
        return TradeResult(success=True, ticket=ticket)

    def partial_close_position(self, ticket: int, volume: float, comment: str = '') -> TradeResult:
        """Close a partial volume of an open position."""
        if ticket not in self._positions:
            return TradeResult(success=False, comment=f"Ticket {ticket} not found")

        pos = self._positions[ticket]
        volume = round(min(volume, pos.volume), 2)

        if volume <= 0:
            return TradeResult(success=False, comment="Volume must be > 0")
        if volume >= pos.volume:
            return self.close_position(ticket, comment)

        if self.mode == 'paper':
            tick = self.get_tick(pos.symbol)
            close_price = tick.bid if pos.type == self.ORDER_BUY else tick.ask
            pnl = ((close_price - pos.price_open) if pos.type == self.ORDER_BUY
                   else (pos.price_open - close_price)) * volume
            self._paper_balance += pnl
            self._positions[ticket].volume = round(pos.volume - volume, 2)
            self._paper_trades.append({
                'ticket': ticket, 'symbol': pos.symbol,
                'type': 'BUY' if pos.type == self.ORDER_BUY else 'SELL',
                'volume': volume, 'price_open': pos.price_open,
                'price_close': close_price, 'pnl': round(pnl, 2),
                'comment': f'partial_{comment}', 'date_close': self.current_bar_date,
            })
            logger.info(
                f"[PAPER] PARTIAL #{ticket} {volume:.2f}lot @ {close_price:.2f} | "
                f"PnL=${pnl:+.2f} | remaining={self._positions[ticket].volume:.2f}"
            )
            return TradeResult(success=True, ticket=ticket)

        tick = self._mt5.symbol_info_tick(pos.symbol)
        close_type = self.ORDER_SELL if pos.type == self.ORDER_BUY else self.ORDER_BUY
        price = tick.bid if pos.type == self.ORDER_BUY else tick.ask
        request = {
            'action': self._mt5.TRADE_ACTION_DEAL,
            'symbol': pos.symbol,
            'volume': float(volume),
            'type': close_type,
            'position': ticket,
            'price': price,
            'comment': f'partial_{comment}',
            'type_time': self._mt5.ORDER_TIME_GTC,
            'type_filling': self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        if result.retcode == self._mt5.TRADE_RETCODE_DONE:
            if ticket in self._positions:
                self._positions[ticket].volume = round(pos.volume - volume, 2)
                if self._positions[ticket].volume <= 0:
                    del self._positions[ticket]
            logger.info(f"[MT5] PARTIAL close #{ticket} {volume:.2f}lot @ {price:.2f}")
            return TradeResult(success=True, ticket=ticket)
        logger.error(f"[MT5] Partial close failed: {result.retcode} {result.comment}")
        return TradeResult(success=False, comment=result.comment)

    def modify_sl(self, ticket: int, sl: float) -> TradeResult:
        """Modify the stop-loss of an open position (e.g. move to breakeven)."""
        if ticket not in self._positions:
            return TradeResult(success=False, comment=f"Ticket {ticket} not found")

        if self.mode == 'paper':
            self._positions[ticket].sl = sl
            logger.info(f"[PAPER] Modified SL #{ticket} → {sl:.2f}")
            return TradeResult(success=True, ticket=ticket)

        pos = self._positions[ticket]
        request = {
            'action': self._mt5.TRADE_ACTION_SLTP,
            'symbol': pos.symbol,
            'position': ticket,
            'sl': sl,
            'tp': pos.tp,
        }
        result = self._mt5.order_send(request)
        if result.retcode == self._mt5.TRADE_RETCODE_DONE:
            self._positions[ticket].sl = sl
            logger.info(f"[MT5] SL modified #{ticket} → {sl:.2f}")
            return TradeResult(success=True, ticket=ticket)
        logger.error(f"[MT5] Modify SL failed: {result.retcode} {result.comment}")
        return TradeResult(success=False, comment=result.comment)

    def close_all_positions(self):
        for ticket in list(self._positions.keys()):
            self.close_position(ticket, comment='close_all')

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if self.mode == 'paper':
            result = list(self._positions.values())
            return [p for p in result if p.symbol == symbol] if symbol else result

        raw = self._mt5.positions_get(symbol=symbol) if symbol else self._mt5.positions_get()
        if not raw:
            return []
        return [Position(ticket=p.ticket, symbol=p.symbol, type=p.type,
                         volume=p.volume, price_open=p.price_open,
                         sl=p.sl, tp=p.tp, profit=p.profit, comment=p.comment)
                for p in raw]

    # ── Paper helpers ────────────────────────────────────────────────────────

    def set_paper_data(self, df: pd.DataFrame):
        self._paper_data = df
        self._paper_idx = 0

    def step_paper(self) -> bool:
        """Advance one bar. Returns False when all bars are consumed."""
        if self._paper_data is None:
            return False
        self._paper_idx += 1
        return self._paper_idx < len(self._paper_data)

    def get_paper_trades(self) -> pd.DataFrame:
        return pd.DataFrame(self._paper_trades)

    @property
    def current_bar_index(self) -> int:
        return self._paper_idx

    @property
    def current_bar_date(self) -> str:
        if self.mode == 'live':
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not hasattr(self, '_paper_data') or self._paper_data is None:
            return ''
        return str(self._paper_data.index[self._paper_idx])
