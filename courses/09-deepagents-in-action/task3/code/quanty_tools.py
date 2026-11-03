"""Deterministic calculations; no model dependency."""
import csv
import math
from datetime import date
from pathlib import Path

DATA = Path(__file__).parent / "data" / "demo.csv"


def wilder_rsi(closes: list, period: int) -> list:
    """SMA seed followed by Wilder smoothing; flat series convention is RSI=50."""
    if type(period) is not int or period < 1:
        raise ValueError("period must be a positive integer")
    if len(closes) < period + 1:
        raise ValueError(f"need at least {period + 1} closing prices")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in closes):
        raise ValueError("closing prices must be finite and positive")
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gain = sum(max(x, 0) for x in changes[:period]) / period
    loss = sum(max(-x, 0) for x in changes[:period]) / period

    def value():
        if gain == loss == 0:
            return 50.0
        if loss == 0:
            return 100.0
        return 100 - 100 / (1 + gain / loss)

    values = [value()]
    for change in changes[period:]:
        gain = (gain * (period - 1) + max(change, 0)) / period
        loss = (loss * (period - 1) + max(-change, 0)) / period
        values.append(value())
    return values


def calculate_rsi(symbol: str = "DEMO", period: int = 14) -> dict:
    """Calculate Wilder RSI from ALL available local synthetic closes, not volume.

    Only DEMO is supported, with 10 closing prices. period is a positive integer,
    defaults to 14; requires at least period+1 prices. If insufficient, return
    status=insufficient_data without inventing a value or changing the period.
    Specify period=3 or 5 explicitly for a computable teaching example.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be nonempty")
    if type(period) is not int or period < 1:
        raise ValueError("period must be a positive integer")
    symbol = symbol.strip().upper()
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["symbol"] == symbol]
    rows.sort(key=lambda r: date.fromisoformat(r["date"]))
    if not rows:
        raise ValueError("unknown symbol; available symbol: DEMO")
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError("duplicate dates")
    closes = [float(r["close"]) for r in rows]
    if any(not math.isfinite(x) or x <= 0 for x in closes):
        raise ValueError("invalid closing prices")
    result = {"symbol": symbol, "period": period, "source": "synthetic teaching fixture; NOT market data",
              "available_prices": len(rows), "required_prices": period + 1,
              "method": "Wilder: SMA seed, then (previous average*(period-1)+current)/period",
              "flat_convention": "zero gains and zero losses => RSI 50",
              "limitations": "Short synthetic history; seed-sensitive; descriptive indicator, not a trading recommendation."}
    if len(rows) < period + 1:
        return {**result, "status": "insufficient_data", "rsi": None}
    values = wilder_rsi(closes, period)
    return {**result, "status": "ok", "rsi": values[-1],
            "as_of": rows[-1]["date"],
            "series": [{"date": row["date"], "rsi": value} for row, value in zip(rows[period:], values)],
            "evidence": [{"date": r["date"], "close": c} for r, c in zip(rows, closes)]}


def analyze_price_volume(symbol: str = "DEMO", window: int = 5) -> dict:
    """Compare latest window sessions against preceding window in local synthetic data.

    Only DEMO exists. Return close-to-close return, mean-volume ratio and raw evidence.
    window must be an integer between 1 and 5. This is descriptive, not causal analysis.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be nonempty")
    if type(window) is not int or not 1 <= window <= 5:
        raise ValueError("window must be an integer between 1 and 5")
    symbol = symbol.strip().upper()
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["symbol"] == symbol]
    rows.sort(key=lambda r: date.fromisoformat(r["date"]))
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError("duplicate dates")
    if len(rows) < 2 * window:
        raise ValueError(f"insufficient data for {symbol}; available symbol: DEMO")
    rows = rows[-2 * window:]
    for row in rows:
        row["close"], row["volume"] = float(row["close"]), float(row["volume"])
        if not all(math.isfinite(row[k]) for k in ("close", "volume")):
            raise ValueError("nonfinite data")
        if row["close"] <= 0 or row["volume"] < 0:
            raise ValueError("invalid price or volume")
    baseline, recent = rows[:window], rows[window:]
    base_volume = sum(r["volume"] for r in baseline) / window
    if base_volume == 0:
        raise ValueError("baseline mean volume is zero")
    recent_volume = sum(r["volume"] for r in recent) / window
    price_return = recent[-1]["close"] / baseline[-1]["close"] - 1
    ratio = recent_volume / base_volume
    return {
        "symbol": symbol, "source": "synthetic teaching fixture; NOT market data",
        "window": window, "baseline_end": baseline[-1]["date"],
        "recent_start": recent[0]["date"], "recent_end": recent[-1]["date"],
        "price_return_pct": round(price_return * 100, 6),
        "baseline_mean_volume": base_volume, "recent_mean_volume": recent_volume,
        "volume_ratio": ratio,
        "volume_expanded": ratio > 1,
        "rise_with_volume_expansion": price_return > 0 and ratio > 1,
        "definition": "return=last recent close/last baseline close-1; expansion=mean-volume ratio>1",
        "limitations": "Descriptive window comparison only; no causality or predictive claim. No adjusted prices or market calendar checks.",
        "evidence": rows,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(analyze_price_volume(), ensure_ascii=False, indent=2))
