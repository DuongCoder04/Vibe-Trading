"""Regression: the vol that prices option legs must annualise on the run's cadence.

``run_options_backtest`` already receives the runner-resolved ``bars_per_year``
and forwards it to ``_calc_options_metrics``, where ``None`` is resolved through
``backtest.metrics.effective_bars_per_year`` -- the shared span-derived factor
whose own docstring promises that "the Sharpe, the annualised volatility, and
the validation Sharpe in a single run card are annualised identically".

``historical_volatility`` is that annualised volatility for the options engine:
it produces the per-bar vol every leg is opened, marked and greeked at. It
annualised at a hardcoded ``np.sqrt(252)`` and its call site never forwarded the
factor the run already knew, so a venue that does not trade 252 bars a year
priced its legs on one cadence and reported its metrics on another. Every crypto
source maps to 365 daily bars, and okx hourly bars to 8760, so an at-the-money
30-day call was marked roughly 16% too cheap on a daily crypto run and far worse
intraday -- silently, into ``trades.csv``, ``greeks.csv``, ``equity.csv``,
``metrics.csv`` and the ``backtest_summary`` tool result.

Same convention as ``8afa3879`` ("fix(portfolio): annualize weekly and monthly
risk bars correctly"), which resolved the risk x-ray's hardcoded 252 through
``calc_bars_per_year``; this is the options engine's copy of that bug.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engines.options_portfolio import historical_volatility, run_options_backtest
from src.quantlib.options import bs_price

DEFAULT_IV = 0.3
WINDOW = 30


def _annualised_hv(close: pd.Series, bars_per_year: float) -> pd.Series:
    """Independent oracle: rolling log-return std scaled by ``sqrt(bars/yr)``."""
    log_ret = np.log(close / close.shift(1))
    scaled = log_ret.rolling(window=WINDOW).std() * math.sqrt(bars_per_year)
    return scaled.fillna(DEFAULT_IV)


def _alternating_close(bars: int = 60) -> pd.Series:
    """Deterministic zigzag whose 30-bar rolling std is non-zero and stable."""
    steps = np.where(np.arange(bars) % 2 == 0, 0.02, -0.015)
    index = pd.date_range("2024-01-01", periods=bars, freq="D")
    return pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)


def test_hv_annualises_on_the_supplied_cadence() -> None:
    """A 365-bar venue must scale by sqrt(365), not the equity-market 252."""
    close = _alternating_close()

    hv = historical_volatility(close, default_iv=DEFAULT_IV, bars_per_year=365)

    pd.testing.assert_series_equal(hv, _annualised_hv(close, 365))
    # The whole point: the same series on another cadence is another volatility.
    assert not hv.equals(historical_volatility(close, default_iv=DEFAULT_IV, bars_per_year=252))


def test_hv_defaults_to_252_so_daily_equity_runs_are_unchanged() -> None:
    """The default keeps every existing caller and the pinned warm-up test."""
    close = _alternating_close()

    pd.testing.assert_series_equal(
        historical_volatility(close),
        _annualised_hv(close, 252),
    )
    pd.testing.assert_series_equal(
        historical_volatility(close),
        historical_volatility(close, bars_per_year=252),
    )


def test_hv_none_resolves_from_the_observed_span() -> None:
    """``None`` means cross-market: measure the cadence, as the metrics do.

    The runner passes ``bars_per_year=None`` for a basket that spans markets
    (#1239). Restated here from the documented convention rather than by
    calling the helper, so the test pins the convention and not the code.
    """
    close = _alternating_close()
    span_days = (close.index[-1] - close.index[0]).days
    expected_factor = int(len(close) / (span_days / 365.25))
    assert expected_factor != 252, "fixture must exercise a non-default factor"

    hv = historical_volatility(close, default_iv=DEFAULT_IV, bars_per_year=None)

    pd.testing.assert_series_equal(hv, _annualised_hv(close, expected_factor))


class _SingleCodeLoader:
    """Serves one deterministic daily series, ignoring the requested window."""

    name = "okx"

    def __init__(self, close: pd.Series) -> None:
        self._close = close

    def fetch(self, codes, start_date, end_date):  # noqa: ANN001
        frame = pd.DataFrame(
            {
                "close": self._close.to_numpy(),
                "open": self._close.to_numpy(),
            },
            index=self._close.index,
        )
        return {code: frame for code in codes}


class _OpenOneCallEngine:
    """Opens one at-the-money call, dated so it fills on the next bar."""

    def __init__(self, signal_date: str, strike: float, expiry: str) -> None:
        self._signal_date = signal_date
        self._strike = strike
        self._expiry = expiry

    def generate(self, data_map):  # noqa: ANN001
        return [
            {
                "date": self._signal_date,
                "action": "open",
                "underlying": "BTC-USDT",
                "legs": [
                    {
                        "type": "call",
                        "strike": self._strike,
                        "expiry": self._expiry,
                        "qty": 1,
                    }
                ],
            }
        ]


def _run_one_call(tmp_path: Path, bars_per_year: int | None) -> tuple[float, float, float]:
    """Open one ATM 30-day call and return (recorded price, spot, sigma at fill)."""
    close = _alternating_close()
    signal_bar = close.index[40]
    fill_bar = close.index[41]
    strike = float(close.at[fill_bar])
    expiry = str((fill_bar + pd.Timedelta(days=30)).date())

    result = run_options_backtest(
        {
            "codes": ["BTC-USDT"],
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "source": "okx",
            "engine": "options",
            "initial_cash": 1_000_000.0,
            "options_config": {
                "risk_free_rate": 0.05,
                "default_iv": DEFAULT_IV,
                "margin_enabled": False,
            },
        },
        _SingleCodeLoader(close),
        _OpenOneCallEngine(str(signal_bar.date()), strike, expiry),
        tmp_path,
        bars_per_year=bars_per_year,
    )

    trades = pd.read_csv(tmp_path / "artifacts" / "trades.csv")
    opens = trades[trades["side"] == "buy"]
    assert len(opens) == 1, f"expected exactly one fill, got {trades.to_dict('records')}"
    assert result["trade_count"] == 1
    return float(opens.iloc[0]["price"]), strike, float(close.at[fill_bar])


def test_options_backtest_prices_the_leg_on_the_run_cadence(tmp_path: Path) -> None:
    """End to end: a 365-bar run must fill at the 365-annualised vol.

    This is the defect the unit tests above cannot catch on their own -- the
    factor has to survive the call site, not just exist as a parameter.
    """
    close = _alternating_close()
    fill_bar = close.index[41]

    price, strike, spot = _run_one_call(tmp_path, bars_per_year=365)

    sigma_365 = float(_annualised_hv(close, 365).at[fill_bar])
    sigma_252 = float(_annualised_hv(close, 252).at[fill_bar])
    expiry_years = 30 / 365.0
    expected = bs_price(spot, strike, expiry_years, 0.05, sigma_365, "call")
    stale = bs_price(spot, strike, expiry_years, 0.05, sigma_252, "call")

    assert price == pytest.approx(expected, abs=1e-4)
    # Quantifies what the hardcoded factor cost: the stale mark sits ~16% below
    # the price this leg should have filled at, and that number is what reached
    # trades.csv, greeks.csv and the run card.
    assert expected > stale
    assert (expected - stale) / expected == pytest.approx(0.16, abs=0.01)


def test_options_backtest_cross_market_none_prices_on_the_observed_span(
    tmp_path: Path,
) -> None:
    """``bars_per_year=None`` must reach pricing too, not just the metrics."""
    close = _alternating_close()
    fill_bar = close.index[41]
    span_days = (close.index[-1] - close.index[0]).days
    factor = int(len(close) / (span_days / 365.25))

    price, strike, spot = _run_one_call(tmp_path, bars_per_year=None)

    sigma = float(_annualised_hv(close, factor).at[fill_bar])
    expected = bs_price(spot, strike, 30 / 365.0, 0.05, sigma, "call")
    assert price == pytest.approx(expected, abs=1e-4)
