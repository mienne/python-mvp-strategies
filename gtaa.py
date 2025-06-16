# -*- coding: utf-8 -*-

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import seaborn as sns
import warnings

from matplotlib.backends.backend_pdf import PdfPages

# Import required libraries for missing functions

import os
import shutil
import tempfile
import sys

sns.set()
warnings.filterwarnings("ignore")


def remove_market_holidays(df, market="NYSE"):
    try:
        cal = mcal.get_calendar(market)
        holidays = cal.holidays().holidays
        df = df[~df.index.isin(holidays)]
        return df
    except:
        return df[df.index.weekday < 5]


def load_gtaa_data(data_path, etf_list, start_date, end_date):
    etf_data = {}

    for etf in etf_list:
        try:

            filename = f"{data_path}/{etf}_data.csv"
            if not os.path.exists(filename):
                continue

            df = pd.read_csv(filename)

            date_cols = ["Date", "DateTime", "Date-Time", "date", "datetime"]
            date_col = next((col for col in date_cols if col in df.columns), None)

            if date_col is None:
                continue

            df["Date"] = pd.to_datetime(df[date_col])
            df.set_index("Date", inplace=True)

            column_mapping = {
                "close": "Close",
                "Close": "Close",
                "open": "Open",
                "Open": "Open",
                "high": "High",
                "High": "High",
                "low": "Low",
                "Low": "Low",
                "volume": "Volume",
                "Volume": "Volume",
                "Last": "Close",
            }

            df.rename(columns=column_mapping, inplace=True)

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    if col == "Volume":
                        df[col] = 1
                    else:
                        continue

            df = df.loc[start_date:end_date]

            if len(df) < 200:
                pass

            df[["Open", "High", "Low", "Close"]].fillna(method="ffill", inplace=True)
            df["Volume"].fillna(1, inplace=True)

            df = remove_market_holidays(df, market="NYSE")
            df = df[df.index.weekday < 5]

            etf_data[etf] = df

        except Exception as e:
            continue

    return etf_data


def calculate_gtaa_signals(etf_data, sma_period=200, rebalance_freq="M", signal_lag=1):
    signals_dict = {}
    cash_etf = "SHV"

    for etf, df in etf_data.items():
        if etf == cash_etf:
            signals_dict[etf] = pd.Series(1, index=df.index, name=f"{etf}_signal")
            continue

        df[f"SMA_{sma_period}"] = (
            df["Close"].rolling(window=sma_period, min_periods=sma_period).mean()
        )

        df["Signal_Raw"] = (df["Close"] > df[f"SMA_{sma_period}"]).astype(int)

        if rebalance_freq == "M":
            monthly_data = df.resample("M").last()
            monthly_signals = monthly_data["Signal_Raw"]

            if signal_lag > 0:
                monthly_signals = monthly_signals.shift(signal_lag)

            daily_signals = monthly_signals.reindex(df.index, method="ffill")

        elif rebalance_freq == "W":
            weekly_data = df.resample("W").last()
            weekly_signals = weekly_data["Signal_Raw"]

            if signal_lag > 0:
                weekly_signals = weekly_signals.shift(signal_lag)

            daily_signals = weekly_signals.reindex(df.index, method="ffill")

        else:
            daily_signals = df["Signal_Raw"]
            if signal_lag > 0:
                daily_signals = daily_signals.shift(signal_lag)

        daily_signals = daily_signals.fillna(0).astype(int)

        signals_dict[etf] = daily_signals

        total_days = len(daily_signals)
        buy_days = (daily_signals == 1).sum()
        signal_changes = (daily_signals.diff() != 0).sum()

    return signals_dict


def calculate_gtaa_weights(signals_dict, etf_weight=0.20):
    cash_etf = "SHV"

    date_indices = [signals.index for signals in signals_dict.values()]
    common_dates = date_indices[0]
    for idx in date_indices[1:]:
        common_dates = common_dates.intersection(idx)

    target_weights_dict = {}

    for date in common_dates:
        daily_weights = {}
        active_etfs = 0

        for etf, signals in signals_dict.items():
            if etf != cash_etf and date in signals.index:
                if signals.loc[date] == 1:
                    active_etfs += 1

        total_etf_weight = 0
        for etf, signals in signals_dict.items():
            if etf == cash_etf:
                continue
            elif date in signals.index and signals.loc[date] == 1:
                daily_weights[etf] = etf_weight
                total_etf_weight += etf_weight
            else:
                daily_weights[etf] = 0.0

        daily_weights[cash_etf] = max(0.0, 1.0 - total_etf_weight)
        target_weights_dict[date] = daily_weights

    weights_df = pd.DataFrame(target_weights_dict).T

    for etf in signals_dict.keys():
        if etf not in weights_df.columns:
            weights_df[etf] = 0.0

    return weights_df


def backtest_gtaa_strategy(
    etf_data,
    signals_dict,
    target_weights,
    initial_capital=1000000,
    transaction_cost=0.003,
):
    portfolio_returns = []
    portfolio_values = [initial_capital]
    trade_log = []

    current_weights = {etf: 0.0 for etf in target_weights.columns}

    for i, date in enumerate(target_weights.index):
        if i == 0:
            current_weights = target_weights.iloc[0].to_dict()
            daily_return = 0.0
            transaction_cost_today = 0.0

            for etf, weight in current_weights.items():
                if weight > 0:
                    trade_log.append(
                        {
                            "date": date,
                            "etf": etf,
                            "action": "INITIAL",
                            "weight_from": 0.0,
                            "weight_to": weight,
                            "price": (
                                etf_data[etf].loc[date, "Close"]
                                if date in etf_data[etf].index
                                else np.nan
                            ),
                        }
                    )
        else:
            target_weights_today = target_weights.iloc[i].to_dict()
            rebalance_needed = _needs_rebalancing(current_weights, target_weights_today)

            transaction_cost_today = 0.0
            if rebalance_needed:
                turnover = sum(
                    abs(target_weights_today[etf] - current_weights[etf])
                    for etf in current_weights.keys()
                )
                transaction_cost_today = turnover * transaction_cost

                for etf in current_weights.keys():
                    weight_change = target_weights_today[etf] - current_weights[etf]
                    if abs(weight_change) > 0.001:
                        trade_log.append(
                            {
                                "date": date,
                                "etf": etf,
                                "action": "BUY" if weight_change > 0 else "SELL",
                                "weight_from": current_weights[etf],
                                "weight_to": target_weights_today[etf],
                                "price": (
                                    etf_data[etf].loc[date, "Close"]
                                    if date in etf_data[etf].index
                                    else np.nan
                                ),
                                "turnover": abs(weight_change),
                                "cost": abs(weight_change) * transaction_cost,
                            }
                        )

                current_weights = target_weights_today.copy()

            etf_returns = {}
            prev_date = target_weights.index[i - 1]

            for etf in current_weights.keys():
                if date in etf_data[etf].index and prev_date in etf_data[etf].index:
                    etf_returns[etf] = (
                        etf_data[etf].loc[date, "Close"]
                        / etf_data[etf].loc[prev_date, "Close"]
                        - 1
                    )
                else:
                    etf_returns[etf] = 0.0

            portfolio_return_gross = sum(
                current_weights[etf] * etf_returns[etf]
                for etf in current_weights.keys()
            )

            daily_return = portfolio_return_gross - transaction_cost_today

        portfolio_returns.append(daily_return)
        new_portfolio_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_portfolio_value)

    returns_series = pd.Series(portfolio_returns, index=target_weights.index)
    values_series = pd.Series(portfolio_values[1:], index=target_weights.index)
    trade_df = pd.DataFrame(trade_log)

    return returns_series, values_series, trade_df


def _needs_rebalancing(current_weights, target_weights, threshold=0.01):
    for etf in current_weights.keys():
        weight_diff = abs(target_weights.get(etf, 0) - current_weights[etf])
        if weight_diff > threshold:
            return True
    return False


def implement_tranche_strategy(etf_data, signals_dict, num_tranches=4):
    cash_etf = "SHV"
    tranche_results = {}
    initial_capital = 1000000

    for tranche_id in range(num_tranches):
        offset_signals = {}

        for etf, signals in signals_dict.items():
            if etf == cash_etf:
                offset_signals[etf] = signals.copy()
            else:
                weekly_signals = signals.resample("W").first()
                offset_weekly = weekly_signals.shift(tranche_id)

                daily_offset = offset_weekly.reindex(signals.index, method="ffill")
                daily_offset = daily_offset.fillna(0).astype(int)

                offset_signals[etf] = daily_offset

        tranche_weights = calculate_gtaa_weights(offset_signals, 0.20)

        tranche_capital = initial_capital / num_tranches
        tranche_returns, tranche_values, tranche_trades = backtest_gtaa_strategy(
            etf_data,
            offset_signals,
            tranche_weights,
            initial_capital=tranche_capital,
            transaction_cost=0.001,
        )

        tranche_results[f"tranche_{tranche_id}"] = {
            "returns": tranche_returns,
            "values": tranche_values,
            "weights": tranche_weights,
            "signals": offset_signals,
            "trades": tranche_trades,
        }

    combined_returns = _combine_tranche_returns(tranche_results)

    tranche_results["combined"] = {
        "returns": combined_returns,
        "values": (1 + combined_returns).cumprod() * initial_capital,
    }

    return tranche_results


def _combine_tranche_returns(tranche_results):
    all_dates = None
    tranche_keys = [k for k in tranche_results.keys() if k.startswith("tranche_")]

    for tranche_key in tranche_keys:
        tranche_dates = tranche_results[tranche_key]["returns"].index
        if all_dates is None:
            all_dates = tranche_dates
        else:
            all_dates = all_dates.intersection(tranche_dates)

    combined_returns = pd.Series(0.0, index=all_dates)
    num_tranches = len(tranche_keys)

    for tranche_key in tranche_keys:
        tranche_returns = tranche_results[tranche_key]["returns"].reindex(
            all_dates, fill_value=0
        )
        combined_returns += tranche_returns / num_tranches

    return combined_returns


def report_stats(daily_ret, freq=252, rf=0.0, name="Strategy"):
    daily_ret = daily_ret.dropna()
    if len(daily_ret) == 0:
        return pd.DataFrame()

    geometric_cum_ret = (1 + daily_ret).cumprod()
    arithmetic_cum_ret = daily_ret.cumsum()
    years = (daily_ret.index[-1] - daily_ret.index[0]).days / 365.25

    geometric_total_return = geometric_cum_ret.iloc[-1] - 1
    arithmetic_total_return = arithmetic_cum_ret.iloc[-1]
    cagr = (1 + geometric_total_return) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = daily_ret.std() * np.sqrt(freq)
    sharpe = (
        (daily_ret.mean() - rf / freq) / daily_ret.std() * np.sqrt(freq)
        if daily_ret.std() > 0
        else 0
    )

    downside_returns = daily_ret[daily_ret < 0]
    sortino = (
        daily_ret.mean() / downside_returns.std() * np.sqrt(freq)
        if len(downside_returns) > 0 and downside_returns.std() > 0
        else 0
    )

    peak = geometric_cum_ret.cummax()
    dd = geometric_cum_ret / peak - 1
    max_dd = dd.min()
    dd_end = dd.idxmin()
    dd_start = (geometric_cum_ret[:dd_end])[
        geometric_cum_ret[:dd_end] == geometric_cum_ret[:dd_end].cummax()
    ].last_valid_index()
    dd_duration = (dd_end - dd_start).days if dd_start else np.nan

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    hit_ratio = (
        (daily_ret > 0).sum() / (daily_ret != 0).sum()
        if (daily_ret != 0).sum() > 0
        else 0
    )
    avg_win = daily_ret[daily_ret > 0].mean() if (daily_ret > 0).sum() > 0 else 0
    avg_loss = daily_ret[daily_ret < 0].mean() if (daily_ret < 0).sum() > 0 else 0
    profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.nan

    up = (daily_ret > 0).astype(int)
    down = (daily_ret < 0).astype(int)
    win_streak = (
        (up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)).max()
        if len(up) > 0
        else 0
    )
    loss_streak = (
        (down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)).max()
        if len(down) > 0
        else 0
    )

    time_in_market = (daily_ret != 0).mean()

    skew = daily_ret.skew()
    kurt = daily_ret.kurt()

    stats = {
        "Total Return (Geometric)": f"{geometric_total_return:.2%}",
        "Total Return (Arithmetic)": f"{arithmetic_total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Annual Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown": f"{max_dd:.2%}",
        "Drawdown Duration (days)": dd_duration,
        "Calmar Ratio": round(calmar, 2),
        "Hit Ratio": f"{hit_ratio:.2%}",
        "Avg Win": f"{avg_win:.2%}",
        "Avg Loss": f"{avg_loss:.2%}",
        "Profit Factor": round(profit_factor, 2),
        "Max Win Streak": win_streak,
        "Max Loss Streak": loss_streak,
        "Time in Market": f"{time_in_market:.2%}",
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2),
    }

    return pd.DataFrame(stats, index=[name])


def report_backtest(
    data,
    filename="backtest_report.pdf",
    output_list=[1, 2, 3, 4, 5],
    strategy_name="Strategy Report",
    author="enne",
):
    daily_pnl_o, daily_pnl_x, daily_ret_o, daily_ret_x, market_ret, df = data

    if daily_ret_o is None:
        return

    if not isinstance(daily_ret_o.index, pd.DatetimeIndex):
        daily_ret_o.index = pd.to_datetime(daily_ret_o.index)

    date_range = (
        daily_ret_o.index.min().strftime("%Y-%m-%d"),
        daily_ret_o.index.max().strftime("%Y-%m-%d"),
    )

    strat_stats = report_stats(daily_ret_o, name="Strategy")

    if market_ret is not None and len(market_ret) > 0:
        market_stats = report_stats(market_ret, name="Market")
        total_stats = pd.concat([strat_stats, market_stats]).T
        total_stats.columns = ["Strategy", "Market"]
    else:
        total_stats = strat_stats.T
        total_stats.columns = ["Strategy"]

    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.8, strategy_name, fontsize=24, ha="center", weight="bold")
        if date_range:
            ax.text(
                0.5,
                0.30,
                f"Backtest Period: {date_range[0]} to {date_range[1]}",
                fontsize=12,
                ha="center",
            )
        ax.text(
            0.5,
            0.35,
            f"Generated on {datetime.datetime.now():%Y-%m-%d}",
            fontsize=12,
            ha="center",
        )
        ax.text(0.5, 0.25, f"Author: {author}", fontsize=12, ha="center")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        table = ax.table(
            cellText=total_stats.values,
            rowLabels=total_stats.index,
            colLabels=total_stats.columns,
            loc="center",
            cellLoc="left",
            colWidths=[0.20] * len(total_stats.columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.7)
        plt.title("Backtest Performance Report", fontsize=20, pad=20)
        pdf.savefig(fig)
        plt.close(fig)

        if 1 in output_list:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot((daily_ret_o * 100).cumsum(), label="Strategy Performance")
            if market_ret is not None and len(market_ret) > 0:
                ax.plot((market_ret * 100).cumsum(), label="Market Performance")
            ax.legend()
            ax.set_title("Equity Curve")
            ax.set_xlabel("Date")
            ax.set_ylabel("Returns(%)")
            pdf.savefig(fig)
            plt.close(fig)

def validate_gtaa_inputs(data_path, etf_list, start_date, end_date, config=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if not os.path.isdir(data_path):
        raise ValueError(f"Data path is not a directory: {data_path}")

    if not isinstance(etf_list, list) or len(etf_list) == 0:
        raise ValueError("ETF list must be a non-empty list")

    if len(set(etf_list)) != len(etf_list):
        raise ValueError("ETF list contains duplicates")

    for etf in etf_list:
        if not isinstance(etf, str) or len(etf) == 0:
            raise ValueError(f"Invalid ETF ticker: {etf}")

    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    if start_dt >= end_dt:
        raise ValueError("Start date must be before end date")

    if (end_dt - start_dt).days < 365:
        raise ValueError("Backtest period must be at least 1 year")

    if config is not None:
        required_config_keys = ["etf_weight", "sma_period", "transaction_cost"]
        for key in required_config_keys:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")

        if not 0 < config["etf_weight"] <= 1.0:
            raise ValueError("ETF weight must be between 0 and 1")

        if config["sma_period"] < 10 or config["sma_period"] > 500:
            raise ValueError("SMA period must be between 10 and 500 days")

        if config["transaction_cost"] < 0 or config["transaction_cost"] > 0.1:
            raise ValueError("Transaction cost must be between 0 and 10%")


def handle_data_quality_issues(etf_data, min_data_points=250):
    cleaned_data = {}
    issues_found = []

    for etf, df in etf_data.items():
        try:
            if len(df) < min_data_points:
                issues_found.append(
                    f"{etf}: Insufficient data ({len(df)} < {min_data_points})"
                )
                continue

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues_found.append(f"{etf}: Missing columns {missing_cols}")
                continue

            missing_count = df[required_cols].isnull().sum().sum()
            if missing_count > 0:
                df[required_cols] = df[required_cols].fillna(method="ffill")
                df[required_cols] = df[required_cols].fillna(method="bfill")

            price_changes = df["Close"].pct_change().abs()
            extreme_moves = (price_changes > 0.5).sum()
            if extreme_moves > 0:
                capped_changes = price_changes.clip(upper=0.5)
                df["Close"] = df["Close"].iloc[0] * (1 + capped_changes).cumprod()

            if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
                issues_found.append(f"{etf}: Contains negative or zero prices")
                continue

            invalid_prices = (
                (df["High"] < df["Low"])
                | (df["Close"] < df["Low"])
                | (df["Close"] > df["High"])
            )
            if invalid_prices.any():
                df.loc[invalid_prices, ["Open", "High", "Low"]] = df.loc[
                    invalid_prices, "Close"
                ].values.reshape(-1, 1)

            cleaned_data[etf] = df

        except Exception as e:
            issues_found.append(f"{etf}: Unexpected error - {str(e)}")
            continue

    if len(cleaned_data) < 2:
        raise ValueError(f"Too few ETFs passed validation: {issues_found}")

    return cleaned_data


def handle_signal_generation_errors(etf_data, sma_period=200):
    signals_dict = {}

    for etf, df in etf_data.items():
        try:
            if etf == "SHV":
                signals_dict[etf] = pd.Series(1, index=df.index)
                continue

            sma = (
                df["Close"]
                .rolling(window=sma_period, min_periods=sma_period // 2)
                .mean()
            )

            if sma.isnull().all():
                signals_dict[etf] = pd.Series(0, index=df.index)
                continue

            signals = (df["Close"] > sma).astype(int)
            signals = signals.fillna(0)
            signals_dict[etf] = signals

        except Exception as e:
            signals_dict[etf] = pd.Series(0, index=df.index)

    return signals_dict


def safe_backtest_execution(etf_data, signals_dict, target_weights, **kwargs):
    try:
        if not etf_data or not signals_dict or target_weights.empty:
            raise ValueError("Invalid input data for backtest")

        all_dates = target_weights.index
        for etf in etf_data.keys():
            etf_dates = etf_data[etf].index
            common_dates = all_dates.intersection(etf_dates)

            if len(common_dates) < len(all_dates) * 0.8:
                raise ValueError("Insufficient common dates for backtest")

        returns, values, trades = backtest_gtaa_strategy(
            etf_data, signals_dict, target_weights, **kwargs
        )

        if returns.isnull().all() or values.isnull().all():
            raise ValueError("Backtest produced all NaN results")

        if len(returns) == 0 or len(values) == 0:
            raise ValueError("Backtest produced empty results")

        return returns, values, trades

    except Exception as e:
        return None, None, None


def run_all_gtaa_tests():
    test_results = []

    try:
        temp_dir = tempfile.mkdtemp()
        test_data = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=100),
                "Close": 100 + np.random.randn(100),
                "Open": 100 + np.random.randn(100),
                "High": 102 + np.random.randn(100),
                "Low": 98 + np.random.randn(100),
                "Volume": 1000000,
            }
        )
        test_data.to_csv(f"{temp_dir}/SPY_data.csv", index=False)

        etf_data = load_gtaa_data(temp_dir, ["SPY"], "2020-01-01", "2020-04-09")
        test_results.append(len(etf_data) > 0)

        shutil.rmtree(temp_dir)

    except Exception as e:
        test_results.append(False)

    try:
        dates = pd.date_range("2020-01-01", periods=300)
        test_etf_data = {
            "SPY": pd.DataFrame(
                {
                    "Close": 100 + np.cumsum(np.random.randn(300) * 0.01),
                    "Open": 100,
                    "High": 101,
                    "Low": 99,
                    "Volume": 1000000,
                },
                index=dates,
            )
        }

        signals = calculate_gtaa_signals(test_etf_data, sma_period=50)
        test_results.append("SPY" in signals and len(signals["SPY"]) > 0)

    except Exception as e:
        test_results.append(False)

    try:
        test_signals = {
            "SPY": pd.Series([1, 0, 1], index=pd.date_range("2020-01-01", periods=3)),
            "SHV": pd.Series([1, 1, 1], index=pd.date_range("2020-01-01", periods=3)),
        }

        weights = calculate_gtaa_weights(test_signals, 0.20)
        test_results.append(len(weights) == 3 and len(weights.columns) == 2)

    except Exception as e:
        test_results.append(False)

    success_rate = sum(test_results) / len(test_results)
    overall_success = success_rate >= 0.8

    return overall_success


class GTAAConfig:
    def __init__(self):
        self.etf_list = ["SPY", "EFA", "IEF", "VNQ", "DBC", "SHV"]
        self.etf_names = {
            "SPY": "SPDR S&P 500 ETF",
            "EFA": "iShares MSCI EAFE ETF",
            "IEF": "iShares 7-10 Year Treasury ETF",
            "VNQ": "Vanguard Real Estate ETF",
            "DBC": "Invesco DB Commodity Index ETF",
            "SHV": "iShares Short Treasury Bond ETF",
        }

        self.cash_etf = "SHV"
        self.sma_period = 200
        self.etf_weight = 0.20
        self.rebalance_freq = "M"
        self.signal_lag = 1

        self.initial_capital = 1000000
        self.transaction_cost = 0.003
        self.min_trade_threshold = 0.01

        self.use_tranches = True
        self.num_tranches = 4

        self.timezone = "US/Eastern"
        self.market_calendar = "NYSE"
        self.min_data_points = 250

        self.risk_free_rate = 0.02
        self.trading_days_per_year = 252
        self.benchmark_etf = "SPY"

        self.generate_pdf_report = True
        self.save_detailed_logs = True
        self.plot_individual_etfs = True

    def get_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")


class GTAAIntegratedExecution:
    def __init__(self, config=None):
        self.config = config or GTAAConfig()
        self.results = {}
        self.execution_log = []
        self.start_time = None
        self.end_time = None

    def log_execution_step(self, step_name, status="SUCCESS", details=""):
        timestamp = datetime.datetime.now()
        log_entry = {
            "timestamp": timestamp,
            "step": step_name,
            "status": status,
            "details": details,
            "duration": None,
        }

        if self.execution_log and "timestamp" in self.execution_log[-1]:
            duration = timestamp - self.execution_log[-1]["timestamp"]
            log_entry["duration"] = duration.total_seconds()

        self.execution_log.append(log_entry)

    def validate_execution_environment(self, data_path):
        self.log_execution_step(
            "Environment Validation", "IN_PROGRESS", "Checking prerequisites..."
        )

        try:
            validate_gtaa_inputs(
                data_path,
                self.config.etf_list,
                "2020-01-01",
                "2025-01-01",
                self.config.get_config_dict(),
            )

            missing_files = []
            for etf in self.config.etf_list:
                file_path = f"{data_path}/{etf}_data.csv"
                if not os.path.exists(file_path):
                    missing_files.append(etf)

            if missing_files:
                self.log_execution_step(
                    "Environment Validation",
                    "WARNING",
                    f"Missing data files: {missing_files}",
                )
            else:
                self.log_execution_step(
                    "Environment Validation", "SUCCESS", "All data files available"
                )

            return len(missing_files) == 0

        except Exception as e:
            self.log_execution_step("Environment Validation", "ERROR", str(e))
            return False

    def execute_data_loading_phase(self, data_path, start_date, end_date):              
        self.log_execution_step(
            "Data Loading",
            "IN_PROGRESS",
            f"Loading {len(self.config.etf_list)} ETFs...",
        )

        try:
            raw_etf_data = load_gtaa_data(
                data_path,
                self.config.etf_list,
                start_date,
                end_date,
            )

            if not raw_etf_data:
                raise ValueError("No ETF data loaded successfully")

            cleaned_etf_data = handle_data_quality_issues(
                raw_etf_data, self.config.min_data_points
            )

            self.results["etf_data"] = cleaned_etf_data
            self.log_execution_step(
                "Data Loading",
                "SUCCESS",
                f"Loaded {len(cleaned_etf_data)} ETFs successfully",
            )

            return True

        except Exception as e:
            self.log_execution_step("Data Loading", "ERROR", str(e))
            return False

    def execute_signal_generation_phase(self):
        self.log_execution_step(
            "Signal Generation", "IN_PROGRESS", "Calculating SMA signals..."
        )

        try:
            signals_dict = handle_signal_generation_errors(
                self.results["etf_data"], self.config.sma_period
            )

            processed_signals = calculate_gtaa_signals(
                self.results["etf_data"],
                sma_period=self.config.sma_period,
                rebalance_freq=self.config.rebalance_freq,
                signal_lag=self.config.signal_lag,
            )

            self.results["signals"] = processed_signals

            signal_stats = self._calculate_signal_statistics(processed_signals)
            self.results["signal_stats"] = signal_stats

            self.log_execution_step(
                "Signal Generation",
                "SUCCESS",
                f"Generated signals for {len(processed_signals)} ETFs",
            )

            return True

        except Exception as e:
            self.log_execution_step("Signal Generation", "ERROR", str(e))
            return False

    def execute_portfolio_construction_phase(self):
        self.log_execution_step(
            "Portfolio Construction", "IN_PROGRESS", "Calculating target weights..."
        )

        try:
            target_weights = calculate_gtaa_weights(
                self.results["signals"], self.config.etf_weight
            )

            self.results["target_weights"] = target_weights

            portfolio_stats = self._calculate_portfolio_statistics(target_weights)
            self.results["portfolio_stats"] = portfolio_stats

            self.log_execution_step(
                "Portfolio Construction",
                "SUCCESS",
                f"Calculated weights for {len(target_weights)} days",
            )

            return True

        except Exception as e:
            self.log_execution_step("Portfolio Construction", "ERROR", str(e))
            return False

    def execute_backtesting_phase(self):        
        self.log_execution_step(
            "Backtesting", "IN_PROGRESS", "Running backtest simulation..."
        )

        try:
            returns, values, trades = safe_backtest_execution(
                self.results["etf_data"],
                self.results["signals"],
                self.results["target_weights"],
                initial_capital=self.config.initial_capital,
                transaction_cost=self.config.transaction_cost,
            )

            if returns is None:
                raise ValueError("Backtest execution failed")

            self.results["portfolio_returns"] = returns
            self.results["portfolio_values"] = values
            self.results["trade_log"] = trades

            benchmark_returns = self._calculate_benchmark_returns()
            self.results["benchmark_returns"] = benchmark_returns

            self.log_execution_step(
                "Backtesting",
                "SUCCESS",
                f"Completed backtest with {len(trades)} trades",
            )

            return True

        except Exception as e:
            self.log_execution_step("Backtesting", "ERROR", str(e))
            return False

    def execute_tranche_analysis_phase(self):
        if not self.config.use_tranches:
            self.log_execution_step(
                "Tranche Analysis", "SKIPPED", "Tranched analysis disabled"
            )
            return True

        self.log_execution_step(
            "Tranche Analysis",
            "IN_PROGRESS",
            f"Running {self.config.num_tranches}-tranche analysis...",
        )

        try:
            tranche_results = implement_tranche_strategy(
                self.results["etf_data"],
                self.results["signals"],
                self.config.num_tranches,
            )

            self.results["tranche_results"] = tranche_results

            tranche_stats = self._calculate_tranche_statistics(tranche_results)
            self.results["tranche_stats"] = tranche_stats

            self.log_execution_step(
                "Tranche Analysis",
                "SUCCESS",
                f"Completed {self.config.num_tranches}-tranche analysis",
            )

            return True

        except Exception as e:
            self.log_execution_step("Tranche Analysis", "ERROR", str(e))
            return False

    def execute_performance_analysis_phase(self):
        self.log_execution_step(
            "Performance Analysis", "IN_PROGRESS", "Calculating performance metrics..."
        )

        try:
            strategy_stats = report_stats(
                self.results["portfolio_returns"],
                freq=self.config.trading_days_per_year,
                rf=self.config.risk_free_rate,
                name="GTAA Strategy",
            )

            benchmark_stats = report_stats(
                self.results["benchmark_returns"],
                freq=self.config.trading_days_per_year,
                rf=self.config.risk_free_rate,
                name="Benchmark (SPY)",
            )

            combined_stats = pd.concat([strategy_stats, benchmark_stats])
            self.results["performance_stats"] = combined_stats

            risk_metrics = self._calculate_risk_metrics()
            self.results["risk_metrics"] = risk_metrics

            attribution_analysis = self._calculate_attribution_analysis()
            self.results["attribution_analysis"] = attribution_analysis

            self.log_execution_step(
                "Performance Analysis", "SUCCESS", "Completed performance analysis"
            )

            return True

        except Exception as e:
            self.log_execution_step("Performance Analysis", "ERROR", str(e))
            return False

    def execute_reporting_phase(self, output_path):
        self.log_execution_step(
            "Reporting", "IN_PROGRESS", "Generating reports and visualizations..."
        )

        try:
            if self.config.generate_pdf_report:
                report_data = self._prepare_report_data()

                pdf_filename = f"{output_path}/gtaa_comprehensive_report.pdf"
                report_backtest(
                    report_data,
                    filename=pdf_filename,
                    strategy_name="Global Tactical Asset Allocation (GTAA)",
                    author="Integrated Strategy Framework",
                )

                self.log_execution_step(
                    "PDF Report", "SUCCESS", f"Generated PDF report: {pdf_filename}"
                )

            self._generate_additional_visualizations(output_path)

            if self.config.save_detailed_logs:
                self._save_detailed_results(output_path)

            self.log_execution_step(
                "Reporting", "SUCCESS", "Completed all reporting tasks"
            )

            return True

        except Exception as e:
            self.log_execution_step("Reporting", "ERROR", str(e))
            return False

    def run_complete_analysis(self, data_path, start_date, end_date, output_path=None):
        self.start_time = datetime.datetime.now()

        print("=" * 80)
        print("GLOBAL TACTICAL ASSET ALLOCATION (GTAA) - INTEGRATED ANALYSIS")
        print("=" * 80)
        print(f"Analysis Period: {start_date} to {end_date}")
        print(
            f"Strategy Config: {self.config.etf_weight:.0%} equal weight, {self.config.sma_period}-day SMA"
        )
        print(f"ETFs: {', '.join(self.config.etf_list)}")
        print(f"Execution Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if output_path is None:
            output_path = data_path

        execution_steps = [
            (
                "Validate Environment",
                lambda: self.validate_execution_environment(data_path),
            ),
            (
                "Load Data",
                lambda: self.execute_data_loading_phase(
                    data_path, start_date, end_date
                ),
            ),
            ("Generate Signals", lambda: self.execute_signal_generation_phase()),
            (
                "Construct Portfolio",
                lambda: self.execute_portfolio_construction_phase(),
            ),
            ("Run Backtest", lambda: self.execute_backtesting_phase()),
            ("Analyze Tranches", lambda: self.execute_tranche_analysis_phase()),
            (
                "Calculate Performance",
                lambda: self.execute_performance_analysis_phase(),
            ),
            ("Generate Reports", lambda: self.execute_reporting_phase(output_path)),
        ]

        for step_name, step_function in execution_steps:
            try:
                success = step_function()
                if not success:
                    print(f"\n❌ Pipeline stopped at: {step_name}")
                    break
            except Exception as e:
                self.log_execution_step(step_name, "ERROR", f"Unexpected error: {e}")
                print(f"\n❌ Pipeline failed at: {step_name}")
                break
        else:
            self.end_time = datetime.datetime.now()
            total_duration = self.end_time - self.start_time

            print(f"\n✅ Analysis completed successfully!")
            print(f"⏱ Total execution time: {total_duration}")

            self._print_results_summary()

        self.results["execution_metadata"] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": total_duration if hasattr(self, "end_time") else None,
            "config": self.config.get_config_dict(),
            "execution_log": self.execution_log,
        }

        return self.results

    def _calculate_signal_statistics(self, signals_dict):
        stats = {}
        for etf, signals in signals_dict.items():
            if etf == self.config.cash_etf:
                continue

            stats[etf] = {
                "total_days": len(signals),
                "buy_signals": (signals == 1).sum(),
                "sell_signals": (signals == 0).sum(),
                "buy_ratio": (signals == 1).mean(),
                "signal_changes": (signals.diff() != 0).sum(),
                "longest_buy_streak": self._calculate_longest_streak(signals, 1),
                "longest_sell_streak": self._calculate_longest_streak(signals, 0),
            }

        return pd.DataFrame(stats).T

    def _calculate_longest_streak(self, series, value):
        try:
            if series is None or len(series) == 0:
                return 0

            if not isinstance(series, pd.Series):
                series = pd.Series(series)

            streaks = (
                (series == value).astype(int).groupby((series != value).cumsum()).sum()
            )
            return streaks.max() if len(streaks) > 0 else 0
        except Exception as e:
            return 0

    def _calculate_portfolio_statistics(self, target_weights):
        stats = {
            "avg_num_positions": (target_weights > 0.01).sum(axis=1).mean(),
            "max_num_positions": (target_weights > 0.01).sum(axis=1).max(),
            "min_num_positions": (target_weights > 0.01).sum(axis=1).min(),
            "avg_cash_allocation": target_weights[self.config.cash_etf].mean(),
            "max_cash_allocation": target_weights[self.config.cash_etf].max(),
            "min_cash_allocation": target_weights[self.config.cash_etf].min(),
            "rebalancing_frequency": len(target_weights),
        }

        return stats

    def _calculate_benchmark_returns(self):
        spy_data = self.results["etf_data"][self.config.benchmark_etf]
        benchmark_returns = spy_data["Close"].pct_change().dropna()

        portfolio_dates = self.results["portfolio_returns"].index
        aligned_returns = benchmark_returns.reindex(portfolio_dates, fill_value=0)

        return aligned_returns

    def _calculate_risk_metrics(self):
        returns = self.results["portfolio_returns"]
        benchmark_returns = self.results["benchmark_returns"]

        risk_metrics = {
            "beta": self._calculate_beta(returns, benchmark_returns),
            "correlation": returns.corr(benchmark_returns),
            "tracking_error": (returns - benchmark_returns).std()
            * np.sqrt(self.config.trading_days_per_year),
            "information_ratio": (returns.mean() - benchmark_returns.mean())
            / (returns - benchmark_returns).std(),
            "downside_deviation": returns[returns < 0].std()
            * np.sqrt(self.config.trading_days_per_year),
            "value_at_risk_5%": returns.quantile(0.05),
            "conditional_var_5%": returns[returns <= returns.quantile(0.05)].mean(),
        }

        return risk_metrics

    def _calculate_beta(self, returns, benchmark_returns):
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 10:
            return np.nan

        covariance = aligned_data.cov().iloc[0, 1]
        benchmark_variance = aligned_data.iloc[:, 1].var()

        return covariance / benchmark_variance if benchmark_variance != 0 else np.nan

    def _calculate_attribution_analysis(self):
        attribution = {}

        for etf in self.config.etf_list:
            if (
                etf in self.results["etf_data"]
                and etf in self.results["target_weights"].columns
            ):
                etf_returns = self.results["etf_data"][etf]["Close"].pct_change()
                etf_weights = self.results["target_weights"][etf]

                common_dates = etf_returns.index.intersection(etf_weights.index)
                if len(common_dates) > 0:
                    aligned_returns = etf_returns.reindex(common_dates, fill_value=0)
                    aligned_weights = etf_weights.reindex(common_dates, fill_value=0)

                    etf_contribution = (
                        aligned_weights.shift(1) * aligned_returns
                    ).sum()

                    attribution[etf] = {
                        "total_contribution": etf_contribution,
                        "avg_weight": aligned_weights.mean(),
                        "etf_total_return": aligned_returns.sum(),
                        "periods_held": (aligned_weights > 0.01).sum(),
                    }

        return pd.DataFrame(attribution).T

    def _calculate_tranche_statistics(self, tranche_results):
        if not tranche_results:
            return {}

        tranche_stats = {}

        for tranche_id in range(self.config.num_tranches):
            tranche_key = f"tranche_{tranche_id}"
            if tranche_key in tranche_results:
                tranche_returns = tranche_results[tranche_key]["returns"]
                tranche_stats[tranche_key] = {
                    "total_return": tranche_returns.sum(),
                    "volatility": tranche_returns.std()
                    * np.sqrt(self.config.trading_days_per_year),
                    "sharpe_ratio": tranche_returns.mean()
                    / tranche_returns.std()
                    * np.sqrt(self.config.trading_days_per_year),
                    "max_drawdown": self._calculate_max_drawdown(tranche_returns),
                }

        if "combined" in tranche_results:
            combined_returns = tranche_results["combined"]["returns"]
            tranche_stats["combined"] = {
                "total_return": combined_returns.sum(),
                "volatility": combined_returns.std()
                * np.sqrt(self.config.trading_days_per_year),
                "sharpe_ratio": combined_returns.mean()
                / combined_returns.std()
                * np.sqrt(self.config.trading_days_per_year),
                "max_drawdown": self._calculate_max_drawdown(combined_returns),
            }

        return tranche_stats

    def _calculate_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _prepare_report_data(self):
        portfolio_returns = self.results["portfolio_returns"]
        benchmark_returns = self.results["benchmark_returns"]

        daily_pnl_o = portfolio_returns * self.config.initial_capital
        daily_pnl_x = daily_pnl_o
        daily_ret_o = portfolio_returns
        daily_ret_x = portfolio_returns
        underlying_ret = benchmark_returns

        df = pd.DataFrame(index=portfolio_returns.index)

        return (daily_pnl_o, daily_pnl_x, daily_ret_o, daily_ret_x, underlying_ret, df)

    def _generate_additional_visualizations(self, output_path):
        try:
            self._plot_etf_signals(output_path)
            self._plot_portfolio_allocation(output_path)
            self._plot_performance_attribution(output_path)
            self._plot_risk_dashboard(output_path)

        except Exception as e:
            self.log_execution_step("Additional Visualizations", "ERROR", str(e))

    def _plot_etf_signals(self, output_path):
        if not self.config.plot_individual_etfs:
            return

        fig, axes = plt.subplots(
            len(self.config.etf_list) - 1,
            1,
            figsize=(15, 3 * (len(self.config.etf_list) - 1)),
        )
        if len(self.config.etf_list) == 2:
            axes = [axes]

        plot_idx = 0
        for etf in self.config.etf_list:
            if etf == self.config.cash_etf:
                continue

            if etf in self.results["etf_data"]:
                data = self.results["etf_data"][etf]
                signals = self.results["signals"][etf]

                ax = axes[plot_idx]

                ax.plot(data.index, data["Close"], label=f"{etf} Price", alpha=0.7)

                if f"SMA_{self.config.sma_period}" in data.columns:
                    ax.plot(
                        data.index,
                        data[f"SMA_{self.config.sma_period}"],
                        label=f"{self.config.sma_period}-day SMA",
                        alpha=0.7,
                    )

                signal_changes = signals.diff() != 0
                buy_signals = signals & signal_changes
                sell_signals = (~signals.astype(bool)) & signal_changes

                if buy_signals.any():
                    ax.scatter(
                        data.index[buy_signals],
                        data["Close"][buy_signals],
                        color="green",
                        marker="^",
                        s=50,
                        label="Buy Signal",
                        zorder=5,
                    )

                if sell_signals.any():
                    ax.scatter(
                        data.index[sell_signals],
                        data["Close"][sell_signals],
                        color="red",
                        marker="v",
                        s=50,
                        label="Sell Signal",
                        zorder=5,
                    )

                ax.set_title(f"{etf} - {self.config.etf_names.get(etf, etf)}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plot_idx += 1

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/etf_signals_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_portfolio_allocation(self, output_path):
        weights = self.results["target_weights"]

        fig, ax = plt.subplots(figsize=(15, 8))

        weights.plot.area(ax=ax, stacked=True, alpha=0.7)

        ax.set_title("GTAA Portfolio Allocation Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Weight")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/portfolio_allocation.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_performance_attribution(self, output_path):
        attribution = self.results["attribution_analysis"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        attribution["total_contribution"].plot.bar(ax=ax1)
        ax1.set_title("Total Return Contribution by ETF")
        ax1.set_xlabel("ETF")
        ax1.set_ylabel("Contribution to Portfolio Return")
        ax1.tick_params(axis="x", rotation=45)

        attribution["avg_weight"].plot.bar(ax=ax2)
        ax2.set_title("Average Portfolio Weight by ETF")
        ax2.set_xlabel("ETF")
        ax2.set_ylabel("Average Weight")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/performance_attribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_risk_dashboard(self, output_path):
        returns = self.results["portfolio_returns"]
        benchmark_returns = self.results["benchmark_returns"]

        fig, ax = plt.subplots(figsize=(15, 6))

        rolling_window = 63
        portfolio_vol = returns.rolling(rolling_window).std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.rolling(rolling_window).std() * np.sqrt(252)

        ax.plot(portfolio_vol.index, portfolio_vol, label="GTAA Strategy", alpha=0.8)
        ax.plot(benchmark_vol.index, benchmark_vol, label="Benchmark (SPY)", alpha=0.8)

        ax.set_title(f"Rolling {rolling_window}-Day Annualized Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/risk_dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _save_detailed_results(self, output_path):
        try:
            self.results["portfolio_returns"].to_csv(
                f"{output_path}/portfolio_returns.csv"
            )

            self.results["target_weights"].to_csv(f"{output_path}/target_weights.csv")

            if not self.results["trade_log"].empty:
                self.results["trade_log"].to_csv(
                    f"{output_path}/trade_log.csv", index=False
                )

            self.results["performance_stats"].to_csv(
                f"{output_path}/performance_stats.csv"
            )

            if "signal_stats" in self.results:
                self.results["signal_stats"].to_csv(f"{output_path}/signal_stats.csv")

            pd.DataFrame(self.execution_log).to_csv(
                f"{output_path}/execution_log.csv", index=False
            )

        except Exception as e:
            self.log_execution_step("Save Results", "ERROR", str(e))

    def _print_results_summary(self):
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)

        portfolio_return = self.results["portfolio_returns"].sum()
        benchmark_return = self.results["benchmark_returns"].sum()

        print(f"Portfolio Total Return: {portfolio_return:.2%}")
        print(f"Benchmark Total Return: {benchmark_return:.2%}")
        print(f"Excess Return: {portfolio_return - benchmark_return:.2%}")

        if "performance_stats" in self.results:
            strategy_stats = self.results["performance_stats"].loc["GTAA Strategy"]
            print(f"Sharpe Ratio: {strategy_stats['Sharpe Ratio']}")
            print(f"Max Drawdown: {strategy_stats['Max Drawdown']}")
            print(f"Calmar Ratio: {strategy_stats['Calmar Ratio']}")

        if "trade_log" in self.results and not self.results["trade_log"].empty:
            num_trades = len(self.results["trade_log"])
            print(f"Total Trades: {num_trades}")

        print("=" * 60)


if __name__ == "__main__":
    DATA_PATH = "/Users/enne/Documents/dev/python-gtaa-strategy/data"
    START_DATE = "2006-01-01"
    END_DATE = "2023-12-31"
    OUTPUT_PATH = "output"

    if not os.path.exists(DATA_PATH):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        possible_data_dirs = []
        for root, dirs, files in os.walk(parent_dir):
            if any(f.endswith("_data.csv") for f in files):
                possible_data_dirs.append(root)

        if possible_data_dirs:
            for path in possible_data_dirs:
                print(f" - {path}")
                
        sys.exit(1)

    etf_list = ["SPY", "EFA", "IEF", "VNQ", "DBC", "SHV"]
    missing_files = []
    for etf in etf_list:
        file_path = f"{DATA_PATH}/{etf}_data.csv"
        if not os.path.exists(file_path):
            missing_files.append(etf)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    config = GTAAConfig()

    config.update_config(
        use_tranches=True,
        num_tranches=4,
        transaction_cost=0.003,
        etf_weight=0.20
    )

    executor = GTAAIntegratedExecution(config)

    try:
        results = executor.run_complete_analysis(
            data_path=DATA_PATH,
            start_date=START_DATE,
            end_date=END_DATE,
            output_path=OUTPUT_PATH,
        )

        print(f"📊 Results available in: {OUTPUT_PATH}")
        print(f"📋 Execution log contains {len(executor.execution_log)} steps")
        
        test_success = run_all_gtaa_tests()
        print(f"🧪 Unit tests {'passed' if test_success else 'failed'}")

    except Exception as e:        
        print("📋 Check execution log for details")

        if hasattr(executor, "execution_log") and executor.execution_log:
            print("\nExecution Log:")
            for log_entry in executor.execution_log[-5:]:
                print(
                    f"  {log_entry['step']}: {log_entry['status']} - {log_entry['details']}"
                )
        else:
            print("❌ No execution log available.")
            import traceback

            print("\n상세 에러 정보:")
            traceback.print_exc()

    print(f"\n📝 For detailed analysis, check the generated reports in: {OUTPUT_PATH}")
