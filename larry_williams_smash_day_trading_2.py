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

from downloader_with_fdr import FDRDownloader

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


def load_smash_day_data(data_path, etf_list, start_date, end_date):
    etf_data = {}
    downloader = FDRDownloader()
    
    # ETF name to ticker mapping
    etf_tickers = {
        "KODEX 200": "069500",
        "KODEX 코스닥 150": "229200",
        "KODEX 레버리지": "122630",
        "KODEX 레버리지 인버스": "114800",
        "KODEX 코스닥150 레버리지": "233740",
        "KODEX 코스닥150 인버스": "251340",
    }

    for etf in etf_list:
        try:
            # Check if data file exists first
            filename = f"{data_path}/{etf}_data.csv"
            if os.path.exists(filename):
                # Try to load from file
                df = pd.read_csv(filename)
                date_cols = ["Date", "DateTime", "Date-Time", "date", "datetime"]
                date_col = next((col for col in date_cols if col in df.columns), None)
                
                if date_col is not None:
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
            else:
                # Download data using FDRDownloader
                ticker = etf_tickers.get(etf)
                if not ticker:
                    continue
                    
                result = downloader.download(
                    tickers=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if ticker not in result or result[ticker] is None:
                    continue
                    
                df = result[ticker]

            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    if col == "Volume":
                        df[col] = 1
                    else:
                        continue

            df = df.loc[start_date:end_date]

            if len(df) < 50:  # Smash Day needs less historical data
                continue

            df[["Open", "High", "Low", "Close"]].fillna(method="ffill", inplace=True)
            df["Volume"].fillna(1, inplace=True)

            df = remove_market_holidays(df, market="KRX")
            df = df[df.index.weekday < 5]

            etf_data[etf] = df

        except Exception as e:
            print(f"Error loading data for {etf}: {e}")
            continue

    return etf_data


def calculate_smash_day_signals(etf_data, lookback_period=3):
    """
    Larry Williams Smash Day 패턴 시그널 생성
    - Buy Signal: 전일 저점 하향 돌파 후 다음날 당일 고점 상향 돌파
    - Sell Signal: 전일 고점 상향 돌파 후 다음날 당일 저점 하향 돌파
    """
    signals_dict = {}

    for etf, df in etf_data.items():
        # Skip inverse ETFs for signal generation
        if "인버스" in etf or "INVERSE" in etf.upper():
            continue

        # Calculate previous day's high/low
        df["prev_high"] = df["High"].shift(1)
        df["prev_low"] = df["Low"].shift(1)

        # Calculate lookback period high/low
        df["lookback_high"] = df["High"].rolling(lookback_period).max().shift(1)
        df["lookback_low"] = df["Low"].rolling(lookback_period).min().shift(1)

        # Identify Smash Days
        # Bearish Smash Day (for buy setup)
        df["bearish_smash"] = (df["Close"] < df["prev_low"]) & (
            df["Close"] < df["lookback_low"]
        )

        # Bullish Smash Day (for sell setup)
        df["bullish_smash"] = (df["Close"] > df["prev_high"]) & (
            df["Close"] > df["lookback_high"]
        )

        # Generate actual entry signals (next day)
        # Buy when next open breaks above smash day high
        df["smash_day_high"] = df["High"].shift(1)
        df["smash_day_low"] = df["Low"].shift(1)

        buy_signal = pd.Series(0, index=df.index)
        sell_signal = pd.Series(0, index=df.index)

        for i in range(2, len(df)):
            # Check for buy entry (after bearish smash day)
            if (
                df["bearish_smash"].iloc[i - 1]
                and df["Open"].iloc[i] > df["smash_day_high"].iloc[i]
            ):
                buy_signal.iloc[i] = 1

            # Check for sell entry (after bullish smash day)
            if (
                df["bullish_smash"].iloc[i - 1]
                and df["Open"].iloc[i] < df["smash_day_low"].iloc[i]
            ):
                sell_signal.iloc[i] = 1

        signals_dict[etf] = {
            "buy": buy_signal,
            "sell": sell_signal,
            "bearish_smash": df["bearish_smash"],
            "bullish_smash": df["bullish_smash"],
        }

    return signals_dict


def calculate_smash_day_weights(signals_dict, etf_mapping, position_size=1.0):
    """
    Smash Day 신호에 따른 포지션 가중치 계산
    - 레버리지 ETF 사용으로 position_size는 보수적으로 설정
    """
    # Get all dates from signals
    all_dates = set()
    for etf_signals in signals_dict.values():
        if "buy" in etf_signals:
            all_dates.update(etf_signals["buy"].index)
        if "sell" in etf_signals:
            all_dates.update(etf_signals["sell"].index)
    all_dates = sorted(list(all_dates))
    
    print(f"DEBUG calculate_smash_day_weights: signals_dict keys: {list(signals_dict.keys())}")
    print(f"DEBUG calculate_smash_day_weights: all_dates count: {len(all_dates)}")

    # Initialize weights dataframe with actual ETF names/tickers
    all_etfs = []
    for mapping in etf_mapping.values():
        if "long" in mapping:
            all_etfs.append(mapping["long"])
        if "inverse" in mapping:
            all_etfs.append(mapping["inverse"])
    weights_df = pd.DataFrame(0.0, index=all_dates, columns=all_etfs)

    # Map signals to appropriate ETFs
    for date in all_dates:
        for base_etf, signal_data in signals_dict.items():
            if date not in signal_data["buy"].index:
                continue

            # Get corresponding leveraged ETFs
            long_etf = etf_mapping.get(base_etf, {}).get("long")
            inverse_etf = etf_mapping.get(base_etf, {}).get("inverse")

            # Buy signal -> Long leveraged ETF
            if date in signal_data["buy"].index and signal_data["buy"].loc[date] == 1 and long_etf:
                weights_df.loc[date, long_etf] = position_size

            # Sell signal -> Inverse ETF
            elif date in signal_data["sell"].index and signal_data["sell"].loc[date] == 1 and inverse_etf:
                weights_df.loc[date, inverse_etf] = position_size

    return weights_df


def backtest_smash_day_strategy(
    etf_data,
    signals_dict,
    target_weights,
    etf_mapping,
    initial_capital=1000000,
    transaction_cost=0.003,
    stop_loss_pct=0.02,  # 2% stop loss for leveraged ETFs
    bailout_exit=True,  # Exit on first positive return
    max_holding_days=3,  # Maximum holding period
):
    portfolio_returns = []
    portfolio_values = [initial_capital]
    trade_log = []

    # Track active positions
    active_positions = {}  # {etf: {'entry_date', 'entry_price', 'size'}}
    
    # Create ticker to ETF name mapping
    ticker_to_name = {
        "122630": "KODEX 레버리지",
        "114800": "KODEX 레버리지 인버스", 
        "233740": "KODEX 코스닥150 레버리지",
        "251340": "KODEX 코스닥150 인버스"
    }
    
    # Only keep mappings for ETFs that exist in our data
    ticker_to_name = {k: v for k, v in ticker_to_name.items() if v in etf_data and k in target_weights.columns}
    
    # Debug mapping
    print(f"DEBUG: Ticker to name mapping: {ticker_to_name}")
    print(f"DEBUG: ETF data keys: {list(etf_data.keys())}")
    print(f"DEBUG: Target weights columns: {list(target_weights.columns)}")

    for i, date in enumerate(target_weights.index):
        daily_return = 0.0

        if i == 0:
            continue

        prev_date = target_weights.index[i - 1]

        # Check for new positions
        for etf in target_weights.columns:
            target_weight = target_weights.loc[date, etf]

            # New position entry
            if target_weight > 0 and etf not in active_positions:
                # Get the corresponding ETF name
                etf_name = ticker_to_name.get(etf)
                
                if etf_name and date in etf_data[etf_name].index:
                    entry_price = etf_data[etf_name].loc[date, "Open"]
                    active_positions[etf] = {
                        "entry_date": date,
                        "entry_price": entry_price,
                        "size": target_weight,
                        "entry_value": portfolio_values[-1] * target_weight,
                    }

                    trade_log.append(
                        {
                            "date": date,
                            "etf": etf,
                            "action": "BUY",
                            "price": entry_price,
                            "size": target_weight,
                            "reason": "Smash Day Signal",
                        }
                    )

        # Calculate returns and check exit conditions
        positions_to_close = []

        for etf, pos_info in active_positions.items():
            etf_name = ticker_to_name.get(etf, etf)  # Use mapping or original name
            if etf_name in etf_data and date in etf_data[etf_name].index and prev_date in etf_data[etf_name].index:
                # Calculate position return
                current_price = etf_data[etf_name].loc[date, "Close"]
                prev_price = etf_data[etf_name].loc[prev_date, "Close"]

                pos_return = (current_price / prev_price - 1) * pos_info["size"]
                daily_return += pos_return

                # Check exit conditions
                total_return = current_price / pos_info["entry_price"] - 1
                days_held = (date - pos_info["entry_date"]).days

                # 1. Bailout exit - first positive return
                if bailout_exit and total_return > 0:
                    positions_to_close.append((etf, "Bailout Exit"))

                # 2. Stop loss
                elif total_return <= -stop_loss_pct:
                    positions_to_close.append((etf, "Stop Loss"))

                # 3. Max holding period
                elif max_holding_days and days_held >= max_holding_days:
                    positions_to_close.append((etf, "Time Exit"))

                # 4. Opposite signal
                else:
                    # Check if opposite signal triggered
                    base_etf = None
                    for base, mapping in etf_mapping.items():
                        if mapping.get("long") == etf or mapping.get("inverse") == etf:
                            base_etf = base
                            break

                    if base_etf and base_etf in signals_dict:
                        is_long = etf_mapping[base_etf].get("long") == etf
                        opposite_signal = signals_dict[base_etf][
                            "sell" if is_long else "buy"
                        ]

                        if (
                            date in opposite_signal.index
                            and opposite_signal.loc[date] == 1
                        ):
                            positions_to_close.append((etf, "Opposite Signal"))

        # Close positions
        for etf, reason in positions_to_close:
            etf_name = ticker_to_name.get(etf, etf)
            if etf_name in etf_data and date in etf_data[etf_name].index:
                exit_price = etf_data[etf_name].loc[date, "Close"]
                pos_info = active_positions[etf]
                total_return = exit_price / pos_info["entry_price"] - 1

                trade_log.append(
                    {
                        "date": date,
                        "etf": etf,
                        "action": "SELL",
                        "price": exit_price,
                        "size": pos_info["size"],
                        "reason": reason,
                        "return": total_return,
                    }
                )

                del active_positions[etf]

        # Transaction costs
        transaction_cost_today = 0.0
        if positions_to_close or any(target_weights.loc[date] > 0):
            turnover = sum(
                abs(target_weights.loc[date, etf]) for etf, _ in positions_to_close
            )
            turnover += sum(
                target_weights.loc[date, etf]
                for etf in target_weights.columns
                if target_weights.loc[date, etf] > 0 and etf not in active_positions
            )
            transaction_cost_today = turnover * transaction_cost

        daily_return -= transaction_cost_today

        portfolio_returns.append(daily_return)
        new_portfolio_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_portfolio_value)

    # Ensure correct indexing - portfolio_returns has one less element than target_weights.index
    if len(portfolio_returns) > 0:
        returns_series = pd.Series(portfolio_returns, index=target_weights.index[1:])
        values_series = pd.Series(portfolio_values[1:], index=target_weights.index[1:])
    else:
        returns_series = pd.Series(dtype=float)
        values_series = pd.Series(dtype=float)
    
    trade_df = pd.DataFrame(trade_log)

    return returns_series, values_series, trade_df


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

    if daily_ret_o is None or len(daily_ret_o) == 0:
        print("Warning: No returns data available for report generation")
        return

    if not isinstance(daily_ret_o.index, pd.DatetimeIndex):
        daily_ret_o.index = pd.to_datetime(daily_ret_o.index)

    # Check if index has valid dates
    if len(daily_ret_o.index) == 0 or pd.isna(daily_ret_o.index.min()):
        print("Warning: Invalid date index in returns data")
        return
        
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


def validate_smash_day_inputs(data_path, etf_list, start_date, end_date, config=None):
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

    if (end_dt - start_dt).days < 90:
        raise ValueError(
            "Backtest period must be at least 90 days for Smash Day strategy"
        )

    if config is not None:
        required_config_keys = [
            "position_size",
            "lookback_period",
            "stop_loss_pct",
            "transaction_cost",
        ]
        for key in required_config_keys:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")

        if not 0 < config["position_size"] <= 1.0:
            raise ValueError("Position size must be between 0 and 1")

        if config["lookback_period"] < 1 or config["lookback_period"] > 10:
            raise ValueError("Lookback period must be between 1 and 10 days")

        if config["stop_loss_pct"] < 0.01 or config["stop_loss_pct"] > 0.1:
            raise ValueError("Stop loss must be between 1% and 10%")

        if config["transaction_cost"] < 0 or config["transaction_cost"] > 0.01:
            raise ValueError("Transaction cost must be between 0 and 1%")


def handle_data_quality_issues(etf_data, min_data_points=50):
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

            # Check for price consistency (High >= Low, etc.)
            invalid_prices = (
                (df["High"] < df["Low"])
                | (df["Close"] < df["Low"])
                | (df["Close"] > df["High"])
                | (df["Open"] < df["Low"])
                | (df["Open"] > df["High"])
            )
            if invalid_prices.any():
                issues_found.append(f"{etf}: Contains invalid OHLC relationships")
                # Fix by adjusting to Close price
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


class SmashDayConfig:
    def __init__(self):
        # Korean leveraged ETF mappings
        self.etf_mapping = {
            "KOSPI200": {"long": "122630", "inverse": "114800"},
            "KOSDAQ150": {
                "long": "233740",
                "inverse": "251340",
            },
        }

        # All ETFs to load (including base indices for signal generation)
        self.etf_list = [
            "KODEX 200",  # Base index for signals
            "KODEX 코스닥 150",  # Base index for signals
            "KODEX 레버리지",
            "KODEX 레버리지 인버스",
            "KODEX 코스닥150 레버리지",
            "KODEX 코스닥150 인버스",
        ]

        # ETF name mappings
        self.etf_names = {
            "KODEX 200": "069500",
            "KODEX 코스닥 150": "229200",
            "KODEX 레버리지": "122630",
            "KODEX 레버리지 인버스": "114800",
            "KODEX 코스닥150 레버리지": "233740",
            "KODEX 코스닥150 인버스": "251340",
        }

        # Smash Day parameters
        self.lookback_period = 3  # Days to look back for high/low
        self.position_size = 0.5  # Conservative for leveraged ETFs
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.bailout_exit = True  # Exit on first positive return
        self.max_holding_days = 3  # Maximum holding period

        # Backtest parameters
        self.initial_capital = 1000000
        self.transaction_cost = 0.003

        # Risk management
        self.max_positions = 1  # Only one position at a time
        self.volatility_filter = False  # Optional: only trade in high volatility
        self.min_volatility = 0.01  # Minimum daily volatility to trade

        # Report settings
        self.generate_pdf_report = True
        self.save_detailed_logs = True
        self.plot_individual_signals = True

        # Market settings
        self.timezone = "Asia/Seoul"
        self.market_calendar = "KRX"
        self.min_data_points = 50

        # Performance benchmarks
        self.risk_free_rate = 0.02
        self.trading_days_per_year = 250  # Korean market
        self.benchmark_etf = "KODEX 200"

    def get_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")


class SmashDayIntegratedExecution:
    def __init__(self, config=None):
        self.config = config or SmashDayConfig()
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
            validate_smash_day_inputs(
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
                self.log_execution_step(
                    "Environment Validation",
                    "INFO",
                    "Will attempt to download missing data using FDRDownloader",
                )
            else:
                self.log_execution_step(
                    "Environment Validation", "SUCCESS", "All data files available"
                )

            # Continue even if files are missing - FDRDownloader will handle it
            return True

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
            raw_etf_data = load_smash_day_data(
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
            "Signal Generation", "IN_PROGRESS", "Calculating Smash Day patterns..."
        )

        try:
            signals_dict = calculate_smash_day_signals(
                self.results["etf_data"], lookback_period=self.config.lookback_period
            )

            self.results["signals"] = signals_dict

            # Calculate signal statistics
            signal_stats = self._calculate_signal_statistics(signals_dict)
            self.results["signal_stats"] = signal_stats

            self.log_execution_step(
                "Signal Generation",
                "SUCCESS",
                f"Generated signals for {len(signals_dict)} ETFs",
            )

            return True

        except Exception as e:
            self.log_execution_step("Signal Generation", "ERROR", str(e))
            return False

    def execute_portfolio_construction_phase(self):
        self.log_execution_step(
            "Portfolio Construction",
            "IN_PROGRESS",
            "Mapping signals to leveraged ETFs...",
        )

        try:
            # Create mapping for base ETFs to signals
            base_mapping = {"KODEX 200": "KOSPI200", "KODEX 코스닥 150": "KOSDAQ150"}

            # Debug: show what signals we have
            self.log_execution_step(
                "Portfolio Construction",
                "DEBUG",
                f"Available signals for ETFs: {list(self.results['signals'].keys())}"
            )
            
            # Filter signals for base ETFs only
            base_signals = {
                base_mapping.get(etf, etf): signals
                for etf, signals in self.results["signals"].items()
                if etf in base_mapping
            }
            
            # If no base signals found, use leveraged ETF signals directly
            if not base_signals:
                self.log_execution_step(
                    "Portfolio Construction",
                    "WARNING",
                    "No base ETF signals found, using leveraged ETF signals directly"
                )
                # Map leveraged ETFs to base names
                leveraged_mapping = {
                    "KODEX 레버리지": "KOSPI200",
                    "KODEX 코스닥150 레버리지": "KOSDAQ150"
                }
                base_signals = {
                    leveraged_mapping.get(etf, etf): signals
                    for etf, signals in self.results["signals"].items()
                    if etf in leveraged_mapping
                }

            target_weights = calculate_smash_day_weights(
                base_signals,
                self.config.etf_mapping,
                position_size=self.config.position_size,
            )

            self.results["target_weights"] = target_weights

            # Calculate portfolio statistics
            portfolio_stats = self._calculate_portfolio_statistics(target_weights)
            self.results["portfolio_stats"] = portfolio_stats
            
            # Debug target weights
            non_zero_weights = (target_weights != 0).sum().sum()
            self.log_execution_step(
                "Portfolio Construction", 
                "DEBUG",
                f"Non-zero weights: {non_zero_weights}, Total days: {len(target_weights)}"
            )

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
            "Backtesting", "IN_PROGRESS", "Running Smash Day backtest simulation..."
        )

        try:
            returns, values, trades = backtest_smash_day_strategy(
                self.results["etf_data"],
                self.results["signals"],
                self.results["target_weights"],
                self.config.etf_mapping,
                initial_capital=self.config.initial_capital,
                transaction_cost=self.config.transaction_cost,
                stop_loss_pct=self.config.stop_loss_pct,
                bailout_exit=self.config.bailout_exit,
                max_holding_days=self.config.max_holding_days,
            )

            if returns is None:
                raise ValueError("Backtest execution failed")
                
            # Debug information
            self.log_execution_step(
                "Backtesting",
                "DEBUG",
                f"Returns shape: {returns.shape if hasattr(returns, 'shape') else len(returns)}, Values shape: {values.shape if hasattr(values, 'shape') else len(values)}"
            )
            
            # More debug info
            if isinstance(trades, pd.DataFrame) and len(trades) > 0:
                self.log_execution_step(
                    "Backtesting",
                    "INFO",
                    f"Total trades: {len(trades)}, Buy trades: {len(trades[trades['action'] == 'BUY'])}, Sell trades: {len(trades[trades['action'] == 'SELL'])}"
                )
            elif isinstance(trades, list) and len(trades) > 0:
                self.log_execution_step(
                    "Backtesting",
                    "INFO", 
                    f"Total trades: {len(trades)}"
                )
            else:
                self.log_execution_step(
                    "Backtesting",
                    "WARNING",
                    "No trades were executed during backtest"
                )

            self.results["portfolio_returns"] = returns
            self.results["portfolio_values"] = values
            self.results["trade_log"] = trades

            # Calculate benchmark returns
            benchmark_returns = self._calculate_benchmark_returns()
            self.results["benchmark_returns"] = benchmark_returns

            self.log_execution_step(
                "Backtesting",
                "SUCCESS",
                f"Completed backtest with {len(trades)} trades",
            )

            return True

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.log_execution_step("Backtesting", "ERROR", str(e))
            self.log_execution_step("Backtesting", "ERROR_DETAIL", error_detail)
            print(f"\n백테스트 에러 상세:\n{error_detail}")
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
                name="Smash Day Strategy",
            )

            benchmark_stats = report_stats(
                self.results["benchmark_returns"],
                freq=self.config.trading_days_per_year,
                rf=self.config.risk_free_rate,
                name=f"Benchmark ({self.config.benchmark_etf})",
            )

            combined_stats = pd.concat([strategy_stats, benchmark_stats])
            self.results["performance_stats"] = combined_stats

            # Calculate additional Smash Day specific metrics
            smash_metrics = self._calculate_smash_day_metrics()
            self.results["smash_metrics"] = smash_metrics

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

                pdf_filename = f"{output_path}/smash_day_report.pdf"
                report_backtest(
                    report_data,
                    filename=pdf_filename,
                    strategy_name="Larry Williams Smash Day Strategy (Korean Leveraged ETFs)",
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
            import traceback
            error_detail = traceback.format_exc()
            self.log_execution_step("Reporting", "ERROR", str(e))
            self.log_execution_step("Reporting", "ERROR_DETAIL", error_detail)
            print(f"\n리포트 생성 에러 상세:\n{error_detail}")
            return False

    def run_complete_analysis(self, data_path, start_date, end_date, output_path=None):
        self.start_time = datetime.datetime.now()

        print("=" * 80)
        print("LARRY WILLIAMS SMASH DAY STRATEGY - KOREAN LEVERAGED ETFs")
        print("=" * 80)
        print(f"Analysis Period: {start_date} to {end_date}")
        print(
            f"Strategy Config: {self.config.position_size:.0%} position size, {self.config.lookback_period}-day lookback"
        )
        print(
            f"Risk Management: {self.config.stop_loss_pct:.1%} stop loss, Bailout: {self.config.bailout_exit}"
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

        # Calculate total_duration if end_time exists
        total_duration = None
        if hasattr(self, "end_time") and self.end_time is not None:
            total_duration = self.end_time - self.start_time

        self.results["execution_metadata"] = {
            "start_time": self.start_time,
            "end_time": getattr(self, "end_time", None),
            "total_duration": total_duration,
            "config": self.config.get_config_dict(),
            "execution_log": self.execution_log,
        }

        return self.results

    def _calculate_signal_statistics(self, signals_dict):
        stats = {}
        for etf, signal_data in signals_dict.items():
            buy_signals = signal_data["buy"]
            sell_signals = signal_data["sell"]
            bearish_smash = signal_data["bearish_smash"]
            bullish_smash = signal_data["bullish_smash"]

            stats[etf] = {
                "total_days": len(buy_signals),
                "buy_signals": buy_signals.sum(),
                "sell_signals": sell_signals.sum(),
                "bearish_smash_days": bearish_smash.sum(),
                "bullish_smash_days": bullish_smash.sum(),
                "signal_ratio": (
                    (buy_signals.sum() + sell_signals.sum()) / len(buy_signals)
                    if len(buy_signals) > 0
                    else 0
                ),
            }

        return pd.DataFrame(stats).T

    def _calculate_portfolio_statistics(self, target_weights):
        active_positions = (target_weights.abs() > 0).sum(axis=1)

        stats = {
            "avg_positions_per_day": active_positions.mean(),
            "max_positions": active_positions.max(),
            "days_with_positions": (active_positions > 0).sum(),
            "total_days": len(target_weights),
            "position_utilization": (active_positions > 0).mean(),
            "avg_position_size": (
                target_weights[target_weights > 0].mean().mean()
                if (target_weights > 0).any().any()
                else 0
            ),
        }

        return stats

    def _calculate_benchmark_returns(self):
        # Check if benchmark ETF exists in data
        if self.config.benchmark_etf not in self.results["etf_data"]:
            # Try to find an alternative benchmark
            available_etfs = list(self.results["etf_data"].keys())
            if available_etfs:
                # Use the first available ETF as benchmark
                benchmark_etf = available_etfs[0]
                self.log_execution_step(
                    "Benchmark",
                    "WARNING",
                    f"Benchmark ETF '{self.config.benchmark_etf}' not found, using '{benchmark_etf}' instead"
                )
            else:
                # No data available, return empty series
                return pd.Series(index=self.results["portfolio_returns"].index, data=0)
        else:
            benchmark_etf = self.config.benchmark_etf
            
        benchmark_data = self.results["etf_data"][benchmark_etf]
        benchmark_returns = benchmark_data["Close"].pct_change().dropna()

        portfolio_dates = self.results["portfolio_returns"].index
        aligned_returns = benchmark_returns.reindex(portfolio_dates, fill_value=0)

        return aligned_returns

    def _calculate_smash_day_metrics(self):
        if self.results["trade_log"].empty:
            return {}

        trades_df = self.results["trade_log"]

        # Filter for exit trades
        exit_trades = trades_df[trades_df["action"] == "SELL"].copy()

        if exit_trades.empty:
            return {}

        # Calculate metrics by exit reason
        exit_reasons = exit_trades["reason"].value_counts()

        # Calculate average returns by exit reason
        avg_returns_by_reason = exit_trades.groupby("reason")["return"].mean()

        # Calculate holding periods
        entry_trades = trades_df[trades_df["action"] == "BUY"]

        metrics = {
            "total_trades": len(exit_trades),
            "trades_per_year": len(exit_trades)
            / (
                (
                    self.results["portfolio_returns"].index[-1]
                    - self.results["portfolio_returns"].index[0]
                ).days
                / 365.25
            ),
            "bailout_exits": exit_reasons.get("Bailout Exit", 0),
            "stop_loss_exits": exit_reasons.get("Stop Loss", 0),
            "time_exits": exit_reasons.get("Time Exit", 0),
            "opposite_signal_exits": exit_reasons.get("Opposite Signal", 0),
            "avg_return_bailout": avg_returns_by_reason.get("Bailout Exit", 0),
            "avg_return_stop_loss": avg_returns_by_reason.get("Stop Loss", 0),
            "avg_return_time_exit": avg_returns_by_reason.get("Time Exit", 0),
            "avg_return_opposite": avg_returns_by_reason.get("Opposite Signal", 0),
        }

        return metrics

    def _prepare_report_data(self):
        portfolio_returns = self.results.get("portfolio_returns", pd.Series())
        benchmark_returns = self.results.get("benchmark_returns", pd.Series())
        
        # Check if we have valid returns data
        if portfolio_returns.empty:
            self.log_execution_step("Report Data", "WARNING", "No portfolio returns data available")
            # Create dummy data to avoid errors
            dummy_index = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D')
            portfolio_returns = pd.Series(0, index=dummy_index)
            benchmark_returns = pd.Series(0, index=dummy_index)

        daily_pnl_o = portfolio_returns * self.config.initial_capital
        daily_pnl_x = daily_pnl_o
        daily_ret_o = portfolio_returns
        daily_ret_x = portfolio_returns
        underlying_ret = benchmark_returns

        df = pd.DataFrame(index=portfolio_returns.index)

        return (daily_pnl_o, daily_pnl_x, daily_ret_o, daily_ret_x, underlying_ret, df)

    def _generate_additional_visualizations(self, output_path):
        try:
            self._plot_smash_day_patterns(output_path)
            self._plot_trade_analysis(output_path)
            self._plot_exit_reason_analysis(output_path)
            self._plot_signal_distribution(output_path)

        except Exception as e:
            self.log_execution_step("Additional Visualizations", "ERROR", str(e))

    def _plot_smash_day_patterns(self, output_path):
        if not self.config.plot_individual_signals:
            return

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot KOSPI 200 patterns
        if (
            "KODEX 200" in self.results["etf_data"]
            and "KODEX 200" in self.results["signals"]
        ):
            self._plot_single_etf_pattern(axes[0], "KODEX 200", "KOSPI 200")

        # Plot KOSDAQ 150 patterns
        if (
            "KODEX 코스닥 150" in self.results["etf_data"]
            and "KODEX 코스닥 150" in self.results["signals"]
        ):
            self._plot_single_etf_pattern(axes[1], "KODEX 코스닥 150", "KOSDAQ 150")

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/smash_day_patterns.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_single_etf_pattern(self, ax, etf_name, title):
        data = self.results["etf_data"][etf_name]
        signals = self.results["signals"][etf_name]

        # Plot last 250 days
        plot_data = data.tail(250)
        plot_signals = {k: v.tail(250) for k, v in signals.items()}

        # Plot price
        ax.plot(plot_data.index, plot_data["Close"], label="Close Price", alpha=0.7)

        # Mark Smash Days
        bearish_smash = plot_signals["bearish_smash"]
        bullish_smash = plot_signals["bullish_smash"]

        # Bearish Smash Days (potential buy setup)
        bearish_dates = plot_data.index[bearish_smash]
        if len(bearish_dates) > 0:
            ax.scatter(
                bearish_dates,
                plot_data.loc[bearish_dates, "Low"] * 0.99,
                color="green",
                marker="^",
                s=100,
                label="Bearish Smash",
                zorder=5,
            )

        # Bullish Smash Days (potential sell setup)
        bullish_dates = plot_data.index[bullish_smash]
        if len(bullish_dates) > 0:
            ax.scatter(
                bullish_dates,
                plot_data.loc[bullish_dates, "High"] * 1.01,
                color="red",
                marker="v",
                s=100,
                label="Bullish Smash",
                zorder=5,
            )

        # Mark actual entry signals
        buy_signals = plot_signals["buy"]
        sell_signals = plot_signals["sell"]

        buy_dates = plot_data.index[buy_signals == 1]
        if len(buy_dates) > 0:
            ax.scatter(
                buy_dates,
                plot_data.loc[buy_dates, "Close"],
                color="darkgreen",
                marker="o",
                s=150,
                label="Buy Entry",
                edgecolors="black",
                linewidths=2,
                zorder=6,
            )

        sell_dates = plot_data.index[sell_signals == 1]
        if len(sell_dates) > 0:
            ax.scatter(
                sell_dates,
                plot_data.loc[sell_dates, "Close"],
                color="darkred",
                marker="o",
                s=150,
                label="Sell Entry",
                edgecolors="black",
                linewidths=2,
                zorder=6,
            )

        ax.set_title(f"{title} - Smash Day Pattern Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_trade_analysis(self, output_path):
        if self.results["trade_log"].empty:
            return

        trades_df = self.results["trade_log"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Trade returns distribution
        exit_trades = trades_df[trades_df["action"] == "SELL"]
        if not exit_trades.empty and "return" in exit_trades.columns:
            returns = exit_trades["return"] * 100  # Convert to percentage

            ax1.hist(returns, bins=30, alpha=0.7, color="blue", edgecolor="black")
            ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
            ax1.set_title("Trade Returns Distribution")
            ax1.set_xlabel("Return (%)")
            ax1.set_ylabel("Frequency")

            # Add statistics
            mean_return = returns.mean()
            median_return = returns.median()
            ax1.axvline(
                x=mean_return,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_return:.2f}%",
            )
            ax1.axvline(
                x=median_return,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_return:.2f}%",
            )
            ax1.legend()

        # Plot 2: Exit reasons
        if not exit_trades.empty:
            exit_reasons = exit_trades["reason"].value_counts()
            colors = [
                (
                    "green"
                    if "Bailout" in reason
                    else (
                        "red"
                        if "Stop" in reason
                        else "blue" if "Time" in reason else "orange"
                    )
                )
                for reason in exit_reasons.index
            ]

            ax2.bar(
                range(len(exit_reasons)), exit_reasons.values, color=colors, alpha=0.7
            )
            ax2.set_xticks(range(len(exit_reasons)))
            ax2.set_xticklabels(exit_reasons.index, rotation=45, ha="right")
            ax2.set_title("Trade Exit Reasons")
            ax2.set_ylabel("Count")

            # Add percentage labels
            total_exits = exit_reasons.sum()
            for i, (reason, count) in enumerate(exit_reasons.items()):
                ax2.text(
                    i, count + 0.5, f"{count/total_exits:.1%}", ha="center", va="bottom"
                )

        plt.tight_layout()
        plt.savefig(f"{output_path}/trade_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_exit_reason_analysis(self, output_path):
        if self.results["trade_log"].empty:
            return

        trades_df = self.results["trade_log"]
        exit_trades = trades_df[trades_df["action"] == "SELL"]

        if exit_trades.empty or "return" not in exit_trades.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate average return by exit reason
        avg_returns = exit_trades.groupby("reason")["return"].agg(
            ["mean", "count", "std"]
        )
        avg_returns["mean"] *= 100  # Convert to percentage
        avg_returns["std"] *= 100

        # Create bar plot with error bars
        x = range(len(avg_returns))
        colors = ["green" if ret > 0 else "red" for ret in avg_returns["mean"]]

        bars = ax.bar(
            x,
            avg_returns["mean"],
            yerr=avg_returns["std"],
            color=colors,
            alpha=0.7,
            capsize=5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(avg_returns.index, rotation=45, ha="right")
        ax.set_title("Average Return by Exit Reason")
        ax.set_ylabel("Average Return (%)")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Add count labels
        for i, (idx, row) in enumerate(avg_returns.iterrows()):
            ax.text(
                i,
                row["mean"] + row["std"] + 0.2,
                f"n={row['count']}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/exit_reason_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_signal_distribution(self, output_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Get signal data
        signal_stats = self.results["signal_stats"]

        # Plot 1: Signal frequency by ETF
        ax = axes[0]
        etfs = signal_stats.index
        buy_signals = signal_stats["buy_signals"]
        sell_signals = signal_stats["sell_signals"]

        x = np.arange(len(etfs))
        width = 0.35

        ax.bar(
            x - width / 2,
            buy_signals,
            width,
            label="Buy Signals",
            color="green",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            sell_signals,
            width,
            label="Sell Signals",
            color="red",
            alpha=0.7,
        )

        ax.set_xlabel("ETF")
        ax.set_ylabel("Number of Signals")
        ax.set_title("Signal Frequency by ETF")
        ax.set_xticks(x)
        ax.set_xticklabels(etfs, rotation=45, ha="right")
        ax.legend()

        # Plot 2: Smash Day frequency
        ax = axes[1]
        bearish_smash = signal_stats["bearish_smash_days"]
        bullish_smash = signal_stats["bullish_smash_days"]

        ax.bar(
            x - width / 2,
            bearish_smash,
            width,
            label="Bearish Smash Days",
            color="darkgreen",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            bullish_smash,
            width,
            label="Bullish Smash Days",
            color="darkred",
            alpha=0.7,
        )

        ax.set_xlabel("ETF")
        ax.set_ylabel("Number of Smash Days")
        ax.set_title("Smash Day Pattern Frequency")
        ax.set_xticks(x)
        ax.set_xticklabels(etfs, rotation=45, ha="right")
        ax.legend()

        # Plot 3: Monthly signal distribution
        ax = axes[2]
        portfolio_returns = self.results["portfolio_returns"]
        monthly_active = (
            (self.results["target_weights"].abs() > 0).any(axis=1).resample("M").sum()
        )

        ax.bar(range(len(monthly_active)), monthly_active.values, alpha=0.7)
        ax.set_title("Monthly Trading Activity")
        ax.set_xlabel("Month")
        ax.set_ylabel("Days with Positions")

        # Plot 4: Performance metrics summary
        ax = axes[3]
        ax.axis("off")

        if "smash_metrics" in self.results:
            metrics = self.results["smash_metrics"]
            text = "Smash Day Strategy Metrics\n\n"
            text += f"Total Trades: {metrics.get('total_trades', 0)}\n"
            text += f"Trades per Year: {metrics.get('trades_per_year', 0):.1f}\n\n"
            text += "Exit Breakdown:\n"
            text += f"  Bailout Exits: {metrics.get('bailout_exits', 0)} "
            text += f"(Avg: {metrics.get('avg_return_bailout', 0)*100:.2f}%)\n"
            text += f"  Stop Loss Exits: {metrics.get('stop_loss_exits', 0)} "
            text += f"(Avg: {metrics.get('avg_return_stop_loss', 0)*100:.2f}%)\n"
            text += f"  Time Exits: {metrics.get('time_exits', 0)} "
            text += f"(Avg: {metrics.get('avg_return_time_exit', 0)*100:.2f}%)\n"
            text += f"  Opposite Signal: {metrics.get('opposite_signal_exits', 0)} "
            text += f"(Avg: {metrics.get('avg_return_opposite', 0)*100:.2f}%)\n"

            ax.text(
                0.1,
                0.9,
                text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                fontfamily="monospace",
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/signal_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _save_detailed_results(self, output_path):
        try:
            # Save returns
            self.results["portfolio_returns"].to_csv(
                f"{output_path}/portfolio_returns.csv"
            )

            # Save weights
            self.results["target_weights"].to_csv(f"{output_path}/target_weights.csv")

            # Save trade log
            if not self.results["trade_log"].empty:
                self.results["trade_log"].to_csv(
                    f"{output_path}/trade_log.csv", index=False
                )

            # Save performance stats
            self.results["performance_stats"].to_csv(
                f"{output_path}/performance_stats.csv"
            )

            # Save signal statistics
            if "signal_stats" in self.results:
                self.results["signal_stats"].to_csv(f"{output_path}/signal_stats.csv")

            # Save Smash Day specific metrics
            if "smash_metrics" in self.results:
                pd.Series(self.results["smash_metrics"]).to_csv(
                    f"{output_path}/smash_metrics.csv"
                )

            # Save execution log
            pd.DataFrame(self.execution_log).to_csv(
                f"{output_path}/execution_log.csv", index=False
            )

        except Exception as e:
            self.log_execution_step("Save Results", "ERROR", str(e))

    def _print_results_summary(self):
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY - SMASH DAY STRATEGY")
        print("=" * 60)

        portfolio_return = self.results["portfolio_returns"].sum()
        benchmark_return = self.results["benchmark_returns"].sum()

        print(f"Portfolio Total Return: {portfolio_return:.2%}")
        print(f"Benchmark Total Return: {benchmark_return:.2%}")
        print(f"Excess Return: {portfolio_return - benchmark_return:.2%}")

        if "performance_stats" in self.results:
            strategy_stats = self.results["performance_stats"].loc["Smash Day Strategy"]
            print(f"Sharpe Ratio: {strategy_stats['Sharpe Ratio']}")
            print(f"Max Drawdown: {strategy_stats['Max Drawdown']}")
            print(f"Hit Ratio: {strategy_stats['Hit Ratio']}")

        if "smash_metrics" in self.results:
            metrics = self.results["smash_metrics"]
            print(f"\nTotal Trades: {metrics.get('total_trades', 0)}")
            print(f"Trades per Year: {metrics.get('trades_per_year', 0):.1f}")
            print(
                f"Bailout Success Rate: {metrics.get('bailout_exits', 0)/metrics.get('total_trades', 1)*100:.1f}%"
            )

        print("=" * 60)


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/Users/enne/Documents/dev/python-mvp-strategies"
    START_DATE = "2018-01-01"
    END_DATE = "2023-12-31"
    OUTPUT_PATH = "output/2"

    # Check data path
    if not os.path.exists(DATA_PATH):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        possible_data_dirs = []
        for root, dirs, files in os.walk(parent_dir):
            if any(f.endswith("_data.csv") for f in files):
                possible_data_dirs.append(root)

        if possible_data_dirs:
            print("Found possible data directories:")
            for path in possible_data_dirs:
                print(f" - {path}")

        sys.exit(1)

    # Create output directory
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Initialize configuration
    config = SmashDayConfig()

    # Optional: Update configuration
    config.update_config(
        position_size=0.5,  # Conservative for leveraged ETFs
        stop_loss_pct=0.02,  # 2% stop loss
        lookback_period=3,  # 3-day lookback for high/low
        bailout_exit=True,  # Exit on first positive return
        max_holding_days=3,  # Maximum 3-day holding period
    )

    # Create and run executor
    executor = SmashDayIntegratedExecution(config)

    try:
        results = executor.run_complete_analysis(
            data_path=DATA_PATH,
            start_date=START_DATE,
            end_date=END_DATE,
            output_path=OUTPUT_PATH,
        )

        print(f"📊 Results available in: {OUTPUT_PATH}")
        print(f"📋 Execution log contains {len(executor.execution_log)} steps")

        # Run validation tests
        # test_success = run_all_smash_day_tests()
        # print(f"🧪 Unit tests {'passed' if test_success else 'failed'}")

    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
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


def run_all_smash_day_tests():
    """Run unit tests for Smash Day strategy components"""
    test_results = []

    # Test 1: Data loading
    try:
        temp_dir = tempfile.mkdtemp()

        # Create test data with OHLC structure
        dates = pd.date_range("2020-01-01", periods=100)
        test_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": 100 + np.random.randn(100) * 2,
                "High": 102 + np.random.randn(100) * 2,
                "Low": 98 + np.random.randn(100) * 2,
                "Close": 100 + np.random.randn(100) * 2,
                "Volume": 1000000 + np.random.randint(-100000, 100000, 100),
            }
        )

        # Ensure OHLC relationships
        test_data["High"] = test_data[["Open", "High", "Close"]].max(axis=1)
        test_data["Low"] = test_data[["Open", "Low", "Close"]].min(axis=1)

        test_data.to_csv(f"{temp_dir}/KODEX 200_data.csv", index=False)

        etf_data = load_smash_day_data(
            temp_dir, ["KODEX 200"], "2020-01-01", "2020-04-09"
        )
        test_results.append(len(etf_data) > 0)

        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"Test 1 failed: {e}")
        test_results.append(False)

    # Test 2: Signal generation
    try:
        dates = pd.date_range("2020-01-01", periods=300)
        test_etf_data = {
            "KODEX 200": pd.DataFrame(
                {
                    "Open": 100 + np.cumsum(np.random.randn(300) * 0.5),
                    "High": 102 + np.cumsum(np.random.randn(300) * 0.5),
                    "Low": 98 + np.cumsum(np.random.randn(300) * 0.5),
                    "Close": 100 + np.cumsum(np.random.randn(300) * 0.5),
                    "Volume": 1000000,
                },
                index=dates,
            )
        }

        # Fix OHLC relationships
        for etf, df in test_etf_data.items():
            df["High"] = df[["Open", "High", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

        signals = calculate_smash_day_signals(test_etf_data, lookback_period=3)
        test_results.append(
            "KODEX 200" in signals
            and "buy" in signals["KODEX 200"]
            and len(signals["KODEX 200"]["buy"]) > 0
        )

    except Exception as e:
        print(f"Test 2 failed: {e}")
        test_results.append(False)

    # Test 3: Weight calculation
    try:
        test_signals = {
            "KOSPI200": {
                "buy": pd.Series(
                    [1, 0, 0], index=pd.date_range("2020-01-01", periods=3)
                ),
                "sell": pd.Series(
                    [0, 1, 0], index=pd.date_range("2020-01-01", periods=3)
                ),
            }
        }

        test_mapping = {
            "KOSPI200": {"long": "KODEX 레버리지", "inverse": "KODEX 인버스"}
        }

        weights = calculate_smash_day_weights(test_signals, test_mapping, 0.5)
        test_results.append(len(weights) == 3 and weights.shape[1] >= 2)

    except Exception as e:
        print(f"Test 3 failed: {e}")
        test_results.append(False)

    # Test 4: Backtest execution
    try:
        dates = pd.date_range("2020-01-01", periods=50)

        test_etf_data = {
            "KODEX 레버리지": pd.DataFrame(
                {
                    "Open": 100 + np.random.randn(50) * 2,
                    "High": 102 + np.random.randn(50) * 2,
                    "Low": 98 + np.random.randn(50) * 2,
                    "Close": 100 + np.random.randn(50) * 2,
                    "Volume": 1000000,
                },
                index=dates,
            )
        }

        # Fix OHLC
        for etf, df in test_etf_data.items():
            df["High"] = df[["Open", "High", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

        test_weights = pd.DataFrame(
            np.random.choice([0, 0.5], size=(50, 1)),
            index=dates,
            columns=["KODEX 레버리지"],
        )

        test_signals = {
            "KOSPI200": {
                "buy": pd.Series(np.random.choice([0, 1], 50), index=dates),
                "sell": pd.Series(np.random.choice([0, 1], 50), index=dates),
            }
        }

        returns, values, trades = backtest_smash_day_strategy(
            test_etf_data,
            test_signals,
            test_weights,
            {"KOSPI200": {"long": "KODEX 레버리지", "inverse": "KODEX 인버스"}},
            initial_capital=1000000,
            transaction_cost=0.003,
            stop_loss_pct=0.02,
        )

        test_results.append(returns is not None and len(returns) > 0)

    except Exception as e:
        print(f"Test 4 failed: {e}")
        test_results.append(False)

    # Test 5: Performance metrics calculation
    try:
        test_returns = pd.Series(
            np.random.randn(252) * 0.01, index=pd.date_range("2020-01-01", periods=252)
        )

        stats = report_stats(test_returns, name="Test Strategy")
        test_results.append(len(stats) > 0 and "Sharpe Ratio" in stats.columns)

    except Exception as e:
        print(f"Test 5 failed: {e}")
        test_results.append(False)

    # Calculate success rate
    success_rate = sum(test_results) / len(test_results)
    overall_success = success_rate >= 0.8

    print(
        f"\nTest Results: {sum(test_results)}/{len(test_results)} passed ({success_rate:.0%})"
    )

    return overall_success


def validate_smash_day_inputs(data_path, etf_list, start_date, end_date, config=None):
    """Enhanced input validation for Smash Day strategy"""

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

    if (end_dt - start_dt).days < 90:
        raise ValueError(
            "Backtest period must be at least 90 days for Smash Day strategy"
        )

    if config is not None:
        required_config_keys = [
            "position_size",
            "lookback_period",
            "stop_loss_pct",
            "transaction_cost",
        ]
        for key in required_config_keys:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")

        if not 0 < config["position_size"] <= 1.0:
            raise ValueError("Position size must be between 0 and 1")

        if config["lookback_period"] < 1 or config["lookback_period"] > 10:
            raise ValueError("Lookback period must be between 1 and 10 days")

        if config["stop_loss_pct"] < 0.01 or config["stop_loss_pct"] > 0.1:
            raise ValueError("Stop loss must be between 1% and 10%")

        if config["transaction_cost"] < 0 or config["transaction_cost"] > 0.01:
            raise ValueError("Transaction cost must be between 0 and 1%")

    return True


def handle_signal_generation_errors(etf_data, lookback_period=3):
    """Error handling for Smash Day signal generation"""
    signals_dict = {}

    for etf, df in etf_data.items():
        try:
            # Skip inverse ETFs
            if "인버스" in etf or "INVERSE" in etf.upper():
                continue

            # Initialize empty signals
            signals_dict[etf] = {
                "buy": pd.Series(0, index=df.index),
                "sell": pd.Series(0, index=df.index),
                "bearish_smash": pd.Series(False, index=df.index),
                "bullish_smash": pd.Series(False, index=df.index),
            }

        except Exception as e:
            print(f"Error generating signals for {etf}: {e}")
            # Return empty signals
            signals_dict[etf] = {
                "buy": pd.Series(0, index=df.index),
                "sell": pd.Series(0, index=df.index),
                "bearish_smash": pd.Series(False, index=df.index),
                "bullish_smash": pd.Series(False, index=df.index),
            }

    return signals_dict


def safe_backtest_execution(
    etf_data, signals_dict, target_weights, etf_mapping, **kwargs
):
    """Safe wrapper for backtest execution with error handling"""
    try:
        if not etf_data or not signals_dict or target_weights.empty:
            raise ValueError("Invalid input data for backtest")

        # Validate data alignment
        all_dates = target_weights.index
        for etf in target_weights.columns:
            if etf in etf_data:
                etf_dates = etf_data[etf].index
                common_dates = all_dates.intersection(etf_dates)

                if len(common_dates) < len(all_dates) * 0.8:
                    print(f"Warning: {etf} has insufficient data overlap")

        returns, values, trades = backtest_smash_day_strategy(
            etf_data, signals_dict, target_weights, etf_mapping, **kwargs
        )

        if returns is None or returns.isnull().all():
            raise ValueError("Backtest produced invalid results")

        if len(returns) == 0:
            raise ValueError("Backtest produced empty results")

        return returns, values, trades

    except Exception as e:
        print(f"Backtest execution error: {e}")
        # Return empty results
        empty_returns = pd.Series(0, index=target_weights.index)
        empty_values = pd.Series(
            kwargs.get("initial_capital", 1000000), index=target_weights.index
        )
        empty_trades = pd.DataFrame()

        return empty_returns, empty_values, empty_trades
