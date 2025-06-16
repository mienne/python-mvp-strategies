#!/usr/bin/env python3

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


def remove_market_holidays(df, market="XKRX"):
    try:
        cal = mcal.get_calendar(market)
        holidays = cal.holidays().holidays
        df = df[~df.index.isin(holidays)]
        return df
    except:
        return df[df.index.weekday < 5]


def load_korean_etf_data(data_path, etf_list, start_date, end_date):
    etf_data = {}

    for etf in etf_list:
        try:
            filename = f"{data_path}/{etf}_data.csv"
            if not os.path.exists(filename):
                continue

            df = pd.read_csv(filename)

            date_cols = ["Date", "DateTime", "Date-Time", "date", "datetime", "날짜"]
            date_col = next((col for col in date_cols if col in df.columns), None)

            if date_col is None:
                continue

            df["Date"] = pd.to_datetime(df[date_col])
            df.set_index("Date", inplace=True)

            column_mapping = {
                "close": "Close",
                "Close": "Close",
                "종가": "Close",
                "open": "Open",
                "Open": "Open",
                "시가": "Open",
                "high": "High",
                "High": "High",
                "고가": "High",
                "low": "Low",
                "Low": "Low",
                "저가": "Low",
                "volume": "Volume",
                "Volume": "Volume",
                "거래량": "Volume",
            }

            df.rename(columns=column_mapping, inplace=True)

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    if col == "Volume":
                        df[col] = 100000
                    else:
                        continue

            df = df.loc[start_date:end_date]

            if len(df) < 250:
                continue

            df[["Open", "High", "Low", "Close"]].fillna(method="ffill", inplace=True)
            df["Volume"].fillna(100000, inplace=True)

            # 한국 시장 휴일 제거
            df = remove_market_holidays(df, market="XKRX")
            df = df[df.index.weekday < 5]

            etf_data[etf] = df

        except Exception as e:
            continue

    return etf_data


def calculate_ema_signals(etf_data, ema_period=112, lookback=250):
    """EMA 기반 트렌드 신호 계산"""
    signals_dict = {}
    returns_dict = {}

    for etf, df in etf_data.items():
        # 일간 수익률 계산
        df["Returns"] = df["Close"].pct_change()

        # 정규화된 수익률 (lookback 기간)
        rolling_mean = (
            df["Returns"].rolling(window=lookback, min_periods=lookback // 2).mean()
        )
        rolling_std = (
            df["Returns"].rolling(window=lookback, min_periods=lookback // 2).std()
        )
        df["Normalized_Returns"] = (df["Returns"] - rolling_mean) / (rolling_std + 1e-6)

        # EMA 계산
        df[f"EMA_{ema_period}"] = (
            df["Normalized_Returns"].ewm(span=ema_period, adjust=False).mean()
        )

        # 신호 생성 (EMA가 양수면 1, 음수면 -1)
        df["Signal_Raw"] = np.where(df[f"EMA_{ema_period}"] > 0, 1, -1)

        # 신호 강도 (EMA 절대값)
        df["Signal_Strength"] = np.abs(df[f"EMA_{ema_period}"])

        signals_dict[etf] = df[["Signal_Raw", "Signal_Strength"]].copy()
        returns_dict[etf] = df["Returns"].copy()

    return signals_dict, returns_dict


def calculate_correlation_weights(returns_dict, lookback=250):
    """상관관계 매트릭스 기반 가중치 계산"""
    # 모든 ETF의 수익률을 하나의 DataFrame으로 결합
    returns_df = pd.DataFrame(returns_dict)

    # 날짜별 상관관계 가중치 저장
    correlation_weights = {}

    for date in returns_df.index[lookback:]:
        # lookback 기간의 상관관계 매트릭스 계산
        window_data = returns_df.loc[:date].tail(lookback)
        corr_matrix = window_data.corr()

        # 상관관계 역제곱근 계산 (다각화 효과)
        # 각 ETF의 평균 상관관계
        avg_corr = corr_matrix.mean(axis=1)

        # 상관관계가 낮을수록 높은 가중치
        weights = 1 / (avg_corr + 0.5)  # 0.5는 안정성을 위한 상수
        weights = weights / weights.sum()  # 정규화

        correlation_weights[date] = weights

    return pd.DataFrame(correlation_weights).T


def calculate_position_sizes(
    signals_dict, correlation_weights, target_vol=0.15, num_tranches=5
):
    """포지션 크기 계산 (분할 진입 고려)"""
    position_sizes = {}

    # 모든 날짜의 통합
    all_dates = set()
    for etf_signals in signals_dict.values():
        all_dates.update(etf_signals.index)
    all_dates = sorted(all_dates)

    for date in all_dates:
        daily_positions = {}

        # 해당 날짜의 상관관계 가중치
        if date in correlation_weights.index:
            corr_weights = correlation_weights.loc[date]
        else:
            # 이전 가중치 사용
            prev_dates = correlation_weights.index[correlation_weights.index < date]
            if len(prev_dates) > 0:
                corr_weights = correlation_weights.loc[prev_dates[-1]]
            else:
                continue

        # 각 ETF의 신호와 포지션 계산
        total_signal_strength = 0
        etf_signals = {}

        for etf in signals_dict.keys():
            if date in signals_dict[etf].index:
                signal = signals_dict[etf].loc[date, "Signal_Raw"]
                strength = signals_dict[etf].loc[date, "Signal_Strength"]

                # 상관관계 가중치 적용
                adjusted_strength = strength * corr_weights.get(
                    etf, 1 / len(signals_dict)
                )

                etf_signals[etf] = {"signal": signal, "strength": adjusted_strength}
                total_signal_strength += adjusted_strength

        # 포지션 크기 계산 (전체 자본의 비율)
        if total_signal_strength > 0:
            for etf, signal_info in etf_signals.items():
                # 기본 포지션 크기
                base_position = (
                    signal_info["strength"] / total_signal_strength
                ) * signal_info["signal"]

                # 분할 진입을 위한 트랜치 크기 (20% 단위)
                tranche_size = base_position / num_tranches

                daily_positions[etf] = {
                    "target_position": base_position,
                    "tranche_size": tranche_size,
                    "signal": signal_info["signal"],
                }

        position_sizes[date] = daily_positions

    return position_sizes


def implement_position_management(position_sizes, rebalance_freq="W"):
    """포지션 관리 및 리밸런싱"""
    managed_positions = {}
    current_positions = {}

    position_df = pd.DataFrame(position_sizes).T

    # 주간 리밸런싱
    if rebalance_freq == "W":
        # 금요일마다 리밸런싱
        rebalance_dates = position_df.index[position_df.index.weekday == 4]
    else:
        rebalance_dates = position_df.index

    for date in position_df.index:
        daily_positions = {}

        # 리밸런싱 날짜인지 확인
        is_rebalance_day = date in rebalance_dates

        if date in position_sizes and position_sizes[date]:
            for etf, target_info in position_sizes[date].items():
                if is_rebalance_day:
                    # 목표 포지션으로 조정
                    target_pos = target_info["target_position"]
                    current_pos = current_positions.get(etf, 0)

                    # 분할 진입/청산 로직
                    if (
                        abs(target_pos - current_pos)
                        > target_info["tranche_size"] * 0.5
                    ):
                        # 한 트랜치씩 조정
                        if target_pos > current_pos:
                            new_pos = current_pos + target_info["tranche_size"]
                        else:
                            new_pos = current_pos - target_info["tranche_size"]

                        # 목표 초과 방지
                        if abs(new_pos) > abs(target_pos):
                            new_pos = target_pos

                        current_positions[etf] = new_pos
                        daily_positions[etf] = new_pos
                    else:
                        daily_positions[etf] = current_pos
                else:
                    # 리밸런싱 날이 아니면 현재 포지션 유지
                    daily_positions[etf] = current_positions.get(etf, 0)

        managed_positions[date] = daily_positions

    return pd.DataFrame(managed_positions).T.fillna(0)


def backtest_korean_etf_strategy(
    etf_data,
    managed_positions,
    initial_capital=100000000,  # 1억원
    transaction_cost=0.002,  # 0.2% (세금 포함)
    stop_loss=-0.15,  # -15% 손절
):
    """백테스트 실행"""
    portfolio_returns = []
    portfolio_values = [initial_capital]
    trade_log = []
    position_values = {}

    # 포지션별 진입 가격 추적
    entry_prices = {}

    for i, date in enumerate(managed_positions.index):
        if i == 0:
            daily_return = 0.0
            for etf in managed_positions.columns:
                if etf in etf_data and date in etf_data[etf].index:
                    position = managed_positions.loc[date, etf]
                    if position != 0:
                        entry_prices[etf] = etf_data[etf].loc[date, "Close"]
                        trade_log.append(
                            {
                                "date": date,
                                "etf": etf,
                                "action": "INITIAL",
                                "position": position,
                                "price": entry_prices[etf],
                            }
                        )
        else:
            prev_date = managed_positions.index[i - 1]
            daily_pnl = 0.0
            transaction_costs = 0.0

            for etf in managed_positions.columns:
                if etf not in etf_data:
                    continue

                current_pos = managed_positions.loc[date, etf]
                prev_pos = managed_positions.loc[prev_date, etf]

                if date in etf_data[etf].index and prev_date in etf_data[etf].index:
                    # 수익률 계산
                    price_change = (
                        etf_data[etf].loc[date, "Close"]
                        / etf_data[etf].loc[prev_date, "Close"]
                        - 1
                    )

                    # 포지션 수익
                    position_return = prev_pos * price_change
                    daily_pnl += position_return

                    # 손절 체크
                    if etf in entry_prices and prev_pos != 0:
                        current_price = etf_data[etf].loc[date, "Close"]
                        entry_price = entry_prices[etf]
                        position_pnl = (current_price / entry_price - 1) * np.sign(
                            prev_pos
                        )

                        if position_pnl <= stop_loss:
                            # 손절 실행
                            managed_positions.loc[date:, etf] = 0
                            current_pos = 0
                            trade_log.append(
                                {
                                    "date": date,
                                    "etf": etf,
                                    "action": "STOP_LOSS",
                                    "position": 0,
                                    "price": current_price,
                                    "pnl": position_pnl,
                                }
                            )

                    # 포지션 변경 시 거래비용
                    position_change = abs(current_pos - prev_pos)
                    if position_change > 0.001:
                        transaction_costs += position_change * transaction_cost

                        # 진입가격 업데이트
                        if current_pos != 0 and prev_pos == 0:
                            entry_prices[etf] = etf_data[etf].loc[date, "Close"]
                        elif current_pos == 0:
                            entry_prices.pop(etf, None)

                        # 거래 기록
                        trade_log.append(
                            {
                                "date": date,
                                "etf": etf,
                                "action": "BUY" if current_pos > prev_pos else "SELL",
                                "position_from": prev_pos,
                                "position_to": current_pos,
                                "price": etf_data[etf].loc[date, "Close"],
                                "cost": position_change * transaction_cost,
                            }
                        )

            # 일간 수익률
            daily_return = daily_pnl - transaction_costs

        portfolio_returns.append(daily_return)
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)

    returns_series = pd.Series(portfolio_returns, index=managed_positions.index)
    values_series = pd.Series(portfolio_values[1:], index=managed_positions.index)
    trade_df = pd.DataFrame(trade_log)

    return returns_series, values_series, trade_df


def report_stats(daily_ret, freq=252, rf=0.02, name="Strategy"):
    """성과 통계 계산"""
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
    """백테스트 리포트 생성"""
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
        # 표지
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

        # 성과 요약
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

        # 수익률 곡선
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


def validate_korean_etf_inputs(data_path, etf_list, start_date, end_date, config=None):
    """입력값 검증"""
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
        required_config_keys = ["ema_period", "target_vol", "transaction_cost"]
        for key in required_config_keys:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")

        if config["ema_period"] < 20 or config["ema_period"] > 200:
            raise ValueError("EMA period must be between 20 and 200 days")

        if config["target_vol"] < 0.05 or config["target_vol"] > 0.30:
            raise ValueError("Target volatility must be between 5% and 30%")

        if config["transaction_cost"] < 0 or config["transaction_cost"] > 0.01:
            raise ValueError("Transaction cost must be between 0 and 1%")


class KoreanETFConfig:
    """한국 ETF 전략 설정"""

    def __init__(self):
        # ETF 리스트 (주요 섹터/테마)
        self.etf_list = [
            "069500",
            "229200",
            "466810",
            "367380",
            "357870",
            "139290",
            "117460",
            "425040",
            "228790",
            "227830",
            "365040",
            "091230",
            "391600",
            "381180",
            "395160",
            "098560",
            "365000",
            "157490",
            "091170",
            "091220",
            "117680",
            "228810",
            "387280",
            "448320",
            "140710",
            "091160",
            "448300",
            "448310",
            "395170",
            "091180",
            "139270",
            "305540",
            "395150",
        ]

        self.etf_names = {
            "069500": "KODEX 200",
            "229200": "KODEX 코스닥150",
            "466810": "PLUS 우주항공&UAM",
            "367380": "PLUS K방산",
            "357870": "PLUS 태양광&ESS",
            "139290": "TIGER 200 철강소재",
            "117460": "TIGER 200 에너지화학",
            "425040": "TIGER 미디어컨텐츠",
            "228790": "TIGER 여행레저",
            "227830": "TIGER 지주회사",
            "365040": "TIGER 게임TOP10",
            "091230": "TIGER 200 헬스케어",
            "391600": "TIGER 코스닥150바이오테크",
            "381180": "KIWOOM Fn유전자혁신기술",
            "395160": "SOL 의료기기소부장Fn",
            "098560": "TIGER 방송통신",
            "365000": "TIGER 인터넷TOP10",
            "157490": "TIGER 소프트웨어",
            "091170": "KODEX 은행",
            "091220": "TIGER 200 금융",
            "117680": "KODEX 증권",
            "228810": "TIGER 화장품",
            "387280": "HANARO K-뷰티",
            "448320": "SOL 조선TOP3플러스",
            "140710": "KODEX 운송",
            "091160": "KODEX 반도체",
            "448300": "SOL 반도체전공정",
            "448310": "SOL 반도체후공정",
            "395170": "SOL 자동차소부장Fn",
            "091180": "KODEX 자동차",
            "139270": "TIGER 200 건설",
            "305540": "TIGER 2차전지TOP10",
            "395150": "SOL 2차전지소부장Fn",
        }

        # 전략 파라미터
        self.ema_period = 112  # 원본 전략 유지
        self.lookback_period = 250  # 상관관계 계산 기간
        self.target_vol = 0.15  # 목표 변동성 15%
        self.rebalance_freq = "W"  # 주간 리밸런싱
        self.num_tranches = 5  # 5회 분할 진입

        # 리스크 관리
        self.stop_loss = -0.15  # -15% 손절
        self.position_limit = 0.20  # 개별 종목 최대 20%

        # 백테스트 설정
        self.initial_capital = 100000000  # 1억원
        self.transaction_cost = 0.002  # 0.2% (세금 포함)
        self.min_trade_threshold = 0.01  # 최소 거래 단위

        # 시장 설정
        self.timezone = "Asia/Seoul"
        self.market_calendar = "XKRX"
        self.min_data_points = 250

        # 벤치마크
        self.benchmark_etf = "069500"  # KODEX 200
        self.risk_free_rate = 0.0325  # 한국 기준금리
        self.trading_days_per_year = 248  # 한국 시장

        # 리포트 설정
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


class KoreanETFIntegratedExecution:
    """한국 ETF 전략 통합 실행 클래스"""

    def __init__(self, config=None):
        self.config = config or KoreanETFConfig()
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
            validate_korean_etf_inputs(
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
            etf_data = load_korean_etf_data(
                data_path,
                self.config.etf_list,
                start_date,
                end_date,
            )

            if not etf_data:
                raise ValueError("No ETF data loaded successfully")

            self.results["etf_data"] = etf_data
            self.log_execution_step(
                "Data Loading",
                "SUCCESS",
                f"Loaded {len(etf_data)} ETFs successfully",
            )

            return True

        except Exception as e:
            self.log_execution_step("Data Loading", "ERROR", str(e))
            return False

    def execute_signal_generation_phase(self):
        self.log_execution_step(
            "Signal Generation", "IN_PROGRESS", "Calculating EMA signals..."
        )

        try:
            signals_dict, returns_dict = calculate_ema_signals(
                self.results["etf_data"],
                ema_period=self.config.ema_period,
                lookback=self.config.lookback_period,
            )

            self.results["signals"] = signals_dict
            self.results["returns"] = returns_dict

            # 신호 통계
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

    def execute_correlation_analysis_phase(self):
        self.log_execution_step(
            "Correlation Analysis", "IN_PROGRESS", "Calculating correlation weights..."
        )

        try:
            correlation_weights = calculate_correlation_weights(
                self.results["returns"], lookback=self.config.lookback_period
            )

            self.results["correlation_weights"] = correlation_weights

            # 상관관계 통계
            corr_stats = self._calculate_correlation_statistics(correlation_weights)
            self.results["correlation_stats"] = corr_stats

            self.log_execution_step(
                "Correlation Analysis",
                "SUCCESS",
                f"Calculated correlation weights for {len(correlation_weights)} days",
            )

            return True

        except Exception as e:
            self.log_execution_step("Correlation Analysis", "ERROR", str(e))
            return False

    def execute_position_sizing_phase(self):
        self.log_execution_step(
            "Position Sizing", "IN_PROGRESS", "Calculating position sizes..."
        )

        try:
            position_sizes = calculate_position_sizes(
                self.results["signals"],
                self.results["correlation_weights"],
                target_vol=self.config.target_vol,
                num_tranches=self.config.num_tranches,
            )

            self.results["position_sizes"] = position_sizes

            self.log_execution_step(
                "Position Sizing",
                "SUCCESS",
                f"Calculated position sizes for {len(position_sizes)} days",
            )

            return True

        except Exception as e:
            self.log_execution_step("Position Sizing", "ERROR", str(e))
            return False

    def execute_position_management_phase(self):
        self.log_execution_step(
            "Position Management", "IN_PROGRESS", "Implementing position management..."
        )

        try:
            managed_positions = implement_position_management(
                self.results["position_sizes"],
                rebalance_freq=self.config.rebalance_freq,
            )

            self.results["managed_positions"] = managed_positions

            # 포지션 통계
            position_stats = self._calculate_position_statistics(managed_positions)
            self.results["position_stats"] = position_stats

            self.log_execution_step(
                "Position Management",
                "SUCCESS",
                f"Managed positions for {len(managed_positions)} days",
            )

            return True

        except Exception as e:
            self.log_execution_step("Position Management", "ERROR", str(e))
            return False

    def execute_backtesting_phase(self):
        self.log_execution_step(
            "Backtesting", "IN_PROGRESS", "Running backtest simulation..."
        )

        try:
            returns, values, trades = backtest_korean_etf_strategy(
                self.results["etf_data"],
                self.results["managed_positions"],
                initial_capital=self.config.initial_capital,
                transaction_cost=self.config.transaction_cost,
                stop_loss=self.config.stop_loss,
            )

            self.results["portfolio_returns"] = returns
            self.results["portfolio_values"] = values
            self.results["trade_log"] = trades

            # 벤치마크 수익률
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

    def execute_performance_analysis_phase(self):
        self.log_execution_step(
            "Performance Analysis", "IN_PROGRESS", "Calculating performance metrics..."
        )

        try:
            strategy_stats = report_stats(
                self.results["portfolio_returns"],
                freq=self.config.trading_days_per_year,
                rf=self.config.risk_free_rate,
                name="Korean ETF Strategy",
            )

            benchmark_stats = report_stats(
                self.results["benchmark_returns"],
                freq=self.config.trading_days_per_year,
                rf=self.config.risk_free_rate,
                name="Benchmark (KODEX 200)",
            )

            combined_stats = pd.concat([strategy_stats, benchmark_stats])
            self.results["performance_stats"] = combined_stats

            # 추가 리스크 메트릭
            risk_metrics = self._calculate_risk_metrics()
            self.results["risk_metrics"] = risk_metrics

            # 수익 기여도 분석
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

                pdf_filename = f"{output_path}/korean_etf_strategy_report.pdf"
                report_backtest(
                    report_data,
                    filename=pdf_filename,
                    strategy_name="Korean ETF Trend Following Strategy (EMA-based)",
                    author="Korean ETF Strategy Framework",
                )

                self.log_execution_step(
                    "PDF Report", "SUCCESS", f"Generated PDF report: {pdf_filename}"
                )

            # 추가 시각화
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
        """전체 분석 실행"""
        self.start_time = datetime.datetime.now()

        print("=" * 80)
        print("KOREAN ETF TREND FOLLOWING STRATEGY - INTEGRATED ANALYSIS")
        print("=" * 80)
        print(f"Analysis Period: {start_date} to {end_date}")
        print(
            f"Strategy Config: {self.config.ema_period}-day EMA, {self.config.target_vol:.0%} target vol"
        )
        print(f"ETFs: {len(self.config.etf_list)} Korean sector/theme ETFs")
        print(f"Execution Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if output_path is None:
            output_path = data_path

        # 실행 단계
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
            ("Analyze Correlations", lambda: self.execute_correlation_analysis_phase()),
            ("Calculate Position Sizes", lambda: self.execute_position_sizing_phase()),
            ("Manage Positions", lambda: self.execute_position_management_phase()),
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

        self.results["execution_metadata"] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": total_duration if hasattr(self, "end_time") else None,
            "config": self.config.get_config_dict(),
            "execution_log": self.execution_log,
        }

        return self.results

    # Helper 메서드들
    def _calculate_signal_statistics(self, signals_dict):
        stats = {}
        for etf, signal_df in signals_dict.items():
            signal = signal_df["Signal_Raw"]
            strength = signal_df["Signal_Strength"]

            stats[etf] = {
                "total_days": len(signal),
                "long_days": (signal == 1).sum(),
                "short_days": (signal == -1).sum(),
                "long_ratio": (signal == 1).mean(),
                "avg_strength": strength.mean(),
                "signal_changes": (signal.diff() != 0).sum(),
            }

        return pd.DataFrame(stats).T

    def _calculate_correlation_statistics(self, correlation_weights):
        return {
            "avg_weight_dispersion": correlation_weights.std(axis=1).mean(),
            "max_weight": correlation_weights.max().max(),
            "min_weight": correlation_weights.min().min(),
            "weight_concentration": (correlation_weights**2).sum(axis=1).mean(),
        }

    def _calculate_position_statistics(self, managed_positions):
        return {
            "avg_num_positions": (managed_positions != 0).sum(axis=1).mean(),
            "max_num_positions": (managed_positions != 0).sum(axis=1).max(),
            "avg_position_size": managed_positions.abs().mean().mean(),
            "position_turnover": managed_positions.diff().abs().sum().sum(),
        }

    def _calculate_benchmark_returns(self):
        benchmark_data = self.results["etf_data"][self.config.benchmark_etf]
        benchmark_returns = benchmark_data["Close"].pct_change().dropna()

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
                and etf in self.results["managed_positions"].columns
            ):
                etf_returns = self.results["etf_data"][etf]["Close"].pct_change()
                etf_positions = self.results["managed_positions"][etf]

                common_dates = etf_returns.index.intersection(etf_positions.index)
                if len(common_dates) > 0:
                    aligned_returns = etf_returns.reindex(common_dates, fill_value=0)
                    aligned_positions = etf_positions.reindex(
                        common_dates, fill_value=0
                    )

                    etf_contribution = (
                        aligned_positions.shift(1) * aligned_returns
                    ).sum()

                    attribution[etf] = {
                        "total_contribution": etf_contribution,
                        "avg_position": aligned_positions.mean(),
                        "etf_total_return": aligned_returns.sum(),
                        "periods_held": (aligned_positions != 0).sum(),
                    }

        return pd.DataFrame(attribution).T

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
            self._plot_position_heatmap(output_path)
            self._plot_correlation_evolution(output_path)
            self._plot_performance_attribution(output_path)

        except Exception as e:
            self.log_execution_step("Additional Visualizations", "ERROR", str(e))

    def _plot_etf_signals(self, output_path):
        if not self.config.plot_individual_etfs:
            return

        # 상위 5개 ETF만 플롯
        top_etfs = list(self.config.etf_list[:5])

        fig, axes = plt.subplots(len(top_etfs), 1, figsize=(15, 3 * len(top_etfs)))
        if len(top_etfs) == 1:
            axes = [axes]

        for idx, etf in enumerate(top_etfs):
            if etf in self.results["etf_data"]:
                data = self.results["etf_data"][etf]
                signal = self.results["signals"][etf]["Signal_Raw"]

                ax = axes[idx]

                # 가격
                ax.plot(
                    data.index,
                    data["Close"],
                    label=f"{self.config.etf_names.get(etf, etf)} Price",
                    alpha=0.7,
                )

                # 매수/매도 구간 표시
                long_periods = signal == 1
                short_periods = signal == -1

                ax.fill_between(
                    data.index,
                    data["Close"].min(),
                    data["Close"].max(),
                    where=long_periods,
                    alpha=0.2,
                    color="green",
                    label="Long",
                )
                ax.fill_between(
                    data.index,
                    data["Close"].min(),
                    data["Close"].max(),
                    where=short_periods,
                    alpha=0.2,
                    color="red",
                    label="Short",
                )

                ax.set_title(f"{etf} - {self.config.etf_names.get(etf, etf)}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/etf_signals_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_position_heatmap(self, output_path):
        positions = self.results["managed_positions"]

        # 월별 평균 포지션
        monthly_positions = positions.resample("M").mean()

        fig, ax = plt.subplots(figsize=(15, 8))

        # ETF 이름으로 컬럼명 변경
        renamed_positions = monthly_positions.rename(columns=self.config.etf_names)

        sns.heatmap(
            renamed_positions.T,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Position Size"},
        )

        ax.set_title("Monthly Average Positions Heatmap")
        ax.set_xlabel("Date")
        ax.set_ylabel("ETF")

        plt.tight_layout()
        plt.savefig(f"{output_path}/position_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_correlation_evolution(self, output_path):
        corr_weights = self.results["correlation_weights"]

        fig, ax = plt.subplots(figsize=(15, 8))

        # 상위 5개 ETF의 상관관계 가중치 추이
        top_etfs = list(self.config.etf_list[:5])

        for etf in top_etfs:
            if etf in corr_weights.columns:
                ax.plot(
                    corr_weights.index,
                    corr_weights[etf],
                    label=self.config.etf_names.get(etf, etf),
                    alpha=0.8,
                )

        ax.set_title("Correlation-based Weights Evolution")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/correlation_weights.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_performance_attribution(self, output_path):
        attribution = self.results["attribution_analysis"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ETF별 수익 기여도
        contribution = attribution["total_contribution"].sort_values(ascending=True)
        contribution.plot.barh(ax=ax1)
        ax1.set_title("Total Return Contribution by ETF")
        ax1.set_xlabel("Contribution")

        # ETF별 평균 포지션
        avg_position = attribution["avg_position"].sort_values(ascending=True)
        avg_position.plot.barh(ax=ax2)
        ax2.set_title("Average Position by ETF")
        ax2.set_xlabel("Average Position")

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/performance_attribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _save_detailed_results(self, output_path):
        try:
            # 수익률 저장
            self.results["portfolio_returns"].to_csv(
                f"{output_path}/portfolio_returns.csv"
            )

            # 포지션 저장
            self.results["managed_positions"].to_csv(
                f"{output_path}/managed_positions.csv"
            )

            # 거래 내역 저장
            if not self.results["trade_log"].empty:
                self.results["trade_log"].to_csv(
                    f"{output_path}/trade_log.csv", index=False
                )

            # 성과 통계 저장
            self.results["performance_stats"].to_csv(
                f"{output_path}/performance_stats.csv"
            )

            # 신호 통계 저장
            if "signal_stats" in self.results:
                self.results["signal_stats"].to_csv(f"{output_path}/signal_stats.csv")

            # 실행 로그 저장
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
            strategy_stats = self.results["performance_stats"].loc[
                "Korean ETF Strategy"
            ]
            print(f"Sharpe Ratio: {strategy_stats['Sharpe Ratio']}")
            print(f"Max Drawdown: {strategy_stats['Max Drawdown']}")
            print(f"Calmar Ratio: {strategy_stats['Calmar Ratio']}")

        if "trade_log" in self.results and not self.results["trade_log"].empty:
            num_trades = len(self.results["trade_log"])
            print(f"Total Trades: {num_trades}")

        print("=" * 60)


# 테스트 함수들
def run_korean_etf_tests():
    """한국 ETF 전략 테스트"""
    test_results = []

    # 테스트 1: 데이터 로딩
    try:
        temp_dir = tempfile.mkdtemp()
        test_data = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=300),
                "Close": 10000 + np.cumsum(np.random.randn(300) * 100),
                "Open": 10000,
                "High": 10100,
                "Low": 9900,
                "Volume": 1000000,
            }
        )
        test_data.to_csv(f"{temp_dir}/069500_data.csv", index=False)

        etf_data = load_korean_etf_data(
            temp_dir, ["069500"], "2020-01-01", "2020-12-31"
        )
        test_results.append(len(etf_data) > 0)

        shutil.rmtree(temp_dir)

    except Exception as e:
        test_results.append(False)

    # 테스트 2: 신호 생성
    try:
        dates = pd.date_range("2020-01-01", periods=300)
        test_etf_data = {
            "069500": pd.DataFrame(
                {
                    "Close": 10000 + np.cumsum(np.random.randn(300) * 100),
                    "Open": 10000,
                    "High": 10100,
                    "Low": 9900,
                    "Volume": 1000000,
                },
                index=dates,
            )
        }

        signals, returns = calculate_ema_signals(test_etf_data, ema_period=50)
        test_results.append("069500" in signals and len(signals["069500"]) > 0)

    except Exception as e:
        test_results.append(False)

    # 테스트 3: 상관관계 가중치
    try:
        test_returns = {
            "ETF1": pd.Series(
                np.random.randn(300), index=pd.date_range("2020-01-01", periods=300)
            ),
            "ETF2": pd.Series(
                np.random.randn(300), index=pd.date_range("2020-01-01", periods=300)
            ),
        }

        corr_weights = calculate_correlation_weights(test_returns, lookback=100)
        test_results.append(len(corr_weights) > 0)

    except Exception as e:
        test_results.append(False)

    success_rate = sum(test_results) / len(test_results)
    overall_success = success_rate >= 0.8

    return overall_success


if __name__ == "__main__":
    # 데이터 경로 설정
    DATA_PATH = "/Users/enne/Documents/dev/python-mvp-strategies/data"
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    OUTPUT_PATH = "output/2"

    # 데이터 경로 확인
    if not os.path.exists(DATA_PATH):
        # 현재 디렉토리에서 data 폴더 찾기
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

    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # 설정 생성
    config = KoreanETFConfig()

    # 설정 조정 (필요시)
    config.update_config(
        ema_period=112,  # 원본 전략 유지
        target_vol=0.15,  # 15% 목표 변동성
        transaction_cost=0.002,  # 0.2% (세금 포함)
        num_tranches=5,  # 5회 분할 진입
    )

    # 실행기 생성
    executor = KoreanETFIntegratedExecution(config)

    try:
        # 전체 분석 실행
        results = executor.run_complete_analysis(
            data_path=DATA_PATH,
            start_date=START_DATE,
            end_date=END_DATE,
            output_path=OUTPUT_PATH,
        )

        print(f"\n📊 Results available in: {OUTPUT_PATH}")
        print(f"📋 Execution log contains {len(executor.execution_log)} steps")

        # 단위 테스트 실행
        test_success = run_korean_etf_tests()
        print(f"🧪 Unit tests {'passed' if test_success else 'failed'}")

    except Exception as e:
        print(f"\n❌ Strategy execution failed: {str(e)}")
        print("📋 Check execution log for details")

        if hasattr(executor, "execution_log") and executor.execution_log:
            print("\nExecution Log:")
