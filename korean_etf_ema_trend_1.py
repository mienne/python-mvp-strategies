#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime, timedelta

import platform
import warnings

warnings.filterwarnings("ignore")


def set_korean_font():
    system = platform.system()

    if system == "Darwin":
        font_list = ["Apple SD Gothic Neo", "AppleGothic", "Arial Unicode MS"]
    elif system == "Windows":
        font_list = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic"]
    else:
        font_list = ["NanumGothic", "NanumBarunGothic", "UnDotum"]

    for font_name in font_list:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


class KoreanETFARPStrategy:
    """
    한국 ETF 시장에 맞춘 ARP-EMA 트렌드팔로잉 전략
    - 10-15개 섹터/테마 ETF
    - 112일 EMA 기반 트렌드 신호
    - 상관관계 매트릭스 리스크 조정
    - 분할 진입 및 주간 리밸런싱
    """

    def __init__(
        self,
        etf_list,
        lookback_ema=112,
        lookback_corr=250,
        target_vol=0.15,
        max_loss=-0.15,
        rebalance_freq="W",
    ):
        self.etf_list = etf_list
        self.lookback_ema = lookback_ema
        self.lookback_corr = lookback_corr
        self.target_vol = target_vol
        self.max_loss = max_loss
        self.rebalance_freq = rebalance_freq

        self.position_scale = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.current_positions = {}
        self.entry_prices = {}

        self.etf_mapping = {
            "KODEX 200": "069500",
            "KODEX 코스닥150": "229200",
            "PLUS 우주항공&UAM": "466810",
            "PLUS K방산": "367380",
            "PLUS 태양광&ESS": "357870",
            "TIGER 200 철강소재": "139290",
            "TIGER 200 에너지화학": "117460",
            "TIGER 미디어컨텐츠": "425040",
            "TIGER 여행레저": "228790",
            "TIGER 지주회사": "227830",
            "TIGER 게임TOP10": "365040",
            "TIGER 200 헬스케어": "091230",
            "TIGER 코스닥150바이오테크": "391600",
            "KIWOOM Fn유전자혁신기술": "381180",
            "SOL 의료기기소부장Fn": "395160",
            "TIGER 방송통신": "098560",
            "TIGER 인터넷TOP10": "365000",
            "TIGER 소프트웨어": "157490",
            "KODEX 은행": "091170",
            "TIGER 200 금융": "091220",
            "KODEX 증권": "117680",
            "TIGER 화장품": "228810",
            "HANARO K-뷰티": "387280",
            "SOL 조선TOP3플러스": "448320",
            "KODEX 운송": "140710",
            "KODEX 반도체": "091160",
            "SOL 반도체전공정": "448300",
            "SOL 반도체후공정": "448310",
            "SOL 자동차소부장Fn": "395170",
            "KODEX 자동차": "091180",
            "TIGER 200 건설": "139270",
            "TIGER 2차전지TOP10": "305540",
            "SOL 2차전지소부장Fn": "395150",
        }

    def fetch_data(self, start_date, end_date):
        data = {}

        for ticker in self.etf_list:
            try:
                if ticker in self.etf_mapping:
                    code = self.etf_mapping[ticker]
                    file_path = f"data/{code}_data.csv"

                    df = pd.read_csv(file_path)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)

                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                    if len(df) > 0:
                        data[ticker] = df

            except Exception as e:
                pass

        if data:
            prices = pd.DataFrame({ticker: df["Close"] for ticker, df in data.items()})
            returns = prices.pct_change().dropna()
        else:
            prices = pd.DataFrame()
            returns = pd.DataFrame()

        return prices, returns

    def calculate_ema_signals(self, prices):
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)

        for ticker in prices.columns:
            ema = prices[ticker].ewm(span=self.lookback_ema, adjust=False).mean()
            price_to_ema = (prices[ticker] - ema) / prices[ticker].rolling(20).std()
            signals[ticker] = np.tanh(price_to_ema)

        return signals

    def calculate_correlation_weights(self, returns):
        corr_matrix = returns.rolling(self.lookback_corr).corr()

        weights = pd.DataFrame(index=returns.index, columns=returns.columns)

        for date in corr_matrix.index.get_level_values(0).unique():
            if date in returns.index:
                try:
                    daily_corr = corr_matrix.loc[date]
                    inv_corr = np.linalg.inv(
                        daily_corr + 0.01 * np.eye(len(daily_corr))
                    )
                    w = inv_corr.sum(axis=1)
                    w = w / w.sum()
                    weights.loc[date] = w
                except:
                    weights.loc[date] = 1 / len(returns.columns)

        return weights

    def position_sizing(self, signals, weights, returns):
        vol = returns.rolling(20).std() * np.sqrt(252)

        raw_positions = signals * weights

        portfolio_vol = (raw_positions * vol).sum(axis=1)
        scaling_factor = self.target_vol / portfolio_vol

        positions = raw_positions * scaling_factor.values.reshape(-1, 1)
        positions = positions.clip(-1, 1)

        return positions

    def apply_risk_management(self, positions, prices):
        managed_positions = positions.copy()

        for ticker in positions.columns:
            for date in positions.index:
                if ticker in self.current_positions:
                    entry_price = self.entry_prices.get(
                        ticker, prices.loc[date, ticker]
                    )
                    current_price = prices.loc[date, ticker]

                    pnl = (current_price - entry_price) / entry_price

                    if pnl < self.max_loss:
                        managed_positions.loc[date, ticker] = 0

        return managed_positions

    def backtest(self, start_date, end_date):
        prices, returns = self.fetch_data(
            start_date - timedelta(days=self.lookback_corr * 2), end_date
        )

        # 신호 계산
        signals = self.calculate_ema_signals(prices)
        weights = self.calculate_correlation_weights(returns)

        # 포지션 계산
        positions = self.position_sizing(signals, weights, returns)
        positions = self.apply_risk_management(positions, prices)

        # 주간 리밸런싱 적용
        if self.rebalance_freq == "W":
            positions = (
                positions.resample("W-FRI")
                .last()
                .reindex(positions.index)
                .fillna(method="ffill")
            )

        # 수익률 계산
        strategy_returns = (positions.shift(1) * returns).sum(axis=1)

        # 성과 지표 계산
        results = self.calculate_performance_metrics(strategy_returns)

        return {
            "positions": positions,
            "returns": strategy_returns,
            "metrics": results,
            "signals": signals,
        }

    def calculate_performance_metrics(self, returns):
        # 연환산 수익률
        annual_return = returns.mean() * 252

        # 연환산 변동성
        annual_vol = returns.std() * np.sqrt(252)

        # 샤프 비율
        sharpe_ratio = annual_return / annual_vol

        # 최대 낙폭
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 승률
        win_rate = (returns > 0).mean()

        # Calmar 비율
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "연환산 수익률": f"{annual_return:.2%}",
            "연환산 변동성": f"{annual_vol:.2%}",
            "샤프 비율": f"{sharpe_ratio:.2f}",
            "최대 낙폭": f"{max_drawdown:.2%}",
            "승률": f"{win_rate:.2%}",
            "Calmar 비율": f"{calmar_ratio:.2f}",
        }

    def visualize_results(self, results, prices):
        # 전체 그리드 설정
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. 누적 수익률 곡선
        ax1 = fig.add_subplot(gs[0, :2])
        cumulative_returns = (1 + results["returns"]).cumprod()
        cumulative_returns.plot(ax=ax1, label="Strategy", linewidth=2, color="darkblue")

        # 벤치마크 (KODEX 200) 비교
        if "KODEX 200" in prices.columns:
            benchmark_returns = prices["KODEX 200"].pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_cumulative = benchmark_cumulative.reindex(
                cumulative_returns.index
            )
            benchmark_cumulative.plot(
                ax=ax1, label="KODEX 200", linewidth=2, color="gray", alpha=0.7
            )

        ax1.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 낙폭 차트
        ax2 = fig.add_subplot(gs[0, 2])
        cumulative = (1 + results["returns"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        drawdown.plot(ax=ax2, color="red", linewidth=1.5)
        ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        ax2.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # 3. 월별 수익률 히트맵
        ax3 = fig.add_subplot(gs[1, :])
        monthly_returns = (
            results["returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        )
        monthly_pivot = pd.pivot_table(
            pd.DataFrame(
                {
                    "Year": monthly_returns.index.year,
                    "Month": monthly_returns.index.month,
                    "Return": monthly_returns.values * 100,
                }
            ),
            values="Return",
            index="Month",
            columns="Year",
        )

        sns.heatmap(
            monthly_pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Return (%)"},
        )
        ax3.set_title("Monthly Returns Heatmap (%)", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Month")

        # 4. 포지션 변화 추이
        ax4 = fig.add_subplot(gs[2, :])
        positions = results["positions"]

        # 상위 5개 ETF만 표시
        top_etfs = positions.abs().mean().nlargest(5).index
        for etf in top_etfs:
            ax4.plot(positions.index, positions[etf] * 100, label=etf, linewidth=1.5)

        ax4.set_title("Position Changes - Top 5 ETFs", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Position Size (%)")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linewidth=0.5)

        # 5. 롤링 성과 지표
        ax5 = fig.add_subplot(gs[3, 0])
        rolling_returns = results["returns"].rolling(252)
        rolling_sharpe = (rolling_returns.mean() * 252) / (
            rolling_returns.std() * np.sqrt(252)
        )

        rolling_sharpe.plot(ax=ax5, color="darkgreen", linewidth=2)
        ax5.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax5.axhline(y=1, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
        ax5.set_title("Rolling Sharpe Ratio (1Y)", fontsize=14, fontweight="bold")
        ax5.set_ylabel("Sharpe Ratio")
        ax5.grid(True, alpha=0.3)

        # 6. 신호 강도 분포
        ax6 = fig.add_subplot(gs[3, 1])
        signals = results["signals"]
        current_signals = signals.iloc[-1].dropna()

        colors = ["red" if x < 0 else "green" for x in current_signals.values]
        ax6.bar(
            range(len(current_signals)), current_signals.values, color=colors, alpha=0.7
        )
        ax6.set_xticks(range(len(current_signals)))
        ax6.set_xticklabels(current_signals.index, rotation=45, ha="right")
        ax6.set_title("Current Signal Strength", fontsize=14, fontweight="bold")
        ax6.set_ylabel("Signal Strength")
        ax6.axhline(y=0, color="black", linewidth=0.5)
        ax6.grid(True, alpha=0.3)

        # 7. 성과 지표 테이블
        ax7 = fig.add_subplot(gs[3, 2])
        ax7.axis("tight")
        ax7.axis("off")

        metrics_data = []
        for key, value in results["metrics"].items():
            metrics_data.append([key, value])

        table = ax7.table(
            cellText=metrics_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax7.set_title("Performance Metrics", fontsize=14, fontweight="bold", pad=20)

        plt.suptitle(
            "Korean ETF ARP-EMA Strategy Backtest Results",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        return fig

    def plot_correlation_heatmap(self, returns):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 전체 기간 상관관계
        corr_full = returns.corr()
        sns.heatmap(
            corr_full,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax1,
            square=True,
        )
        ax1.set_title("Full Period Correlation Matrix", fontsize=14, fontweight="bold")

        # 최근 60일 상관관계
        corr_recent = returns.tail(60).corr()
        sns.heatmap(
            corr_recent,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax2,
            square=True,
        )
        ax2.set_title(
            "Recent 60 Days Correlation Matrix", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        return fig

    def plot_risk_return_scatter(self, returns, positions):
        fig, ax = plt.subplots(figsize=(10, 8))

        # 개별 ETF 성과
        annual_returns = returns.mean() * 252
        annual_vols = returns.std() * np.sqrt(252)

        # 현재 포지션 크기
        current_positions = positions.iloc[-1].abs()

        # 산점도
        scatter = ax.scatter(
            annual_vols * 100,
            annual_returns * 100,
            s=current_positions * 1000,
            alpha=0.6,
            c=annual_returns,
            cmap="RdYlGn",
        )

        # ETF 이름 표시
        for etf in returns.columns:
            ax.annotate(
                etf,
                (annual_vols[etf] * 100, annual_returns[etf] * 100),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # 효율적 프론티어 근사선
        ax.plot(
            [0, annual_vols.max() * 100],
            [0, annual_vols.max() * 100 * 0.5],
            "k--",
            alpha=0.5,
            label="0.5 Sharpe Ratio",
        )

        ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
        ax.set_ylabel("Annualized Return (%)", fontsize=12)
        ax.set_title("Risk-Return Profile of ETFs", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Annual Return (%)", rotation=270, labelpad=20)

        return fig

    def visualize_results_separate(self, results, prices, save_path="output/1"):
        import os

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 1. 누적 수익률 곡선
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        cumulative_returns = (1 + results["returns"]).cumprod()
        cumulative_returns.plot(ax=ax1, label="Strategy", linewidth=2, color="darkblue")

        # 벤치마크 (KODEX 200) 비교
        if "KODEX 200" in prices.columns:
            benchmark_returns = prices["KODEX 200"].pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_cumulative = benchmark_cumulative.reindex(
                cumulative_returns.index
            )
            benchmark_cumulative.plot(
                ax=ax1, label="KODEX 200", linewidth=2, color="gray", alpha=0.7
            )

        ax1.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/1_cumulative_returns.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # 2. 낙폭 차트
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        cumulative = (1 + results["returns"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        drawdown.plot(ax=ax2, color="red", linewidth=1.5)
        ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        ax2.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/2_drawdown.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 3. 월별 수익률 히트맵
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        monthly_returns = (
            results["returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        )
        monthly_pivot = pd.pivot_table(
            pd.DataFrame(
                {
                    "Year": monthly_returns.index.year,
                    "Month": monthly_returns.index.month,
                    "Return": monthly_returns.values * 100,
                }
            ),
            values="Return",
            index="Month",
            columns="Year",
        )

        sns.heatmap(
            monthly_pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Return (%)"},
        )
        ax3.set_title("Monthly Returns Heatmap (%)", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Month")
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/3_monthly_returns_heatmap.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # 4. 포지션 변화 추이
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        positions = results["positions"]

        # 상위 5개 ETF만 표시
        top_etfs = positions.abs().mean().nlargest(5).index
        for etf in top_etfs:
            ax4.plot(positions.index, positions[etf] * 100, label=etf, linewidth=1.5)

        ax4.set_title("Position Changes - Top 5 ETFs", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Position Size (%)")
        ax4.legend(loc="best")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"{save_path}/4_position_changes.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 5. 롤링 샤프 비율
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        rolling_returns = results["returns"].rolling(252)
        rolling_sharpe = (rolling_returns.mean() * 252) / (
            rolling_returns.std() * np.sqrt(252)
        )

        rolling_sharpe.plot(ax=ax5, color="darkgreen", linewidth=2)
        ax5.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax5.axhline(y=1, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
        ax5.set_title("Rolling Sharpe Ratio (1Y)", fontsize=14, fontweight="bold")
        ax5.set_ylabel("Sharpe Ratio")
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/5_rolling_sharpe.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 6. 신호 강도 분포
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        signals = results["signals"]
        current_signals = signals.iloc[-1].dropna()

        colors = ["red" if x < 0 else "green" for x in current_signals.values]
        ax6.bar(
            range(len(current_signals)), current_signals.values, color=colors, alpha=0.7
        )
        ax6.set_xticks(range(len(current_signals)))
        ax6.set_xticklabels(current_signals.index, rotation=45, ha="right")
        ax6.set_title("Current Signal Strength", fontsize=14, fontweight="bold")
        ax6.set_ylabel("Signal Strength")
        ax6.axhline(y=0, color="black", linewidth=0.5)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/6_signal_strength.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 7. 성과 지표 테이블
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        ax7.axis("tight")
        ax7.axis("off")

        metrics_data = []
        for key, value in results["metrics"].items():
            metrics_data.append([key, value])

        table = ax7.table(
            cellText=metrics_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 2)

        ax7.set_title("Performance Metrics", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/7_performance_metrics.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"모든 그래프가 {save_path} 폴더에 저장되었습니다.")


if __name__ == "__main__":
    etf_list = [
        "KODEX 200",
        "KODEX 코스닥150",
        "PLUS 우주항공&UAM",
        "PLUS K방산",
        "PLUS 태양광&ESS",
        "TIGER 200 철강소재",
        "TIGER 200 에너지화학",
        "TIGER 미디어컨텐츠",
        "TIGER 여행레저",
        "TIGER 지주회사",
        "TIGER 게임TOP10",
        "TIGER 200 헬스케어",
        "TIGER 코스닥150바이오테크",
        "KIWOOM Fn유전자혁신기술",
        "SOL 의료기기소부장Fn",
        "TIGER 방송통신",
        "TIGER 인터넷TOP10",
        "TIGER 소프트웨어",
        "KODEX 은행",
        "TIGER 200 금융",
        "KODEX 증권",
        "TIGER 화장품",
        "HANARO K-뷰티",
        "SOL 조선TOP3플러스",
        "KODEX 운송",
        "KODEX 반도체",
        "SOL 반도체전공정",
        "SOL 반도체후공정",
        "SOL 자동차소부장Fn",
        "KODEX 자동차",
        "TIGER 200 건설",
        "TIGER 2차전지TOP10",
        "SOL 2차전지소부장Fn",
    ]

    strategy = KoreanETFARPStrategy(
        etf_list=etf_list,
        lookback_ema=112,
        lookback_corr=250,
        target_vol=0.15,
        max_loss=-0.15,
        rebalance_freq="W",
    )

    results = strategy.backtest(
        start_date=datetime(2020, 1, 1), end_date=datetime(2024, 12, 31)
    )

    print("=== 한국 ETF ARP-EMA 전략 백테스트 결과 ===")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value}")

    latest_positions = results["positions"].iloc[-1]
    print("\n=== 현재 추천 포지션 ===")
    for ticker, position in latest_positions.items():
        if abs(position) > 0.01:
            print(f"{ticker}: {position:.1%}")

    print("\n=== 백테스트 결과 시각화 ===")

    prices, returns = strategy.fetch_data(
        datetime(2020, 1, 1) - timedelta(days=500), datetime(2024, 12, 31)
    )

    # 1. 개별 그래프로 시각화 저장
    strategy.visualize_results_separate(results, prices, save_path="output/1")

    # 2. 상관관계 히트맵
    fig_corr = strategy.plot_correlation_heatmap(returns)
    plt.savefig("output/1/8_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. 리스크-수익률 산점도
    fig_scatter = strategy.plot_risk_return_scatter(returns, results["positions"])
    plt.savefig("output/1/9_risk_return_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 추가 분석 출력
    print("\n=== 추가 분석 결과 ===")

    # 포지션 회전율
    turnover = results["positions"].diff().abs().sum(axis=1).mean()
    print(f"일평균 포지션 회전율: {turnover:.1%}")

    # 최대 포지션 ETF
    max_position_etf = results["positions"].abs().mean().idxmax()
    avg_position = results["positions"][max_position_etf].mean()
    print(f"평균 최대 포지션 ETF: {max_position_etf} ({avg_position:.1%})")

    # 수익 기여도 분석
    position_returns = results["positions"].shift(1) * returns
    contribution = position_returns.sum() / position_returns.sum().sum()

    print("\n=== 수익 기여도 Top 5 ===")
    for etf, contrib in contribution.nlargest(5).items():
        print(f"{etf}: {contrib:.1%}")
