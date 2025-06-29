import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from downloader_with_fdr import FDRDownloader

warnings.filterwarnings("ignore")

# Korean font setup
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class SmashDayStrategy:
    """
    Larry Williams Smash Day Strategy adapted for Korean Leveraged ETFs
    - KOSPI 200 Leverage (2x)
    - KOSDAQ 150 Leverage (2x)
    - Using inverse ETFs for short positions
    """

    def __init__(
        self,
        initial_capital=10000000,
        position_size=1.0,
        stop_loss=-0.02,
        lookback_days=3,
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size  # 1.0 = 100% of capital
        self.stop_loss = stop_loss  # -2% for leveraged ETFs
        self.lookback_days = lookback_days

        # Korean Leveraged ETF tickers
        self.etf_mapping = {
            "KOSPI_2X": "122630",  # KODEX Leverage
            "KOSPI_INV_2X": "114800",  # KODEX Inverse 2X
            "KOSDAQ_2X": "233740",  # KODEX KOSDAQ 150 Leverage
            "KOSDAQ_INV": "251340",  # KODEX KOSDAQ 150 Inverse
        }

    def generate_smash_day_signals(self, df):
        """
        Generate Smash Day buy/sell signals
        Buy: Close < Yesterday's Low, Next Open > Today's High
        Sell: Close > Yesterday's High, Next Open < Today's Low
        """
        # Create a copy to avoid SettingWithCopyWarning
        signals = pd.DataFrame(index=df.index)

        # Previous day's high/low
        signals["prev_high"] = df["High"].shift(1)
        signals["prev_low"] = df["Low"].shift(1)

        # Rolling max/min for lookback period
        signals["lookback_high"] = df["High"].rolling(self.lookback_days).max().shift(1)
        signals["lookback_low"] = df["Low"].rolling(self.lookback_days).min().shift(1)

        # Smash Day conditions
        signals["smash_down"] = (df["Close"] < signals["prev_low"]) & (
            df["Close"] < signals["lookback_low"]
        )
        signals["smash_up"] = (df["Close"] > signals["prev_high"]) & (
            df["Close"] > signals["lookback_high"]
        )

        # Entry conditions (next day's open)
        signals["next_open"] = df["Open"].shift(-1)
        signals["today_high"] = df["High"]
        signals["today_low"] = df["Low"]

        # Final signals
        signals["buy_signal"] = signals["smash_down"].shift(1) & (
            df["Open"] > signals["today_high"].shift(1)
        )
        signals["sell_signal"] = signals["smash_up"].shift(1) & (
            df["Open"] < signals["today_low"].shift(1)
        )

        return signals

    def run_backtest(self, df, signals):
        """Execute backtest with position management"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []

        # Daily tracking
        daily_capital = []
        daily_position = []

        for i in range(len(df)):
            date = df.index[i]
            current_price = df["Close"].iloc[i]

            # Update position value
            if position != 0:
                current_value = capital + position * (current_price - entry_price)
                pnl_pct = (current_price - entry_price) / entry_price

                # Bailout exit (first positive return)
                if pnl_pct > 0:
                    realized_pnl = position * (current_price - entry_price)
                    capital += realized_pnl
                    trades.append(
                        {
                            "date": date,
                            "type": "bailout_exit",
                            "price": current_price,
                            "position": -position,
                            "pnl": realized_pnl,
                            "pnl_pct": pnl_pct,
                        }
                    )
                    position = 0
                    entry_price = 0

                # Stop loss exit
                elif pnl_pct <= self.stop_loss:
                    realized_pnl = position * (current_price - entry_price)
                    capital += realized_pnl
                    trades.append(
                        {
                            "date": date,
                            "type": "stop_loss",
                            "price": current_price,
                            "position": -position,
                            "pnl": realized_pnl,
                            "pnl_pct": pnl_pct,
                        }
                    )
                    position = 0
                    entry_price = 0

            # Entry signals (only if no position)
            if position == 0:
                if signals["buy_signal"].iloc[i]:
                    position_size = capital * self.position_size
                    position = position_size / current_price
                    entry_price = current_price
                    trades.append(
                        {
                            "date": date,
                            "type": "buy_entry",
                            "price": current_price,
                            "position": position,
                            "pnl": 0,
                            "pnl_pct": 0,
                        }
                    )

                elif signals["sell_signal"].iloc[i]:
                    # Use inverse ETF for short
                    position_size = capital * self.position_size
                    position = -position_size / current_price  # Negative for tracking
                    entry_price = current_price
                    trades.append(
                        {
                            "date": date,
                            "type": "sell_entry",
                            "price": current_price,
                            "position": position,
                            "pnl": 0,
                            "pnl_pct": 0,
                        }
                    )

            # Track daily values
            if position != 0:
                current_capital = capital + position * (current_price - entry_price)
            else:
                current_capital = capital

            daily_capital.append(current_capital)
            daily_position.append(abs(position) if position != 0 else 0)

        # Create results DataFrame
        results_df = pd.DataFrame(
            {"capital": daily_capital, "position": daily_position}, index=df.index
        )

        results_df["returns"] = results_df["capital"].pct_change()
        results_df["cumulative_returns"] = (1 + results_df["returns"]).cumprod()

        return results_df, pd.DataFrame(trades)

    def calculate_performance_metrics(self, results_df, trades_df):
        """Calculate comprehensive performance metrics"""
        returns = results_df["returns"].dropna()

        # Basic metrics
        total_return = (results_df["capital"].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (1 + total_return / 100) ** (252 / len(results_df)) - 1
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = (
            (annual_return * 100) / (volatility * 100) if volatility > 0 else 0
        )

        # Maximum drawdown
        cumulative = results_df["cumulative_returns"]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Trade statistics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df["pnl"] > 0]
            losing_trades = trades_df[trades_df["pnl"] < 0]

            win_rate = (
                len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            )
            avg_win = (
                winning_trades["pnl_pct"].mean() * 100 if len(winning_trades) > 0 else 0
            )
            avg_loss = (
                losing_trades["pnl_pct"].mean() * 100 if len(losing_trades) > 0 else 0
            )

            # Profit factor
            total_wins = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
            total_losses = (
                abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 1
            )
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        metrics = {
            "Total Return (%)": f"{total_return:.2f}",
            "Annual Return (%)": f"{annual_return*100:.2f}",
            "Volatility (%)": f"{volatility*100:.2f}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown (%)": f"{max_drawdown:.2f}",
            "Win Rate (%)": f"{win_rate:.2f}",
            "Avg Win (%)": f"{avg_win:.2f}",
            "Avg Loss (%)": f"{avg_loss:.2f}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Total Trades": len(trades_df),
            "Skewness": f"{skewness:.2f}",
            "Kurtosis": f"{kurtosis:.2f}",
        }

        return metrics

    def visualize_results(self, results_df, trades_df, signals_df, ticker_name):
        """Create comprehensive multi-panel visualization"""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)

        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        results_df["cumulative_returns"].plot(ax=ax1, linewidth=2, color="darkblue")
        ax1.set_title(
            f"Cumulative Returns - {ticker_name} Smash Day Strategy",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Cumulative Return")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color="black", linestyle="--", alpha=0.5)

        # Mark entry/exit points
        for _, trade in trades_df.iterrows():
            if trade["type"] in ["buy_entry", "sell_entry"]:
                ax1.axvline(
                    x=trade["date"],
                    color="green" if "buy" in trade["type"] else "red",
                    alpha=0.3,
                    linestyle="--",
                )

        # 2. Drawdown Chart
        ax2 = fig.add_subplot(gs[1, :])
        cumulative = results_df["cumulative_returns"]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        drawdown.plot(ax=ax2, color="red", linewidth=1.5)
        ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        ax2.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # 3. Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[2, :])
        monthly_returns = (
            results_df["returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
            * 100
        )

        if len(monthly_returns) > 0:
            monthly_pivot = pd.pivot_table(
                pd.DataFrame(
                    {
                        "Year": monthly_returns.index.year,
                        "Month": monthly_returns.index.month,
                        "Return": monthly_returns.values,
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

        # 4. Daily Returns Distribution
        ax4 = fig.add_subplot(gs[3, 0])
        returns = results_df["returns"].dropna() * 100
        returns.hist(bins=50, ax=ax4, alpha=0.7, color="darkblue")
        ax4.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax4.set_title("Daily Returns Distribution", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Return (%)")
        ax4.set_ylabel("Frequency")

        # Add distribution stats
        ax4.text(
            0.05,
            0.95,
            f"Skew: {returns.skew():.2f}\nKurt: {returns.kurtosis():.2f}",
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 5. Annual Returns Bar Chart
        ax5 = fig.add_subplot(gs[3, 1])
        annual_returns = (
            results_df["returns"].resample("Y").apply(lambda x: (1 + x).prod() - 1)
            * 100
        )

        if len(annual_returns) > 0:
            colors = ["green" if x > 0 else "red" for x in annual_returns]
            annual_returns.plot(kind="bar", ax=ax5, color=colors, alpha=0.7)
            ax5.set_title("Annual Returns", fontsize=12, fontweight="bold")
            ax5.set_xlabel("Year")
            ax5.set_ylabel("Return (%)")
            ax5.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            # Format x-axis labels
            ax5.set_xticklabels(
                [str(x.year) for x in annual_returns.index], rotation=45
            )

        # 6. Rolling Sharpe Ratio
        ax6 = fig.add_subplot(gs[3, 2])
        rolling_returns = results_df["returns"].rolling(252)
        rolling_sharpe = (rolling_returns.mean() * 252) / (
            rolling_returns.std() * np.sqrt(252)
        )

        rolling_sharpe.plot(ax=ax6, color="darkgreen", linewidth=2)
        ax6.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax6.axhline(y=1, color="green", linestyle="--", alpha=0.3)
        ax6.set_title("Rolling Sharpe Ratio (1Y)", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Sharpe Ratio")
        ax6.grid(True, alpha=0.3)

        # 7. Trade Analysis
        ax7 = fig.add_subplot(gs[4, 0])
        if len(trades_df) > 0:
            trade_types = trades_df["type"].value_counts()
            trade_types.plot(kind="bar", ax=ax7, color="darkblue", alpha=0.7)
            ax7.set_title("Trade Type Distribution", fontsize=12, fontweight="bold")
            ax7.set_xlabel("Trade Type")
            ax7.set_ylabel("Count")
            ax7.tick_params(axis="x", rotation=45)

        # 8. Win/Loss Distribution
        ax8 = fig.add_subplot(gs[4, 1])
        if len(trades_df) > 0:
            pnl_values = trades_df[trades_df["pnl"] != 0]["pnl_pct"] * 100
            if len(pnl_values) > 0:
                pnl_values.hist(bins=30, ax=ax8, alpha=0.7, color="darkblue")
                ax8.axvline(x=0, color="black", linestyle="--", alpha=0.5)
                ax8.set_title("Trade P&L Distribution", fontsize=12, fontweight="bold")
                ax8.set_xlabel("Return (%)")
                ax8.set_ylabel("Frequency")

        # 9. Smash Day Signal Frequency
        ax9 = fig.add_subplot(gs[4, 2])
        signal_counts = pd.DataFrame(
            {
                "Buy Signals": signals_df["buy_signal"].resample("M").sum(),
                "Sell Signals": signals_df["sell_signal"].resample("M").sum(),
            }
        )

        if len(signal_counts) > 0:
            signal_counts.plot(
                kind="bar", ax=ax9, stacked=True, color=["green", "red"], alpha=0.7
            )
            ax9.set_title("Monthly Signal Frequency", fontsize=12, fontweight="bold")
            ax9.set_xlabel("Month")
            ax9.set_ylabel("Signal Count")
            ax9.legend()

            # Format x-axis
            ax9.set_xticklabels(
                [f"{x.strftime('%Y-%m')}" for x in signal_counts.index],
                rotation=45,
                ha="right",
            )

        # 10. Performance Summary Table
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis("tight")
        ax10.axis("off")

        # Calculate metrics
        metrics = self.calculate_performance_metrics(results_df, trades_df)

        # Create table data
        table_data = []
        for key, value in metrics.items():
            table_data.append([key, value])

        table = ax10.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(table_data) + 1):
            table[(i, 0)].set_facecolor("#E8E8E8")
            table[(i, 1)].set_facecolor("#F8F8F8")

        ax10.set_title("Performance Summary", fontsize=14, fontweight="bold", pad=20)

        plt.suptitle(
            f"Smash Day Strategy Backtest Results - {ticker_name}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        return fig


# Main execution
if __name__ == "__main__":
    # Initialize strategy
    strategy = SmashDayStrategy(
        initial_capital=10000000,  # 10M KRW
        position_size=1.0,  # 100% capital per trade
        stop_loss=-0.02,  # -2% stop loss for leveraged ETFs
        lookback_days=3,  # 3-day lookback for smash pattern
    )

    # Backtest parameters
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    # Test on KOSPI 200 Leverage ETF
    print("=== Smash Day Strategy Backtest ===")
    print(f"Testing on KODEX Leverage (KOSPI 200 2X)")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: 10,000,000 KRW")
    print(f"Stop Loss: {strategy.stop_loss*100:.1f}%")
    print(f"Lookback Days: {strategy.lookback_days}")
    print("-" * 50)

    # Fetch data
    reuslt = FDRDownloader().download(
        tickers=strategy.etf_mapping["KOSPI_2X"],
        start_date=start_date,
        end_date=end_date,
    )

    df = reuslt[strategy.etf_mapping["KOSPI_2X"]]

    if len(df) > 0:
        # Generate signals
        signals = strategy.generate_smash_day_signals(df)

        # Run backtest
        results, trades = strategy.run_backtest(df, signals)

        # Calculate and display metrics
        metrics = strategy.calculate_performance_metrics(results, trades)

        print("\n=== Performance Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        # Visualize results
        fig = strategy.visualize_results(results, trades, signals, "KODEX Leverage")
        plt.savefig("smash_day_backtest_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Trade summary
        if len(trades) > 0:
            print("\n=== Trade Summary ===")
            print(f"Total Trades: {len(trades)}")
            print(f"Buy Entries: {len(trades[trades['type'] == 'buy_entry'])}")
            print(f"Sell Entries: {len(trades[trades['type'] == 'sell_entry'])}")
            print(f"Bailout Exits: {len(trades[trades['type'] == 'bailout_exit'])}")
            print(f"Stop Loss Exits: {len(trades[trades['type'] == 'stop_loss'])}")

            # Recent trades
            print("\n=== Recent 10 Trades ===")
            print(trades.tail(10).to_string())
    else:
        print("Error: No data fetched for the specified ticker and date range.")
