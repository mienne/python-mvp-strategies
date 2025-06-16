#!/usr/bin/env python3

import FinanceDataReader as fdr
import os
import pandas as pd

from datetime import datetime
from typing import List, Optional, Dict, Union


class FDRDownloader:
    def __init__(self):
        self.default_tickers = {
            # 삼성전자, SK하이닉스, LG화학, NAVER, 카카오
            "kr_stock": [
                "005930",
                "000660",
                "051910",
                "035420",
                "035720",
            ],
            # KODEX 200, KODEX 코스닥150, KODEX 인버스 등
            "kr_etf": [
                "069500",
                "229200",
                "091180",
                "102110",
                "305720",
            ],
            "us_stock": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "us_etf": ["SPY", "QQQ", "EFA", "IEF", "VNQ", "DBC"],
            "index": ["KS11", "KQ11", "S&P500", "NASDAQ", "DJI"],
            "forex": ["USD/KRW", "EUR/USD", "JPY/KRW", "EUR/KRW"],
            "crypto": ["BTC/USD", "ETH/USD", "BTC/KRW"],
        }

    def download(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        market: str = "kr_etf",
        save_path: str = "data",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        save_format: str = "csv",
    ) -> Dict[str, pd.DataFrame]:
        if tickers is None:
            tickers = self.default_tickers.get(market, [])
        elif isinstance(tickers, str):
            tickers = [tickers]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        os.makedirs(save_path, exist_ok=True)

        data_dict = {}

        for ticker in tickers:
            try:
                df = fdr.DataReader(ticker, start_date, end_date)

                if df is not None and not df.empty:
                    df = self._standardize_columns(df)
                    df.index.name = "Date"
                    self._save_data(df, ticker, save_path, save_format)
                    data_dict[ticker] = df

            except Exception as e:
                pass

        return data_dict

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
            "Change": "Change",
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
            "변동률": "Change",
        }

        new_df = pd.DataFrame()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                new_df[new_col] = df[old_col]

        return new_df

    def _save_data(
        self, df: pd.DataFrame, ticker: str, save_path: str, save_format: str
    ) -> str:
        filename = f"{save_path}/{ticker}_data.{save_format}"

        if save_format == "csv":
            df.to_csv(filename)
        elif save_format == "parquet":
            df.to_parquet(filename)
        elif save_format == "excel":
            df.to_excel(filename)
        else:
            raise ValueError(f"Unsupported format: {save_format}")

        return filename

    def get_ticker_list(self, market: str) -> pd.DataFrame:
        try:
            if market in ["KRX", "KOSPI", "KOSDAQ"]:
                df = fdr.StockListing(market)
            else:
                df = fdr.StockListing(market)
            return df
        except Exception as e:
            return pd.DataFrame()


def main():
    downloader = FDRDownloader()
    downloader.download(
        tickers=[
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
        ],
        save_path="data",
        start_date="2020-01-01",
    )


if __name__ == "__main__":
    main()
