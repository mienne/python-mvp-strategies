#!/usr/bin/env python3

import FinanceDataReader as fdr
import os
import pandas as pd

from datetime import datetime
from typing import List, Optional, Dict, Union


class FDRDownloader:
    def __init__(self):
        self.default_tickers = {
            'kr_stock': ['005930', '000660', '051910', '035420', '035720'],  # 삼성전자, SK하이닉스, LG화학, NAVER, 카카오
            'kr_etf': ['069500', '229200', '091180', '102110', '305720'],  # KODEX 200, KODEX 코스닥150, KODEX 인버스 등
            'us_stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'us_etf': ['SPY', 'QQQ', 'EFA', 'IEF', 'VNQ', 'DBC'],
            'index': ['KS11', 'KQ11', 'S&P500', 'NASDAQ', 'DJI'],
            'forex': ['USD/KRW', 'EUR/USD', 'JPY/KRW', 'EUR/KRW'],
            'crypto': ['BTC/USD', 'ETH/USD', 'BTC/KRW']
        }
        
    def download(self, 
                 tickers: Optional[Union[str, List[str]]] = None,
                 market: str = 'kr_etf',
                 save_path: str = 'data',
                 start_date: str = "2020-01-01",
                 end_date: Optional[str] = None,
                 save_format: str = 'csv') -> Dict[str, pd.DataFrame]:

        
        if tickers is None:
            tickers = self.default_tickers.get(market, [])
        elif isinstance(tickers, str):
            tickers = [tickers]
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        os.makedirs(save_path, exist_ok=True)
        
        data_dict = {}
        
        for ticker in tickers:
            try:
                print(f"📊 Downloading {ticker}...")
                
                df = fdr.DataReader(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    df = self._standardize_columns(df)
                    df.index.name = 'Date'
                    filename = self._save_data(df, ticker, save_path, save_format)
                    data_dict[ticker] = df
                    print(f"✅ {ticker} saved to {filename}")
                else:
                    print(f"⚠️ No data found for {ticker}")
                    
            except Exception as e:
                print(f"❌ Error downloading {ticker}: {e}")
                
        return data_dict
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Change': 'Change',
            '시가': 'Open',
            '고가': 'High',
            '저가': 'Low',
            '종가': 'Close',
            '거래량': 'Volume',
            '변동률': 'Change'
        }
        
        new_df = pd.DataFrame()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                new_df[new_col] = df[old_col]
                
        return new_df
    
    def _save_data(self, df: pd.DataFrame, ticker: str, save_path: str, save_format: str) -> str:
        filename = f"{save_path}/{ticker}_data.{save_format}"
        
        if save_format == 'csv':
            df.to_csv(filename)
        elif save_format == 'parquet':
            df.to_parquet(filename)
        elif save_format == 'excel':
            df.to_excel(filename)
        else:
            raise ValueError(f"Unsupported format: {save_format}")
            
        return filename
    
    def get_ticker_list(self, market: str) -> pd.DataFrame:
        try:
            if market in ['KRX', 'KOSPI', 'KOSDAQ']:
                df = fdr.StockListing(market)
            else:
                df = fdr.StockListing(market)
            return df
        except Exception as e:
            print(f"❌ Error getting ticker list for {market}: {e}")
            return pd.DataFrame()
    
    def download_korean_etfs(self, 
                            etf_codes: Optional[List[str]] = None,
                            save_path: str = 'data',
                            start_date: str = "2020-01-01",
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        if etf_codes is None:
            etf_codes = [
                '069500',  # KODEX 200
                '229200',  # KODEX 코스닥150
                '091180',  # KODEX 인버스
                '114800',  # KODEX 인버스2X
                '251340',  # KODEX 코스닥150레버리지
                '122630',  # KODEX 레버리지
                '233740',  # KODEX 코스닥150 인버스
                '102110',  # TIGER 200
                '252670',  # KODEX 200선물인버스2X
                '305720',  # KODEX 2차전지산업
                '091170',  # KODEX 은행
                '117680',  # KODEX 철강
                '140700',  # KODEX 보험
                '244580',  # KODEX 바이오
                '266360',  # KODEX MSCI Korea ESG
            ]
            
        return self.download(etf_codes, market='kr_etf', save_path=save_path, 
                           start_date=start_date, end_date=end_date)


def main():
    downloader = FDRDownloader()
    
    print("=== 한국 ETF 다운로드 ===")
    kr_etf_data = downloader.download_korean_etfs(
        save_path='data/kr_etf',
        start_date='2020-01-01'
    )
    
    print("\n=== 미국 ETF 다운로드 ===")
    us_etf_data = downloader.download(
        tickers=['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'],
        market='us_etf',
        save_path='data/us_etf',
        start_date='2020-01-01'
    )
    
    print("\n=== 한국 주식 다운로드 ===")
    kr_stock_data = downloader.download(
        tickers=['005930', '000660', '051910'],  # 삼성전자, SK하이닉스, LG화학
        market='kr_stock',
        save_path='data/kr_stock',
        start_date='2020-01-01'
    )
    
    print("\n=== 환율 데이터 다운로드 ===")
    forex_data = downloader.download(
        tickers=['USD/KRW', 'EUR/KRW', 'JPY/KRW'],
        market='forex',
        save_path='data/forex',
        start_date='2020-01-01'
    )
    
    print("\n=== 지수 데이터 다운로드 ===")
    index_data = downloader.download(
        tickers=['KS11', 'KQ11', 'DJI', 'IXIC'],  # KOSPI, KOSDAQ, 다우존스, 나스닥
        market='index',
        save_path='data/index',
        start_date='2020-01-01'
    )
    
    print("\n=== KRX 종목 리스트 조회 ===")
    krx_list = downloader.get_ticker_list('KRX')
    print(f"KRX 전체 종목 수: {len(krx_list)}")
    print(krx_list.head())


if __name__ == "__main__":
    main()