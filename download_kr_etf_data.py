#!/usr/bin/env python3
"""
한국 ETF 데이터 다운로드 스크립트
FinanceDataReader를 사용하여 주요 한국 ETF 데이터를 다운로드합니다.
"""

from fdr_downloader import FDRDownloader
from datetime import datetime, timedelta


def download_kr_etf_for_strategy():
    """전략에 사용할 한국 ETF 데이터 다운로드"""
    
    # ETF 코드와 이름 매핑
    etf_mapping = {
        '069500': 'KODEX 200',
        '139260': 'TIGER 200IT',
        '091170': 'KODEX 은행',
        '228790': 'TIGER 의료기기',
        '117680': 'KODEX 철강',
        '371460': 'TIGER 차이나전기차',
        '102110': 'KODEX 삼성그룹',
        '285000': 'KODEX 원유선물',
        '091180': 'KODEX 인버스',
        '305720': 'TIGER 미국채10년',
        '244580': 'KODEX 리츠',
        '266360': 'KODEX KRX300',
        '227550': 'TIGER 반도체',
        '148020': 'TIGER 미국S&P500',
        '140700': 'KODEX 금',
    }
    
    # 다운로더 초기화
    downloader = FDRDownloader()
    
    # ETF 코드 리스트
    etf_codes = list(etf_mapping.keys())
    
    # 데이터 다운로드 (최근 5년)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    print(f"📊 한국 ETF 데이터 다운로드 시작")
    print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"ETF 개수: {len(etf_codes)}개")
    print("="*50)
    
    # 다운로드 실행
    data_dict = downloader.download(
        tickers=etf_codes,
        market='kr_etf',
        save_path='data',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        save_format='csv'
    )
    
    # 결과 출력
    print("\n📊 다운로드 완료!")
    print("="*50)
    
    success_count = 0
    for code, name in etf_mapping.items():
        if code in data_dict:
            df = data_dict[code]
            print(f"✅ {name} ({code}): {len(df)} rows")
            success_count += 1
        else:
            print(f"❌ {name} ({code}): 다운로드 실패")
    
    print(f"\n총 {success_count}/{len(etf_codes)}개 ETF 다운로드 성공")
    
    # 데이터 샘플 출력
    if data_dict:
        sample_code = list(data_dict.keys())[0]
        sample_name = etf_mapping.get(sample_code, sample_code)
        print(f"\n📊 {sample_name} ({sample_code}) 데이터 샘플:")
        print(data_dict[sample_code].tail())
    
    return data_dict


def download_additional_data():
    """추가 데이터 다운로드 (지수, 환율 등)"""
    
    downloader = FDRDownloader()
    
    # 1. 주요 지수 다운로드
    print("\n📊 주요 지수 다운로드")
    index_data = downloader.download(
        tickers=['KS11', 'KQ11', 'DJI', 'IXIC', 'SPX'],  # KOSPI, KOSDAQ, 다우, 나스닥, S&P500
        market='index',
        save_path='data/index',
        start_date='2020-01-01'
    )
    
    # 2. 환율 다운로드
    print("\n📊 환율 데이터 다운로드")
    forex_data = downloader.download(
        tickers=['USD/KRW', 'EUR/KRW', 'JPY/KRW'],
        market='forex',
        save_path='data/forex',
        start_date='2020-01-01'
    )
    
    return index_data, forex_data


if __name__ == "__main__":
    # 1. 전략용 한국 ETF 데이터 다운로드
    etf_data = download_kr_etf_for_strategy()
    
    # 2. 추가 데이터 다운로드 (선택사항)
    # index_data, forex_data = download_additional_data()