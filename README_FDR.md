# FinanceDataReader Downloader

FinanceDataReader를 활용한 범용 금융 데이터 다운로더입니다.

## 특징

- 🇰🇷 한국 주식/ETF 데이터 지원
- 🇺🇸 미국 주식/ETF 데이터 지원
- 📊 주요 지수 데이터 (KOSPI, S&P500, NASDAQ 등)
- 💱 환율 데이터 (USD/KRW, EUR/KRW 등)
- 🪙 암호화폐 데이터 (BTC/USD, ETH/USD 등)
- 📁 다양한 저장 형식 지원 (CSV, Parquet, Excel)

## 설치

```bash
pip install finance-datareader
```

## 사용법

### 1. 기본 사용법

```python
from fdr_downloader import FDRDownloader

# 다운로더 초기화
downloader = FDRDownloader()

# 한국 ETF 다운로드
data = downloader.download(
    tickers=['069500', '229200', '091180'],  # KODEX 200, KODEX 코스닥150, KODEX 인버스
    market='kr_etf',
    save_path='data',
    start_date='2020-01-01'
)
```

### 2. 한국 ETF 전용 다운로드

```python
# 주요 한국 ETF 자동 다운로드
kr_etf_data = downloader.download_korean_etfs(
    save_path='data/kr_etf',
    start_date='2020-01-01'
)
```

### 3. 전략용 데이터 다운로드

```bash
# 스크립트 실행
python download_kr_etf_data.py
```

### 4. 다양한 시장 데이터 다운로드

```python
# 미국 주식
us_stocks = downloader.download(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    market='us_stock',
    save_path='data/us_stock'
)

# 지수 데이터
indices = downloader.download(
    tickers=['KS11', 'KQ11', 'SPX'],  # KOSPI, KOSDAQ, S&P500
    market='index',
    save_path='data/index'
)

# 환율 데이터
forex = downloader.download(
    tickers=['USD/KRW', 'EUR/KRW'],
    market='forex',
    save_path='data/forex'
)
```

## 지원 마켓 타입

- `kr_stock`: 한국 주식
- `kr_etf`: 한국 ETF
- `us_stock`: 미국 주식
- `us_etf`: 미국 ETF
- `index`: 주요 지수
- `forex`: 환율
- `crypto`: 암호화폐

## 주요 한국 ETF 코드

| 코드 | ETF명 |
|------|-------|
| 069500 | KODEX 200 |
| 229200 | KODEX 코스닥150 |
| 091180 | KODEX 인버스 |
| 114800 | KODEX 인버스2X |
| 122630 | KODEX 레버리지 |
| 102110 | TIGER 200 |
| 305720 | KODEX 2차전지산업 |
| 091170 | KODEX 은행 |
| 117680 | KODEX 철강 |
| 244580 | KODEX 바이오 |

## 출력 예시

```
📊 Downloading 069500...
✅ 069500 saved to data/069500_data.csv
📊 Downloading 229200...
✅ 229200 saved to data/229200_data.csv
```

## 주의사항

- 대량 다운로드 시 서버 부하를 고려하여 적절한 딜레이를 추가하세요
- 일부 데이터는 거래소 정책에 따라 제한될 수 있습니다
- 실시간 데이터가 아닌 일별 종가 기준 데이터입니다