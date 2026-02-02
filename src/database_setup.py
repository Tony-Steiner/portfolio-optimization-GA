import requests
import pandas as pd
import os
import datetime
import io
import yfinance as yf
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# --- GLOBAL CONFIGURATION ---
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")
SERIES_ID = "SF60633"

# DYNAMIC DATE CONFIGURATION
START_DATE = "2006-01-01"
# Set END_DATE to tomorrow to ensure we capture today's closing data
END_DATE = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")


def get_db_engine():
    """Creates and returns the SQLAlchemy engine."""
    if not DB_PASS:
        raise ValueError("DB_PASS is missing in .env file")
    connection_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_str)


def populate_cetes_data(engine):
    """Fetches synthetic CETES data from Banxico and uploads to DB."""
    print("\n--- 1. Processing CETES Data ---")

    print("Clearing old CETES data...")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE cetes_data"))

    print("Connecting to Banxico API...")
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{SERIES_ID}/datos/{START_DATE}/{END_DATE}"
    headers = {"Bmx-Token": BANXICO_TOKEN}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code}")

        data = response.json()
        cetes_data = data["bmx"]["series"][0]["datos"]
        df = pd.DataFrame(cetes_data)
        df.columns = ["Date", "Yield"]

        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df["Yield"] = pd.to_numeric(df["Yield"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        df_daily = df.resample("D").ffill()
        df_daily["Daily_Rate"] = (df_daily["Yield"] / 100) / 360
        df_daily["Price"] = 100 * (1 + df_daily["Daily_Rate"]).cumprod()

        upload_df = df_daily[["Yield", "Price"]].copy()
        upload_df.columns = ["yield", "price"]
        upload_df.index.name = "date"

        print(f"Generated {len(upload_df)} days of synthetic CETES prices.")
        upload_df.to_sql("cetes_data", con=engine, if_exists="append", index=True)
        print("Success! CETES data refreshed.")

    except Exception as e:
        print(f"Failed to process CETES data: {e}")


def populate_metadata(engine):
    """Scrapes S&P 500 tickers, adds ETFs, and uploads metadata to DB."""
    print("\n--- 2. Processing Assets Metadata ---")

    print("Clearing old Metadata...")
    with engine.begin() as conn:
        # --- THE CRITICAL FIX: CASCADE ---
        # This allows deleting metadata even if market_data is attached to it
        conn.execute(text("TRUNCATE TABLE assets_metadata CASCADE"))

    print("Fetching S&P 500 list from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        sp500_df = tables[0]

        df_stocks = sp500_df[["Symbol", "Security", "GICS Sector"]].copy()
        df_stocks.columns = ["ticker", "name", "sector"]
        df_stocks["ticker"] = df_stocks["ticker"].str.replace(".", "-", regex=False)
        df_stocks["asset_type"] = "Stock"

        etfs = [
            {
                "ticker": "VT",
                "name": "Vanguard Total World",
                "sector": "Global ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VPL",
                "name": "Vanguard Pacific",
                "sector": "Pacific ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VGK",
                "name": "Vanguard Europe",
                "sector": "Europe ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VWO",
                "name": "Vanguard Emerging",
                "sector": "Emerging Mkts ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VOO",
                "name": "Vanguard S&P 500",
                "sector": "US Large Cap ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VEA",
                "name": "Vanguard Developed",
                "sector": "Dev Markets ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VNQ",
                "name": "Vanguard Real Estate",
                "sector": "US REIT ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "VNQI",
                "name": "Vanguard Global REIT",
                "sector": "Intl REIT ETF",
                "asset_type": "ETF",
            },
            {
                "ticker": "SPY",
                "name": "SPDR S&P 500",
                "sector": "US Large Cap ETF",
                "asset_type": "ETF",
            },
        ]
        df_etfs = pd.DataFrame(etfs)

        full_metadata = pd.concat([df_stocks, df_etfs], ignore_index=True)
        print(f"Prepared {len(full_metadata)} assets.")

        full_metadata.to_sql(
            "assets_metadata", con=engine, if_exists="append", index=False
        )
        print("Success! Metadata refreshed.")

    except Exception as e:
        print(f"Metadata Error: {e}")


def download_market_data(engine):
    """Downloads historical price data IN BATCHES."""
    print("\n--- 3. Processing Market Prices (Yahoo Finance) ---")

    print("WIPING old market data...")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE market_data"))

    try:
        # 1. Get Tickers
        with engine.connect() as conn:
            result = conn.execute(text("SELECT ticker FROM assets_metadata"))
            valid_tickers = [row[0] for row in result]

        print(f"Found {len(valid_tickers)} tickers to download.")

        # 2. Batch Download
        BATCH_SIZE = 50
        all_data = []

        for i in range(0, len(valid_tickers), BATCH_SIZE):
            batch = valid_tickers[i : i + BATCH_SIZE]
            print(f"Downloading batch {i} to {i + BATCH_SIZE}...")

            try:
                data = yf.download(
                    batch,
                    start=START_DATE,
                    end=END_DATE,
                    group_by="column",
                    auto_adjust=False,
                    progress=False,
                )

                if not data.empty:
                    if len(batch) == 1:
                        data.columns = pd.MultiIndex.from_product([data.columns, batch])

                    df_long = data.stack(level=1, future_stack=True).reset_index()
                    all_data.append(df_long)
            except Exception as e:
                print(f"  Error in batch {i}: {e}")

        if not all_data:
            print("No data downloaded.")
            return

        print("Concatenating batches...")
        final_df = pd.concat(all_data)

        # 3. Clean & Rename
        final_df.rename(
            columns={
                "Date": "date",
                "Ticker": "ticker",
                "Adj Close": "adj_close",
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Open": "open",
                "Volume": "volume",
            },
            inplace=True,
        )

        final_df.dropna(inplace=True)

        print(f"Uploading {len(final_df)} rows to database...")
        final_df.to_sql(
            "market_data",
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )
        print("Success! Full history refreshed.")

    except Exception as e:
        print(f"Market Data Error: {e}")


if __name__ == "__main__":
    print("Initializing Database Update...")
    db_engine = get_db_engine()

    populate_cetes_data(db_engine)
    populate_metadata(db_engine)
    download_market_data(db_engine)

    print("\n--- Setup Complete ---")
