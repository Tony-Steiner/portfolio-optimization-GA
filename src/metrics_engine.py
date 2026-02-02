import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


class FinancialMetrics:
    def __init__(
        self, db_user=None, db_pass=None, db_host=None, db_name=None, db_port=None
    ):
        """
        Initializes connection.
        Priority: 1. Arguments passed in. 2. Environment Variables (.env)
        """
        user = db_user or os.getenv("DB_USER")
        password = db_pass or os.getenv("DB_PASS")
        host = db_host or os.getenv("DB_HOST")
        name = db_name or os.getenv("DB_NAME")
        port = db_port or os.getenv("DB_PORT", "5432")  # Added Port support

        if not user or not password:
            raise ValueError("Database credentials missing. Check your .env file.")

        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{name}"
        )
        self.prices = None

    def load_prices(self, tickers=None, start_date=None, end_date=None):
        """
        Fetches 'adj_close' prices from DB and pivots them so
        Columns = Tickers, Index = Date.
        """
        print("Fetching data from database...")

        # Base Query
        query = "SELECT date, ticker, adj_close FROM market_data"
        conditions = []

        # Add Filters if requested
        if tickers:
            # Safe(r) way to format list for SQL
            ticker_list = "', '".join(tickers)
            conditions.append(f"ticker IN ('{ticker_list}')")
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date ASC"

        # Load to Pandas
        df = pd.read_sql(query, self.engine)

        if df.empty:
            print("Warning: Database returned no data for these criteria.")
            self.prices = pd.DataFrame()
            return self.prices

        # --- SAFETY FIX: Force Date Format ---
        # This prevents the "String vs Timestamp" error in notebooks
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Pivot: Long -> Wide
        self.prices = df.pivot(index="date", columns="ticker", values="adj_close")

        # Forward fill missing days (crucial for weekends/holidays in finance)
        self.prices = self.prices.ffill()

        print(
            f"Loaded prices for {len(self.prices.columns)} assets over {len(self.prices)} days."
        )
        return self.prices

    def get_log_returns(self):
        """
        Calculates Logarithmic Returns: ln(Pt / Pt-1)
        """
        if self.prices is None or self.prices.empty:
            raise Exception("Prices not loaded. Call load_prices() first.")

        # Vectorized calculation
        # shift(1) moves the price down one day to align t with t-1
        log_returns = np.log(self.prices / self.prices.shift(1))

        return log_returns.dropna()

    def get_rolling_volatility(self, window=252):
        """
        Calculates annualized volatility over a rolling window.
        Default window=252 (One trading year).
        """
        returns = self.get_log_returns()

        # 1. Calculate std dev
        # 2. Multiply by sqrt(252) to annualize
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)

        return rolling_std.dropna()

    def get_correlation_matrix(self, window=None):
        """
        Calculates correlation between assets.
        If window is None, calculates over the entire history.
        """
        returns = self.get_log_returns()

        if window:
            # Returns a 3D object (Date, Ticker, Ticker)
            return returns.rolling(window=window).corr()
        else:
            # Returns a standard 2D Matrix (Ticker x Ticker)
            return returns.corr()
