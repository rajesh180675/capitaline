# /mount/src/data/financial_engine.py

"""
Backend Financial Engine - Core Logic for Financial Data Processing
This module handles all data fetching, processing, and analysis operations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
import time
import re
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinancialDataCache:
    """Simple in-memory cache for financial data with a Time-To-Live (TTL)."""
    def __init__(self, ttl_minutes: int = 15):
        self.cache = {}
        self.ttl_seconds = ttl_minutes * 60
        logger.info(f"Cache initialized with TTL of {ttl_minutes} minutes.")

    def _get_cache_key(self, symbol: str, data_type: str) -> str:
        return f"{symbol}_{data_type}"

    def get(self, symbol: str, data_type: str) -> Optional[Any]:
        key = self._get_cache_key(symbol, data_type)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (time.time() - timestamp) < self.ttl_seconds:
                logger.info(f"Cache HIT for key: {key}")
                return data
            else:
                logger.info(f"Cache EXPIRED for key: {key}")
                del self.cache[key]
        logger.info(f"Cache MISS for key: {key}")
        return None

    def set(self, symbol: str, data_type: str, data: Any) -> None:
        key = self._get_cache_key(symbol, data_type)
        self.cache[key] = (data, time.time())
        logger.info(f"Cache SET for key: {key}")

    def clear(self) -> None:
        self.cache.clear()
        logger.info("Cache has been cleared.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_entries': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'ttl_minutes': self.ttl_seconds / 60
        }


def log_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else ''
        func_name = f"{class_name}.{func.__name__}"
        logger.info(f"Executing {func_name}...")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func_name} completed in {end_time - start_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}", exc_info=False) # Set exc_info to False for cleaner logs
            raise
    return wrapper


class FinancialDataProcessor:
    def __init__(self, cache_ttl_minutes: int = 15):
        self.cache = FinancialDataCache(cache_ttl_minutes)
        self.MILLION = 1_000_000
        self.BILLION = 1_000_000_000

    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        if not symbol or not isinstance(symbol, str):
            return False, "Symbol cannot be empty."
        symbol = symbol.strip().upper()
        symbol = re.sub(r'[^A-Z0-9.-]', '', symbol)
        if not re.match(r'^[A-Z0-9.-]{1,10}(\.[A-Z]{1,5})?$', symbol):
            return False, "Symbol contains invalid characters."
        return True, symbol

    @staticmethod
    def safe_division(numerator: Union[pd.Series, float], denominator: Union[pd.Series, float]) -> Union[pd.Series, float]:
        """
        BACKEND FIX: Perform safe division for both scalars and Pandas Series.
        This function now correctly handles vectorized operations.
        """
        if isinstance(denominator, pd.Series):
            # For a Series, replace 0s and NaNs with NaN, then perform division.
            # The result will be NaN where division by zero would occur.
            # Fill the resulting NaNs with 0.
            den_safe = denominator.replace(0, np.nan)
            return (numerator / den_safe).fillna(0)
        else:
            # Original scalar logic
            if denominator is None or denominator == 0 or np.isnan(denominator):
                return 0
            if numerator is None or np.isnan(numerator):
                return 0
            return numerator / denominator

    @log_operation
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        cached_data = self.cache.get(symbol, 'stock_info')
        if cached_data:
            return cached_data
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info or info.get('trailingPE') is None and info.get('regularMarketPrice') is None:
                logger.warning(f"No valid info found for symbol: {symbol}.")
                return None
            self.cache.set(symbol, 'stock_info', info)
            return info
        except Exception as e:
            logger.error(f"yfinance error fetching info for {symbol}: {e}")
            return None

    @log_operation
    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        cached_data = self.cache.get(symbol, 'financial_statements')
        if cached_data:
            return cached_data
        statements = {}
        try:
            stock = yf.Ticker(symbol)
            statement_map = {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow
            }
            for name, data in statement_map.items():
                if not data.empty:
                    df = data.T.reset_index()
                    df.rename(columns={'index': 'Year'}, inplace=True)
                    df['Year'] = pd.to_datetime(df['Year']).dt.strftime('%Y')
                    statements[name] = df
                else:
                    statements[name] = pd.DataFrame()
            if any(not df.empty for df in statements.values()):
                self.cache.set(symbol, 'financial_statements', statements)
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {e}")
        return statements

    @log_operation
    def calculate_financial_ratios(self, statements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        income_stmt = statements.get('income_statement', pd.DataFrame())
        balance_sheet = statements.get('balance_sheet', pd.DataFrame())

        if income_stmt.empty or balance_sheet.empty:
            return pd.DataFrame()

        merged_df = pd.merge(income_stmt, balance_sheet, on="Year", how="inner", suffixes=('_inc', '_bal'))
        if merged_df.empty:
            return pd.DataFrame()

        ratios_df = pd.DataFrame()
        ratios_df['Year'] = merged_df['Year']

        def get_col(df, options):
            for col in options:
                if col in df.columns:
                    return pd.to_numeric(df[col], errors='coerce')
            return pd.Series(0, index=df.index)

        net_income = get_col(merged_df, ['Net Income', 'Net Income Common Stockholders'])
        total_revenue = get_col(merged_df, ['Total Revenue', 'Revenue'])
        total_assets = get_col(merged_df, ['Total Assets'])
        stockholder_equity = get_col(merged_df, ['Stockholder Equity', 'Total Stockholder Equity'])
        current_assets = get_col(merged_df, ['Current Assets', 'Total Current Assets'])
        current_liabilities = get_col(merged_df, ['Current Liabilities', 'Total Current Liabilities'])
        total_liabilities = get_col(merged_df, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])

        ratios_df['Net Profit Margin (%)'] = (self.safe_division(net_income, total_revenue) * 100).round(2)
        ratios_df['ROE (%)'] = (self.safe_division(net_income, stockholder_equity) * 100).round(2)
        ratios_df['Current Ratio'] = self.safe_division(current_assets, current_liabilities).round(2)
        ratios_df['Debt to Equity'] = self.safe_division(total_liabilities, stockholder_equity).round(2)

        return ratios_df.sort_values('Year', ascending=False)

    @log_operation
    def get_historical_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        cache_key = f"historical_{period}"
        cached_data = self.cache.get(symbol, cache_key)
        if cached_data is not None:
            return cached_data
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            if hist.empty:
                return pd.DataFrame()
            hist.reset_index(inplace=True)
            self.cache.set(symbol, cache_key, hist)
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def format_large_number(self, value) -> str:
        if not isinstance(value, (int, float)) or pd.isna(value):
            return "N/A"
        if abs(value) >= self.BILLION:
            return f"{value / self.BILLION:.2f}B"
        if abs(value) >= self.MILLION:
            return f"{value / self.MILLION:.2f}M"
        return f"{value:,.0f}"

    @log_operation
    def get_company_analysis(self, symbol: str, period: str = '1y') -> Optional[Dict[str, Any]]:
        stock_info = self.get_stock_info(symbol)
        if not stock_info:
            return None
        
        statements = self.get_financial_statements(symbol)
        ratios = self.calculate_financial_ratios(statements)
        historical_data = self.get_historical_data(symbol, period=period)

        price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice'))
        prev_close = stock_info.get('previousClose')

        return {
            'basic_info': {
                'symbol': symbol,
                'company_name': stock_info.get('longName', 'N/A'),
                'sector': stock_info.get('sector', 'N/A'),
                'industry': stock_info.get('industry', 'N/A'),
                'summary': stock_info.get('longBusinessSummary', 'N/A'),
                'website': stock_info.get('website', '#'),
                'current_price': price,
                'change_percent': self.safe_division(price - prev_close, prev_close) * 100 if price and prev_close else 0,
                'market_cap': stock_info.get('marketCap'),
                'market_cap_formatted': self.format_large_number(stock_info.get('marketCap')),
                'volume': stock_info.get('volume'),
                'fifty_two_week_high': stock_info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': stock_info.get('fiftyTwoWeekLow'),
                'dividend_yield': stock_info.get('dividendYield'),
            },
            'valuation_metrics': {'pe_ratio': stock_info.get('trailingPE'), 'pb_ratio': stock_info.get('priceToBook')},
            'profitability_metrics': {'roe': stock_info.get('returnOnEquity'), 'profit_margin': stock_info.get('profitMargins')},
            'historical_data': historical_data,
            'statements': statements,
            'ratios': ratios
        }

# --- Convenience Functions for Frontend ---
financial_processor = FinancialDataProcessor()

def get_stock_analysis(symbol: str, period: str = '1y') -> Optional[Dict[str, Any]]:
    return financial_processor.get_company_analysis(symbol, period=period)
def validate_symbol(symbol: str) -> Tuple[bool, str]:
    return FinancialDataProcessor.validate_symbol(symbol)
def clear_cache():
    financial_processor.cache.clear()
def get_cache_stats() -> Dict[str, Any]:
    return financial_processor.cache.get_stats()
