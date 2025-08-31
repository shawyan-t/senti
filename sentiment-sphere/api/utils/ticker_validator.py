"""
NASDAQ/NYSE Ticker Validation System
Validates stock tickers and provides company information for financial sentiment analysis.
"""

import re
import requests
import json
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import yfinance as yf

class TickerValidator:
    """Validates NASDAQ and NYSE stock tickers and retrieves company information."""
    
    def __init__(self):
        """Initialize the ticker validator with common tickers and exchanges."""
        # Major NASDAQ/NYSE tickers for quick validation
        self.known_tickers = {
            # Major Tech Stocks (NASDAQ)
            'AAPL': {'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
            'MSFT': {'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology'},
            'GOOGL': {'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
            'AMZN': {'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Discretionary'},
            'NVDA': {'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology'},
            'META': {'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
            'TSLA': {'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Discretionary'},
            'NFLX': {'name': 'Netflix Inc.', 'exchange': 'NASDAQ', 'sector': 'Communication Services'},
            
            # Major NYSE Stocks
            'JPM': {'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'sector': 'Financial Services'},
            'JNJ': {'name': 'Johnson & Johnson', 'exchange': 'NYSE', 'sector': 'Healthcare'},
            'WMT': {'name': 'Walmart Inc.', 'exchange': 'NYSE', 'sector': 'Consumer Staples'},
            'PG': {'name': 'Procter & Gamble Co.', 'exchange': 'NYSE', 'sector': 'Consumer Staples'},
            'HD': {'name': 'Home Depot Inc.', 'exchange': 'NYSE', 'sector': 'Consumer Discretionary'},
            'BAC': {'name': 'Bank of America Corp.', 'exchange': 'NYSE', 'sector': 'Financial Services'},
            'DIS': {'name': 'Walt Disney Co.', 'exchange': 'NYSE', 'sector': 'Communication Services'},
            'KO': {'name': 'Coca-Cola Co.', 'exchange': 'NYSE', 'sector': 'Consumer Staples'},
            
            # ETFs
            'VOO': {'name': 'Vanguard S&P 500 ETF', 'exchange': 'NYSE', 'sector': 'ETF'},
            'SPY': {'name': 'SPDR S&P 500 ETF Trust', 'exchange': 'NYSE', 'sector': 'ETF'},
            'QQQ': {'name': 'Invesco QQQ Trust', 'exchange': 'NASDAQ', 'sector': 'ETF'},
            'VTI': {'name': 'Vanguard Total Stock Market ETF', 'exchange': 'NYSE', 'sector': 'ETF'},
        }
    
    def validate_ticker(self, ticker_input: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Validate if input is a valid NASDAQ/NYSE ticker.
        
        Args:
            ticker_input (str): User input to validate
            
        Returns:
            Tuple[bool, Optional[Dict], str]: (is_valid, company_info, error_message)
        """
        # Clean and normalize input
        ticker = ticker_input.strip().upper()
        
        # Basic format validation
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            return False, None, f"'{ticker_input}' is not a valid ticker format. Please enter a 1-5 letter stock symbol (e.g., AAPL, NVDA, META)."
        
        # Check known tickers first (fast lookup)
        if ticker in self.known_tickers:
            return True, self.known_tickers[ticker], ""
        
        # Use yfinance to validate unknown tickers
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if not info or 'symbol' not in info:
                return False, None, f"'{ticker}' is not a recognized stock ticker. Please enter a valid NASDAQ or NYSE ticker."
            
            # Check if it's from NASDAQ or NYSE
            exchange = info.get('exchange', '').upper()
            if exchange not in ['NASDAQ', 'NMS', 'NYSE', 'NYQ']:
                return False, None, f"'{ticker}' is not traded on NASDAQ or NYSE. This system only analyzes NASDAQ and NYSE stocks."
            
            # Extract company information
            company_info = {
                'name': info.get('longName', info.get('shortName', ticker)),
                'exchange': 'NASDAQ' if exchange in ['NASDAQ', 'NMS'] else 'NYSE',
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency', 'USD')
            }
            
            return True, company_info, ""
            
        except Exception as e:
            return False, None, f"Could not validate ticker '{ticker}'. Please ensure it's a valid NASDAQ or NYSE stock symbol."
    
    def get_financial_context(self, ticker: str, company_info: Dict) -> Dict:
        """
        Get additional financial context for the ticker.
        
        Args:
            ticker (str): Valid ticker symbol
            company_info (Dict): Company information
            
        Returns:
            Dict: Financial context for sentiment analysis
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get recent price data
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0] if len(hist) > 1 else current_price
                price_change = ((current_price - prev_price) / prev_price) * 100
            else:
                current_price = None
                price_change = 0
            
            # Get basic info
            info = stock.info
            
            financial_context = {
                'ticker': ticker,
                'company_name': company_info['name'],
                'exchange': company_info['exchange'],
                'sector': company_info['sector'],
                'current_price': float(current_price) if current_price else None,
                'price_change_5d': round(price_change, 2),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume'),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return financial_context
            
        except Exception as e:
            print(f"Error getting financial context for {ticker}: {e}")
            return {
                'ticker': ticker,
                'company_name': company_info['name'],
                'exchange': company_info['exchange'],
                'sector': company_info['sector'],
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def generate_search_queries(self, ticker: str, company_info: Dict) -> List[str]:
        """
        Generate comprehensive search queries in hierarchical order for financial sentiment analysis.
        
        Hierarchy: Financial Outlets -> Reputable News -> Discussion Boards -> Social Media -> General Web
        
        Args:
            ticker (str): Ticker symbol
            company_info (Dict): Company information
            
        Returns:
            List[str]: List of search queries optimized for diverse, hierarchical sources
        """
        company_name = company_info['name']
        
        queries = [
            # TIER 1: Premium Financial Sources (Highest Priority)
            f"${ticker} site:bloomberg.com OR site:reuters.com OR site:marketwatch.com OR site:wsj.com OR site:ft.com",
            f'"{company_name}" earnings site:barrons.com OR site:cnbc.com OR site:finance.yahoo.com',
            
            # TIER 2: Reputable News Sources
            f"${ticker} stock news site:cnn.com OR site:bbc.com OR site:forbes.com OR site:reuters.com",
            f'"{company_name}" financial news site:nytimes.com OR site:washingtonpost.com OR site:ap.org',
            
            # TIER 3: Financial Analysis & Discussion Boards
            f"${ticker} analysis site:seekingalpha.com OR site:fool.com OR site:zacks.com OR site:investopedia.com",
            f'"{company_name}" discussion site:stocktwits.com OR site:investorshub.com',
            
            # TIER 4: Reddit & Social Discussion (Filtered for quality)
            f"${ticker} reddit site:reddit.com/r/investing OR site:reddit.com/r/stocks OR site:reddit.com/r/SecurityAnalysis",
            f"${ticker} sentiment site:reddit.com/r/wallstreetbets",
            
            # TIER 5: Broader Financial Community & News
            f"${ticker} stock opinion OR earnings OR forecast",
            f'"{company_name}" latest news OR analyst rating OR price target',
            
            # TIER 6: Recent Breaking News (Any reputable source)
            f"${ticker} breaking news OR latest update OR just released",
            f'"{company_name}" today OR this week OR recent'
        ]
        
        return queries[:3]  # Reduced to 3 to avoid Google API rate limits
    
    def is_market_hours(self) -> bool:
        """Check if markets are currently open (rough estimate)."""
        from datetime import datetime
        import pytz
        
        try:
            # Get current time in Eastern timezone
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            
            # Check if it's a weekday and between market hours
            if now.weekday() >= 5:  # Weekend
                return False
            
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except:
            return True  # Assume markets could be open if we can't determine


# Global instance
_ticker_validator = None

def get_ticker_validator() -> TickerValidator:
    """Get singleton instance of ticker validator."""
    global _ticker_validator
    if _ticker_validator is None:
        _ticker_validator = TickerValidator()
    return _ticker_validator