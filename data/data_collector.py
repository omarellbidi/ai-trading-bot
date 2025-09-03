"""
Data collection pipeline for market data, news, and sentiment analysis
"""

import pandas as pd
import requests
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import api_config, data_config, asset_config
from utils.logger import logger

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    
@dataclass 
class NewsData:
    """News data structure"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    relevance_score: float
    symbols: List[str]

class DatabaseManager:
    """Manage SQLite database operations"""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # News data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT NOT NULL,
                published_at DATETIME NOT NULL,
                url TEXT UNIQUE NOT NULL,
                relevance_score REAL DEFAULT 0.0,
                sentiment_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # News-symbol mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_symbols (
                news_id INTEGER,
                symbol TEXT,
                FOREIGN KEY (news_id) REFERENCES news_data (id),
                PRIMARY KEY (news_id, symbol)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_symbol_time ON market_data(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news_data(published_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_symbols(symbol)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_market_data(self, data: MarketData):
        """Insert market data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (data.symbol, data.timestamp, data.open_price, data.high_price, 
                  data.low_price, data.close_price, data.volume))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert market data for {data.symbol}", error=e)
        finally:
            conn.close()
    
    def insert_news_data(self, data: NewsData) -> Optional[int]:
        """Insert news data and return news_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news_data 
                (title, content, source, published_at, url, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (data.title, data.content, data.source, data.published_at, 
                  data.url, data.relevance_score))
            
            news_id = cursor.lastrowid
            if news_id and data.symbols:
                # Insert symbol mappings
                for symbol in data.symbols:
                    cursor.execute("""
                        INSERT OR IGNORE INTO news_symbols (news_id, symbol)
                        VALUES (?, ?)
                    """, (news_id, symbol))
            
            conn.commit()
            return news_id
        except Exception as e:
            logger.error("Failed to insert news data", error=e)
            return None
        finally:
            conn.close()

class MarketDataCollector:
    """Collect real-time and historical market data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data from Yahoo Finance"""
        try:
            url = f"{api_config.YAHOO_FINANCE_BASE_URL}/v8/finance/chart/{symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                
                # Get latest data point
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                if timestamps and quotes:
                    latest_idx = -1
                    return MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamps[latest_idx]),
                        open_price=quotes['open'][latest_idx] or 0,
                        high_price=quotes['high'][latest_idx] or 0,
                        low_price=quotes['low'][latest_idx] or 0,
                        close_price=quotes['close'][latest_idx] or 0,
                        volume=quotes['volume'][latest_idx] or 0
                    )
            
            logger.warning(f"Failed to get data for {symbol}: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting realtime data for {symbol}", error=e)
            return None
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> List[MarketData]:
        """Get historical market data"""
        try:
            url = f"{api_config.YAHOO_FINANCE_BASE_URL}/v8/finance/chart/{symbol}"
            params = {"range": period, "interval": "1d"}
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                market_data = []
                for i in range(len(timestamps)):
                    if all(quotes[key][i] is not None for key in ['open', 'high', 'low', 'close']):
                        market_data.append(MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(timestamps[i]),
                            open_price=quotes['open'][i],
                            high_price=quotes['high'][i],
                            low_price=quotes['low'][i],
                            close_price=quotes['close'][i],
                            volume=quotes['volume'][i] or 0
                        ))
                
                return market_data
            
            logger.warning(f"Failed to get historical data for {symbol}: {response.status_code}")
            return []
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}", error=e)
            return []
    
    def collect_all_symbols(self, symbols: List[str], historical: bool = False):
        """Collect data for all symbols"""
        logger.info(f"Starting data collection for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                if historical:
                    # Get historical data first
                    historical_data = self.get_historical_data(symbol)
                    for data_point in historical_data:
                        self.db_manager.insert_market_data(data_point)
                    logger.info(f"Collected {len(historical_data)} historical records for {symbol}")
                
                # Get real-time data
                realtime_data = self.get_realtime_data(symbol)
                if realtime_data:
                    self.db_manager.insert_market_data(realtime_data)
                    logger.debug(f"Updated realtime data for {symbol}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}", error=e)
                continue
        
        logger.info("Data collection completed")

class NewsCollector:
    """Collect financial news and analyze sentiment"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.session = requests.Session()
    
    def get_financial_news(self, keywords: List[str] = None) -> List[NewsData]:
        """Get financial news from News API"""
        if keywords is None:
            keywords = ["stock market", "trading", "finance", "economy"]
        
        news_articles = []
        
        for keyword in keywords:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': api_config.NEWS_API_KEY,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20,
                    'from': (datetime.now() - timedelta(hours=6)).isoformat()
                }
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        if article.get('title') and article.get('url'):
                            # Extract relevant symbols from title and description
                            symbols = self._extract_symbols(
                                f"{article.get('title', '')} {article.get('description', '')}"
                            )
                            
                            news_articles.append(NewsData(
                                title=article['title'],
                                content=article.get('description', ''),
                                source=article.get('source', {}).get('name', 'Unknown'),
                                published_at=datetime.fromisoformat(
                                    article['publishedAt'].replace('Z', '+00:00')
                                ),
                                url=article['url'],
                                relevance_score=len(symbols) / 10.0,  # Simple relevance scoring
                                symbols=symbols
                            ))
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error getting news for keyword '{keyword}'", error=e)
                continue
        
        return news_articles
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract relevant symbols from news text"""
        text_upper = text.upper()
        relevant_symbols = []
        
        # Check all tracked symbols
        from config.settings import get_all_symbols
        all_symbols = get_all_symbols()
        for symbol in all_symbols:
            if symbol.upper() in text_upper:
                relevant_symbols.append(symbol)
        
        # Check for common company names
        company_mappings = {
            'APPLE': 'AAPL',
            'MICROSOFT': 'MSFT', 
            'GOOGLE': 'GOOGL',
            'TESLA': 'TSLA',
            'NVIDIA': 'NVDA',
            'BITCOIN': 'BTCUSD',
            'ETHEREUM': 'ETHUSD'
        }
        
        for company, symbol in company_mappings.items():
            if company in text_upper and symbol not in relevant_symbols:
                relevant_symbols.append(symbol)
        
        return relevant_symbols
    
    def collect_news(self):
        """Collect and store news data"""
        logger.info("Starting news collection")
        
        try:
            news_articles = self.get_financial_news()
            
            for article in news_articles:
                news_id = self.db_manager.insert_news_data(article)
                if news_id:
                    logger.debug(f"Stored news: {article.title[:50]}...")
            
            logger.info(f"Collected {len(news_articles)} news articles")
            
        except Exception as e:
            logger.error("Failed to collect news", error=e)

class DataCollectionOrchestrator:
    """Orchestrate all data collection activities"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.market_collector = MarketDataCollector(self.db_manager)
        self.news_collector = NewsCollector(self.db_manager)
    
    def run_initial_collection(self):
        """Run initial data collection with historical data"""
        logger.info("Starting initial data collection")
        
        # Collect historical market data
        from config.settings import get_all_symbols
        all_symbols = get_all_symbols()
        self.market_collector.collect_all_symbols(all_symbols, historical=True)
        
        # Collect initial news
        self.news_collector.collect_news()
        
        logger.info("Initial data collection completed")
    
    def run_realtime_collection(self):
        """Run real-time data collection"""
        logger.info("Starting real-time data collection")
        
        # Collect real-time market data
        from config.settings import get_all_symbols
        all_symbols = get_all_symbols()
        self.market_collector.collect_all_symbols(all_symbols, historical=False)
        
        # Collect news every 5 minutes
        if int(time.time()) % data_config.NEWS_UPDATE_INTERVAL == 0:
            self.news_collector.collect_news()
    
    def get_latest_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get latest market data for a symbol"""
        conn = sqlite3.connect(self.db_manager.db_path)
        
        query = """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM market_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df

if __name__ == "__main__":
    # Test the data collection pipeline
    orchestrator = DataCollectionOrchestrator()
    orchestrator.run_initial_collection()