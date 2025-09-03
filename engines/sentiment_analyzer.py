"""
News sentiment analysis engine for trading decisions
"""

import pandas as pd
import numpy as np
import sqlite3
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Optional sentiment analysis dependencies
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    # Will fall back to keyword-based analysis

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import data_config, asset_config
from utils.logger import logger

@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float

@dataclass
class MarketSentiment:
    """Market sentiment for a specific symbol"""
    symbol: str
    timestamp: datetime
    sentiment_score: float
    news_volume: int
    sentiment_trend: str  # "bullish", "bearish", "neutral"
    confidence: float

class SentimentAnalyzer:
    """Analyze sentiment from financial news"""
    
    def __init__(self):
        if SENTIMENT_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_keywords = self._load_financial_keywords()
    
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load financial sentiment keywords"""
        return {
            'bullish': [
                'bullish', 'bull', 'rising', 'surge', 'rally', 'gain', 'profit', 
                'growth', 'increase', 'up', 'positive', 'strong', 'beat', 'exceed',
                'outperform', 'upgrade', 'buy', 'optimistic', 'confidence', 'boost',
                'momentum', 'breakout', 'breakthrough', 'success', 'record', 'high'
            ],
            'bearish': [
                'bearish', 'bear', 'falling', 'drop', 'decline', 'loss', 'weak',
                'decrease', 'down', 'negative', 'miss', 'below', 'underperform',
                'downgrade', 'sell', 'pessimistic', 'concern', 'worry', 'fear',
                'crash', 'collapse', 'plunge', 'tumble', 'slide', 'low', 'risk'
            ],
            'uncertainty': [
                'uncertain', 'volatility', 'volatile', 'fluctuate', 'mixed',
                'unclear', 'confusion', 'debate', 'question', 'doubt', 'caution'
            ]
        }
    
    def analyze_text_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of a text using multiple methods"""
        if not SENTIMENT_AVAILABLE:
            # Fallback to keyword-based analysis
            return self._keyword_based_sentiment(text)
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Financial keyword enhancement
        keyword_sentiment = self._keyword_based_sentiment(text)
        
        # Combine sentiments (weighted average)
        compound_score = (
            0.4 * vader_scores['compound'] +
            0.3 * textblob_polarity +
            0.3 * keyword_sentiment.compound
        )
        
        # Calculate confidence based on agreement between methods
        confidence = 1.0 - abs(vader_scores['compound'] - textblob_polarity) / 2.0
        confidence = min(confidence + textblob_subjectivity * 0.1, 1.0)
        
        return SentimentScore(
            positive=max(vader_scores['pos'], (textblob_polarity + 1) / 2 if textblob_polarity > 0 else 0),
            negative=max(vader_scores['neg'], (-textblob_polarity + 1) / 2 if textblob_polarity < 0 else 0),
            neutral=vader_scores['neu'],
            compound=compound_score,
            confidence=confidence
        )
    
    def _keyword_based_sentiment(self, text: str) -> SentimentScore:
        """Fallback keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.financial_keywords['bullish'] if word in text_lower)
        bearish_count = sum(1 for word in self.financial_keywords['bearish'] if word in text_lower)
        uncertainty_count = sum(1 for word in self.financial_keywords['uncertainty'] if word in text_lower)
        
        total_count = bullish_count + bearish_count + uncertainty_count
        
        if total_count == 0:
            return SentimentScore(0.33, 0.33, 0.33, 0.0, 0.1)
        
        positive = bullish_count / total_count
        negative = bearish_count / total_count
        neutral = uncertainty_count / total_count
        
        compound = (bullish_count - bearish_count) / max(total_count, 1)
        confidence = min(total_count / 10.0, 1.0)  # More keywords = higher confidence
        
        return SentimentScore(positive, negative, neutral, compound, confidence)
    
    def analyze_news_for_symbol(self, symbol: str, hours_back: int = 24) -> MarketSentiment:
        """Analyze sentiment for a specific symbol"""
        try:
            # Get news data from database
            db_path = "data/trading_bot.db"
            conn = sqlite3.connect(db_path)
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            query = """
                SELECT n.title, n.content, n.published_at, n.relevance_score
                FROM news_data n
                JOIN news_symbols ns ON n.id = ns.news_id
                WHERE ns.symbol = ? AND n.published_at > ?
                ORDER BY n.published_at DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, cutoff_time))
            conn.close()
            
            if df.empty:
                logger.warning(f"No news found for {symbol} in the last {hours_back} hours")
                return MarketSentiment(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    sentiment_score=0.0,
                    news_volume=0,
                    sentiment_trend="neutral",
                    confidence=0.0
                )
            
            # Analyze sentiment for each news article
            sentiments = []
            total_relevance = 0
            
            for _, row in df.iterrows():
                text = f"{row['title']} {row['content'] if pd.notna(row['content']) else ''}"
                sentiment = self.analyze_text_sentiment(text)
                
                # Weight by relevance and recency
                relevance_weight = row['relevance_score'] if pd.notna(row['relevance_score']) else 0.5
                
                # Time decay (more recent news has higher weight)
                hours_ago = (datetime.now() - pd.to_datetime(row['published_at'])).total_seconds() / 3600
                time_weight = np.exp(-hours_ago / 12)  # Half-life of 12 hours
                
                weight = relevance_weight * time_weight
                sentiments.append((sentiment, weight))
                total_relevance += weight
            
            # Calculate weighted average sentiment
            if total_relevance > 0:
                weighted_sentiment = sum(s.compound * w for s, w in sentiments) / total_relevance
                avg_confidence = sum(s.confidence * w for s, w in sentiments) / total_relevance
            else:
                weighted_sentiment = 0.0
                avg_confidence = 0.0
            
            # Determine trend
            if weighted_sentiment > 0.1:
                trend = "bullish"
            elif weighted_sentiment < -0.1:
                trend = "bearish"
            else:
                trend = "neutral"
            
            market_sentiment = MarketSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=weighted_sentiment,
                news_volume=len(df),
                sentiment_trend=trend,
                confidence=avg_confidence
            )
            
            logger.info(f"Analyzed sentiment for {symbol}: {trend} ({weighted_sentiment:.3f})")
            return market_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}", error=e)
            return MarketSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=0.0,
                news_volume=0,
                sentiment_trend="neutral",
                confidence=0.0
            )
    
    def get_market_sentiment_overview(self, symbols: List[str] = None) -> Dict[str, MarketSentiment]:
        """Get sentiment overview for multiple symbols"""
        if symbols is None:
            from config.settings import get_all_symbols
        symbols = get_all_symbols()
        
        sentiment_overview = {}
        
        for symbol in symbols:
            sentiment_overview[symbol] = self.analyze_news_for_symbol(symbol)
        
        return sentiment_overview
    
    def get_sentiment_signals(self, symbol: str) -> Dict[str, float]:
        """Get trading signals based on sentiment"""
        sentiment = self.analyze_news_for_symbol(symbol)
        
        signals = {
            'sentiment_signal': 0.0,
            'news_volume_signal': 0.0,
            'confidence_signal': 0.0,
            'combined_signal': 0.0
        }
        
        # Sentiment signal (-1 to 1)
        signals['sentiment_signal'] = np.tanh(sentiment.sentiment_score * 2)
        
        # News volume signal (0 to 1) - high news volume can indicate volatility
        if sentiment.news_volume > 0:
            signals['news_volume_signal'] = min(sentiment.news_volume / 20.0, 1.0)
        
        # Confidence signal (0 to 1)
        signals['confidence_signal'] = sentiment.confidence
        
        # Combined signal (weighted)
        signals['combined_signal'] = (
            0.6 * signals['sentiment_signal'] * signals['confidence_signal'] +
            0.2 * signals['news_volume_signal'] +
            0.2 * (1 if sentiment.sentiment_trend in ['bullish', 'bearish'] else 0)
        )
        
        return signals

class SentimentDatabase:
    """Manage sentiment data in database"""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        self.init_sentiment_tables()
    
    def init_sentiment_tables(self):
        """Initialize sentiment-related tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                sentiment_score REAL NOT NULL,
                news_volume INTEGER DEFAULT 0,
                sentiment_trend TEXT DEFAULT 'neutral',
                confidence REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Update news_data table to include sentiment_score if not exists
        cursor.execute("PRAGMA table_info(news_data)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'sentiment_score' not in columns:
            cursor.execute("ALTER TABLE news_data ADD COLUMN sentiment_score REAL DEFAULT 0.0")
        
        conn.commit()
        conn.close()
    
    def store_sentiment(self, sentiment: MarketSentiment):
        """Store sentiment analysis result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_scores 
                (symbol, timestamp, sentiment_score, news_volume, sentiment_trend, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sentiment.symbol,
                sentiment.timestamp,
                sentiment.sentiment_score,
                sentiment.news_volume,
                sentiment.sentiment_trend,
                sentiment.confidence
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store sentiment for {sentiment.symbol}", error=e)
        finally:
            conn.close()
    
    def get_sentiment_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get sentiment history for a symbol"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.now() - timedelta(days=days)
        query = """
            SELECT timestamp, sentiment_score, news_volume, sentiment_trend, confidence
            FROM sentiment_scores
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, cutoff_time))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

class SentimentEngine:
    """Main sentiment analysis engine"""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.database = SentimentDatabase()
    
    def update_all_sentiments(self):
        """Update sentiment scores for all tracked symbols"""
        logger.info("Starting sentiment update for all symbols")
        
        from config.settings import get_all_symbols
        symbols = get_all_symbols()
        
        for symbol in symbols:
            try:
                sentiment = self.analyzer.analyze_news_for_symbol(symbol)
                self.database.store_sentiment(sentiment)
                logger.debug(f"Updated sentiment for {symbol}: {sentiment.sentiment_trend}")
            except Exception as e:
                logger.error(f"Failed to update sentiment for {symbol}", error=e)
        
        logger.info("Completed sentiment update for all symbols")
    
    def get_trading_sentiment_signals(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get sentiment-based trading signals for multiple symbols"""
        signals = {}
        
        for symbol in symbols:
            try:
                signals[symbol] = self.analyzer.get_sentiment_signals(symbol)
            except Exception as e:
                logger.error(f"Failed to get sentiment signals for {symbol}", error=e)
                signals[symbol] = {
                    'sentiment_signal': 0.0,
                    'news_volume_signal': 0.0,
                    'confidence_signal': 0.0,
                    'combined_signal': 0.0
                }
        
        return signals

