#!/usr/bin/env python3
"""
Main entry point for the AI Trading Bot
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.trading_engine import main as trading_main
from data.data_collector import DataCollectionOrchestrator
from models.price_predictor import ModelManager
from engines.sentiment_analyzer import SentimentEngine
from utils.logger import logger

def run_data_collection():
    """Run initial data collection"""
    logger.info("Running data collection...")
    orchestrator = DataCollectionOrchestrator()
    orchestrator.run_initial_collection()
    logger.info("Data collection completed")

def train_models():
    """Train ML models"""
    logger.info("Training ML models...")
    manager = ModelManager()
    results = manager.train_all_models()
    
    for symbol, performance in results.items():
        if performance:
            logger.info(f"Trained models for {symbol}")
            for model_name, perf in performance.items():
                logger.info(f"  {model_name}: Accuracy={perf.accuracy:.3f}")
        else:
            logger.warning(f"Failed to train models for {symbol}")

def update_sentiment():
    """Update sentiment analysis"""
    logger.info("Updating sentiment analysis...")
    engine = SentimentEngine()
    engine.update_all_sentiments()
    logger.info("Sentiment analysis completed")

def run_trading():
    """Run the main trading engine"""
    logger.info("Starting trading engine...")
    trading_main()

def run_test():
    """Run system tests"""
    logger.info("Running system tests...")
    
    # Test data collection
    try:
        orchestrator = DataCollectionOrchestrator()
        test_data = orchestrator.get_latest_data("AAPL", limit=10)
        if not test_data.empty:
            logger.info("✅ Data collection working")
        else:
            logger.warning("⚠️ No data available")
    except Exception as e:
        logger.error("❌ Data collection failed", error=e)
    
    # Test sentiment analysis
    try:
        sentiment_engine = SentimentEngine()
        signals = sentiment_engine.get_trading_sentiment_signals(["AAPL"])
        if "AAPL" in signals:
            logger.info("✅ Sentiment analysis working")
        else:
            logger.warning("⚠️ Sentiment analysis no data")
    except Exception as e:
        logger.error("❌ Sentiment analysis failed", error=e)
    
    # Test model prediction
    try:
        model_manager = ModelManager()
        prediction = model_manager.get_prediction("AAPL")
        if prediction:
            logger.info("✅ Model prediction working")
        else:
            logger.warning("⚠️ No trained model available")
    except Exception as e:
        logger.error("❌ Model prediction failed", error=e)
    
    logger.info("System test completed")

def main():
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('command', choices=[
        'collect', 'train', 'sentiment', 'trade', 'test'
    ], help='Command to run')
    
    parser.add_argument('--symbols', nargs='+', 
                       help='Symbols to process (default: all configured symbols)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Route to appropriate function
    if args.command == 'collect':
        run_data_collection()
    elif args.command == 'train':
        train_models()
    elif args.command == 'sentiment':
        update_sentiment()
    elif args.command == 'trade':
        run_trading()
    elif args.command == 'test':
        run_test()

if __name__ == "__main__":
    main()