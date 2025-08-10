import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import json
import time

class MarketDataIntegrator:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    def get_stock_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Get comprehensive stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            
            # Get current info
            info = stock.info
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(hist)
            
            # Get financial statements
            financials = self._get_financial_statements(stock)
            
            return {
                "ticker": ticker,
                "current_price": info.get("regularMarketPrice", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 1.0),
                "volume": info.get("volume", 0),
                "historical_data": hist.to_dict('records') if not hist.empty else [],
                "technical_indicators": technical_indicators,
                "financials": financials,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return {"error": str(e)}
    
    def _calculate_technical_indicators(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from historical data"""
        if hist_data.empty:
            return {}
        
        try:
            # Moving averages
            hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
            hist_data['EMA_12'] = hist_data['Close'].ewm(span=12).mean()
            hist_data['EMA_26'] = hist_data['Close'].ewm(span=26).mean()
            
            # RSI
            delta = hist_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist_data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            hist_data['MACD'] = hist_data['EMA_12'] - hist_data['EMA_26']
            hist_data['MACD_Signal'] = hist_data['MACD'].ewm(span=9).mean()
            hist_data['MACD_Histogram'] = hist_data['MACD'] - hist_data['MACD_Signal']
            
            # Bollinger Bands
            hist_data['BB_Middle'] = hist_data['Close'].rolling(window=20).mean()
            bb_std = hist_data['Close'].rolling(window=20).std()
            hist_data['BB_Upper'] = hist_data['BB_Middle'] + (bb_std * 2)
            hist_data['BB_Lower'] = hist_data['BB_Middle'] - (bb_std * 2)
            
            # Get latest values
            latest = hist_data.iloc[-1] if len(hist_data) > 0 else None
            
            if latest is not None:
                return {
                    "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else None,
                    "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
                    "ema_12": float(latest['EMA_12']) if pd.notna(latest['EMA_12']) else None,
                    "ema_26": float(latest['EMA_26']) if pd.notna(latest['EMA_26']) else None,
                    "rsi": float(latest['RSI']) if pd.notna(latest['RSI']) else None,
                    "macd": float(latest['MACD']) if pd.notna(latest['MACD']) else None,
                    "macd_signal": float(latest['MACD_Signal']) if pd.notna(latest['MACD_Signal']) else None,
                    "bb_upper": float(latest['BB_Upper']) if pd.notna(latest['BB_Upper']) else None,
                    "bb_middle": float(latest['BB_Middle']) if pd.notna(latest['BB_Middle']) else None,
                    "bb_lower": float(latest['BB_Lower']) if pd.notna(latest['BB_Lower']) else None,
                    "current_price": float(latest['Close'])
                }
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
        
        return {}
    
    def _get_financial_statements(self, stock: yf.Ticker) -> Dict[str, Any]:
        """Get financial statements data"""
        try:
            # Get income statement
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            return {
                "income_statement": income_stmt.to_dict() if not income_stmt.empty else {},
                "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {}
            }
        except Exception as e:
            print(f"Error fetching financial statements: {e}")
            return {}
    
    def get_market_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get market sentiment data (placeholder for real sentiment API)"""
        try:
            # This would integrate with real sentiment APIs like:
            # - News sentiment APIs
            # - Social media sentiment
            # - Analyst ratings
            
            # For now, return mock sentiment data
            sentiment_score = np.random.uniform(-1, 1)  # Random sentiment between -1 and 1
            
            if sentiment_score > 0.3:
                overall_sentiment = "positive"
            elif sentiment_score < -0.3:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            return {
                "ticker": ticker,
                "sentiment_score": sentiment_score,
                "overall_sentiment": overall_sentiment,
                "confidence": abs(sentiment_score),
                "sources": ["news", "social_media", "analyst_ratings"],
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_earnings_calendar(self, ticker: str) -> Dict[str, Any]:
        """Get upcoming earnings calendar"""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and not calendar.empty:
                return {
                    "ticker": ticker,
                    "next_earnings_date": calendar.index[0].strftime("%Y-%m-%d") if len(calendar) > 0 else None,
                    "estimated_eps": float(calendar.iloc[0]['Earnings Average']) if len(calendar) > 0 else None,
                    "estimated_revenue": float(calendar.iloc[0]['Revenue Average']) if len(calendar) > 0 else None
                }
            else:
                return {"ticker": ticker, "next_earnings_date": None}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_analyst_recommendations(self, ticker: str) -> Dict[str, Any]:
        """Get analyst recommendations"""
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                # Get latest recommendations
                latest_recs = recommendations.tail(10)
                
                # Count recommendations
                rec_counts = latest_recs['To Grade'].value_counts()
                
                return {
                    "ticker": ticker,
                    "buy_count": int(rec_counts.get('Buy', 0)),
                    "hold_count": int(rec_counts.get('Hold', 0)),
                    "sell_count": int(rec_counts.get('Sell', 0)),
                    "strong_buy_count": int(rec_counts.get('Strong Buy', 0)),
                    "total_recommendations": len(latest_recs),
                    "latest_recommendations": latest_recs.to_dict('records')
                }
            else:
                return {"ticker": ticker, "no_recommendations": True}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview"""
        try:
            # Get major indices
            indices = {
                "SPY": "S&P 500",
                "QQQ": "NASDAQ",
                "DIA": "Dow Jones",
                "IWM": "Russell 2000"
            }
            
            market_data = {}
            for ticker, name in indices.items():
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    market_data[name] = {
                        "ticker": ticker,
                        "price": info.get("regularMarketPrice", 0),
                        "change": info.get("regularMarketChange", 0),
                        "change_percent": info.get("regularMarketChangePercent", 0),
                        "volume": info.get("volume", 0)
                    }
                except:
                    continue
            
            return {
                "market_data": market_data,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 1.0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "fifty_day_average": info.get("fiftyDayAverage", 0),
                "two_hundred_day_average": info.get("twoHundredDayAverage", 0),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_historical_returns(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Calculate historical returns and volatility"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": "No historical data available"}
            
            # Calculate returns
            hist['Returns'] = hist['Close'].pct_change()
            
            # Calculate metrics
            total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(hist)) - 1
            volatility = hist['Returns'].std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            hist['Cumulative'] = (1 + hist['Returns']).cumprod()
            hist['Running_Max'] = hist['Cumulative'].expanding().max()
            hist['Drawdown'] = (hist['Cumulative'] - hist['Running_Max']) / hist['Running_Max']
            max_drawdown = hist['Drawdown'].min()
            
            return {
                "ticker": ticker,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "returns_data": hist[['Returns', 'Drawdown']].to_dict('records')
            }
            
        except Exception as e:
            return {"error": str(e)} 