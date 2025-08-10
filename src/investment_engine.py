import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
import requests
from dataclasses import dataclass

@dataclass
class InvestmentSignal:
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH
    time_horizon: str  # SHORT, MEDIUM, LONG
    target_price: float
    stop_loss: float
    take_profit: float
    key_factors: List[str]
    timestamp: datetime

class InvestmentDecisionEngine:
    def __init__(self):
        self.risk_thresholds = {
            "conservative": {"max_risk": 0.3, "min_confidence": 0.8},
            "moderate": {"max_risk": 0.5, "min_confidence": 0.7},
            "aggressive": {"max_risk": 0.7, "min_confidence": 0.6}
        }
        
    def generate_investment_decision(self, 
                                   financial_analysis: Dict[str, Any],
                                   market_forecast: Dict[str, Any],
                                   sentiment_analysis: Dict[str, Any],
                                   risk_profile: str = "moderate") -> InvestmentSignal:
        """
        Generate investment decision based on comprehensive analysis
        """
        
        # Calculate composite score
        financial_score = self._calculate_financial_score(financial_analysis)
        forecast_score = self._calculate_forecast_score(market_forecast)
        sentiment_score = self._calculate_sentiment_score(sentiment_analysis)
        
        # Weighted decision score
        decision_score = (
            financial_score * 0.4 +  # Financial analysis weight
            forecast_score * 0.4 +   # Market forecast weight
            sentiment_score * 0.2    # Sentiment weight
        )
        
        # Determine action based on score
        action, confidence = self._determine_action(decision_score, risk_profile)
        
        # Calculate price targets
        current_price = self._get_current_price(financial_analysis.get("ticker", ""))
        target_price, stop_loss, take_profit = self._calculate_price_targets(
            current_price, action, confidence, market_forecast
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            financial_analysis, market_forecast, sentiment_analysis, action
        )
        
        # Identify key factors
        key_factors = self._identify_key_factors(
            financial_analysis, market_forecast, sentiment_analysis
        )
        
        return InvestmentSignal(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            risk_level=self._assess_risk_level(decision_score),
            time_horizon=self._determine_time_horizon(market_forecast),
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            key_factors=key_factors,
            timestamp=datetime.now()
        )
    
    def _calculate_financial_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate financial health score (0-1)"""
        score = 0.0
        factors = 0
        
        # Revenue growth
        if "revenue_growth" in analysis:
            growth = analysis["revenue_growth"]
            if growth > 0.15:
                score += 0.25
            elif growth > 0.05:
                score += 0.15
            elif growth > 0:
                score += 0.05
            factors += 1
        
        # Profit margin
        if "profit_margin" in analysis:
            margin = analysis["profit_margin"]
            if margin > 0.20:
                score += 0.25
            elif margin > 0.10:
                score += 0.15
            elif margin > 0.05:
                score += 0.05
            factors += 1
        
        # Debt to equity
        if "debt_to_equity" in analysis:
            debt_ratio = analysis["debt_to_equity"]
            if debt_ratio < 0.3:
                score += 0.25
            elif debt_ratio < 0.5:
                score += 0.15
            elif debt_ratio < 1.0:
                score += 0.05
            factors += 1
        
        # Return on equity
        if "return_on_equity" in analysis:
            roe = analysis["return_on_equity"]
            if roe > 0.15:
                score += 0.25
            elif roe > 0.10:
                score += 0.15
            elif roe > 0.05:
                score += 0.05
            factors += 1
        
        return score / max(factors, 1)
    
    def _calculate_forecast_score(self, forecast: Dict[str, Any]) -> float:
        """Calculate market forecast score (0-1)"""
        score = 0.0
        factors = 0
        
        # Market outlook
        outlook = forecast.get("market_outlook", "").lower()
        if outlook == "bullish":
            score += 0.4
        elif outlook == "neutral":
            score += 0.2
        factors += 1
        
        # Revenue growth forecast
        if "revenue_growth" in forecast:
            growth = forecast["revenue_growth"]
            if isinstance(growth, str) and "%" in growth:
                try:
                    growth_val = float(growth.replace("%", "")) / 100
                    if growth_val > 0.10:
                        score += 0.3
                    elif growth_val > 0.05:
                        score += 0.2
                    elif growth_val > 0:
                        score += 0.1
                except:
                    pass
            factors += 1
        
        # Confidence level
        confidence = forecast.get("confidence", 0.5)
        score += confidence * 0.3
        factors += 1
        
        return score / max(factors, 1)
    
    def _calculate_sentiment_score(self, sentiment: Dict[str, Any]) -> float:
        """Calculate sentiment score (0-1)"""
        score = 0.0
        factors = 0
        
        # Overall sentiment
        overall_sentiment = sentiment.get("overall_sentiment", "").lower()
        if overall_sentiment == "positive":
            score += 0.5
        elif overall_sentiment == "neutral":
            score += 0.25
        factors += 1
        
        # Sentiment score
        sentiment_score = sentiment.get("sentiment_score", 0)
        score += (sentiment_score + 1) * 0.25  # Convert -1 to 1 range to 0 to 1
        factors += 1
        
        return score / max(factors, 1)
    
    def _determine_action(self, decision_score: float, risk_profile: str) -> Tuple[str, float]:
        """Determine buy/sell/hold action based on score and risk profile"""
        thresholds = self.risk_thresholds.get(risk_profile, self.risk_thresholds["moderate"])
        min_confidence = thresholds["min_confidence"]
        
        if decision_score > 0.7 and decision_score >= min_confidence:
            return "BUY", decision_score
        elif decision_score < 0.3 and decision_score >= min_confidence:
            return "SELL", decision_score
        else:
            return "HOLD", decision_score
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current stock price"""
        try:
            if ticker:
                stock = yf.Ticker(ticker)
                info = stock.info
                return info.get("regularMarketPrice", 100.0)
        except:
            pass
        return 100.0  # Default price
    
    def _calculate_price_targets(self, current_price: float, action: str, 
                               confidence: float, forecast: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate target price, stop loss, and take profit levels"""
        
        # Base volatility assumption
        volatility = 0.15  # 15% volatility
        
        if action == "BUY":
            # Target price: current + expected growth
            growth_factor = forecast.get("expected_growth", 0.10)
            target_price = current_price * (1 + growth_factor)
            
            # Stop loss: current - volatility
            stop_loss = current_price * (1 - volatility)
            
            # Take profit: target + additional upside
            take_profit = target_price * (1 + volatility)
            
        elif action == "SELL":
            # Target price: current - expected decline
            decline_factor = forecast.get("expected_decline", 0.10)
            target_price = current_price * (1 - decline_factor)
            
            # Stop loss: current + volatility
            stop_loss = current_price * (1 + volatility)
            
            # Take profit: target - additional downside
            take_profit = target_price * (1 - volatility)
            
        else:  # HOLD
            target_price = current_price
            stop_loss = current_price * (1 - volatility * 0.5)
            take_profit = current_price * (1 + volatility * 0.5)
        
        return round(target_price, 2), round(stop_loss, 2), round(take_profit, 2)
    
    def _generate_reasoning(self, financial_analysis: Dict[str, Any], 
                          market_forecast: Dict[str, Any], 
                          sentiment_analysis: Dict[str, Any], 
                          action: str) -> str:
        """Generate detailed reasoning for the investment decision"""
        
        reasoning_parts = []
        
        # Financial analysis reasoning
        if financial_analysis:
            if "revenue_growth" in financial_analysis:
                growth = financial_analysis["revenue_growth"]
                reasoning_parts.append(f"Revenue growth: {growth:.1%}")
            
            if "profit_margin" in financial_analysis:
                margin = financial_analysis["profit_margin"]
                reasoning_parts.append(f"Profit margin: {margin:.1%}")
        
        # Market forecast reasoning
        if market_forecast:
            outlook = market_forecast.get("market_outlook", "")
            reasoning_parts.append(f"Market outlook: {outlook}")
            
            confidence = market_forecast.get("confidence", 0)
            reasoning_parts.append(f"Forecast confidence: {confidence:.1%}")
        
        # Sentiment reasoning
        if sentiment_analysis:
            sentiment = sentiment_analysis.get("overall_sentiment", "")
            reasoning_parts.append(f"Market sentiment: {sentiment}")
        
        # Action-specific reasoning
        if action == "BUY":
            reasoning_parts.append("Strong fundamentals and positive outlook suggest upside potential")
        elif action == "SELL":
            reasoning_parts.append("Weak fundamentals and negative outlook suggest downside risk")
        else:
            reasoning_parts.append("Mixed signals suggest maintaining current position")
        
        return ". ".join(reasoning_parts)
    
    def _identify_key_factors(self, financial_analysis: Dict[str, Any], 
                            market_forecast: Dict[str, Any], 
                            sentiment_analysis: Dict[str, Any]) -> List[str]:
        """Identify key factors influencing the decision"""
        factors = []
        
        # Financial factors
        if financial_analysis.get("revenue_growth", 0) > 0.1:
            factors.append("Strong revenue growth")
        if financial_analysis.get("profit_margin", 0) > 0.15:
            factors.append("High profit margins")
        if financial_analysis.get("debt_to_equity", 1) < 0.5:
            factors.append("Low debt levels")
        
        # Market factors
        if market_forecast.get("market_outlook") == "bullish":
            factors.append("Bullish market outlook")
        if market_forecast.get("confidence", 0) > 0.7:
            factors.append("High forecast confidence")
        
        # Sentiment factors
        if sentiment_analysis.get("overall_sentiment") == "positive":
            factors.append("Positive market sentiment")
        
        return factors[:5]  # Top 5 factors
    
    def _assess_risk_level(self, decision_score: float) -> str:
        """Assess risk level based on decision score"""
        if decision_score > 0.7:
            return "LOW"
        elif decision_score > 0.4:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _determine_time_horizon(self, market_forecast: Dict[str, Any]) -> str:
        """Determine investment time horizon"""
        # Default to medium term
        return "MEDIUM"

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def optimize_portfolio(self, investment_signals: List[InvestmentSignal], 
                         risk_profile: str = "moderate") -> Dict[str, Any]:
        """Optimize portfolio allocation based on investment signals"""
        
        # Filter signals by confidence and risk profile
        valid_signals = self._filter_signals(investment_signals, risk_profile)
        
        if not valid_signals:
            return {"error": "No valid investment signals found"}
        
        # Calculate optimal weights using risk-adjusted returns
        weights = self._calculate_optimal_weights(valid_signals)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(valid_signals, weights)
        
        return {
            "allocations": weights,
            "metrics": portfolio_metrics,
            "signals": valid_signals
        }
    
    def _filter_signals(self, signals: List[InvestmentSignal], 
                       risk_profile: str) -> List[InvestmentSignal]:
        """Filter signals based on risk profile and confidence"""
        min_confidence = {
            "conservative": 0.8,
            "moderate": 0.7,
            "aggressive": 0.6
        }.get(risk_profile, 0.7)
        
        return [s for s in signals if s.confidence >= min_confidence]
    
    def _calculate_optimal_weights(self, signals: List[InvestmentSignal]) -> Dict[str, float]:
        """Calculate optimal portfolio weights using risk-adjusted returns"""
        
        # Calculate risk-adjusted scores
        scores = []
        for signal in signals:
            # Risk-adjusted score = confidence * (1 - risk_factor)
            risk_factor = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}.get(signal.risk_level, 0.5)
            score = signal.confidence * (1 - risk_factor)
            scores.append(score)
        
        # Normalize weights
        total_score = sum(scores)
        if total_score == 0:
            # Equal weights if no valid scores
            weight = 1.0 / len(signals)
            return {f"signal_{i}": weight for i in range(len(signals))}
        
        weights = {}
        for i, signal in enumerate(signals):
            weights[f"signal_{i}"] = scores[i] / total_score
        
        return weights
    
    def _calculate_portfolio_metrics(self, signals: List[InvestmentSignal], 
                                   weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        
        # Expected return
        expected_return = sum(
            weights[f"signal_{i}"] * signals[i].confidence 
            for i in range(len(signals))
        )
        
        # Portfolio risk (weighted average of individual risks)
        risk_levels = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}
        portfolio_risk = sum(
            weights[f"signal_{i}"] * risk_levels.get(signals[i].risk_level, 0.5)
            for i in range(len(signals))
        )
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / max(portfolio_risk, 0.01)
        
        return {
            "expected_return": expected_return,
            "portfolio_risk": portfolio_risk,
            "sharpe_ratio": sharpe_ratio,
            "num_positions": len(signals)
        } 