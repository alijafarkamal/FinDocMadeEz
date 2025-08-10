import openai
from openai import OpenAI
from typing import Dict, List, Any
import json
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class FinancialAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            self.llm = True
        else:
            self.client = None
            self.llm = None
    
    def analyze_financial_document(self, text: str, company: str, year: str) -> Dict[str, Any]:
        if not self.llm:
            return self._mock_analysis(company, year)
        
        system_prompt = """You are a financial analyst expert. Analyze the provided financial document and extract key insights. Focus on:
1. Financial performance metrics
2. Key trends and changes
3. Risk factors
4. Growth opportunities
5. Investment implications

Provide a structured analysis with specific numbers and insights."""
        
        user_prompt = f"""Analyze this financial document for {company} ({year}):

{text[:8000]}

Provide a comprehensive financial analysis including:
- Key financial metrics (revenue, profit, debt, etc.)
- Performance trends
- Risk assessment
- Investment outlook
- Key insights and recommendations"""
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://findocgpt.com",
                    "X-Title": "FinDocGPT",
                },
                extra_body={},
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            return self._parse_analysis_response(completion.choices[0].message.content, company, year)
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return self._mock_analysis(company, year)
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        if not self.llm:
            return self._extract_metrics_from_text(text)
        
        prompt = f"""Extract key financial metrics from this financial document text. Look for specific numbers and calculations.

Return ONLY a valid JSON object with this exact structure (use null for missing values):
{{
    "revenue": number or null,
    "net_income": number or null,
    "total_assets": number or null,
    "total_liabilities": number or null,
    "debt": number or null,
    "equity": number or null,
    "cash_flow": number or null,
    "profit_margin": number or null,
    "debt_to_equity": number or null,
    "return_on_equity": number or null
}}

Important: Extract actual numbers from the text. Look for:
- Revenue/Sales/Net sales
- Net income/Net earnings/Profit
- Total assets
- Total liabilities
- Debt/Long-term debt
- Equity/Shareholders equity
- Cash flow from operations
- Calculate profit_margin = net_income/revenue if both exist
- Calculate debt_to_equity = debt/equity if both exist
- Calculate return_on_equity = net_income/equity if both exist

Document text: {text[:6000]}"""
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://findocgpt.com",
                    "X-Title": "FinDocGPT",
                },
                extra_body={},
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            response_text = completion.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            try:
                metrics = json.loads(response_text)
                # Validate that we got some real data
                if any(v is not None and v != 0 for v in metrics.values()):
                    return metrics
                else:
                    return self._extract_metrics_from_text(text)
            except json.JSONDecodeError:
                return self._extract_metrics_from_text(text)
                
        except Exception as e:
            print(f"API call failed: {e}")
            return self._extract_metrics_from_text(text)
    
    def _extract_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """Fallback method to extract metrics using regex patterns"""
        import re
        
        metrics = {
            "revenue": None,
            "net_income": None,
            "total_assets": None,
            "total_liabilities": None,
            "debt": None,
            "equity": None,
            "cash_flow": None,
            "profit_margin": None,
            "debt_to_equity": None,
            "return_on_equity": None
        }
        
        # Extract numbers with currency symbols and commas
        amount_pattern = r'\$?([\d,]+\.?\d*)'
        
        # Look for revenue/sales
        revenue_patterns = [
            r'revenue[:\s]*\$?([\d,]+\.?\d*)',
            r'net sales[:\s]*\$?([\d,]+\.?\d*)',
            r'total revenue[:\s]*\$?([\d,]+\.?\d*)',
            r'sales[:\s]*\$?([\d,]+\.?\d*)',
            r'net revenues[:\s]*\$?([\d,]+\.?\d*)',
            r'total net revenues[:\s]*\$?([\d,]+\.?\d*)',
            r'consolidated revenue[:\s]*\$?([\d,]+\.?\d*)',
            r'operating revenue[:\s]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics["revenue"] = float(match.group(1).replace(',', ''))
                    break
                except:
                    pass
        
        # Look for net income
        income_patterns = [
            r'net income[:\s]*\$?([\d,]+\.?\d*)',
            r'net earnings[:\s]*\$?([\d,]+\.?\d*)',
            r'profit[:\s]*\$?([\d,]+\.?\d*)',
            r'net profit[:\s]*\$?([\d,]+\.?\d*)',
            r'income[:\s]*\$?([\d,]+\.?\d*)',
            r'earnings[:\s]*\$?([\d,]+\.?\d*)',
            r'consolidated net income[:\s]*\$?([\d,]+\.?\d*)',
            r'net income attributable[:\s]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics["net_income"] = float(match.group(1).replace(',', ''))
                    break
                except:
                    pass
        
        # Look for total assets
        assets_patterns = [
            r'total assets[:\s]*\$?([\d,]+\.?\d*)',
            r'assets[:\s]*\$?([\d,]+\.?\d*)',
            r'consolidated assets[:\s]*\$?([\d,]+\.?\d*)',
            r'total consolidated assets[:\s]*\$?([\d,]+\.?\d*)',
            r'property, plant and equipment[:\s]*\$?([\d,]+\.?\d*)',
            r'pp&e[:\s]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in assets_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics["total_assets"] = float(match.group(1).replace(',', ''))
                    break
                except:
                    pass
        
        # Look for total liabilities
        liabilities_patterns = [
            r'total liabilities[:\s]*\$?([\d,]+\.?\d*)',
            r'liabilities[:\s]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in liabilities_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics["total_liabilities"] = float(match.group(1).replace(',', ''))
                    break
                except:
                    pass
        
        # Look for debt
        debt_patterns = [
            r'debt[:\s]*\$?([\d,]+\.?\d*)',
            r'long-term debt[:\s]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in debt_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics["debt"] = float(match.group(1).replace(',', ''))
                    break
                except:
                    pass
        
        # Look for equity
        equity_patterns = [
            r'equity[:\s]*\$?([\d,]+\.?\d*)',
            r'shareholders equity[:\s]*\$?([\d,]+\.?\d*)',
            r'stockholders equity[:\s]*\$?([\d,]+\.?\d*)',
            r'total equity[:\s]*\$?([\d,]+\.?\d*)',
            r'consolidated equity[:\s]*\$?([\d,]+\.?\d*)',
            r'retained earnings[:\s]*\$?([\d,]+\.?\d*)',
            r'common stock[:\s]*\$?([\d,]+\.?\d*)',
            r'paid-in capital[:\s]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in equity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    metrics["equity"] = float(match.group(1).replace(',', ''))
                    break
                except:
                    pass
        
        # Calculate derived metrics
        if metrics["net_income"] and metrics["revenue"] and metrics["revenue"] != 0:
            metrics["profit_margin"] = metrics["net_income"] / metrics["revenue"]
        
        if metrics["debt"] and metrics["equity"] and metrics["equity"] != 0:
            metrics["debt_to_equity"] = metrics["debt"] / metrics["equity"]
        
        if metrics["net_income"] and metrics["equity"] and metrics["equity"] != 0:
            metrics["return_on_equity"] = metrics["net_income"] / metrics["equity"]
        
        return metrics
    
    def _extract_metrics_with_debug(self, text: str) -> Dict[str, Any]:
        """Extract metrics and return debug information"""
        metrics = self._extract_metrics_from_text(text)
        
        # Find what patterns were matched
        import re
        debug_info = {
            "revenue_found": False,
            "income_found": False,
            "assets_found": False,
            "liabilities_found": False,
            "debt_found": False,
            "equity_found": False,
            "patterns_matched": []
        }
        
        # Check what was found
        if metrics["revenue"]:
            debug_info["revenue_found"] = True
            debug_info["patterns_matched"].append("revenue")
        if metrics["net_income"]:
            debug_info["income_found"] = True
            debug_info["patterns_matched"].append("net_income")
        if metrics["total_assets"]:
            debug_info["assets_found"] = True
            debug_info["patterns_matched"].append("total_assets")
        if metrics["total_liabilities"]:
            debug_info["liabilities_found"] = True
            debug_info["patterns_matched"].append("total_liabilities")
        if metrics["debt"]:
            debug_info["debt_found"] = True
            debug_info["patterns_matched"].append("debt")
        if metrics["equity"]:
            debug_info["equity_found"] = True
            debug_info["patterns_matched"].append("equity")
        
        return {"metrics": metrics, "debug": debug_info}
    
    def generate_investment_recommendation(self, analysis: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return self._mock_recommendation()
        
        prompt = f"""Based on this financial analysis and metrics, provide an investment recommendation:

Analysis: {json.dumps(analysis, indent=2)}
Metrics: {json.dumps(metrics, indent=2)}

Provide a recommendation in JSON format:
{{
    "recommendation": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "risk_level": "LOW/MEDIUM/HIGH",
    "time_horizon": "SHORT/MEDIUM/LONG",
    "key_factors": ["factor1", "factor2", "factor3"]
}}"""
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://findocgpt.com",
                    "X-Title": "FinDocGPT",
                },
                extra_body={},
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return json.loads(completion.choices[0].message.content)
        except:
            return self._mock_recommendation()
    
    def predict_market_trends(self, company_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.llm:
            return self._mock_trends()
        
        prompt = f"""Based on this historical company data, predict market trends:

{json.dumps(company_data, indent=2)}

Provide predictions in JSON format:
{{
    "revenue_growth": "predicted percentage",
    "profit_trend": "increasing/decreasing/stable",
    "market_outlook": "bullish/bearish/neutral",
    "key_drivers": ["driver1", "driver2"],
    "risks": ["risk1", "risk2"],
    "confidence": 0.0-1.0
}}"""
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://findocgpt.com",
                    "X-Title": "FinDocGPT",
                },
                extra_body={},
                model="deepseek/deepseek-chat-v3-0324:4k",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return json.loads(completion.choices[0].message.content)
        except:
            return self._mock_trends()
    
    def _parse_analysis_response(self, response: str, company: str, year: str) -> Dict[str, Any]:
        return {
            "company": company,
            "year": year,
            "analysis": response,
            "summary": response[:500] + "..." if len(response) > 500 else response,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def _mock_analysis(self, company: str, year: str) -> Dict[str, Any]:
        return {
            "company": company,
            "year": year,
            "analysis": f"Financial analysis for {company} in {year}. This analysis provides insights into the company's financial performance, key metrics, and investment outlook.",
            "summary": f"Financial summary for {company} {year}",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def _mock_metrics(self) -> Dict[str, Any]:
        return {
            "revenue": 1000000,
            "net_income": 150000,
            "total_assets": 2000000,
            "total_liabilities": 800000,
            "debt": 500000,
            "equity": 1200000,
            "cash_flow": 200000,
            "profit_margin": 0.15,
            "debt_to_equity": 0.42,
            "return_on_equity": 0.125
        }
    
    def _mock_recommendation(self) -> Dict[str, Any]:
        return {
            "recommendation": "HOLD",
            "confidence": 0.7,
            "reasoning": "Mock investment reasoning based on financial analysis",
            "risk_level": "MEDIUM",
            "time_horizon": "MEDIUM",
            "key_factors": ["Financial performance", "Market conditions", "Industry trends"]
        }
    
    def _mock_trends(self) -> Dict[str, Any]:
        return {
            "revenue_growth": "5-10%",
            "profit_trend": "increasing",
            "market_outlook": "bullish",
            "key_drivers": ["Market expansion", "Product innovation"],
            "risks": ["Economic uncertainty", "Competition"],
            "confidence": 0.75
        } 