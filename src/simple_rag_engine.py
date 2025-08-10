import os
import json
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
import re
from difflib import SequenceMatcher

class SimpleRAGEngine:
    def __init__(self, data_dir: str = "data", demo_mode: bool = False):
        self.data_dir = data_dir
        self.demo_mode = demo_mode
        
        # Initialize paths
        self.questions_file = os.path.join(data_dir, "financebench_open_source.jsonl")
        self.docs_file = os.path.join(data_dir, "financebench_document_information.jsonl")
        
        # Load data and initialize
        self._load_data()
        self._initialize_components()
    
    def _load_data(self):
        """Load FinanceBench data"""
        try:
            if os.path.exists(self.questions_file) and os.path.exists(self.docs_file):
                self.questions = pd.read_json(self.questions_file, lines=True)
                self.docs_meta = pd.read_json(self.docs_file, lines=True)
                
                # Merge to get doc metadata in questions
                self.questions = self.questions.merge(self.docs_meta, on="doc_name", how="left")
                print(f"✅ Loaded {len(self.questions)} questions and {len(self.docs_meta)} documents")
            else:
                print("⚠️ FinanceBench data files not found, using demo mode")
                self.demo_mode = True
                self.questions = pd.DataFrame()
                self.docs_meta = pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.demo_mode = True
            self.questions = pd.DataFrame()
            self.docs_meta = pd.DataFrame()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            # Initialize sentiment analyzer (simplified)
            self.sentiment_analyzer = self._create_simple_sentiment_analyzer()
            print("✅ Simple sentiment analyzer initialized")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            self.demo_mode = True
    
    def _create_simple_sentiment_analyzer(self):
        """Create a simple sentiment analyzer"""
        def analyze_sentiment(text):
            positive_words = ['positive', 'growth', 'increase', 'profit', 'success', 'strong', 'good', 'up', 'rise']
            negative_words = ['negative', 'decline', 'loss', 'decrease', 'weak', 'bad', 'down', 'fall', 'risk']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "positive", 0.7
            elif negative_count > positive_count:
                return "negative", 0.7
            else:
                return "neutral", 0.5
        
        return analyze_sentiment
    
    def get_exact_match(self, user_query: str, threshold: float = 0.8) -> Tuple[Optional[str], Optional[str], float]:
        """Check for exact question matches using string similarity"""
        if self.demo_mode or self.questions.empty:
            return None, None, 0.0
        
        try:
            best_match = None
            best_score = 0.0
            best_answer = None
            
            for _, row in self.questions.iterrows():
                question_text = row["question"]
                gold_answer = row["answer"]
                
                # Calculate similarity using SequenceMatcher
                similarity = SequenceMatcher(None, user_query.lower(), question_text.lower()).ratio()
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = question_text
                    best_answer = gold_answer
            
            return best_answer, best_match, best_score
        except Exception as e:
            print(f"Error in exact match: {e}")
            return None, None, 0.0
    
    def search_evidence(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search through evidence snippets for relevant information"""
        if self.demo_mode or self.questions.empty:
            return []
        
        try:
            relevant_evidence = []
            
            for _, row in self.questions.iterrows():
                evidence_list = row.get("evidence", [])
                
                for evidence in evidence_list:
                    snippet = evidence.get("evidence_text", "").strip()
                    if not snippet:
                        continue
                    
                    # Simple keyword matching
                    query_words = set(user_query.lower().split())
                    snippet_words = set(snippet.lower().split())
                    
                    # Calculate overlap
                    overlap = len(query_words.intersection(snippet_words))
                    if overlap > 0:
                        relevance_score = overlap / len(query_words)
                        
                        relevant_evidence.append({
                            "text": snippet,
                            "relevance": relevance_score,
                            "metadata": {
                                "company": row.get("company_x", ""),
                                "doc_name": row.get("doc_name", ""),
                                "question_type": row.get("question_type", "")
                            }
                        })
            
            # Sort by relevance and return top_k
            relevant_evidence.sort(key=lambda x: x["relevance"], reverse=True)
            return relevant_evidence[:top_k]
            
        except Exception as e:
            print(f"Error in evidence search: {e}")
            return []
    
    def fetch_company_news(self, company_name: str) -> List[str]:
        """Fetch company news from multiple sources"""
        if self.demo_mode:
            return self._get_demo_news(company_name)
        
        sources = [
            lambda: self._fetch_from_google_news_rss(company_name),
            lambda: self._fetch_from_yahoo(company_name),
            lambda: self._fetch_from_newsapi(company_name),
        ]
        
        for source in sources:
            try:
                headlines = source()
                if headlines:
                    return headlines[:5]
            except Exception as e:
                print(f"News source error: {e}")
                continue
        
        return []
    
    def _fetch_from_google_news_rss(self, company_name: str, max_results: int = 5) -> List[str]:
        """Fetch news from Google News RSS"""
        try:
            query = company_name.replace(' ', '+')
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            
            headlines = []
            for item in items[:max_results]:
                title = item.find("title")
                if title is not None and title.text:
                    headlines.append(title.text)
            return headlines
        except Exception as e:
            print(f"Google News RSS error: {e}")
            return []
    
    def _fetch_from_yahoo(self, company_name: str) -> List[str]:
        """Fetch news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(company_name)
            news = ticker.news
            if news:
                return [n.get("title", "") for n in news if n.get("title")]
        except Exception as e:
            print(f"Yahoo Finance error: {e}")
        return []
    
    def _fetch_from_newsapi(self, company_name: str, api_key: str = "b18d6a7fa41a409eb23a5ec4b657864d") -> List[str]:
        """Fetch news from NewsAPI"""
        try:
            url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&language=en&apiKey={api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return [a["title"] for a in resp.json().get("articles", []) if "title" in a]
        except Exception as e:
            print(f"NewsAPI error: {e}")
        return []
    
    def _get_demo_news(self, company_name: str) -> List[str]:
        """Get demo news for testing"""
        demo_news = {
            "3M": [
                "3M Reports Strong Q3 Earnings Growth",
                "3M Announces New Sustainability Initiative",
                "3M Expands Global Manufacturing Operations",
                "3M Launches Innovative Product Line",
                "3M Receives Industry Recognition Award"
            ],
            "Amazon": [
                "Amazon Reports Record Holiday Sales",
                "Amazon Expands Cloud Services Portfolio",
                "Amazon Announces New AI Initiatives",
                "Amazon Strengthens Supply Chain Network",
                "Amazon Launches New Marketplace Features"
            ],
            "Apple": [
                "Apple Reports Strong iPhone Sales",
                "Apple Announces New Product Line",
                "Apple Expands Services Business",
                "Apple Invests in Renewable Energy",
                "Apple Launches New Developer Tools"
            ]
        }
        return demo_news.get(company_name, [
            f"{company_name} Reports Positive Q3 Results",
            f"{company_name} Announces Strategic Partnership",
            f"{company_name} Expands Market Presence",
            f"{company_name} Launches New Products",
            f"{company_name} Receives Industry Awards"
        ])
    
    def analyze_company_sentiment(self, company_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Analyze company sentiment from news headlines"""
        headlines = self.fetch_company_news(company_name)
        if not headlines:
            return "No recent news found", []
        
        try:
            sentiments = []
            total_sentiment_score = 0
            
            for headline in headlines:
                sentiment, score = self.sentiment_analyzer(headline)
                sentiments.append({
                    "headline": headline,
                    "sentiment": sentiment,
                    "score": score
                })
                total_sentiment_score += score if sentiment == "positive" else (-score if sentiment == "negative" else 0)
            
            avg_score = total_sentiment_score / len(sentiments) if sentiments else 0
            overall = "Positive" if avg_score > 0.1 else "Negative" if avg_score < -0.1 else "Neutral"
            
            return overall, sentiments
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return "Analysis failed", []
    
    def answer_query_with_sentiment(self, user_query: str) -> Dict[str, Any]:
        """Answer query using simple RAG with sentiment analysis"""
        try:
            # Get exact match
            gold_answer, matched_question, score = self.get_exact_match(user_query)
            
            if gold_answer:
                rag_answer = gold_answer
                answer_source = "exact_match"
                confidence = score
            else:
                # Search evidence
                evidence_results = self.search_evidence(user_query)
                
                if evidence_results:
                    # Combine evidence for answer
                    evidence_texts = [e["text"] for e in evidence_results[:2]]
                    rag_answer = f"Based on the available evidence: {' '.join(evidence_texts[:200])}..."
                    answer_source = "evidence_search"
                    confidence = evidence_results[0]["relevance"]
                else:
                    rag_answer = "I don't have enough information to answer this question accurately."
                    answer_source = "no_data"
                    confidence = 0.0
            
            # Extract company name from query
            possible_companies = self.questions["company_x"].dropna().unique() if not self.questions.empty else []
            company_in_query = next((c for c in possible_companies if c.lower() in user_query.lower()), None)
            
            # Analyze sentiment if company found
            if company_in_query:
                sentiment, details = self.analyze_company_sentiment(company_in_query)
                return {
                    "answer": rag_answer,
                    "company": company_in_query,
                    "sentiment": sentiment,
                    "sentiment_details": details,
                    "answer_source": answer_source,
                    "confidence": confidence,
                    "matched_question": matched_question
                }
            else:
                return {
                    "answer": rag_answer,
                    "company": None,
                    "sentiment": None,
                    "sentiment_details": [],
                    "answer_source": answer_source,
                    "confidence": confidence,
                    "matched_question": matched_question
                }
                
        except Exception as e:
            print(f"Error in query answering: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your query.",
                "company": None,
                "sentiment": None,
                "sentiment_details": [],
                "answer_source": "error",
                "confidence": 0.0,
                "matched_question": None
            }
    
    def get_available_companies(self) -> List[str]:
        """Get list of available companies"""
        if self.questions.empty:
            return ["3M", "Amazon", "Apple", "Microsoft", "Google"]
        return self.questions["company_x"].dropna().unique().tolist()
    
    def get_sample_questions(self, company: str = None) -> List[str]:
        """Get sample questions for testing"""
        if self.questions.empty:
            return [
                "What is the revenue for the latest quarter?",
                "What are the key financial metrics?",
                "What is the company's debt level?",
                "What are the main risks mentioned?",
                "What is the growth strategy?"
            ]
        
        if company:
            company_questions = self.questions[self.questions["company_x"] == company]
            return company_questions["question"].head(5).tolist()
        else:
            return self.questions["question"].head(10).tolist()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        return {
            "demo_mode": self.demo_mode,
            "questions_loaded": len(self.questions),
            "documents_loaded": len(self.docs_meta),
            "vector_stores_ready": False,  # Using simple search instead
            "embedding_model_ready": False,  # Using simple search instead
            "sentiment_analyzer_ready": self.sentiment_analyzer is not None,
            "available_companies": self.get_available_companies(),
            "last_updated": datetime.now().isoformat()
        } 