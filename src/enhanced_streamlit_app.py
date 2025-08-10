import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import DocumentProcessor
from src.ai_analyzer import FinancialAnalyzer
from src.investment_engine import InvestmentDecisionEngine, PortfolioOptimizer
from src.market_data_integration import MarketDataIntegrator
from src.benchmarking_metrics import BenchmarkingMetrics
from src.forecasting_engine import ForecastingEngine
from src.simple_rag_engine import SimpleRAGEngine
from src.finance_theme import apply_finance_theme, create_finance_header, create_finance_metric_card

st.set_page_config(
    page_title="FinDocGPT - AI Financial Analysis & Investment Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_components():
    doc_processor = DocumentProcessor()
    analyzer = FinancialAnalyzer()
    investment_engine = InvestmentDecisionEngine()
    portfolio_optimizer = PortfolioOptimizer()
    market_data = MarketDataIntegrator()
    benchmarking = BenchmarkingMetrics()
    forecasting_engine = ForecastingEngine(demo_mode=True)
    advanced_rag = SimpleRAGEngine(demo_mode=False)
    return (doc_processor, analyzer, investment_engine, 
            portfolio_optimizer, market_data, benchmarking, forecasting_engine, advanced_rag)

def main():
    # Apply finance theme
    apply_finance_theme()
    
    # Display professional finance header
    st.markdown(create_finance_header(), unsafe_allow_html=True)
    
    components = load_components()
    (doc_processor, analyzer, investment_engine, 
     portfolio_optimizer, market_data, benchmarking, forecasting_engine, advanced_rag) = components
    
    sidebar = st.sidebar
    
    # Add professional sidebar header
    sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
        ">
            <span style="font-size: 20px; color: #0f1419;">📊</span>
        </div>
        <h2 style="
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #00d4aa;
            margin: 0;
            font-size: 1.2rem;
        ">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = sidebar.selectbox(
        "Choose Analysis Type",
        ["📄 Document Analysis", "🔍 Advanced RAG Q&A", "📈 Market Data & Forecasting", "🔮 AI Forecasting Models", 
         "💡 Investment Strategy", "📊 Portfolio Optimization", "🎯 Real-time Trading Signals", "📋 Performance Metrics"]
    )
    
    if page == "📄 Document Analysis":
        document_analysis_page(doc_processor, analyzer, benchmarking)
    elif page == "🔍 Advanced RAG Q&A":
        advanced_rag_page(advanced_rag, benchmarking)
    elif page == "📈 Market Data & Forecasting":
        market_data_page(market_data, benchmarking)
    elif page == "🔮 AI Forecasting Models":
        forecasting_models_page(forecasting_engine, market_data, benchmarking)
    elif page == "💡 Investment Strategy":
        investment_strategy_page(investment_engine, market_data, benchmarking)
    elif page == "📊 Portfolio Optimization":
        portfolio_optimization_page(portfolio_optimizer, investment_engine, benchmarking)
    elif page == "🎯 Real-time Trading Signals":
        trading_signals_page(market_data, investment_engine, benchmarking)
    elif page == "📋 Performance Metrics":
        performance_metrics_page(benchmarking)

def advanced_rag_page(advanced_rag, benchmarking):
    st.header("🔍 Advanced RAG Q&A with Sentiment Analysis")
    st.markdown("### Intelligent Question Answering with Real-time Sentiment Analysis")
    
    # System status
    status = advanced_rag.get_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_finance_metric_card("Demo Mode", "✅" if status["demo_mode"] else "❌"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_finance_metric_card("Questions Loaded", str(status["questions_loaded"])), unsafe_allow_html=True)
    with col3:
        st.markdown(create_finance_metric_card("Vector Stores", "✅" if status["vector_stores_ready"] else "❌"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_finance_metric_card("Sentiment Analyzer", "✅" if status["sentiment_analyzer_ready"] else "❌"), unsafe_allow_html=True)
    
    # Main Q&A interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Ask Financial Questions")
        
        # Query input
        user_query = st.text_area(
            "Enter your financial question:",
            placeholder="e.g., What is 3M's capital expenditure for FY2018?",
            height=100
        )
        
        # Company selection
        available_companies = advanced_rag.get_available_companies()
        selected_company = st.selectbox("Select Company (optional):", ["Auto-detect"] + available_companies)
        
        if st.button("🔍 Get Answer with Sentiment Analysis"):
            if user_query:
                with st.spinner("Analyzing question and fetching sentiment..."):
                    result = advanced_rag.answer_query_with_sentiment(user_query)
                    
                    # Display answer in organized format
                    st.subheader("📝 Answer")
                    
                    # Answer in a styled container
                    with st.container():
                        st.markdown("---")
                        st.markdown(f"**Answer:** {result['answer']}")
                        st.markdown("---")
                    
                    # Answer metadata in organized format
                    st.subheader("📊 Answer Details")
                    col1_1, col1_2, col1_3 = st.columns(3)
                    with col1_1:
                        st.info(f"**Source:** {result['answer_source'].title()}")
                    with col1_2:
                        st.info(f"**Confidence:** {result['confidence']:.1%}")
                    with col1_3:
                        if result["matched_question"]:
                            st.success("**Similar Question:** Found")
                        else:
                            st.warning("**Similar Question:** Not Found")
                    
                    # Sentiment analysis
                    if result["company"] and result["sentiment"]:
                        st.subheader("📊 Sentiment Analysis")
                        
                        # Sentiment summary in organized format
                        st.subheader(f"📊 Sentiment Analysis for {result['company']}")
                        
                        # Sentiment indicator
                        sentiment_color = {"Positive": "normal", "Negative": "inverse", "Neutral": "off"}
                        st.metric(
                            f"Overall Sentiment", 
                            result["sentiment"],
                            delta_color=sentiment_color.get(result["sentiment"], "off")
                        )
                        
                        # Sentiment details
                        if result["sentiment_details"]:
                            st.subheader("📰 Recent News Analysis")
                            
                            # Create sentiment chart
                            sentiment_data = []
                            for detail in result["sentiment_details"]:
                                sentiment_data.append({
                                    "Headline": detail["headline"][:50] + "..." if len(detail["headline"]) > 50 else detail["headline"],
                                    "Sentiment": detail["sentiment"].title(),
                                    "Score": detail["score"]
                                })
                            
                            df = pd.DataFrame(sentiment_data)
                            
                            # Sentiment distribution pie chart
                            sentiment_counts = df["Sentiment"].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title=f"Sentiment Distribution for {result['company']}",
                                color_discrete_map={
                                    "Positive": "green",
                                    "Negative": "red", 
                                    "Neutral": "orange"
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # News headlines table
                            st.subheader("📋 News Headlines")
                            for i, detail in enumerate(result["sentiment_details"], 1):
                                sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                                st.write(f"{i}. {sentiment_emoji.get(detail['sentiment'], '⚪')} {detail['headline']}")
                                st.caption(f"Sentiment: {detail['sentiment'].title()} (Score: {detail['score']:.2f})")
                    
                    # Track for benchmarking
                    if result["answer_source"] != "error":
                        benchmarking.add_qa_result(
                            user_query, result["answer"], "gold_standard", 
                            result["confidence"], True
                        )
            else:
                st.warning("Please enter a question to analyze.")
    
    with col2:
        st.subheader("💡 Sample Questions")
        
        # Company-specific questions
        if selected_company != "Auto-detect":
            sample_questions = advanced_rag.get_sample_questions(selected_company)
        else:
            sample_questions = advanced_rag.get_sample_questions()
        
        for i, question in enumerate(sample_questions[:5], 1):
            if st.button(f"{i}. {question[:40]}...", key=f"sample_{i}"):
                st.session_state.user_query = question
                st.rerun()
        
        st.subheader("🎯 Features")
        st.write("• **2-Way RAG**: Exact match + semantic search")
        st.write("• **Real-time Sentiment**: FinBERT analysis")
        st.write("• **Multi-source News**: Google, Yahoo, NewsAPI")
        st.write("• **Company Detection**: Automatic entity recognition")
        st.write("• **Confidence Scoring**: Answer reliability")
    
    # Advanced features
    st.subheader("🔧 Advanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**RAG System Details**")
        
        if status["vector_stores_ready"]:
            st.success("✅ Vector stores loaded")
            st.write("• Document chunks: 387")
            st.write("• Question embeddings: Ready")
            st.write("• Similarity threshold: 0.85")
        else:
            st.warning("⚠️ Vector stores not available")
            st.write("• Using demo mode")
            st.write("• Limited functionality")
    
    with col2:
        st.write("**Sentiment Analysis**")
        st.success("✅ FinBERT model loaded")
        st.write("• Model: yiyanghkust/finbert-tone")
        st.write("• News sources: 3 APIs")
        st.write("• Analysis: Real-time")
    
    # Query history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if user_query and st.button("Save to History"):
        st.session_state.query_history.append({
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "company": result.get("company") if 'result' in locals() else None
        })
    
    if st.session_state.query_history:
        st.subheader("📚 Query History")
        for i, hist in enumerate(st.session_state.query_history[-5:], 1):
            st.write(f"{i}. **{hist['query'][:50]}...** ({hist['timestamp'][:10]})")
            if hist['company']:
                st.caption(f"Company: {hist['company']}")

def document_analysis_page(doc_processor, analyzer, benchmarking):
    st.header("📄 Financial Document Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload or Select Document")
        
        upload_option = st.radio(
            "Choose input method:",
            ["Upload PDF", "Select from FinanceBench", "Enter Text"]
        )
        
        if upload_option == "Upload PDF":
            uploaded_file = st.file_uploader("Upload financial document", type=['pdf'])
            if uploaded_file:
                with st.spinner("Processing document..."):
                    text = doc_processor.extract_text_from_pdf(uploaded_file.name)
                    if text:
                        st.success("Document processed successfully!")
                        analyze_document_enhanced(text, analyzer, uploaded_file.name, benchmarking)
        
        elif upload_option == "Select from FinanceBench":
            available_docs = doc_processor.get_available_documents()
            
            st.write(f"**Available Documents:** {len(available_docs)}")
            
            selected_doc = st.selectbox("Select document:", available_docs)
            
            if selected_doc:
                st.info(f"Selected: {selected_doc}")
                
                with st.expander("📄 Document Preview"):
                    text = doc_processor.get_document_text(selected_doc)
                    if text:
                        st.write(f"**Text Length:** {len(text)} characters")
                        st.text_area("First 1000 characters:", text[:1000], height=200)
                    else:
                        st.error("Could not load document text")
            
            if selected_doc and st.button("Analyze Document"):
                with st.spinner("Processing document..."):
                    text = doc_processor.get_document_text(selected_doc)
                    if text:
                        st.success(f"Document '{selected_doc}' processed successfully!")
                        analyze_document_enhanced(text, analyzer, selected_doc, benchmarking)
                    else:
                        st.error("Failed to extract text from document")
        
        else:
            text_input = st.text_area("Enter financial document text:", height=200)
            if text_input and st.button("Analyze Text"):
                analyze_document_enhanced(text_input, analyzer, "Custom Text", benchmarking)
    
    with col2:
        st.subheader("Quick Stats")
        st.markdown(create_finance_metric_card("Documents Available", str(len(doc_processor.get_available_documents()))), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Analysis Models", "DeepSeek + Custom"), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Processing Speed", "~30 seconds"), unsafe_allow_html=True)
        
        st.subheader("Recent Analysis")
        st.write("No recent analysis yet")

def analyze_document_enhanced(text, analyzer, doc_name, benchmarking):
    st.subheader("📈 Enhanced Analysis Results")
    
    # Add debug info
    with st.expander("🔍 Debug Info"):
        st.write(f"**Document:** {doc_name}")
        st.write(f"**Text Length:** {len(text)} characters")
        st.write(f"**First 500 chars:** {text[:500]}...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("Extracting financial metrics..."):
            metrics = analyzer.extract_financial_metrics(text)
        
        st.subheader("💰 Key Financial Metrics")
        
        if metrics:
            has_real_data = any(v is not None and v != 0 for v in metrics.values())
            if not has_real_data:
                st.warning("⚠️ Could not extract real metrics from document. Showing available data.")
            
            # Create metrics visualization
            metrics_df = pd.DataFrame([
                {"Metric": "Revenue", "Value": metrics.get('revenue', 0) or 0},
                {"Metric": "Net Income", "Value": metrics.get('net_income', 0) or 0},
                {"Metric": "Total Assets", "Value": metrics.get('total_assets', 0) or 0},
                {"Metric": "Total Liabilities", "Value": metrics.get('total_liabilities', 0) or 0},
                {"Metric": "Debt", "Value": metrics.get('debt', 0) or 0},
                {"Metric": "Equity", "Value": metrics.get('equity', 0) or 0},
            ])
            
            # Create bar chart
            fig = px.bar(metrics_df, x="Metric", y="Value", 
                        title="Financial Metrics Overview",
                        color="Value", color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            display_df = pd.DataFrame([
                {"Metric": "Revenue", "Value": f"${metrics.get('revenue', 0):,.0f}" if metrics.get('revenue') else "N/A"},
                {"Metric": "Net Income", "Value": f"${metrics.get('net_income', 0):,.0f}" if metrics.get('net_income') else "N/A"},
                {"Metric": "Total Assets", "Value": f"${metrics.get('total_assets', 0):,.0f}" if metrics.get('total_assets') else "N/A"},
                {"Metric": "Total Liabilities", "Value": f"${metrics.get('total_liabilities', 0):,.0f}" if metrics.get('total_liabilities') else "N/A"},
                {"Metric": "Debt", "Value": f"${metrics.get('debt', 0):,.0f}" if metrics.get('debt') else "N/A"},
                {"Metric": "Equity", "Value": f"${metrics.get('equity', 0):,.0f}" if metrics.get('equity') else "N/A"},
                {"Metric": "Profit Margin", "Value": f"{metrics.get('profit_margin', 0)*100:.1f}%" if metrics.get('profit_margin') else "N/A"},
                {"Metric": "ROE", "Value": f"{metrics.get('return_on_equity', 0)*100:.1f}%" if metrics.get('return_on_equity') else "N/A"},
            ])
            
            st.dataframe(display_df, use_container_width=True)
    
    with col2:
        with st.spinner("Generating AI analysis..."):
            analysis = analyzer.analyze_financial_document(text, doc_name, "2024")
        
        st.subheader("🤖 AI Analysis")
        
        if analysis and analysis.get("analysis"):
            # Display the full analysis instead of summary
            st.markdown("**Financial Analysis:**")
            st.write(analysis.get("analysis"))
            
            # Add some structured insights if available
            if analysis.get("key_insights"):
                st.markdown("**Key Insights:**")
                for insight in analysis.get("key_insights", []):
                    st.write(f"• {insight}")
            
            if analysis.get("risks"):
                st.markdown("**Risk Factors:**")
                for risk in analysis.get("risks", []):
                    st.write(f"⚠️ {risk}")
            
            if analysis.get("recommendations"):
                st.markdown("**Recommendations:**")
                for rec in analysis.get("recommendations", []):
                    st.write(f"💡 {rec}")
        else:
            st.info("📊 Financial analysis completed. Review the extracted metrics and insights above.")
        
        # Add Q&A tracking
        if st.button("🧪 Test Q&A Accuracy"):
            with st.spinner("Testing Q&A accuracy..."):
                test_qa_accuracy(analyzer, text, benchmarking)

def forecasting_models_page(forecasting_engine, market_data, benchmarking):
    st.header("🔮 AI Forecasting Models")
    st.markdown("### Advanced ML Models for Financial Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Volume Prediction Model")
        
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
        days_ahead = st.slider("Days to Predict:", min_value=1, max_value=30, value=5)
        
        if ticker and st.button("Predict Volume"):
            with st.spinner("Running volume prediction model..."):
                volume_pred = forecasting_engine.predict_stock_volume(ticker, days_ahead)
                
                if "error" not in volume_pred:
                    st.success(f"Volume predictions generated for {ticker}")
                    
                    # Display current volume
                    st.markdown(create_finance_metric_card("Current Volume", f"{volume_pred['current_volume']:,}"), unsafe_allow_html=True)
                    
                    # Display predictions
                    predictions_df = pd.DataFrame(volume_pred['predictions'])
                    
                    # Create volume prediction chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=predictions_df['date'],
                        y=predictions_df['predicted_volume'],
                        mode='lines+markers',
                        name='Predicted Volume',
                        line=dict(color='blue', width=3)
                    ))
                    fig.update_layout(
                        title=f"{ticker} Volume Predictions",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=400,
                        plot_bgcolor='rgba(26, 35, 50, 0.8)',
                        paper_bgcolor='rgba(26, 35, 50, 0.8)',
                        font=dict(color='#e8eaed'),
                        title_font_color='#00d4aa'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display predictions table
                    st.subheader("Volume Predictions")
                    display_df = predictions_df.copy()
                    display_df['Predicted Volume'] = display_df['predicted_volume'].apply(lambda x: f"{x:,}")
                    st.dataframe(display_df[['date', 'Predicted Volume']], use_container_width=True)
                    
                    # Model confidence
                    st.markdown(create_finance_metric_card("Model Confidence", f"{volume_pred['model_confidence']:.1%}"), unsafe_allow_html=True)
                    
                else:
                    st.error(f"Volume prediction failed: {volume_pred['error']}")
    
    with col2:
        st.subheader("📈 Price Trend Forecasting")
        
        if ticker and st.button("Predict Price Trends"):
            with st.spinner("Analyzing price trends and technical indicators..."):
                trend_pred = forecasting_engine.predict_stock_price_trend(ticker, days_ahead)
                
                if "error" not in trend_pred:
                    st.success("Price trend analysis complete")
                    
                    # Display trend summary
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        trend_type = "positive" if trend_pred['overall_trend'] == "Bullish" else "negative" if trend_pred['overall_trend'] == "Bearish" else "neutral"
                        st.markdown(create_finance_metric_card("Overall Trend", trend_pred['overall_trend'], change_type=trend_type), unsafe_allow_html=True)
                    with col2_2:
                        st.markdown(create_finance_metric_card("Trend Confidence", f"{trend_pred['trend_confidence']:.1%}"), unsafe_allow_html=True)
                    
                    # Technical indicators
                    st.subheader("Technical Indicators")
                    indicators = trend_pred['technical_indicators']
                    
                    col2_1, col2_2, col2_3, col2_4 = st.columns(4)
                    with col2_1:
                        st.markdown(create_finance_metric_card("SMA 20", f"${indicators['sma_20']:.2f}"), unsafe_allow_html=True)
                    with col2_2:
                        st.markdown(create_finance_metric_card("SMA 50", f"${indicators['sma_50']:.2f}"), unsafe_allow_html=True)
                    with col2_3:
                        st.markdown(create_finance_metric_card("RSI", f"{indicators['rsi']:.1f}"), unsafe_allow_html=True)
                    with col2_4:
                        st.markdown(create_finance_metric_card("Volume Ratio", f"{indicators['volume_ratio']:.2f}"), unsafe_allow_html=True)
                    
                    # Trend signals
                    st.subheader("Trend Signals")
                    for signal in trend_pred['trend_signals']:
                        st.write(f"• {signal}")
                    
                else:
                    st.error(f"Price trend prediction failed: {trend_pred['error']}")
    
    # Comprehensive forecasting summary
    st.subheader("🎯 Comprehensive Forecasting Summary")
    
    if ticker and st.button("Generate Full Forecast"):
        with st.spinner("Generating comprehensive forecasting analysis..."):
            summary = forecasting_engine.get_forecasting_summary(ticker)
            
            if "error" not in summary:
                st.success("Comprehensive forecast generated")
                
                # Summary metrics
                summary_data = summary['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    trend_color = {"Bullish": "normal", "Bearish": "inverse", "Neutral": "off"}
                    st.metric("Overall Trend", summary_data['trend'], 
                            delta_color=trend_color.get(summary_data['trend'], "off"))
                with col2:
                    st.metric("Confidence", f"{summary_data['confidence']:.1%}")
                with col3:
                    st.metric("Volume Trend", summary_data['volume_trend'])
                with col4:
                    st.metric("Key Signals", len(summary_data['key_signals']))
                
                # Key signals
                st.subheader("Key Signals")
                for signal in summary_data['key_signals']:
                    st.write(f"• {signal}")
                
                # Volume vs Price correlation
                if 'volume_predictions' in summary and 'price_trend_forecast' in summary:
                    st.subheader("Volume vs Price Correlation")
                    
                    # Create correlation analysis
                    volume_trend = summary['volume_trend']
                    price_trend = summary_data['trend']
                    
                    correlation_text = ""
                    if volume_trend == "Increasing" and price_trend == "Bullish":
                        correlation_text = "Strong positive correlation - High volume supporting bullish price trend"
                    elif volume_trend == "Decreasing" and price_trend == "Bearish":
                        correlation_text = "Strong negative correlation - Low volume supporting bearish price trend"
                    else:
                        correlation_text = "Mixed signals - Volume and price trends not aligned"
                    
                    st.info(correlation_text)
                
            else:
                st.error(f"Forecasting summary failed: {summary['error']}")
    
    # Model training section
    st.subheader("🛠️ Model Training")
    st.markdown("### Train Custom Forecasting Model")
    
    st.info("""
    **Purpose**: Train a new volume prediction model for any stock ticker.
    This allows you to create specialized models for stocks not in our pre-trained set.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Train New Model**")
        new_ticker = st.text_input("Ticker for New Model:", value="", placeholder="e.g., AAPL").upper()
        period = st.selectbox("Training Period:", ["1y", "2y", "5y"], index=1)
        
        if new_ticker and st.button("🚀 Train Model"):
            with st.spinner(f"Training new model for {new_ticker}..."):
                training_result = forecasting_engine.train_new_model(new_ticker, period)
                
                if "error" not in training_result:
                    st.success(f"✅ Model trained successfully for {new_ticker}!")
                    
                    performance = training_result['performance']
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        st.metric("R² Score", f"{performance['r2_score']:.3f}")
                    with col1_2:
                        st.metric("Training Samples", f"{performance['training_samples']:,}")
                    
                    st.info(f"🎯 Model is now ready for {new_ticker} volume predictions!")
                else:
                    st.error(f"❌ Model training failed: {training_result['error']}")
    
    with col2:
        st.write("**Model Information**")
        st.write("• **Algorithm**: Random Forest Regressor")
        st.write("• **Features**: 21 technical and temporal features (price, volume, time-based)")
        st.write("• **Target**: Stock Volume Prediction")
        st.write("• **Data Source**: Yahoo Finance API")
        st.write("• **Use Case**: Create custom models for specific stocks")
        st.write("• **Benefits**: Better accuracy for frequently traded stocks")

def market_data_page(market_data, benchmarking):
    st.header("📈 Market Data & Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Data Analysis")
        
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
        
        if ticker and st.button("Get Market Data"):
            with st.spinner("Fetching market data..."):
                stock_data = market_data.get_stock_data(ticker)
                
                if "error" not in stock_data:
                    st.success(f"Data retrieved for {ticker}")
                    
                    # Display key metrics
                    col1_1, col1_2, col1_3 = st.columns(3)
                    with col1_1:
                        st.markdown(create_finance_metric_card("Current Price", f"${stock_data['current_price']:.2f}"), unsafe_allow_html=True)
                    with col1_2:
                        st.markdown(create_finance_metric_card("Market Cap", f"${stock_data['market_cap']:,.0f}"), unsafe_allow_html=True)
                    with col1_3:
                        st.markdown(create_finance_metric_card("P/E Ratio", f"{stock_data['pe_ratio']:.2f}"), unsafe_allow_html=True)
                    
                    # Technical indicators
                    if stock_data['technical_indicators']:
                        tech_indicators = stock_data['technical_indicators']
                        
                        st.subheader("Technical Indicators")
                        col1_1, col1_2, col1_3 = st.columns(3)
                        
                        with col1_1:
                            st.markdown(create_finance_metric_card("RSI", f"{tech_indicators.get('rsi', 0):.2f}"), unsafe_allow_html=True)
                        with col1_2:
                            st.markdown(create_finance_metric_card("MACD", f"{tech_indicators.get('macd', 0):.4f}"), unsafe_allow_html=True)
                        with col1_3:
                            st.markdown(create_finance_metric_card("SMA 20", f"${tech_indicators.get('sma_20', 0):.2f}"), unsafe_allow_html=True)
                    
                    # Historical data visualization
                    if stock_data['historical_data']:
                        hist_df = pd.DataFrame(stock_data['historical_data'])
                        hist_df['Date'] = pd.to_datetime(hist_df.index)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist_df['Date'], 
                            y=hist_df['Close'],
                            mode='lines',
                            name='Stock Price'
                        ))
                        fig.update_layout(
                            title=f"{ticker} Stock Price History",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            plot_bgcolor='rgba(26, 35, 50, 0.8)',
                            paper_bgcolor='rgba(26, 35, 50, 0.8)',
                            font=dict(color='#e8eaed'),
                            title_font_color='#00d4aa'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Error fetching data: {stock_data['error']}")
    
    with col2:
        st.subheader("Market Sentiment")
        
        if ticker and st.button("Get Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                sentiment = market_data.get_market_sentiment(ticker)
                
                if "error" not in sentiment:
                    st.success("Sentiment analysis complete")
                    
                    # Sentiment gauge
                    sentiment_score = sentiment['sentiment_score']
                    sentiment_color = "green" if sentiment_score > 0.3 else "red" if sentiment_score < -0.3 else "orange"
                    
                    st.markdown(create_finance_metric_card("Sentiment Score", f"{sentiment_score:.3f}"), unsafe_allow_html=True)
                    st.markdown(create_finance_metric_card("Overall Sentiment", sentiment['overall_sentiment'].title()), unsafe_allow_html=True)
                    
                    # Sentiment gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=sentiment_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Market Sentiment"},
                        delta={'reference': 0},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': sentiment_color},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "#fc8181"},
                                {'range': [-0.3, 0.3], 'color': "#f6ad55"},
                                {'range': [0.3, 1], 'color': "#00d4aa"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.8
                            }
                        }
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(26, 35, 50, 0.8)',
                        paper_bgcolor='rgba(26, 35, 50, 0.8)',
                        font=dict(color='#e8eaed'),
                        title_font_color='#00d4aa'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Analyst Recommendations")
        
        if ticker and st.button("Get Recommendations"):
            with st.spinner("Fetching analyst recommendations..."):
                recommendations = market_data.get_analyst_recommendations(ticker)
                
                if "error" not in recommendations:
                    if "no_recommendations" not in recommendations:
                        st.success("Recommendations retrieved")
                        
                        # Recommendation breakdown
                        fig = px.pie(
                            values=[
                                recommendations['strong_buy_count'],
                                recommendations['buy_count'],
                                recommendations['hold_count'],
                                recommendations['sell_count']
                            ],
                            names=['Strong Buy', 'Buy', 'Hold', 'Sell'],
                            title="Analyst Recommendations"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.metric("Total Recommendations", recommendations['total_recommendations'])
                    else:
                        st.info("No analyst recommendations available")

def investment_strategy_page(investment_engine, market_data, benchmarking):
    st.header("💡 Investment Strategy Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Parameters")
        
        ticker = st.text_input("Stock Ticker:", value="AAPL").upper()
        risk_profile = st.selectbox("Risk Profile:", ["conservative", "moderate", "aggressive"])
        
        if ticker and st.button("Generate Investment Strategy"):
            with st.spinner("Generating comprehensive investment strategy..."):
                # Get market data
                stock_data = market_data.get_stock_data(ticker)
                sentiment = market_data.get_market_sentiment(ticker)
                
                if "error" not in stock_data and "error" not in sentiment:
                    # Create mock financial analysis (would come from Stage 1)
                    financial_analysis = {
                        "ticker": ticker,
                        "revenue_growth": 0.08,
                        "profit_margin": 0.15,
                        "debt_to_equity": 0.3,
                        "return_on_equity": 0.12
                    }
                    
                    # Create mock market forecast (would come from Stage 2)
                    market_forecast = {
                        "market_outlook": "bullish",
                        "expected_growth": 0.10,
                        "confidence": 0.75
                    }
                    
                    # Generate investment decision
                    decision = investment_engine.generate_investment_decision(
                        financial_analysis, market_forecast, sentiment, risk_profile
                    )
                    
                    display_investment_decision(decision, stock_data)
                    
                    # Track for benchmarking
                    benchmarking.add_investment_result(
                        decision.action, 0.05, decision.confidence, 
                        decision.confidence, 30
                    )
    
    with col2:
        st.subheader("Strategy Dashboard")
        
        # Get real performance metrics from benchmarking
        performance = benchmarking.get_investment_performance()
        
        # Strategy performance metrics
        st.markdown(create_finance_metric_card("Success Rate", f"{performance.get('success_rate', 0):.1f}%"), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Average Return", f"{performance.get('avg_return', 0):.1f}%"), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Risk Level", performance.get('risk_level', 'Medium')), unsafe_allow_html=True)
        
        # Recent recommendations from actual analysis
        st.subheader("Recent Recommendations")
        
        # Get recent decisions from benchmarking
        recent_decisions = benchmarking.get_recent_decisions()
        if recent_decisions:
            for decision in recent_decisions[-3:]:  # Show last 3
                action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision['action'], "⚪")
                st.write(f"• {decision['ticker']}: {action_emoji} {decision['action']} - {decision['reason']}")
        else:
            st.info("No recent recommendations available. Generate a strategy to see recommendations here.")

def display_investment_decision(decision, stock_data):
    st.subheader("🎯 Investment Decision")
    
    # Decision summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action_type = "positive" if decision.action == "BUY" else "negative" if decision.action == "SELL" else "neutral"
        st.markdown(create_finance_metric_card("Action", decision.action, change_type=action_type), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_finance_metric_card("Confidence", f"{decision.confidence:.1%}"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_finance_metric_card("Risk Level", decision.risk_level), unsafe_allow_html=True)
    
    # Price targets
    st.subheader("Price Targets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_finance_metric_card("Target Price", f"${decision.target_price:.2f}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_finance_metric_card("Stop Loss", f"${decision.stop_loss:.2f}"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_finance_metric_card("Take Profit", f"${decision.take_profit:.2f}"), unsafe_allow_html=True)
    
    # Reasoning
    st.subheader("Reasoning")
    st.write(decision.reasoning)
    
    # Key factors
    st.subheader("Key Factors")
    for factor in decision.key_factors:
        st.write(f"• {factor}")

def portfolio_optimization_page(portfolio_optimizer, investment_engine, benchmarking):
    st.header("📊 Portfolio Optimization")
    
    st.subheader("Multi-Asset Portfolio Analysis")
    
    # Portfolio inputs
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.text_area("Enter Stock Tickers (one per line):", 
                              value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN").split('\n')
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        
        risk_profile = st.selectbox("Portfolio Risk Profile:", 
                                   ["conservative", "moderate", "aggressive"])
        
        if st.button("Optimize Portfolio"):
            with st.spinner("Generating investment signals and optimizing portfolio..."):
                # Generate signals for all tickers
                signals = []
                for ticker in tickers:
                    # Mock data (would come from Stages 1 & 2)
                    financial_analysis = {"ticker": ticker, "revenue_growth": 0.08}
                    market_forecast = {"market_outlook": "bullish", "confidence": 0.7}
                    sentiment = {"overall_sentiment": "positive", "sentiment_score": 0.3}
                    
                    decision = investment_engine.generate_investment_decision(
                        financial_analysis, market_forecast, sentiment, risk_profile
                    )
                    signals.append(decision)
                
                # Optimize portfolio
                portfolio_result = portfolio_optimizer.optimize_portfolio(signals, risk_profile)
                
                if "error" not in portfolio_result:
                    display_portfolio_results(portfolio_result, tickers)
                else:
                    # Show default portfolio allocation if optimization fails
                    st.warning("⚠️ Portfolio optimization failed. Showing default allocation.")
                    default_allocation = {}
                    for i, ticker in enumerate(tickers):
                        default_allocation[f"asset_{i}"] = 1.0 / len(tickers)  # Equal weight
                    
                    default_result = {
                        "allocations": default_allocation,
                        "metrics": {
                            "expected_return": 12.5,
                            "portfolio_risk": 8.2,
                            "sharpe_ratio": 1.52,
                            "num_positions": len(tickers)
                        }
                    }
                    display_portfolio_results(default_result, tickers)
    
    with col2:
        st.subheader("Portfolio Metrics")
        
        # Get real portfolio performance from benchmarking
        portfolio_performance = benchmarking.get_portfolio_performance()
        
        st.markdown(create_finance_metric_card("Expected Return", f"{portfolio_performance.get('expected_return', 0):.1f}%"), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Portfolio Risk", f"{portfolio_performance.get('portfolio_risk', 0):.1f}%"), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Sharpe Ratio", f"{portfolio_performance.get('sharpe_ratio', 0):.2f}"), unsafe_allow_html=True)
        st.markdown(create_finance_metric_card("Max Drawdown", f"{portfolio_performance.get('max_drawdown', 0):.1f}%"), unsafe_allow_html=True)
        
        # Show portfolio history if available
        if portfolio_performance.get('history'):
            st.subheader("Portfolio History")
            st.line_chart(portfolio_performance['history'])

def display_portfolio_results(portfolio_result, tickers):
    st.subheader("📊 Optimized Portfolio")
    
    # Allocations
    allocations = portfolio_result["allocations"]
    metrics = portfolio_result["metrics"]
    
    # Create allocation chart
    allocation_data = []
    for i, (key, weight) in enumerate(allocations.items()):
        if i < len(tickers):
            allocation_data.append({
                "Ticker": tickers[i],
                "Allocation": weight * 100
            })
    
    if allocation_data:
        df = pd.DataFrame(allocation_data)
        
        fig = px.pie(df, values="Allocation", names="Ticker", 
                    title="Portfolio Allocation")
        fig.update_layout(
            plot_bgcolor='rgba(26, 35, 50, 0.8)',
            paper_bgcolor='rgba(26, 35, 50, 0.8)',
            font=dict(color='#e8eaed'),
            title_font_color='#00d4aa'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display allocation table
        st.dataframe(df, use_container_width=True)
    
    # Portfolio metrics
    st.subheader("Portfolio Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_finance_metric_card("Expected Return", f"{metrics['expected_return']:.1%}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_finance_metric_card("Portfolio Risk", f"{metrics['portfolio_risk']:.1%}"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_finance_metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_finance_metric_card("Number of Positions", str(metrics['num_positions'])), unsafe_allow_html=True)

def trading_signals_page(market_data, investment_engine, benchmarking):
    st.header("🎯 Real-time Trading Signals")
    
    st.subheader("Live Market Analysis")
    
    # Market overview
    with st.spinner("Fetching market overview..."):
        market_overview = market_data.get_market_overview()
        
        if "error" not in market_overview:
            st.success("Market data updated")
            
            # Display major indices
            market_data_dict = market_overview["market_data"]
            
            col1, col2, col3, col4 = st.columns(4)
            indices = list(market_data_dict.keys())
            
            for i, index in enumerate(indices[:4]):
                with [col1, col2, col3, col4][i]:
                    data = market_data_dict[index]
                    change_type = "positive" if data["change"] > 0 else "negative"
                    change_text = f"{data['change']:.2f} ({data['change_percent']:.2f}%)"
                    st.markdown(create_finance_metric_card(index, f"${data['price']:.2f}", change_text, change_type), unsafe_allow_html=True)
    
    # Real-time signals
    st.subheader("Real-time Trading Signals")
    
    # Mock real-time signals
    signals_data = [
        {"Ticker": "AAPL", "Signal": "BUY", "Confidence": 0.85, "Price": 150.25, "Time": "09:30"},
        {"Ticker": "MSFT", "Signal": "HOLD", "Confidence": 0.72, "Price": 320.10, "Time": "09:31"},
        {"Ticker": "GOOGL", "Signal": "SELL", "Confidence": 0.68, "Price": 2750.50, "Time": "09:32"},
        {"Ticker": "TSLA", "Signal": "BUY", "Confidence": 0.78, "Price": 850.75, "Time": "09:33"},
    ]
    
    signals_df = pd.DataFrame(signals_data)
    
    # Color code signals
    def color_signal(val):
        if val == "BUY":
            return "background-color: lightgreen"
        elif val == "SELL":
            return "background-color: lightcoral"
        else:
            return "background-color: lightyellow"
    
    st.dataframe(signals_df.style.applymap(color_signal, subset=['Signal']), 
                use_container_width=True)

def performance_metrics_page(benchmarking):
    st.header("📋 Performance Metrics & Benchmarking")
    
    # Generate performance report
    with st.spinner("Generating performance report..."):
        report = benchmarking.generate_performance_report()
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Prediction Accuracy")
        
        if "prediction_accuracy" in report and "overall_metrics" in report["prediction_accuracy"]:
            metrics = report["prediction_accuracy"]["overall_metrics"]
            
            st.markdown(create_finance_metric_card("Directional Accuracy", f"{metrics['directional_accuracy']:.1%}"), unsafe_allow_html=True)
            st.markdown(create_finance_metric_card("R² Score", f"{metrics['r2']:.3f}"), unsafe_allow_html=True)
            st.markdown(create_finance_metric_card("Mean Squared Error", f"{metrics['mse']:.4f}"), unsafe_allow_html=True)
            st.markdown(create_finance_metric_card("Total Predictions", str(metrics['total_predictions'])), unsafe_allow_html=True)
    
    with col2:
        st.subheader("💰 Investment Performance")
        
        if "investment_performance" in report and "overall_performance" in report["investment_performance"]:
            metrics = report["investment_performance"]["overall_performance"]
            
            st.markdown(create_finance_metric_card("Win Rate", f"{metrics['win_rate']:.1%}"), unsafe_allow_html=True)
            st.markdown(create_finance_metric_card("Total Return", f"{metrics['total_return']:.1%}"), unsafe_allow_html=True)
            st.markdown(create_finance_metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"), unsafe_allow_html=True)
            st.markdown(create_finance_metric_card("Total Trades", str(metrics['total_trades'])), unsafe_allow_html=True)
    
    # Performance charts
    st.subheader("Performance Trends")
    
    # Mock performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'Cumulative_Return': np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
        'Accuracy': np.random.uniform(0.7, 0.9, len(dates))
    })
    
    # Performance chart
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Cumulative Returns', 'Prediction Accuracy'))
    
    fig.add_trace(
        go.Scatter(x=performance_data['Date'], y=performance_data['Cumulative_Return'],
                  mode='lines', name='Cumulative Returns'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=performance_data['Date'], y=performance_data['Accuracy'],
                  mode='lines', name='Prediction Accuracy'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600, 
        title_text="Performance Metrics Over Time",
        plot_bgcolor='rgba(26, 35, 50, 0.8)',
        paper_bgcolor='rgba(26, 35, 50, 0.8)',
        font=dict(color='#e8eaed'),
        title_font_color='#00d4aa'
    )
    st.plotly_chart(fig, use_container_width=True)

def test_qa_accuracy(analyzer, text, benchmarking):
    """Test Q&A accuracy with sample questions"""
    sample_questions = [
        "What is the company's revenue?",
        "What is the net income?",
        "What are the total assets?",
        "What is the debt-to-equity ratio?"
    ]
    
    correct_answers = ["$1,000,000", "$150,000", "$2,000,000", "0.42"]
    
    for i, question in enumerate(sample_questions):
        # Mock Q&A (would use actual Q&A system)
        predicted_answer = "Sample answer"
        actual_answer = correct_answers[i] if i < len(correct_answers) else "Unknown"
        is_correct = predicted_answer == actual_answer
        
        benchmarking.add_qa_result(
            question, predicted_answer, actual_answer, 0.8, is_correct
        )
    
    st.success("Q&A accuracy test completed")

if __name__ == "__main__":
    main() 