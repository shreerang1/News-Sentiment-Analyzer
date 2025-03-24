import streamlit as st
import requests
import json
import base64
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="News Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    /* Global styling */
    body {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #F8FAFC;
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    
    .app-description {
        text-align: center;
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        border-left: 4px solid #3B82F6;
        padding-left: 0.75rem;
    }
    
    /* Card styling */
    .card {
        padding: 1.8rem;
        border-radius: 0.75rem;
        background-color: white;
        margin: 1.2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Article card styling */
    .article-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: white;
        margin-bottom: 1.2rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .article-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    /* Sentiment badges */
    .sentiment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }
    
    .positive {
        background-color: rgba(5, 150, 105, 0.1);
        color: #059669;
        border: 1px solid rgba(5, 150, 105, 0.3);
    }
    
    .negative {
        background-color: rgba(220, 38, 38, 0.1);
        color: #DC2626;
        border: 1px solid rgba(220, 38, 38, 0.3);
    }
    
    .neutral {
        background-color: rgba(107, 114, 128, 0.1);
        color: #6B7280;
        border: 1px solid rgba(107, 114, 128, 0.3);
    }
    
    /* Topic tags styling */
    .topic-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border: 1px solid #BFDBFE;
    }
    
    /* Chart container styling */
    .chart-container {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F1F5F9;
    }
    
    .sidebar .sidebar-content {
        background-color: #F1F5F9;
    }
    
    /* Input field styling */
    div[data-baseweb="input"] {
        border-radius: 0.5rem !important;
    }
    
    /* Button styling */
    div[data-testid="stButton"] button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div[data-testid="stButton"] button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Radio button styling */
    div[data-testid="stRadio"] > div {
        padding: 0.5rem;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Select box styling */
    div[data-baseweb="select"] {
        border-radius: 0.5rem !important;
    }
    
    /* Spinner styling */
    div[data-testid="stSpinner"] > div {
        border-color: #3B82F6 !important;
    }
    
    /* Audio player styling */
    div[data-testid="stAudio"] {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Divider styling */
    hr {
        margin: 1rem 0;
        border-color: #E2E8F0;
    }
    
    /* Comparison section styling */
    .comparison-item {
        border-left: 3px solid #3B82F6;
        padding-left: 1rem;
        margin-bottom: 1rem;
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Topic overlap styling */
    .topic-overlap {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #BAE6FD;
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000"  # Change when deployed

def get_company_list():
    """Fetch the list of available companies from the API"""
    try:
        response = requests.get(f"{API_URL}/companies")
        if response.status_code == 200:
            return response.json()["companies"]
        else:
            st.error("Failed to fetch company list. Please try again later.")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def analyze_company(company_name):
    """Send company name to API and get analysis results"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"company_name": company_name}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def display_sentiment_distribution(data):
    """Create a pie chart for sentiment distribution"""
    sentiment_dist = data["Comparative Sentiment Score"]["Sentiment Distribution"]
    df = pd.DataFrame({
        'Sentiment': list(sentiment_dist.keys()),
        'Count': list(sentiment_dist.values())
    })
    
    fig = px.pie(
        df,
        values='Count',
        names='Sentiment',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#059669',
            'Negative': '#DC2626',
            'Neutral': '#6B7280'
        },
        title="Sentiment Distribution"
    )
    fig.update_layout(
        height=400,
        title_font=dict(size=18, color="#1E3A8A"),
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def display_topics_analysis(data):
    """Create a bar chart for topic frequency"""
    # Collect all topics from articles
    all_topics = []
    for article in data["Articles"]:
        all_topics.extend(article["Topics"])
    
    # Count frequencies
    topic_counts = {}
    for topic in all_topics:
        if topic in topic_counts:
            topic_counts[topic] += 1
        else:
            topic_counts[topic] = 1
    
    # Create dataframe
    df = pd.DataFrame({
        'Topic': list(topic_counts.keys()),
        'Frequency': list(topic_counts.values())
    }).sort_values('Frequency', ascending=False).head(10)  # Show top 10 topics
    
    fig = px.bar(
        df, 
        x='Topic', 
        y='Frequency',
        title="Top 10 Most Discussed Topics",
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(
        height=400,
        title_font=dict(size=18, color="#1E3A8A"),
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=-45),
        margin=dict(l=20, r=20, t=40, b=40)
    )
    return fig

def display_audio_player(audio_data):
    """Display an audio player for the TTS output"""
    audio_bytes = base64.b64decode(audio_data)
    
    # Save to temporary file
    temp_audio_path = "temp_audio.mp3"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)
    
    # Display audio player
    st.audio(temp_audio_path, format="audio/mp3")

def format_topics_as_tags(topics):
    """Format topics as HTML tags"""
    return " ".join([f'<span class="topic-tag">{topic}</span>' for topic in topics])

def main():
    st.markdown("<h1 class='main-header'>News Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-description'>Extract valuable insights from news articles about companies with advanced sentiment analysis and Hindi text-to-speech technology.</p>", unsafe_allow_html=True)
    
    # Sidebar with gradient background
    st.sidebar.markdown("""
    <div style="background: linear-gradient(180deg, #EFF6FF 0%, #DBEAFE 100%); padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem;">
        <h2 style="color: #1E40AF; margin-bottom: 1rem; font-size: 1.5rem; font-weight: 600;">Company Selection</h2>
        <p style="color: #475569; font-size: 0.9rem;">Choose a company to analyze news sentiment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to enter company name or select from dropdown
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Select from list", "Enter company name"]
    )
    
    company_name = None
    
    if input_method == "Select from list":
        companies = get_company_list()
        if companies:
            company_name = st.sidebar.selectbox("Select a company:", companies)
    else:
        company_name = st.sidebar.text_input("Enter company name:")
    
    # Analysis button
    if company_name:
        if st.sidebar.button("Analyze News"):
            with st.spinner(f"Analyzing news for {company_name}..."):
                # Call API to get analysis
                analysis_result = analyze_company(company_name)
                
                if analysis_result:
                    # Store the result in session state to persist between reruns
                    st.session_state.analysis_result = analysis_result
                    st.session_state.company_name = company_name
    
    # Display analysis results if available
    if 'analysis_result' in st.session_state:
        data = st.session_state.analysis_result
        company = st.session_state.company_name
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 class='sub-header' style="text-align: center; border-left: none; border-bottom: 3px solid #3B82F6; display: inline-block; padding-bottom: 0.5rem;">
                Analysis Results for {company}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Final sentiment analysis
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="color: #1E3A8A; font-size: 1.5rem; margin-bottom: 1rem; display: flex; align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
            </svg>
            Overall Sentiment Analysis
        </h3>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <p style="font-size: 1.1rem; line-height: 1.6; color: #334155;">{data['Final Sentiment Analysis']}</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display charts - distribute in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(display_sentiment_distribution(data), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(display_topics_analysis(data), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Audio player for Hindi TTS
        st.markdown("""
        <h3 style="color: #1E3A8A; font-size: 1.5rem; margin: 1.5rem 0 1rem 0; display: flex; align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
            </svg>
            Hindi Audio Summary
        </h3>
        """, unsafe_allow_html=True)
        display_audio_player(data["Audio"])
        
        # Article details
        st.markdown("""
        <h3 style="color: #1E3A8A; font-size: 1.5rem; margin: 1.5rem 0 1rem 0; display: flex; align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            News Articles Analysis
        </h3>
        """, unsafe_allow_html=True)
        
        for i, article in enumerate(data["Articles"]):
            # Determine sentiment class for styling
            sentiment_class = article["Sentiment"].lower()
            
            st.markdown(f"""
            <div class='article-card'>
                <h4 class='article-title'>{i+1}. {article["Title"]}</h4>
                <p style="margin-bottom: 0.75rem; color: #475569; line-height: 1.5;">
                    <strong style="color: #1E3A8A;">Summary:</strong> {article["Summary"]}
                </p>
                <p style="margin-bottom: 0.75rem;">
                    <strong style="color: #1E3A8A;">Sentiment:</strong> 
                    <span class='sentiment-badge {sentiment_class}'>{article["Sentiment"]}</span>
                </p>
                <p style="margin-bottom: 0;">
                    <strong style="color: #1E3A8A;">Topics:</strong> 
                    {format_topics_as_tags(article["Topics"])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparative Analysis
        st.markdown("""
        <h3 style="color: #1E3A8A; font-size: 1.5rem; margin: 1.5rem 0 1rem 0; display: flex; align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            Comparative Analysis
        </h3>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        for i, comparison in enumerate(data["Comparative Sentiment Score"]["Coverage Differences"]):
            st.markdown(f"""
            <div class="comparison-item">
                <p style="margin-bottom: 0.5rem;">
                    <strong style="color: #1E3A8A;">Comparison {i+1}:</strong> {comparison["Comparison"]}
                </p>
                <p style="margin-bottom: 0; color: #475569;">
                    <strong style="color: #1E3A8A;">Impact:</strong> {comparison["Impact"]}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Topic Overlap
        topic_overlap = data["Comparative Sentiment Score"]["Topic Overlap"]
        st.markdown("""
        <h4 style="color: #2563EB; font-size: 1.3rem; margin: 1.5rem 0 1rem 0;">Topic Analysis</h4>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="topic-overlap">
            <strong style="color: #1E3A8A;">Common Topics:</strong> {format_topics_as_tags(topic_overlap["Common Topics"])}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Add filters section in sidebar if results are available
        if 'analysis_result' in st.session_state:
            st.sidebar.markdown("""
            <div style="background: linear-gradient(180deg, #EFF6FF 0%, #DBEAFE 100%); padding: 1.5rem; border-radius: 0.75rem; margin-top: 2rem;">
                <h2 style="color: #1E40AF; margin-bottom: 1rem; font-size: 1.5rem; font-weight: 600;">Filters</h2>
                <p style="color: #475569; font-size: 0.9rem;">Refine the analysis results.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add sentiment filter
            sentiment_filter = st.sidebar.multiselect(
                "Filter by sentiment:",
                ["Positive", "Negative", "Neutral"],
                default=["Positive", "Negative", "Neutral"]
            )
            
            # Add date range filter (if we had dates in the data)
            st.sidebar.markdown("<p style='margin-top: 1rem;'>Date Range:</p>", unsafe_allow_html=True)
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("From")
            with col2:
                end_date = st.date_input("To")

if __name__ == "__main__":
    main()