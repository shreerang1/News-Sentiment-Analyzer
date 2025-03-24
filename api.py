from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from utils import (
    fetch_news_articles,
    analyze_sentiment,
    generate_comparative_analysis,
    summarize_article,
    extract_topics,
    text_to_speech_hindi
)

app = FastAPI(title="News Sentiment Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample company list (in production, this would come from a database)
COMPANIES = [
    "Tesla", "Apple", "Microsoft", "Google", "Amazon", 
    "Meta", "Nvidia", "Reliance", "Tata", "Infosys"
]

# Models
class CompanyRequest(BaseModel):
    company_name: str

class ArticleResponse(BaseModel):
    Title: str
    Summary: str
    Sentiment: str
    Topics: List[str]

class SentimentDistribution(BaseModel):
    Positive: int
    Negative: int
    Neutral: int

class ComparisonItem(BaseModel):
    Comparison: str
    Impact: str

class TopicOverlap(BaseModel):
    Common_Topics: List[str]
    Unique_Topics: Dict[str, List[str]]

class ComparativeSentimentScore(BaseModel):
    Sentiment_Distribution: SentimentDistribution
    Coverage_Differences: List[ComparisonItem]
    Topic_Overlap: TopicOverlap

class AnalysisResponse(BaseModel):
    Company: str
    Articles: List[ArticleResponse]
    Comparative_Sentiment_Score: ComparativeSentimentScore
    Final_Sentiment_Analysis: str
    Audio: str  # Base64 encoded audio

@app.get("/")
def read_root():
    return {"message": "News Sentiment Analysis API is running"}

@app.get("/companies")
def get_companies():
    """Get list of available companies"""
    return {"companies": COMPANIES}

@app.post("/analyze")
def analyze_company(request: CompanyRequest = Body(...)):
    """Analyze news for a given company"""
    company_name = request.company_name.strip()
    
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name cannot be empty")
    
    try:
        # Fetch news articles
        articles = fetch_news_articles(company_name)
        
        if not articles or len(articles) == 0:
            raise HTTPException(status_code=404, detail=f"No news articles found for {company_name}")
        
        # Limit to 10 articles if more are found
        articles = articles[:10]
        
        # Process each article
        processed_articles = []
        for article in articles:
            # Extract summary
            summary = summarize_article(article["content"])
            
            # Analyze sentiment
            sentiment = analyze_sentiment(article["content"])
            
            # Extract topics
            topics = extract_topics(article["content"])
            
            processed_articles.append({
                "Title": article["title"],
                "Summary": summary,
                "Sentiment": sentiment,
                "Topics": topics,
                "content": article["content"]  # Keep original content for comparative analysis
            })
        
        # Generate comparative analysis
        comparative_analysis = generate_comparative_analysis(processed_articles)
        
        # Determine overall sentiment
        sentiment_counts = comparative_analysis["Sentiment Distribution"]
        if sentiment_counts["Positive"] > sentiment_counts["Negative"]:
            final_sentiment = f"{company_name}'s latest news coverage is mostly positive. Potential stock growth expected."
        elif sentiment_counts["Positive"] < sentiment_counts["Negative"]:
            final_sentiment = f"{company_name}'s latest news coverage is mostly negative. Market caution advised."
        else:
            final_sentiment = f"{company_name}'s latest news coverage is mixed. Market may remain stable."
        
        # Generate Hindi TTS
        # Create summary text for TTS
        tts_text = f"{company_name} के बारे में समाचार विश्लेषण. समग्र भावना: {final_sentiment}"
        audio_base64 = text_to_speech_hindi(tts_text)
        
        # Format for client
        # Remove the content field which was only needed for analysis
        for article in processed_articles:
            del article["content"]
            
        response_data = {
            "Company": company_name,
            "Articles": processed_articles,
            "Comparative Sentiment Score": comparative_analysis,
            "Final Sentiment Analysis": final_sentiment,
            "Audio": audio_base64
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing news: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
