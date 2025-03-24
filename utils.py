import requests
from bs4 import BeautifulSoup
from newspaper import Article
from typing import List, Dict, Any
import base64
import io
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import VitsModel, AutoProcessor
import torch
import json
from collections import Counter
import base64
import os
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
import time
import random
from typing import List, Dict, Any, Optional

def fetch_news_articles(company_name: str, num_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Scrape news articles about a specific company using BeautifulSoup.
    
    Args:
        company_name (str): Name of the company to search for
        num_articles (int): Maximum number of articles to fetch (default: 10)
        
    Returns:
        List[Dict[str, Any]]: List of articles with title, content, url, and date
    """
    articles = []
    
    # Search query URLs (you can add more news sources as needed)
    search_urls = [
        f"https://www.reuters.com/search/news?blob={company_name}",
        f"https://www.cnbc.com/search/?query={company_name}&qsearchterm={company_name}",
        f"https://economictimes.indiatimes.com/searchresult.cms?query={company_name}"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    for search_url in search_urls:
        try:
            # Fetch search results page
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
                
            # Parse the search results
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different parsing logic for different news sources
            if "reuters.com" in search_url:
                article_elements = soup.select('.search-result-title a')
                for article in article_elements:
                    if len(articles) >= num_articles:
                        break
                    
                    article_url = "https://www.reuters.com" + article['href'] if article['href'].startswith('/') else article['href']
                    articles.append(fetch_article_content(article_url, article.text.strip(), headers))
            
            elif "cnbc.com" in search_url:
                article_elements = soup.select('.Card-title a')
                for article in article_elements:
                    if len(articles) >= num_articles:
                        break
                        
                    article_url = "https://www.cnbc.com" + article['href'] if article['href'].startswith('/') else article['href']
                    articles.append(fetch_article_content(article_url, article.text.strip(), headers))
            
            elif "economictimes.indiatimes.com" in search_url:
                article_elements = soup.select('.article-list h3 a')
                for article in article_elements:
                    if len(articles) >= num_articles:
                        break
                        
                    article_url = "https://economictimes.indiatimes.com" + article['href'] if article['href'].startswith('/') else article['href']
                    articles.append(fetch_article_content(article_url, article.text.strip(), headers))
            
            # Add small delay between requests to be respectful
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            print(f"Error scraping {search_url}: {str(e)}")
            continue
    
    # Filter out any None values (failed fetches)
    articles = [article for article in articles if article is not None]
    
    # Limit to requested number
    return articles[:num_articles]

def fetch_article_content(url: str, title: str, headers: dict) -> Optional[Dict[str, Any]]:
    """
    Fetch and parse the content of a specific news article.
    
    Args:
        url (str): URL of the article
        title (str): Title of the article
        headers (dict): HTTP headers for the request
        
    Returns:
        Optional[Dict[str, Any]]: Article data or None if fetching fails
    """
    try:
        # Add delay to be respectful to the website
        time.sleep(random.uniform(0.5, 1.5))
        
        # Fetch article page
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        # Parse the article
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract date (different selectors for different sites)
        date = None
        date_selectors = [
            'time', '.ArticleHeader-date', '.date', '.publish-date',
            '.article-timestamp', '.article__date', '.story-date'
        ]
        
        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                date = date_element.text.strip()
                break
        
        # Extract content (different selectors for different sites)
        content = ""
        content_selectors = [
            '.ArticleBody-body', '.article-content', '.story-content',
            'article p', '.article-text p', '.story-text p'
        ]
        
        for selector in content_selectors:
            paragraphs = soup.select(selector)
            if paragraphs:
                content = ' '.join([p.text.strip() for p in paragraphs])
                break
        
        # If content extraction failed, try a more generic approach
        if not content:
            # Exclude navigation, header, footer, etc.
            for tag in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                tag.decompose()
            
            # Get all paragraphs from main content area
            main_content = soup.select_one('main') or soup.select_one('article') or soup.select_one('.content') or soup
            paragraphs = main_content.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs])
        
        # Return article data
        return {
            "title": title,
            "url": url,
            "date": date,
            "content": content if content else "Content extraction failed."
        }
        
    except Exception as e:
        print(f"Error fetching article {url}: {str(e)}")
        return None

def search_news_by_keywords(keywords: List[str], num_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Search for news articles containing specific keywords.
    
    Args:
        keywords (List[str]): List of keywords to search for
        num_articles (int): Maximum number of articles to fetch (default: 10)
        
    Returns:
        List[Dict[str, Any]]: List of articles matching the keywords
    """
    # Join keywords with '+' for URL
    query = '+'.join(keywords)
    return fetch_news_articles(query, num_articles)
# Download necessary NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize models
def initialize_models():
    """Initialize all models required for analysis"""
    global sentiment_analyzer, summarizer, translator, text_to_speech_model, text_to_speech_processor
    
    # Sentiment analysis model
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    # Summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Translation model (English to Hindi)
    translator_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    translator_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    translator = (translator_model, translator_tokenizer)
    
    # Text-to-speech model
    text_to_speech_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
    text_to_speech_processor = AutoProcessor.from_pretrained("facebook/mms-tts-hin")

# Initialize models on module load
initialize_models()

def fetch_news_articles(company_name: str, num_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch news articles related to a company
    
    Args:
        company_name: Name of the company
        num_articles: Maximum number of articles to fetch
        
    Returns:
        List of article dictionaries with title, url, and content
    """
    # Search query
    search_query = f"{company_name} company news"
    
    # Search for news links (in a real app, use a proper news API)
    # For demonstration, using a Google search scraping approach
    # Note: In production, use a proper news API like NewsAPI, Bing News, or GoogleNews
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # For demonstration purposes, we'll use a sample of fake articles
    # In production, replace this with actual web scraping or API calls
    articles = generate_sample_articles(company_name, num_articles)
    
    return articles

def generate_sample_articles(company_name: str, num_articles: int) -> List[Dict[str, Any]]:
    """Generate sample articles for demonstration purposes"""
    sample_templates = [
        {
            "title_template": "{company}'s Q3 Earnings Beat Expectations",
            "content_template": "{company} reported strong Q3 earnings today, beating analyst expectations by 15%. Revenue increased by 22% year-over-year, driven by strong product sales and expanding market share. The CEO stated that the company is well-positioned for future growth despite market uncertainties. Analysts have maintained a 'buy' rating for the stock, with an average price target increase of 10%.",
            "sentiment": "positive"
        },
        {
            "title_template": "{company} Announces Layoffs Amid Restructuring",
            "content_template": "{company} announced today that it will lay off approximately 5% of its global workforce as part of a restructuring plan. The company cited challenging market conditions and the need to optimize operations. The stock price fell by 3% following the announcement. Some analysts have expressed concerns about the company's future growth prospects, while others view the restructuring as a necessary step toward long-term efficiency.",
            "sentiment": "negative"
        },
        {
            "title_template": "{company} Launches New Product Line",
            "content_template": "{company} unveiled its latest product line today, featuring advanced technology and improved features. The launch event attracted significant media attention and initial customer feedback has been largely positive. Industry experts predict the new products could help the company gain market share in competitive segments. The product line is expected to be available in stores next month.",
            "sentiment": "positive"
        },
        {
            "title_template": "{company} Faces Regulatory Investigation",
            "content_template": "Regulatory authorities have launched an investigation into {company}'s business practices, according to sources familiar with the matter. The investigation reportedly focuses on potential antitrust violations and data privacy concerns. The company stated it is cooperating fully with authorities but maintains that all its practices comply with applicable regulations. Legal experts suggest the investigation could lead to significant fines if violations are found.",
            "sentiment": "negative"
        },
        {
            "title_template": "{company} Partners with Tech Giant for Cloud Solutions",
            "content_template": "{company} announced a strategic partnership with a leading tech company to enhance its cloud infrastructure and digital capabilities. The multi-year agreement will focus on developing AI-powered solutions and modernizing legacy systems. Industry analysts view this partnership positively, noting it could accelerate {company}'s digital transformation efforts. The companies plan to launch their first joint solutions in the coming quarter.",
            "sentiment": "positive"
        },
        {
            "title_template": "{company} Stock Remains Stable Despite Market Volatility",
            "content_template": "Amid broader market fluctuations, {company}'s stock has maintained relative stability over the past month. Analysts attribute this to the company's strong fundamentals and diversified revenue streams. While some sectors have experienced significant downturns, {company} has managed to navigate economic uncertainties effectively. Institutional investors have maintained their positions, signaling confidence in the company's long-term prospects.",
            "sentiment": "neutral"
        },
        {
            "title_template": "{company} Expands Operations in Asian Markets",
            "content_template": "{company} announced plans to significantly expand its presence in key Asian markets, with particular focus on India and Southeast Asia. The expansion includes new manufacturing facilities and increased sales operations. Market analysts project this move could increase the company's global market share by 2-3% within five years. Local businesses have welcomed the expansion, citing potential job creation and economic benefits.",
            "sentiment": "positive"
        },
        {
            "title_template": "Activist Investors Pressure {company} Board for Changes",
            "content_template": "A group of activist investors has acquired a significant stake in {company} and is pushing for strategic changes, including potential spinoffs of underperforming divisions. The investors claim the current leadership has failed to maximize shareholder value. The company's board has acknowledged the investors' concerns but defended its current strategy. Some market analysts believe this pressure could lead to positive restructuring, while others warn of potential disruption to long-term plans.",
            "sentiment": "mixed"
        },
        {
            "title_template": "{company} Achieves Carbon Neutrality Goal Ahead of Schedule",
            "content_template": "{company} announced today that it has achieved carbon neutrality across all operations, reaching its environmental goal two years ahead of schedule. The company cited investments in renewable energy, supply chain improvements, and carbon offset programs as key factors. Environmental groups have praised the achievement while encouraging further sustainability efforts. The company has announced new targets for reducing Scope 3 emissions by 2030.",
            "sentiment": "positive"
        },
        {
            "title_template": "Supply Chain Issues Impact {company}'s Production Capacity",
            "content_template": "{company} reported that ongoing global supply chain disruptions have impacted its production capacity, potentially affecting quarterly earnings. The company has implemented mitigation strategies but warns that shortages of key components may persist. Several competitors face similar challenges, suggesting industry-wide implications. Analysts have adjusted short-term price targets but maintain confidence in long-term performance once supply chains normalize.",
            "sentiment": "negative"
        },
        {
            "title_template": "{company} CEO Featured in Business Leadership Summit",
            "content_template": "The CEO of {company} was a keynote speaker at yesterday's Global Business Leadership Summit, discussing innovation strategies and market trends. The presentation highlighted the company's approach to digital transformation and talent development. The CEO's remarks received positive reception from attendees and industry media. Several key insights about the company's future direction were shared, though specific product announcements were not made.",
            "sentiment": "neutral"
        },
        {
            "title_template": "{company} Reports Mixed Q2 Results",
            "content_template": "{company} released its Q2 financial results today, showing mixed performance across different business segments. While revenue slightly exceeded expectations, profit margins were narrower than projected. The consumer products division showed strong growth, but enterprise solutions underperformed. Management remains confident about meeting full-year guidance despite macroeconomic headwinds. The stock traded within a narrow range following the announcement.",
            "sentiment": "neutral"
        }
    ]
    
    # Generate random articles based on templates
    articles = []
    templates_to_use = random.sample(sample_templates, min(num_articles, len(sample_templates)))
    
    for template in templates_to_use:
        title = template["title_template"].format(company=company_name)
        content = template["content_template"].format(company=company_name)
        
        # Add random details to make content more unique
        content += f" Industry analysts from {random.choice(['Goldman Sachs', 'Morgan Stanley', 'JP Morgan', 'Bank of America'])} have {random.choice(['maintained', 'adjusted', 'upgraded', 'reviewed'])} their outlook on {company_name}."
        
        # Generate a random URL
        url = f"https://news-{random_string(8)}.example.com/{company_name.lower().replace(' ', '-')}-{random_string(6)}"
        
        articles.append({
            "title": title,
            "url": url,
            "content": content
        })
    
    return articles

def random_string(length: int) -> str:
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of an article
    
    Args:
        text: Article content
        
    Returns:
        Sentiment label (Positive, Negative, or Neutral)
    """
    # Use pretrained sentiment analysis model
    try:
        result = sentiment_analyzer(text[:512])[0]  # Use first 512 tokens for efficiency
        
        # Map score to categorical sentiment
        score = result["score"]
        label = result["label"]
        
        if label == "positive" and score > 0.6:
            return "Positive"
        elif label == "negative" and score > 0.6:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return "Neutral"  # Default to neutral on error

def summarize_article(text: str, max_length: int = 150) -> str:
    """
    Generate a concise summary of an article
    
    Args:
        text: Article content
        max_length: Maximum length of summary in tokens
        
    Returns:
        Article summary
    """
    try:
        # For longer texts, split and summarize in chunks
        if len(text) > 1000:
            chunks = split_text(text)
            chunk_summaries = []
            
            for chunk in chunks:
                if len(chunk.strip()) > 100:  # Only summarize substantial chunks
                    summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
                    chunk_summaries.append(summary)
            
            full_summary = " ".join(chunk_summaries)
            
            # If still too long, summarize again
            if len(full_summary) > 500:
                full_summary = summarizer(full_summary, max_length=max_length, min_length=50, do_sample=False)[0]["summary_text"]
            
            return full_summary
        else:
            # For shorter texts, summarize directly
            summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
            return summary
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        # Fallback to extractive summarization on error
        return extractive_summarize(text, 2)

def split_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into chunks based on sentences for processing long texts"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    """Fallback extractive summarization method"""
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    # Simple approach: take first two sentences and last sentence
    if num_sentences == 3 and len(sentences) >= 3:
        return " ".join([sentences[0], sentences[1], sentences[-1]])
    
    # Otherwise just take the first n sentences
    return " ".join(sentences[:num_sentences])

def extract_topics(text: str, num_topics: int = 3) -> List[str]:
    """
    Extract key topics from article text
    
    Args:
        text: Article content
        num_topics: Number of topics to extract
        
    Returns:
        List of topics
    """
    try:
        # Tokenize and normalize text
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words and w.isalpha()]
        
        # Count word frequencies
        word_freq = Counter(filtered_tokens)
        
        # Filter out common words that aren't meaningful as topics
        common_words = {'company', 'business', 'market', 'year', 'report', 'said', 'says', 'according'}
        for word in common_words:
            if word in word_freq:
                del word_freq[word]
        
        # Get most common topics
        common_topics = [word.title() for word, _ in word_freq.most_common(num_topics * 2)]
        
        # Map to predefined business topics for better organization
        business_topics = {
            "Finance": ["Revenue", "Profit", "Earnings", "Investment", "Stock", "Shares", "Financial", "Money", "Fund", "Capital"],
            "Technology": ["Tech", "Software", "Hardware", "Digital", "Innovation", "Product", "App", "Device", "Platform", "System"],
            "Management": ["CEO", "Executive", "Leadership", "Management", "Strategy", "Decision", "Board", "Director", "Operation"],
            "Marketing": ["Brand", "Customer", "Consumer", "Marketing", "Advertisement", "Sales", "Market", "Buyer", "Client"],
            "Regulation": ["Regulation", "Compliance", "Legal", "Law", "Government", "Policy", "Rule", "Standard", "Guidelines"],
            "Growth": ["Growth", "Expansion", "Development", "Scale", "Increase", "Grow", "Rise", "Performance"],
            "Competition": ["Competitor", "Competition", "Rival", "Industry", "Sector", "Market", "Player"],
            "Innovation": ["Innovation", "Research", "Development", "Innovative", "Design", "Creative", "New"],
            "Global": ["Global", "International", "Worldwide", "Export", "Import", "Foreign", "Overseas", "Country"],
            "Sustainability": ["Sustainable", "Environmental", "Green", "Carbon", "Climate", "Eco", "Renewable", "Energy"]
        }
        
        # Match extracted topics to business categories
        matched_topics = set()
        for word in common_topics:
            for category, keywords in business_topics.items():
                if any(keyword.lower() in word.lower() or word.lower() in keyword.lower() for keyword in keywords):
                    matched_topics.add(category)
                    break
        
        # If we don't have enough matches, add some from the common topics
        if len(matched_topics) < num_topics:
            for word in common_topics:
                if len(matched_topics) >= num_topics:
                    break
                matched_topics.add(word)
        
        # Limit to requested number
        return list(matched_topics)[:num_topics]
    
    except Exception as e:
        print(f"Error extracting topics: {str(e)}")
        # Return default topics on error
        return ["Business", "Finance", "Industry"]
def generate_comparative_analysis(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comparative analysis of multiple news articles
    
    Args:
        articles: List of processed articles
        
    Returns:
        Dictionary with comparative analysis results
    """
    # Count sentiment distribution
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        sentiment_counts[article["Sentiment"]] += 1
    
    # Extract all topics
    all_topics = []
    article_topics = {}
    for i, article in enumerate(articles):
        article_topics[f"Article {i+1}"] = article["Topics"]
        all_topics.extend(article["Topics"])
    
    # Find common topics
    topic_counter = Counter(all_topics)
    common_topics = [topic for topic, count in topic_counter.items() if count > 1]
    
    # Find unique topics per article
    unique_topics = {}
    for i, article in enumerate(articles):
        article_key = f"Article {i+1}"
        unique_to_this = []
        for topic in article["Topics"]:
            # Check if topic appears only in this article
            if all_topics.count(topic) == 1:
                unique_to_this.append(topic)
        if unique_to_this:
            unique_topics[article_key] = unique_to_this
    
    # Generate comparisons between articles
    comparisons = []
    # Limit to 5 comparisons for readability
    max_comparisons = min(5, len(articles) * (len(articles) - 1) // 2)
    comparison_count = 0
    
    for i in range(len(articles)):
        for j in range(i+1, len(articles)):
            if comparison_count >= max_comparisons:
                break
                
            # Skip if both articles have the same sentiment
            if articles[i]["Sentiment"] == articles[j]["Sentiment"]:
                continue
                
            article1 = articles[i]
            article2 = articles[j]
            
            # Create comparison text
            if article1["Sentiment"] != article2["Sentiment"]:
                comparison_text = f"Article '{article1['Title']}' ({article1['Sentiment']}) contrasts with '{article2['Title']}' ({article2['Sentiment']})."
                
                # Determine impact based on sentiments
                if article1["Sentiment"] == "Positive" and article2["Sentiment"] == "Negative":
                    impact = "The mixed sentiment may indicate divided opinions about the company's prospects."
                elif article1["Sentiment"] == "Negative" and article2["Sentiment"] == "Positive":
                    impact = "The conflicting coverage suggests viewing both positive developments and challenges."
                else:
                    impact = "The different perspectives provide a more nuanced view of the company's situation."
                
                comparisons.append({
                    "Comparison": comparison_text,
                    "Impact": impact
                })
                comparison_count += 1
    
    # If we didn't find enough sentiment-based comparisons, add topic-based ones
    if comparison_count < max_comparisons:
        for i in range(len(articles)):
            for j in range(i+1, len(articles)):
                if comparison_count >= max_comparisons:
                    break
                    
                article1 = articles[i]
                article2 = articles[j]
                
                # Skip if already compared
                if any(f"'{article1['Title']}'" in comp["Comparison"] and f"'{article2['Title']}'" in comp["Comparison"] for comp in comparisons):
                    continue
                
                # Check for different topics
                diff_topics1 = set(article1["Topics"]) - set(article2["Topics"])
                diff_topics2 = set(article2["Topics"]) - set(article1["Topics"])
                
                if diff_topics1 and diff_topics2:
                    comparison_text = f"Article '{article1['Title']}' focuses on {', '.join(diff_topics1)}, while '{article2['Title']}' focuses on {', '.join(diff_topics2)}."
                    impact = f"This shows how coverage varies across different aspects of the company's operations."
                    
                    comparisons.append({
                        "Comparison": comparison_text,
                        "Impact": impact
                    })
                    comparison_count += 1
    
    # Structure the final comparative analysis
    comparative_analysis = {
        "Sentiment Distribution": sentiment_counts,
        "Coverage Differences": comparisons,
        "Topic Overlap": {
            "Common Topics": common_topics if common_topics else ["No common topics found"],
            "Unique Topics": unique_topics
        }
    }
    
    return comparative_analysis

def text_to_speech_hindi(text):
    """
    Convert text to Hindi speech and return base64 encoded audio
    
    Args:
        text (str): Text to convert to speech
        
    Returns:
        str: Base64 encoded audio data
    """
    try:
        # Create a temporary file to store the audio
        temp_file = "temp_hindi_speech.mp3"
        
        # Generate Hindi speech
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(temp_file)
        
        # Read the file and encode to base64
        with open(temp_file, "rb") as audio_file:
            audio_data = audio_file.read()
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return base64_audio
    
    except Exception as e:
        print(f"Error in text to speech conversion: {str(e)}")
        return ""

def is_english(text: str) -> bool:
    """Check if the text is primarily in English"""
    # Simple heuristic: check if most words are in English
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    english_word_count = sum(1 for word in words if word in stop_words or word.isalpha())
    return english_word_count / len(words) > 0.5 if words else True

def translate_to_hindi(text: str) -> str:
    """Translate English text to Hindi"""
    try:
        model, tokenizer = translator
        
        # Tokenize and translate
        tokenizer.src_lang = "en"
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id("hi"),
                max_length=512
            )
        
        # Decode the translated text
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        # Fallback: return a basic Hindi message
        return "कंपनी के बारे में समाचार विश्लेषण।"  # "News analysis about the company." in Hindi