# News Summarization and Text-to-Speech Application

This application extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi.

## Features

- **News Extraction**: Extracts titles, summaries, and metadata from multiple news articles related to the input company
- **Sentiment Analysis**: Analyzes sentiment (positive, negative, neutral) for each article
- **Comparative Analysis**: Compares sentiment and topics across articles to derive insights
- **Topic Extraction**: Identifies key topics discussed in each article
- **Text-to-Speech**: Converts summarized content to Hindi speech
- **User-friendly Interface**: Simple web interface built with Streamlit

## Architecture

The application follows a client-server architecture:

1. **Frontend**: Streamlit-based web interface
2. **Backend**: FastAPI server that handles data processing and analysis
3. **API Communication**: REST API endpoints for communication between frontend and backend

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/news-analysis-app.git
   cd news-analysis-app
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

### Running the Application

1. Start the FastAPI backend server:
   ```
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```
   streamlit run app.py
   ```

3. Access the application in your web browser at `http://localhost:8501`

## Usage

1. Input a company name in the provided text field or select from the dropdown
2. Click "Analyze News" to initiate the analysis
3. View the sentiment analysis results, comparative analysis, and topic breakdown
4. Listen to the Hindi audio summary via the embedded audio player

## API Endpoints

The application exposes the following RESTful API endpoints:

- `GET /companies`: Returns a list of available companies
- `POST /analyze`: Analyzes news for a given company
  - Request body: `{"company_name": "Company Name"}`
  - Returns analysis results including processed articles, sentiment analysis, and more

### Testing the API with Postman

1. Get the list of companies:
   - Method: GET
   - URL: `http://localhost:8000/companies`

2. Analyze a company:
   - Method: POST
   - URL: `http://localhost:8000/analyze`
   - Headers: `Content-Type: application/json`
   - Body: `{"company_name": "Tesla"}`

## Models Used

- **Sentiment Analysis**: FinBERT model (ProsusAI/finbert) for financial sentiment analysis
- **Summarization**: BART-large-CNN model (facebook/bart-large-cnn)
- **Translation**: M2M100 model (facebook/m2m100_418M) for English to Hindi translation
- **Text-to-Speech**: Facebook MMS-TTS Hindi model (facebook/mms-tts-hin)

## Assumptions & Limitations

- **News Source Limitation**: The demo uses simulated news articles instead of real-time web scraping to avoid rate limiting and inconsistent responses
- **Processing Constraints**: Limited to 10 articles per company for performance reasons
- **Language Support**: Primary analysis is performed in English, with only the summary translated to Hindi
- **Model Size**: Uses smaller, efficient models that balance performance and accuracy
- **Content Generation**: The application generates realistic-looking but synthetic news content for demonstration purposes

## Deployment

This application is deployed on Hugging Face Spaces. You can access it at: [https://huggingface.co/spaces/yourusername/news-analysis-app](https://huggingface.co/spaces/yourusername/news-analysis-app)

### Deployment Steps

1. Create a new Space on Hugging Face
2. Connect your GitHub repository
3. Add the necessary secrets for any API keys
4. Configure the build process using the requirements.txt file

## License

MIT

## Author

Your Name - [your.email@example.com](mailto:your.email@example.com)
