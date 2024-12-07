# AI-Financial Agent with gpt4o-mini🤑
### Prerequisites
Before you begin, ensure you have Python 3.10 or higher installed on your system.

### Setup and Run Locally
1. Clone the Repository:
```
https://github.com/PiSpace/ai_agent
```
2. Install Dependencies:
```
pip install -r requirements.txt
```
3. Environment Configuration:

Secure your OpenAI API key for use in the application.
- Create a .env file in the root directory.
- Add your OpenAI API API key: OpenAI_API_KEY=your_api_key_here.
4. Launch the Application:
Start the chatbot using Streamlit.
```
streamlit run run.py
```
5. Useful links
   - To get openai API: (https://platform.openai.com/api-keys)
   - To deploy your app: https://streamlit.io/
   - ML algo: (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)
   - ML algo: (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
   - OpenAI doc: (https://platform.openai.com/docs/guides/vision)
   - NLP, Sentiment Analysis: (https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)

6. Features
- AI-Driven Stock Analysis: Offers real-time stock data, technical indicators, and next-day price predictions using machine learning models.
- News Sentiment Analysis: Analyzes the sentiment of the latest news articles to gauge market sentiment and its impact on stock prices.
- Interactive Stock Comparison: Enables side-by-side comparison of multiple stocks.
- AI Financial Assistant: Provides dynamic, real-time responses to financial queries and market-related questions, with image analysis capabilities.
- Portfolio simulator: Enables users to choose various portfolio allocations and provides them with the expected return, annual volatility, and Sharpe ratio for a specified period, along with additional features such as risk analysis and optimization suggestions.
- Backtesting: Permits users to backtest diverse strategies, producing performance metrics to assess their effectiveness.
