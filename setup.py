from setuptools import setup, find_packages

setup(
    name="ai_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "yfinance",
        "pandas",
        "numpy",
        "plotly",
        "tensorflow",
        "scikit-learn",
        "pandas-ta",
        "pillow",
        "openai"
    ]
) 