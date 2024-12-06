import streamlit as st
import json
import yfinance as yf

class UserPreferences:
    def __init__(self, username):
        self.username = username
        self.load_preferences()

    def load_preferences(self):
        with open("users.json", "r") as f:
            users = json.load(f)
            self.preferences = users[self.username]["preferences"]
            self.watchlist = users[self.username]["watchlist"]

    def save_preferences(self):
        with open("users.json", "r") as f:
            users = json.load(f)
        users[self.username]["preferences"] = self.preferences
        users[self.username]["watchlist"] = self.watchlist
        with open("users.json", "w") as f:
            json.dump(users, f)

    def display_watchlist(self):
        st.sidebar.title("Watchlist")
        for symbol in self.watchlist:
            stock = yf.Ticker(symbol)
            info = stock.info
            with st.sidebar.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.write(symbol)
                with col2:
                    st.write(f"${info.get('currentPrice', 'N/A')}") 