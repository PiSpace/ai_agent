import streamlit as st
import hashlib
import json
import os
import yfinance as yf

class Auth:
    def __init__(self):
        self.users_file = "users.json"
        self.load_users()

    def load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, "r") as f:
                self.users = json.load(f)
        else:
            self.users = {}

    def save_users(self):
        with open(self.users_file, "w") as f:
            json.dump(self.users, f)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def login_user(self):
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if username in self.users and self.users[username]["password"] == self.hash_password(password):
                st.session_state.user = username
                st.session_state.authenticated = True
                return True
            else:
                st.sidebar.error("Invalid username or password")
        return False

    def register_user(self):
        st.sidebar.title("Register")
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password")
        
        if st.sidebar.button("Register"):
            if new_password != confirm_password:
                st.sidebar.error("Passwords don't match")
            elif new_username in self.users:
                st.sidebar.error("Username already exists")
            else:
                self.users[new_username] = {
                    "password": self.hash_password(new_password),
                    "preferences": {},
                    "watchlist": []
                }
                self.save_users()
                st.sidebar.success("Registration successful!")

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

class AlertSystem:
    def __init__(self, username):
        self.username = username
        self.load_alerts()

    def load_alerts(self):
        with open("users.json", "r") as f:
            users = json.load(f)
            self.alerts = users[self.username].get("alerts", [])

    def add_alert(self, symbol, condition, value):
        alert = {
            "symbol": symbol,
            "condition": condition,
            "value": value,
            "active": True
        }
        self.alerts.append(alert)
        self.save_alerts()

    def check_alerts(self):
        for alert in self.alerts:
            if alert["active"]:
                stock = yf.Ticker(alert["symbol"])
                current_price = stock.info.get('currentPrice', 0)
                if self.evaluate_condition(current_price, alert["condition"], alert["value"]):
                    self.trigger_alert(alert)

    def evaluate_condition(self, current_price, condition, value):
        if condition == "above":
            return current_price > value
        elif condition == "below":
            return current_price < value
        return False

    def save_alerts(self):
        with open("users.json", "r") as f:
            users = json.load(f)
        users[self.username]["alerts"] = self.alerts
        with open("users.json", "w") as f:
            json.dump(users, f)

    def trigger_alert(self, alert):
        st.error(f"Alert triggered: {alert['symbol']} {alert['condition']} {alert['value']}")