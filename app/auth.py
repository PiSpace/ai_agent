import streamlit as st
import hashlib
import json
import os

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