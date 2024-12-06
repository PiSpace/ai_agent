import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
from datetime import datetime
from app.config import Config

class AIFinancialAssistant:
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def run(self):
        """Main run method for the assistant"""
        self.display_sidebar()
        self.display_chat_interface()
        self.handle_user_input()

    def display_sidebar(self):
        """Display sidebar with controls and options"""
        with st.sidebar:
            st.title("Pi - Financial Assistant")
            
            # New chat button
            if st.button("New Chat"):
                st.session_state.messages = []
                st.rerun()
            
            # Debug mode toggle
            if st.checkbox("Debug Mode"):
                st.write("API Status:", "Connected" if self.client else "Not Connected")

            # Upload image option
            uploaded_file = st.file_uploader(
                "Upload Image for Analysis",
                type=['png', 'jpg', 'jpeg']
            )
            if uploaded_file:
                self.handle_image_upload(uploaded_file)

    def display_chat_interface(self):
        """Display the main chat interface"""
        st.title("AI Financial Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "image" in message:
                    st.image(message["image"])
                st.markdown(message["content"])

    def handle_user_input(self):
        """Handle user text input"""
        if prompt := st.chat_input("Ask me anything about finance..."):
            self.add_message("user", prompt)
            self.generate_response(prompt)

    def generate_response(self, prompt):
        """Generate AI response"""
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Create messages context
                messages = [
                    {"role": "system", "content": "You are Pi, an expert financial advisor. Provide clear and practical financial advice."},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-5:]],
                    {"role": "user", "content": prompt}
                ]
                
                # Get streaming response
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                self.add_message("assistant", full_response)
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    def handle_image_upload(self, uploaded_file):
        """Handle image upload and analysis"""
        try:
            image = Image.open(uploaded_file)
            
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Add image message
            self.add_message("user", "Analyzing uploaded image...", image)
            
            # Get image analysis
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this financial chart or image. Focus on key trends and insights."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            analysis = response.choices[0].message.content
            self.add_message("assistant", analysis)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    def add_message(self, role, content, image=None):
        """Add a message to the chat history"""
        message = {"role": role, "content": content}
        if image:
            message["image"] = image
        st.session_state.messages.append(message)
