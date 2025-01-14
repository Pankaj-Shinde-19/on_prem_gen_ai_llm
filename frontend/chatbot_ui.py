import streamlit as st
import requests
from datetime import datetime

# Constants for user and bot avatars
USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🤖"

# Configure Streamlit page settings
st.set_page_config(
    page_title="On Prem Gen AI LLM",
    page_icon="C:\\Users\\PankajShinde\\PycharmProjects\\testing-on-premise-gen-ai\\frontend\\logo.png",
    layout="wide"
)

# Initialize session state variables if not already set
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = datetime.now().timestamp()
    st.session_state.messages = []  # Stores the chat history

# Load custom CSS for styling
with open('C:\\Users\\PankajShinde\\PycharmProjects\\testing-on-premise-gen-ai\\frontend\\style.css') as f:
    css = f.read()

# Apply the custom CSS
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def set_question(question):
    """Set the user query into the session state."""
    st.session_state["my_question"] = question

# Initial bot message
st.chat_message("assistant", avatar=BOT_AVATAR).write(
    "Greetings! I’m your on-premise assistant, here to help you with any questions from our knowledge base."
)

# Retrieve the current question from session state
my_question = st.session_state.get("my_question", default=None)

# Display chat history from session state
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input field
if my_question := st.chat_input("Ask me a question"):
    set_question(None)  # Clear the current question
    st.session_state.messages.append({"role": "user", "content": my_question})

    # Display the user message
    user_message = st.chat_message("user", avatar=USER_AVATAR)
    user_message.write(f"{my_question}")

    try:
        # Send the user query to the backend API
        response = requests.post(
            "http://127.0.0.1:5000/ask",
            json={"query": my_question}
        )
        response_data = response.json()

        if response.status_code == 200:
            # Successful response from the API
            assistant_response = response_data.get("response", "I couldn't generate a response for that question.")
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            # Display the bot's response
            assistant_message = st.chat_message("assistant", avatar=BOT_AVATAR)
            assistant_message.write(assistant_response)

        else:
            # API returned an error
            error_message = "Failed to retrieve data from the server."
            st.session_state.messages.append({"role": "assistant", "content": error_message})

            assistant_message = st.chat_message("assistant", avatar=BOT_AVATAR)
            assistant_message.error(error_message)
            st.write(f"Error: Unable to connect to chatbot. Status code: {response.status_code}")

    except Exception as ex:
        # Handle exceptions during API call
        error_message = "An error occurred while processing the request. Please try again later."
        st.session_state.messages.append({"role": "assistant", "content": error_message})

        assistant_message = st.chat_message("assistant", avatar=BOT_AVATAR)
        assistant_message.error(error_message)
        st.write(f"Error: {ex}")
        print(f"Error: {ex}")
