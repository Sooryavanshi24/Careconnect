from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

# Set your OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the LLM
llm = OpenAI(api_key=api_key)

# Define the prompt template for conversation using a simpler PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
    You are 🌍 Careconnect, a helpful assistant. The user said: {user_input}. Respond accordingly.
    """
)

# Create a LangChain for conversation
conversation_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit app layout
st.title("🌍 Careconnect Chatbot")
st.write("Hello! I'm 🌍 Careconnect. How can I assist you today?")

# Chat interface with chat_input and chat_message
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store messages in session state

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input field for chat messages
user_input = st.chat_input("Enter your message:")

if user_input:
    # Append user message to the session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # Get the response from LangChain
        bot_response = conversation_chain.run(user_input=user_input)

        # Append assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.write(bot_response)

    except Exception as e:
        st.write(f"An error occurred: {e}")

