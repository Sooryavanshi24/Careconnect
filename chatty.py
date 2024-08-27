from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

# Set your OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the LLM
llm = OpenAI(api_key=api_key)

# Define the prompt template for conversation with memory
def generate_prompt(messages):
    conversation_history = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
    return f"""
    You are 🌍 Careconnect, a helpful assistant. Here is the conversation so far:
    {conversation_history}
    The user said: {messages[-1]['content']}
    Respond accordingly.
    """

# Create a LangChain for conversation with memory
def get_conversation_chain():
    prompt_template = PromptTemplate(
        input_variables=["messages"],
        template=generate_prompt
    )
    return LLMChain(llm=llm, prompt=prompt_template)

conversation_chain = get_conversation_chain()

# Streamlit app layout
st.title("🌍 Careconnect Chatbot")
st.write("Hello! I'm 🌍 Careconnect. How can I assist you today?")

# Initialize or reset the conversation history if the session state is empty
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
        # Get the response from LangChain with conversation context
        bot_response = conversation_chain.run(messages=st.session_state.messages)

        # Append assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.write(bot_response)

    except Exception as e:
        st.write(f"An error occurred: {e}")

    # Optionally limit the memory size to prevent excessive prompt length
    max_memory_length = 20  # Limit the number of messages to remember
    if len(st.session_state.messages) > max_memory_length:
        st.session_state.messages = st.session_state.messages[-max_memory_length:]



