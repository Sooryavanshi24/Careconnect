import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Retrieve the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI model with the API key
llm = OpenAI(
    openai_api_key=openai_api_key,
    temperature=0.5  # Adjust temperature for more creative responses
)

# Define a prompt template for general conversation
conversation_prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
    You are 🌍 Careconnect, a helpful assistant. The user said: {user_input}. Respond accordingly.
    """
)

# Define a more detailed prompt template for dialect understanding
detailed_prompt_template = PromptTemplate(
    input_variables=["data_description", "question"],
    template="""
    You are a health expert with expertise in understanding various dialects and mental health issues. Your task is to interpret the following data and answer the question in a manner that respects dialect nuances and also provide health-related issues and health centers to the user.
    
    Data:
    {data_description}
    
    Question:
    {question}
    
    Please consider any dialect variations in your response.
    """
)

# Create LangChains for different purposes
conversation_chain = LLMChain(llm=llm, prompt=conversation_prompt_template)
detailed_chain = LLMChain(llm=llm, prompt=detailed_prompt_template)

def get_response(data_description, question):
    # Normalize the question for dialect
    normalized_question = normalize_dialect(question)
    
    # Running the chain to get a response
    response = detailed_chain.run(data_description=data_description, question=normalized_question)
    
    return response

def normalize_dialect(input_text):
    # Example function to normalize common dialect-specific phrases
    # Extend this function as needed
    normalized_text = input_text.replace("y'all", "you all").replace("gonna", "going to")
    return normalized_text

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Streamlit application
def main():
    st.set_page_config(page_title="CareConnect - Empathetic Mental Health Companion", layout="wide")

    st.title("CareConnect")

    # Chat interface with chat_input and chat_message
    st.write("Hello! I'm 🌍 Careconnect. How can I assist you today?")

    # Display previous messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input field for chat messages
    user_input = st.chat_input("Enter your message:")

    if user_input:
        # Append user message to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Get the response from LangChain for a general conversation
            bot_response = conversation_chain.run(user_input=user_input)

            # Append assistant's response to the session state
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # Display the assistant's response
            with st.chat_message("assistant"):
                st.write(bot_response)

        except Exception as e:
            st.write(f"An error occurred: {e}")

    # Input fields for detailed analysis
    st.sidebar.title("Health and Dialect Analysis")
    data_description = st.sidebar.text_area("Enter Data Description", "Data includes various facts about Health, Mental health challenges, and Health centers in St.Kitts.")
    question = st.sidebar.text_input("Enter Your Question")
    
    # Submit button
    if st.sidebar.button("Get Detailed Response"):
        if question:
            answer = get_response(data_description, question)
            # Save the question and answer to the session state
            st.session_state['messages'].append(f"*You:* {question}")
            st.session_state['messages'].append(f"*CareConnect:* {answer}")
        else:
            st.warning("Please enter a question.")

    # Footer
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 336px; /* Adjust this value to match the width of your sidebar */
            width: calc(100% - 336px); /* Adjust this value to match the width of your sidebar */
            text-align: center; /* Center text within the available width */
            color: grey;
            padding: 10px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
            background: white; /* Ensure footer background is white to match the page background */
        }
        </style>
        <div class="footer">
            Created by Code Brawlers. This chatbot does not replace human interaction. Seek help from nearby health centres.
        </div>
        """,
        unsafe_allow_html=True
    )
    
if __name__ == "__main__":
    main()

    
if __name__ == "__main__":
    main()

