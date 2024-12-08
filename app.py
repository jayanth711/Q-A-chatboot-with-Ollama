import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracking setup
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] ="true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with OLLAMA"

# Prompt template setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries."),
        ("user","Question:{question}")
    ]
)

def generate_response(question, engine,temperature, max_tokens):
    
    # Initialize the LLM with the chosen parameters
    llm = Ollama(model=engine)
    
    # Define the output parser and chain
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Generate answer
    answer = chain.invoke({"question": question})
    return answer


# Dropdown for model selection
engine = st.sidebar.selectbox("Select an OpenAI model", ["mistral", "gemma2","phi3"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

# Generate and display response if user_input is provided
if user_input:
    response = generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("Provide a query to get a response.")  
