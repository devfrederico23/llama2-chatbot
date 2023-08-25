import streamlit as st
import os
from utils import debounce_replicate_run

# External libraries
import replicate

# Set initial page configuration
st.set_page_config(
    page_title="LLaMA2Chat",
    page_icon=":volleyball:",
    layout="wide"
)

# Global variables
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', default='')

# Define model endpoints as independent variables
LLaMA2_7B_ENDPOINT = os.environ.get('MODEL_ENDPOINT70B', default='')
LLaMA2_13B_ENDPOINT = os.environ.get('MODEL_ENDPOINT70B', default='')
LLaMA2_70B_ENDPOINT = os.environ.get('MODEL_ENDPOINT70B', default='')

PRE_PROMPT = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."

def render_app():
    # Set up containers
    response_container = st.container()
    main_container = st.container()

    # Set up Session State variables
    st.session_state.setdefault('chat_dialogue', [])
    
    # Use the appropriate model endpoint based on user selection
    selected_model = st.sidebar.selectbox('Choose a LLaMA2 model:', ['LLaMA2-70B', 'LLaMA2-13B', 'LLaMA2-7B'], key='model')
    if selected_model == 'LLaMA2-7B':
        llm_endpoint = LLaMA2_7B_ENDPOINT
    elif selected_model == 'LLaMA2-13B':
        llm_endpoint = LLaMA2_13B_ENDPOINT
    else:
        llm_endpoint = LLaMA2_70B_ENDPOINT
    
    st.session_state.setdefault('llm', llm_endpoint)
    st.session_state.setdefault('temperature', 0.1)
    st.session_state.setdefault('top_p', 0.9)
    st.session_state.setdefault('max_seq_len', 512)
    st.session_state.setdefault('pre_prompt', PRE_PROMPT)
    st.session_state.setdefault('string_dialogue', '')

     # Set up left sidebar
    st.sidebar.header("LLaMA2 Chatbot")
    
    # Container for the chat history
    response_container = st.container()
    
    # Container for the user's text input
    user_input_container = st.container()
    
    # Set up/Initialize Session State variables:
    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = []

    # Model hyperparameters:
    st.session_state['temperature'] = st.sidebar.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    st.session_state['top_p'] = st.sidebar.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state['max_seq_len'] = st.sidebar.slider('Max Sequence Length:', min_value=64, max_value=4096, value=2048, step=8)

    new_prompt = st.sidebar.text_area('Prompt before the chat starts. Edit here if desired:', PRE_PROMPT, height=60)
    if new_prompt != PRE_PROMPT and new_prompt != "" and new_prompt != None:
        st.session_state['pre_prompt'] = new_prompt + "\n\n"
    else:
        st.session_state['pre_prompt'] = PRE_PROMPT

    btn_col1, btn_col2 = st.sidebar.columns(2)

    # Display chat history
    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Type your question here to talk to LLaMA2"):
        st.session_state.chat_dialogue.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            string_dialogue = st.session_state['pre_prompt']
            for dict_message in st.session_state.chat_dialogue:
                speaker = "User" if dict_message["role"] == "user" else "Assistant"
                string_dialogue += f"{speaker}: {dict_message['content']}\n\n"
            output = debounce_replicate_run(
                st.session_state['llm'],
                string_dialogue + "Assistant: ",
                st.session_state['max_seq_len'],
                st.session_state['temperature'],
                st.session_state['top_p'],
                REPLICATE_API_TOKEN
            )
            for item in output:
                full_response += item
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.chat_dialogue.append({"role": "assistant", "content": full_response})

def main():
    render_app()

if __name__ == "__main__":
    main()
