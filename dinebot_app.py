import re
import streamlit as st
from backend.src.agent.graph import graph
from backend.src.agent.utils import _print_event
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import uuid
from dotenv import load_dotenv
import langgraph 
from langgraph.pregel.io import AddableValuesDict

# Initialize unique thread ID for the session
thread_id = str(uuid.uuid4())

# Configuration dictionary
config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

# Set to track printed events
_printed = set()

# Streamlit UI
st.set_page_config(page_title="DineBot - Your Cheif on the go", page_icon="🍽️")

st.title("🍽️ Welcome to Dinebot")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2856b0;
    }
    .title {
        color: #4c79a1;
    }
    .header-text {
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("image2.png", use_column_width=True)


st.markdown("<h2 class='header-text'>Your Personal Restaurant Assistant 🤖</h2>", unsafe_allow_html=True)
st.markdown("Dinebot can assist you with cab booking, table reservations, and provide restaurant information. Ask me anything!")

# Session state for storing responses and agent state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# Continuous conversation: Create a container to display conversation
conversation_container = st.container()


def get_response(query):
    persona = "General Agent"
    # Call the agent with user input
    events = graph.stream(
        {"messages": {"General Agent": ("user", query)},
         "current_persona": persona},
        config,
        stream_mode="values"
    )
    # Process the events and display them
    for event in events:
        _print_event(event, _printed)

        # Ensure 'event' is a dictionary before accessing its contents
        if isinstance(event, dict):
            # Retrieve the 'messages' dictionary and the 'current_persona'
            messages = event.get('messages', {})
            current_persona = event.get('current_persona', 'General Agent')

            # Get the list of messages for the current persona
            persona_messages = messages.get(current_persona, [])

            # Find the most recent AIMessage with non-empty content
            recent_ai_message_content = None
            for msg in reversed(persona_messages):
                if isinstance(msg, AIMessage) and msg.content.strip():
                    recent_ai_message_content = msg.content
                    break

            # Check if a recent AI message content was found and display it
            if recent_ai_message_content:
                st.markdown(recent_ai_message_content)
                return recent_ai_message_content  # Return to append to chat history
        else:
            # Log or handle cases where 'event' is not a dictionary, if needed
            print("Unexpected event type:", type(event))

    st.markdown("No AI response available.")
    return None

  
#conversation 
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("You"):
            st.markdown(message.content)
    else:
        with st.chat_message("DineBot"):
            st.markdown(message.content)
            
# User input
user_query = st.chat_input("You:")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("You"):
        st.markdown(user_query)
    with st.chat_message("DineBot"):
        
        ai_resp = get_response(query=user_query)
        st.session_state.chat_history.append(AIMessage(ai_resp))
   