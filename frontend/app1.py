import streamlit as st
import pandas as pd
from datetime import datetime
import random

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_case' not in st.session_state:
        st.session_state.current_case = None
    if 'cases' not in st.session_state:
        st.session_state.cases = []

def generate_case_number():
    return f"CASE-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

def create_new_case(case_type, description):
    case = {
        'case_number': generate_case_number(),
        'type': case_type,
        'status': 'New',
        'assigned_agent': 'Unassigned',
        'description': description,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.cases.append(case)
    return case

def main():
    st.set_page_config(page_title="Multi-Agent Support System", layout="wide")
    init_session_state()

    # Sidebar for case management
    with st.sidebar:
        st.title("Case Management")
        
        # New Case Creation
        st.subheader("Create New Case")
        case_type = st.selectbox(
            "Case Type",
            ["Deposit Operations", "Investment Operations", "General Inquiry"]
        )
        case_description = st.text_area("Case Description")
        if st.button("Create Case"):
            new_case = create_new_case(case_type, case_description)
            st.session_state.current_case = new_case
            st.success(f"Created case: {new_case['case_number']}")

        # Case List
        st.subheader("Active Cases")
        for case in st.session_state.cases:
            if st.button(
                f"{case['case_number']} - {case['type']}",
                key=case['case_number']
            ):
                st.session_state.current_case = case

    # Main chat interface
    st.title("Multi-Agent Support System")

    # Current Case Information
    if st.session_state.current_case:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Case Number", st.session_state.current_case['case_number'])
        with col2:
            st.metric("Status", st.session_state.current_case['status'])
        with col3:
            st.metric("Assigned Agent", st.session_state.current_case['assigned_agent'])

        # Agent Selection
        agent_type = st.selectbox(
            "Select Agent Type",
            [
                "Deposit Ops Agent",
                "Invest Ops Agent",
                "Deposit Supervisor",
                "General Agent"
            ]
        )

        # Chat Interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Simulate agent response (replace with actual agent logic)
            response = f"[{agent_type}] Response to: {prompt}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            # Update case status (simplified)
            st.session_state.current_case['status'] = 'In Progress'
            st.session_state.current_case['assigned_agent'] = agent_type

        # Case Actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Mark as Resolved"):
                st.session_state.current_case['status'] = 'Resolved'
        with col2:
            if st.button("Escalate Case"):
                st.session_state.current_case['status'] = 'Escalated'
        with col3:
            if st.button("Reset Chat"):
                st.session_state.messages = []

    else:
        st.info("Please select or create a case to start chatting")

    # Display all cases in a table
    st.subheader("Case Overview")
    if st.session_state.cases:
        df = pd.DataFrame(st.session_state.cases)
        st.dataframe(df, hide_index=True)

if __name__ == "__main__":
    main()