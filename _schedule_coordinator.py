# Copyright (c) 2024 Tongyang Systems.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import streamlit as st
import datetime
import uuid
import pandas as pd
from typing import List, Dict, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import time

# Load the dataframe
df = pd.read_csv('./data_schedule/simulation_3-Table 1.csv')

# Initialize Groq LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)


# Create pandas dataframe agent
pandas_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)

# Simulated email configuration
OUTBOX_DIR = "simulated_outbox"
INBOX_DIR = "simulated_inbox"
os.makedirs(OUTBOX_DIR, exist_ok=True)
os.makedirs(INBOX_DIR, exist_ok=True)

# Define the state
class State(TypedDict):
    question: str
    answer: str
    raw_data: str
    next: str
    email_sent: bool
    email_responses: List[str]
    sent_emails: List[Dict[str, str]] 

# Function to send simulated email
def send_simulated_email(recipient, subject, body, attachment=None):
    email_content = {
        "to": recipient,
        "subject": subject,
        "body": body,
        "attachment": attachment.to_dict() if attachment is not None else None
    }
    filename = f"{OUTBOX_DIR}/{time.time()}-{uuid.uuid4()}.json"
    with open(filename, 'w') as f:
        json.dump(email_content, f)
    return email_content


# Function to receive simulated email
def receive_simulated_email():
    responses = []
    for filename in os.listdir(INBOX_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(INBOX_DIR, filename), 'r') as f:
                email_content = json.load(f)
                responses.append(email_content['body'])
    return responses

# Define the nodes
def send_emails(state: State) -> State:
    recipients = pd.read_csv('email_recipients.csv')
    sent_emails = []
    for _, recipient in recipients.iterrows():
        subject = f"Schedule Data Review Request - {recipient['department']}"
        body = f"""
From: CHERRY Shipping Line
To: {recipient['name']}
Subject: Urgent: Revised Coastal Schedule - Delays and Recovery Plan

Dear {recipient['name']},

I hope this email finds you well. As our valued partner and {recipient['role']} in the {recipient['department']} department, we are writing to inform you about recent developments affecting our coastal schedule.

Please find attached the revised coastal schedule for your immediate review and attention.

Key updates:
1. Delay in Singapore: We experienced operational delays during our port call in Singapore.
2. Engine Trouble: We encountered engine issues in the Dubai-Suez segment, causing additional delays.
3. Recovery Plan: We have implemented a maximum speed increase to mitigate these delays.
4. Expected Outcome: Despite the setbacks, we anticipate an on-time arrival in MARSEILLE.

Points for your consideration:
1. Updated vessel arrival times at each port
2. Revised berth allocation, if applicable
3. Any potential conflicts with other scheduled vessels
4. Impact on cargo operations and connections

We kindly request your understanding regarding these unforeseen circumstances. Our team has been working diligently to minimize the impact on our overall schedule and to ensure the least possible disruption to our shared operations.

Action Required:
Please review the attached revised schedule and provide your feedback or any concerns at your earliest convenience. If you foresee any issues or require additional information, please don't hesitate to inform us immediately.

We value our partnership and appreciate your cooperation in these challenging situations. Should you have any questions or need further clarification, please feel free to reach out to us.

Thank you for your prompt attention to this matter and for your continued support.

Best regards,
Cherry Vessel Operation Team
CHERRY Shipping Line

Attachment: Revised_Coastal_Schedule.pdf
"""
        email_content = send_simulated_email(recipient['email'], subject, body, df)
        sent_emails.append(email_content)
    state['email_sent'] = True
    state['sent_emails'] = sent_emails
    return state


def wait_for_responses(state: State) -> State:
    time.sleep(1)  # Wait for 10 seconds in the simulation
    responses = receive_simulated_email()
    state['email_responses'] = responses
    return state


def summarize_responses(state: State) -> State:
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following email responses regarding the schedule data:\n{responses}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"responses": "\n".join(state['email_responses'])})
    state['answer'] = summary
    return state


# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("send_emails", send_emails)
workflow.add_node("wait_for_responses", wait_for_responses)
workflow.add_node("summarize_responses", summarize_responses)

# Add edges
workflow.set_entry_point("send_emails")
workflow.add_edge("send_emails", "wait_for_responses")
workflow.add_edge("wait_for_responses", "summarize_responses")
workflow.add_edge("summarize_responses", END)

# Compile the graph
app = workflow.compile()

# Function for processing main workflows
def schedule_coordinator(prompt: str):
    response = app.invoke({
        "question": prompt,
        "answer": "",
        "raw_data": "",
        "next": ""
    })
    return response['answer']

def get_inbox_contents():
    inbox_contents = []
    for filename in os.listdir(INBOX_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(INBOX_DIR, filename), 'r') as f:
                email_content = json.load(f)
                inbox_contents.append(email_content['body'])
    return "\n\n".join(inbox_contents)

# LLM을 사용하여 inbox 내용을 기반으로 답변을 생성하는 체인을 만듭니다
inbox_prompt = ChatPromptTemplate.from_template(
    "Given the following email responses:\n\n{inbox_contents}\n\nPlease answer the following question: {question}"
)

# 체인 생성
inbox_chain = inbox_prompt | llm | StrOutputParser()




# Function for processing streamlit questions
def run_streamlit():

   # Select columns to display
    columns_to_display = ['VESSEL', 'VOYAGE', 'PORT', 
                          'LONG_TERM_SCHEDULE_ETA', 'SIMULATED_SCHEDULE_ETA', 'DELAY_ETA']

    # Display the dataframe as a table with selected columns
    st.markdown("#### Simulated Schedule (Selected Columns)")
    st.dataframe(df[columns_to_display], height=300)

    # Add a button to show all columns
    if st.button("Show All Columns"):
        st.markdown("#### Full Schedule Data")
        st.dataframe(df, height=300)


    # Add a button to trigger the email workflow
    if st.button("Send Schedule and Get Feedback (Simulation)"):
        with st.spinner("Simulating email sending and waiting for responses..."):
            result = app.invoke({
                "question": "Summarize feedback on schedule data",
                "answer": "",
                "raw_data": "",
                "next": "",
                "email_sent": False,
                "email_responses": [],
                "sent_emails": []  # 초기 상태에 추가
            })
        st.success("Simulation completed. Here's the summary of responses:")
        st.write(result['answer'])
        
        # Display sent emails
        st.subheader("Sent Emails")
        for email in result['sent_emails']:
            with st.expander(f"Email to: {email['to']}"):
                st.write(f"Subject: {email['subject']}")
                st.write("Body:")
                st.text(email['body'])
        
        # Display DataFrame
        st.subheader("Schedule Data (First 10 rows)")
        st.dataframe(df.head(10))

    # Add a button to simulate receiving responses
    if st.button("Simulate Receiving Responses"):
        # Simulated responses
        simulated_responses = [
        '''From: OLIVE Shipping Line
            Subject: Re: Revised Coastal Schedule

            Dear Valued Partner,

            Thank you for your email regarding the revised coastal schedule. We acknowledge receipt and have reviewed the changes.

            However, we must bring to your attention that according to our Vessel Sharing Agreement (VSA), we are entitled to compensation from the slot cost for this voyage due to the 24 hour delays.

            We kindly request that you check this compensation as per our VSA. Please let us know if you need any further information or clarification on this matter.

            Best regards,
            OLIVE Shipping Line''',
        '''From: PROTECT Shipping Line
            Subject: Re: Revised Schedule Confirmation

            Dear Team,

            Thank you for your email containing the revised schedule. We have carefully reviewed the changes and can confirm that the updated schedule is acceptable to us.

            We will proceed with arranging our outbound containers in accordance with this new schedule. If there are any further changes or updates, please inform us promptly.

            Thank you for your continued cooperation.

            Best regards,
            PROTECT Shipping Line''',
        '''From: GUARDIAN Shipping Line
            Subject: Re: Updated Schedule and Outbound Container Preparation

            Dear Colleagues,

            We appreciate your email detailing the updated schedule. Thank you for your diligent efforts in managing and minimizing delays under these challenging circumstances.

            In light of the revised schedule, we are now preparing our outbound containers for DUBAI and SUEZ. We will ensure that all necessary arrangements are made to align with the new timelines.

            Please don't hesitate to contact us if there are any additional changes or if you require any further information from our end.

            Thank you for your continued support and collaboration.

            Best regards,
            GUARDIAN Shipping Line'''
        ]
        for response in simulated_responses:
            filename = f"{INBOX_DIR}/{time.time()}-{uuid.uuid4()}.json"
            with open(filename, 'w') as f:
                json.dump({"body": response}, f)
        st.success("Simulated responses have been added to the inbox.")

    # Initialize session state for schedule messages
    if "schedule_messages" not in st.session_state:
        st.session_state.schedule_messages = []

    # Display chat messages
    for message in st.session_state.schedule_messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}\n\n<div style='font-size:0.8em; color:#888;'>{message['timestamp']}</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Any question about the received responses?")

    if prompt:
        # 사용자 메시지 추가 및 표시 (이 부분은 그대로 유지)
        user_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.schedule_messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        
        with st.chat_message("user"):
            st.markdown(f"{prompt}\n\n<div style='font-size:0.8em; color:#888;'>{user_timestamp}</div>", unsafe_allow_html=True)
        
        # AI 응답 생성
        with st.spinner("Thinking..."):
            inbox_contents = get_inbox_contents()
            response = inbox_chain.invoke({"inbox_contents": inbox_contents, "question": prompt})
            ai_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # AI 응답 추가 및 표시
        st.session_state.schedule_messages.append({"role": "assistant", "content": response, "timestamp": ai_timestamp})
        with st.chat_message("assistant"):
            st.markdown(f"{response}\n\n<div style='font-size:0.8em; color:#888;'>{ai_timestamp}</div>", unsafe_allow_html=True)
        
        st.rerun()

# Run the main functions
def run():
    return schedule_coordinator

# Run the streamlit function
if __name__ == "__main__":
    run_streamlit()