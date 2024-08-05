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
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType

# Load the dataframe
df = pd.read_csv('./data_credit/chapter_3_credit_checker.csv')

# Initialize Groq LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)


# Create pandas dataframe agent
if isinstance(llm, ChatOpenAI):
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
else:
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

# Define the prompt templates
analysis_prompt_template = """
You are a ship captain's credit officer with deep knowledge of maritime logistics and financial operations.
You have access to a comprehensive database containing information about worldwide maritime trends, vessel credit, customer payments, and financial transactions.
 
Use the following information to answer the user's question in detail:
 
1. The user's question is provided in the {question} placeholder below. Make sure to address this question directly in your response.
2. Analyze the database and provide insights based on the question.
3. The pandas agent has performed calculations or data manipulations based on the question. The results are provided in the {pandas_response} placeholder below. Use this information to support your analysis.
4. Provide clear and concise answers, using bullet points where appropriate.
5. If the question is unclear, ask for clarification.
6. If the answer requires comparing multiple transactions or summarizing trends, make sure to highlight key findings.
7. Include relevant statistics or metrics from the database to support your answer.
8. If asked about specific columns, refer to the column descriptions provided.
9. If you don't know the answer or if the required data is not in the database, say so honestly.
10. Refer to the dataframe column names and descriptions provided below.
 
Dataframe column descriptions:
SERVICE: service where the vessel is deployed
VESSEL: vessel full name
VOYAGE: voyage number with direction code
PORT: calling port name
TERMINAL: berthing terminal in the calling port
CREDIT_LIMIT: credit limit for the customer
CURRENT_BALANCE: current balance for the customer
PAYMENT_DUE_DATE: payment due date for the current transaction
PAYMENT_RECEIVED: amount of payment received for the current transaction
OUTSTANDING_BALANCE: outstanding balance for the customer
WORLDWIDE_TRADE_INDEX: worldwide trade index (e.g., Dow Jones Industrial Average, S&P 500)
WORLDWIDE_SHIPPING_INDEX: worldwide shipping index (e.g., Global Container Index, Freightos)
WORLDWIDE_CRUDE_OIL_INDEX: worldwide crude oil index (e.g., WTI Crude Oil, Brent Crude Oil)
WORLDWIDE_GOLD_INDEX: worldwide gold index (e.g., London Gold Fix, New York Gold Fix)
WORLDWIDE_INFLATION_RATE: worldwide inflation rate (e.g., Consumer Price Index, GDP Deflator)

Question: {question}

Pandas Agent Response: {pandas_response}

Answer:
[Provide your detailed answer here, addressing the question directly and using the information from the pandas agent response to support your analysis.]
"""

refinement_prompt_template = """
You are a ship captain's credit officer with deep knowledge of maritime logistics and financial operations.
Review the following analysis and raw data, then provide a refined and coherent response to the original question.

Guidelines:
1. Directly address the original question without adding unnecessary speculation.
2. If the analysis or raw data is insufficient to fully answer the question, clearly state what information is missing or unclear.
3. Provide numerical answers with appropriate units and context when applicable.
4. If the data reveals any patterns or notable insights, briefly mention them.
5. Keep your response concise but informative, typically within 3-5 sentences.
6. If the question asks for an average, include the exact calculated value and round it to two decimal places for clarity.
7. Mention any data cleaning or preprocessing steps if they were necessary to arrive at the answer.

Original Question: {question}
(This is the question asked by the user. Your refined answer should directly address this question.)

Analysis: {analysis}
(This section contains the results of the initial analysis. Use this information to inform your refined answer.)

Raw Data: {raw_data}
(This section provides relevant raw data. Use this to verify or supplement the analysis if necessary.)

Refined Answer:
(Provide your refined answer here, following the guidelines above and using the information from the question, analysis, and raw data. Ensure that your response directly addresses the original question, incorporates insights from the analysis, and is supported by the raw data where applicable.)
"""

# Create the prompts
analysis_prompt = ChatPromptTemplate.from_template(analysis_prompt_template)
refinement_prompt = ChatPromptTemplate.from_template(refinement_prompt_template)

# Create the chains
analysis_chain = analysis_prompt | llm | StrOutputParser()
refinement_chain = refinement_prompt | llm | StrOutputParser()

# Define the state
class State(TypedDict):
    question: str
    answer: str
    raw_data: str
    next: str

# Define the nodes
def credit_question(state: State) -> State:
    question = state['question'].lower()
    if any(keyword in question for keyword in ['calculate', 'average', 'sum', 'mean', 'median', 'max', 'min']):
        state['next'] = "use_dataframe_tool"
    else:
        state['next'] = "use_analysis_chain"
    return state

def use_dataframe_tool(state: State) -> State:
    result = pandas_agent.run(state['question'])
    state['raw_data'] = result
    state['next'] = "refine_result"
    return state

def use_analysis_chain(state: State) -> State:
    pandas_response = pandas_agent.run(state['question'])
    result = analysis_chain.invoke({
        "question": state['question'],
        "pandas_response": pandas_response
    })
    state['raw_data'] = pandas_response
    state['answer'] = result
    state['next'] = "refine_result"
    return state

def refine_result(state: State) -> State:
    refined_answer = refinement_chain.invoke({
        "question": state['question'],
        "analysis": state['answer'],
        "raw_data": state['raw_data']
    })
    state['answer'] = refined_answer
    return state

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("credit_question", credit_question)
workflow.add_node("use_dataframe_tool", use_dataframe_tool)
workflow.add_node("use_analysis_chain", use_analysis_chain)
workflow.add_node("refine_result", refine_result)

# Add edges
workflow.set_entry_point("credit_question")
workflow.add_conditional_edges(
    "credit_question",
    lambda state: state['next'],
    {
        "use_dataframe_tool": "use_dataframe_tool",
        "use_analysis_chain": "use_analysis_chain"
    }
)
workflow.add_edge("use_dataframe_tool", "refine_result")
workflow.add_edge("use_analysis_chain", "refine_result")
workflow.add_edge("refine_result", END)

# Compile the graph
app = workflow.compile()

def run():

    st.dataframe(df, height=300)

    # Initialize session state for schedule messages
    if "booking_messages" not in st.session_state:
        st.session_state.booking_messages = []

    # Display chat messages
    for message in st.session_state.booking_messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}\n\n<div style='font-size:0.8em; color:#888;'>{message['timestamp']}</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Any question about credit?")

    if prompt:
        # Add user message
        user_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.booking_messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"{prompt}\n\n<div style='font-size:0.8em; color:#888;'>{user_timestamp}</div>", unsafe_allow_html=True)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = app.invoke({
                "question": prompt,
                "answer": "",
                "raw_data": "",
                "next": ""
            })
            ai_response = response['answer']
            ai_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add and display AI response
        st.session_state.booking_messages.append({"role": "assistant", "content": ai_response, "timestamp": ai_timestamp})
        with st.chat_message("assistant"):
            st.markdown(f"{ai_response}\n\n<div style='font-size:0.8em; color:#888;'>{ai_timestamp}</div>", unsafe_allow_html=True)
        
        st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    run()