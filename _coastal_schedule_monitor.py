# Copyright (c) 2024 Tongyang Systems.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import streamlit as st
import datetime
import uuid
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, TypedDict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType

# Load the dataframe
coastal_schedule = pd.read_csv('./data_schedule/coastal_schedule-Table 1.csv')

# Initialize Groq LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Create pandas dataframe agent
if isinstance(llm, ChatOpenAI):
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        coastal_schedule,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
else:
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        coastal_schedule,
        verbose=True,
        allow_dangerous_code=True
    )

# Define the prompt templates
analysis_prompt = """ You are a container vessel schedule expert with deep knowledge of maritime operations. You have access to a dataframe containing information about vessel voyages, schedules, and performance metrics.

Use the following information to answer the user's question in detail:
1. The user's question is provided in the {question} placeholder. Make sure to address this question directly in your response.
2. The pandas agent has performed calculations or data manipulations based on the question. The results are provided in the {pandas_response} placeholder. Use this information to support your analysis.
3. If the pandas agent encountered an error, explain what might have caused it and suggest alternative approaches.
4. Provide clear and concise answers, using bullet points where appropriate.
5. If the question is unclear, ask for clarification.
6. If the answer requires comparing multiple voyages or summarizing trends, make sure to highlight key findings.
7. Include relevant statistics or metrics from the dataframe to support your answer.
8. If you don't know the answer or if the required data is not in the dataframe, say so honestly.
9. If {question} is delated to DELAY, check If DELAY_ETA is greater than 0.

Dataframe column descriptions:
SERVICE: service where the vessel is deployed
VESSEL: vessel full name
VOYAGE: voyage number with direction code
PORT: calling port name
TERMINAL: berthing terminal in the calling port
LONG_TERM_SCHEDULE_ETA: estimated time of arrival per long term schedule
LONG_TERM_SCHEDULE_ETB: estimated time of berth per long term schedule
LONG_TERM_SCHEDULE_ETD: estimated time of departure per long term schedule
COASTAL_SCHEDULE_ETA: estimated time of arrival per coastal schedule
COASTAL_SCHEDULE_ETB: estimated time of berth per coastal schedule
COASTAL_SCHEDULE_ETD: estimated time of departure per coastal schedule
DELAY_ETA: delays in arrival by the coastal schedule against long term schedule
DELAY_ETB: delays in berth by the coastal schedule against long term schedule
DELAY_ETD: delays in departure by the coastal schedule against long term schedule
ZD: zone description
PILOT_IN: pilot in time between arrival to berth
PILOT_OUT: pilot out time between unberth to pilot left
BERTH_HOUR: berthing hours between berthing to unberthing
DISTANCE: distance in nautical miles between calling port to next calling port
SEA_SPEED: speed in knots for sailing between calling port to next calling port

Question: {question}

Pandas Agent Response: {pandas_response}

Answer: """


refinement_prompt = """
You are a maritime logistics expert tasked with providing clear, accurate, and contextually relevant summaries of vessel schedule data.
Review the following analysis and raw data, then provide a refined and coherent response to the original question.

Guidelines:
1. Directly address the original question without adding unnecessary speculation.
2. If the analysis or raw data is insufficient to fully answer the question, clearly state what information is missing or unclear.
3. If the pandas agent encountered an error, explain the implications and suggest alternative ways to approach the question.
4. Provide numerical answers with appropriate units and context when applicable.
5. If the data reveals any patterns or notable insights, briefly mention them.
6. Keep your response concise but informative, typically within 3-5 sentences.
7. If the question asks for an average, include the exact calculated value and round it to two decimal places for clarity.
8. Mention any data cleaning or preprocessing steps if they were necessary to arrive at the answer.

Original Question: {question}
(This is the question asked by the user. Your refined answer should directly address this question.)

Analysis: {analysis}
(This section contains the results of the initial analysis. Use this information to inform your refined answer.)

Raw Data: {raw_data}
(This section provides relevant raw data or error messages. Use this to verify or supplement the analysis if necessary.)

Refined Answer:
(Provide your refined answer here, following the guidelines above and using the information from the question, analysis, and raw data.)
"""

# Create the prompts
analysis_prompt = ChatPromptTemplate.from_template(analysis_prompt)
refinement_prompt = ChatPromptTemplate.from_template(refinement_prompt)


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
def route_question(state: State) -> State:
    question = state['question'].lower()
    if any(keyword in question for keyword in ['calculate', 'average', 'sum', 'mean', 'median', 'max', 'min']):
        state['next'] = "use_dataframe_tool"
    else:
        state['next'] = "use_analysis_chain"
    return state

def use_dataframe_tool(state: State) -> State:
    try:
        result = pandas_agent.run(state['question'])
        state['raw_data'] = result
    except Exception as e:
        state['raw_data'] = f"Error occurred: {str(e)}"
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
workflow.add_node("route_question", route_question)
workflow.add_node("use_dataframe_tool", use_dataframe_tool)
workflow.add_node("use_analysis_chain", use_analysis_chain)
workflow.add_node("refine_result", refine_result)

# Add edges
workflow.set_entry_point("route_question")
workflow.add_conditional_edges(
    "route_question",
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

# Function for processing main workflows
def coastal_monitor(prompt: str):
    response = app.invoke({
        "question": prompt,
        # "answer": "",
        # "raw_data": "",
        # "next": ""
    })
    return response['answer']

# Function for processing streamlit questions
def run_streamlit():
    # Select columns to display
    columns_to_display = ['VESSEL', 'VOYAGE', 'PORT', 
                          'LONG_TERM_SCHEDULE_ETA', 'COASTAL_SCHEDULE_ETA', 'DELAY_ETA']

    # Display the dataframe as a table with selected columns
    st.markdown("#### Coastal Schedule (Selected Columns)")
    st.dataframe(coastal_schedule[columns_to_display], height=300)

    # Add a button to show all columns
    if st.button("Show All Columns"):
        st.markdown("#### Full Schedule Data")
        st.dataframe(coastal_schedule, height=300)

    # Initialize session state for schedule messages
    if "schedule_messages" not in st.session_state:
        st.session_state.schedule_messages = []

    # Display chat messages
    for message in st.session_state.schedule_messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}\n\n<div style='font-size:0.8em; color:#888;'>{message['timestamp']}</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Any question about schedule delays?")

    if prompt:
        # Add user message
        user_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.schedule_messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"{prompt}\n\n<div style='font-size:0.8em; color:#888;'>{user_timestamp}</div>", unsafe_allow_html=True)
        
        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = coastal_monitor(prompt)
            ai_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add and display AI response
        st.session_state.schedule_messages.append({"role": "assistant", "content": ai_response, "timestamp": ai_timestamp})
        with st.chat_message("assistant"):
            st.markdown(f"{ai_response}\n\n<div style='font-size:0.8em; color:#888;'>{ai_timestamp}</div>", unsafe_allow_html=True)
        
        st.rerun()

# Run the main functions
def run():
    return coastal_monitor

# Run the streamlit function
if __name__ == "__main__":
    run_streamlit()