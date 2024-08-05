import os
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from langgraph.graph import Graph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load the dataframe
coastal_schedule = pd.read_csv('./data_schedule/coastal_schedule-Table 1.csv')
simulation = pd.read_csv('./data_schedule/simulation_3-Table 1.csv')

# Initialize Groq LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)


# State definition
class State(TypedDict):
    coastal_schedule: pd.DataFrame
    simulation_result: pd.DataFrame
    current_port: str
    delays: List[dict]
    has_delays: bool
    simulation_request: Dict
    original_delays: str
    simulation_results: str
    anlaysis_result: str

# Pydantic model for structured simulation request output
class SimulationRequest(BaseModel):
    focus_area: str = Field(description="The main area of focus for the simulation and analysis")
    specific_ports: List[str] = Field(description="List of specific ports to focus on, if any")
    metrics_of_interest: List[str] = Field(description="Metrics to pay special attention to in the analysis")
    additional_context: str = Field(description="Any additional context or requirements for the simulation")

# Data loading function
def load_data(file_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        print(f"Current working directory: {os.getcwd()}")
        raise

# Delay detection function
# def check_delays(state: State) -> State:
#     delays = []
#     for _, row in state['coastal_schedule'].iterrows():
#         eta_delay = pd.Timedelta(row['DELAY_ETA'])
#         if eta_delay > timedelta(hours=0):
#             delays.append({
#                 'port': row['PORT'],
#                 'delay': eta_delay,
#                 'original_eta': row['LONG_TERM_SCHEDULE_ETA'],
#                 'new_eta': row['COASTAL_SCHEDULE_ETA']
#             })
#     state['delays'] = delays
#     state['has_delays'] = len(delays) > 0
#     return state

def check_delays(state: State) -> State:
    delays = []
    for _, row in state['coastal_schedule'].iterrows():
        eta_delay = pd.Timedelta(row['DELAY_ETA'])
        if eta_delay > timedelta(hours=0):
            delays.append({
                'port': row['PORT'],
                'delay': eta_delay,
                'original_eta': row['LONG_TERM_SCHEDULE_ETA'],
                'new_eta': row['COASTAL_SCHEDULE_ETA']
            })
    state['delays'] = delays
    state['has_delays'] = len(delays) > 0
    return state



# Date processing function
def process_date(date_str, time_saved):
    try:
        date = pd.to_datetime(date_str)
        if pd.isna(date):
            return ''
        return (date - time_saved).strftime('%Y%m%d %H:%M')
    except ValueError:
        return date_str  # Return original value if not a date format

# Simulation function
def run_simulation(state: State) -> State:
    if not state['has_delays']:
        state['simulation_result'] = state['coastal_schedule'].copy()
        return state

    simulation_df = state['coastal_schedule'].copy()
    suez_index = simulation_df[simulation_df['PORT'] == 'SUEZ'].index[0]
    marseille_index = simulation_df[simulation_df['PORT'] == 'MARSEILLE'].index[0]
    
    # Increase speed from SUEZ to MARSEILLE to 23.8
    simulation_df.loc[suez_index, 'SEA_SPEED'] = 23.8
    
    # Adjust ETA, ETB, ETD for ports after MARSEILLE
    time_saved = timedelta(hours=24)  # Assume 24 hours saved as an example
    for i in range(marseille_index, len(simulation_df)):
        simulation_df.loc[i, 'COASTAL_SCHEDULE_ETA'] = process_date(simulation_df.loc[i, 'COASTAL_SCHEDULE_ETA'], time_saved)
        simulation_df.loc[i, 'COASTAL_SCHEDULE_ETB'] = process_date(simulation_df.loc[i, 'COASTAL_SCHEDULE_ETB'], time_saved)
        simulation_df.loc[i, 'COASTAL_SCHEDULE_ETD'] = process_date(simulation_df.loc[i, 'COASTAL_SCHEDULE_ETD'], time_saved)
        simulation_df.loc[i, 'DELAY_ETA'] = '0d 0h'
        simulation_df.loc[i, 'DELAY_ETB'] = '0d 0h'
        simulation_df.loc[i, 'DELAY_ETD'] = '0d 0h'

    state['simulation_result'] = simulation_df
    return state

# Create a prompt template for simulation request
simulation_request_prompt = PromptTemplate(
    template="Parse the following user request for a schedule recovery simulation and analysis:\n{user_input}\n{format_instructions}\n",
    input_variables=["user_input"],
    partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=SimulationRequest).get_format_instructions()}
)

# Create the simulation request chain
simulation_request_chain = simulation_request_prompt | llm | PydanticOutputParser(pydantic_object=SimulationRequest)

# Create a prompt template for analysis
analysis_prompt = PromptTemplate.from_template(
    """
    Analyze the following shipping schedule delay information and simulation results:

    Original Delays:
    {delays}

    Simulation Results:
    {simulation_results}

    User's Simulation Request:
    Focus Area: {focus_area}
    Specific Ports: {specific_ports}
    Metrics of Interest: {metrics_of_interest}
    Additional Context: {additional_context}

    Please provide:
    1. A summary of the delay situation and simulation results
    2. The difference between the original schedule and the simulation results 
    3. A focused analysis addressing the user's specific simulation request, paying special attention to:
       - The main focus area
       - Any specified ports
       - The metrics of interest
       - Additional context provided

    Ensure that your analysis emphasizes the aspects mentioned in the user's request.
    """
)

# Create a partial prompt with some pre-filled values
analysis_prompt = analysis_prompt.partial(
    focus_area="",
    specific_ports="",
    metrics_of_interest="",
    additional_context=""
)


# Create the analysis chain
analysis_chain = analysis_prompt | llm | StrOutputParser()

# Results reporting function
def report_results(state: State) -> State:
    if not state['has_delays']:
        state['original_delays'] = "No delays detected in the original schedule."
        state['simulation_results'] = ""
        state['analysis_result'] = "No analysis performed as there were no delays."
        return state

    delays_str = "Original schedule delays:\n"
    for delay in state['delays']:
        delays_str += f"Delay of {delay['delay']} at {delay['port']}. Original ETA: {delay['original_eta']}, New ETA: {delay['new_eta']}\n"
    
    state['original_delays'] = delays_str

    simulation_str = "Simulation results:\n"
    for _, row in state['simulation_result'].iterrows():
        simulation_str += f"{row['PORT']} - ETA: {row['COASTAL_SCHEDULE_ETA']}, Delay: {row['DELAY_ETA']}\n"
    
    state['simulation_results'] = simulation_str

    # Run the analysis chain
    analysis_result = analysis_chain.invoke({
        "delays": delays_str,
        "simulation_results": simulation_str,
        **state['simulation_request']
    })

    state['analysis_result'] = analysis_result
    
    return state

# Graph definition
workflow = Graph()

# Node definition
workflow.add_node("check_delays", check_delays)
workflow.add_node("run_simulation", run_simulation)
workflow.add_node("report_results", report_results)

# Entry point definition
workflow.set_entry_point("check_delays")

# Edge definition
workflow.add_edge('check_delays', 'run_simulation')
workflow.add_edge('run_simulation', 'report_results')
workflow.add_edge('report_results', END)

# Graph compilation
app = workflow.compile()


# Function for processing streamlit questions
def run_streamlit():
    st.title("Container Vessel Schedule Simulator")

    # Select columns to display
    columns_to_display = ['VESSEL', 'VOYAGE', 'PORT', 
                          'LONG_TERM_SCHEDULE_ETA', 'SIMULATED_SCHEDULE_ETA', 'DELAY_ETA']

    # Display the dataframe as a table with selected columns
    st.markdown("#### Current Schedule (Selected Columns)")
    st.dataframe(simulation[columns_to_display], height=300)

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
    prompt = st.chat_input("Enter your simulation request or question:")

    if prompt:
        # Add user message
        user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.schedule_messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"{prompt}\n\n<div style='font-size:0.8em; color:#888;'>{user_timestamp}</div>", unsafe_allow_html=True)
        
        # Process user input
        with st.spinner("Processing your request..."):
            processed_request = simulation_request_chain.invoke({"user_input": prompt})
            
            # Prepare initial state
            initial_state = State(
                coastal_schedule=coastal_schedule,
                simulation_result=simulation.copy(),  # Use the simulation DataFrame
                current_port='',
                delays=[],
                has_delays=False,
                simulation_request=processed_request.dict(),
                original_delays="",
                simulation_results="",
                analysis_result=""
            )
            
            # Run the workflow
            final_state = app.invoke(initial_state)
            
            # 결과 표시
            st.markdown("#### Original Schedule Delays")
            st.text(final_state['original_delays'])
            
            st.markdown("#### Simulation Results")
            st.text(final_state['simulation_results'])
            
            st.markdown("#### Analysis")
            st.markdown(final_state['analysis_result'])
            
            ai_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add and display AI response (전체 결과를 하나의 메시지로 통합)
        full_response = f"""
        {final_state['original_delays']}
        
        {final_state['simulation_results']}
        
        Analysis:
        {final_state['analysis_result']}
        """
        st.session_state.schedule_messages.append({"role": "assistant", "content": full_response, "timestamp": ai_timestamp})
        with st.chat_message("assistant"):
            st.markdown(f"{full_response}\n\n<div style='font-size:0.8em; color:#888;'>{ai_timestamp}</div>", unsafe_allow_html=True)
        
        # If there's a simulation result, display it
        if not final_state['simulation_result'].empty:
            st.markdown("#### Detailed Simulation Result")
            try:
                st.dataframe(final_state['simulation_result'][columns_to_display], height=300)
            except KeyError as e:
                st.error(f"Error displaying simulation result: {str(e)}")
                st.write("Available columns:", final_state['simulation_result'].columns.tolist())
        
        st.rerun()

# Run the main functions
def run():
    return contingency_simulator

# Run the streamlit function
if __name__ == "__main__":
    run_streamlit()