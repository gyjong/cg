import streamlit as st
import os
import datetime
import uuid
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
import json
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Environment setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "containergenie.ai"

# Initialize Groq LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

def is_valid_file(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

def extract_text_from_pdf(pdf_path):
    if not is_valid_file(pdf_path):
        st.error(f"Invalid or inaccessible file: {pdf_path}")
        return ""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        return "\n".join([page.page_content for page in pages])
    except Exception as e:
        st.error(f"Error processing file {pdf_path}: {str(e)}")
        return ""

# Prompt templates
email_read_prompt = PromptTemplate.from_template("""
Parse the following email text and extract the booking information in a structured format.
Return the result as a Python dictionary, without any additional text or formatting.

Email text:
{email_text}

- The missing information should not be filled.

Extract the following information:
- To
- From
- Subject
- Salutation
- Booking Request Summary
- Vessel Name
- Voyage Number
- Customer Name
- Shipper Name
- HS Code
- Commodity Items
- Container Size
- Container Type
- Container Unit
- Place of Loading (without code)
- Estimated Arrival Time (Loading Port)
- Estimated Departure Time (Loading Port)
- Booking Close Time
- Document Close Time
- Cargo Close Time
- Place of Discharging (without code)
- Estimated Arrival Time (Discharging Port)
- Remark (if any)

Format the dates as YYYY-MM-DD HH:MM.
For any fields not found in the email except for REMARK, use an "MISSING" as the value.
Ensure all keys in the dictionary are in lowercase, with spaces instead of underscores.

Notes:
- The 'Salutation' should capture the greeting (e.g., "Dear, Sir").
- The 'Subject' should capture the entire booking request line.
- Reorganize 'Subject'
- The 'Booking Request Summary' should capture the entire booking request line.
- Reorganize 'Booking Request Summary'
""")

email_validation_prompt = PromptTemplate.from_template("""
Analyze the following parsed email information to determine if it contains all critical booking information and is free from major inconsistencies. Focus on the essential elements for a shipping booking, using the remark as supplementary information.

Parsed email information:
{parsed_email_dict}

Perform the following checks:

1. Critical Information Check:
Ensure the following essential fields are present and not empty:
[List of essential fields]

2. Major Consistency Check:
[List of consistency checks]

3. Date Logic Check:
[List of date logic checks]

4. Remark Cross-Check:
[Remark cross-check instructions]

After performing these checks, respond with one of the following:
1. If any critical information is missing, there are major inconsistencies, or the remark contradicts important booking details:
ISSUE FOUND
Then, provide a list of only the critical issues found, focusing on missing essential information or significant inconsistencies.

2. If all critical information is present, there are no major inconsistencies, and the remark (if present) doesn't contradict other information:
EMAIL CONFIRMED

Important Notes:
[Additional notes on validation]
""")

# Chain for parsing email
email_parse_runnable = (
    {"email_text": RunnablePassthrough()}
    | email_read_prompt
    | llm
    | StrOutputParser()
)


# Chain for validating email
email_validation_runnable = (
    {"parsed_email_dict": RunnablePassthrough()}
    | email_validation_prompt
    | llm
    | StrOutputParser()
)


# Define State type
class State(TypedDict):
    messages: Annotated[list[AnyMessage], ...]
    iteration: int
    email_path: str
    next: str
    thread_id: str
    status: str
    parsed_result: dict
    validated_result: str
    processed_emails: List[str]

# Manager class
class Manager:
    def __init__(self):
        self.emails = []
        self.iter_email = 0
        self.total_num_email = 0
        self.processed_emails = set()

    def __call__(self, state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        dir_path = configuration.get("dir_path", None)
        state = {**state}
        
        if dir_path and not self.emails and not state["next"]:
            self.emails = self.get_emails(dir_path)
            self.total_num_email = len(self.emails)
        elif not self.emails:
            return {"messages": "DONE", "next": "end", "status": "Processing complete", "processed_emails": list(self.processed_emails)}
        
        if state["next"]:
            if "EMAIL CONFIRMED" in state.get("validated_result", ""):
                state["status"] = "Email Confirmed"
            elif "ISSUE FOUND" in state.get("validated_result", ""):
                state["status"] = "Issue Found"
            self.processed_emails.add(state.get("email_path", ""))
        else:
            state["status"] = "Starting Process"

        if self.emails:
            email_path = self.emails.pop(0)
            self.iter_email += 1
            next_node = "E-mail Parser"
            return {
                "email_path": email_path,
                "next": next_node,
                "status": f"Processing email {self.iter_email} of {self.total_num_email}: {state['status']}",
                "processed_emails": list(self.processed_emails)
            }
        else:
            return {"messages": "DONE", "next": "end", "status": "Processing complete", "processed_emails": list(self.processed_emails)}

    def get_emails(self, dir_path):
        return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]

# Email Parser Agent
class EmailParserAgent:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        email_path = state.get("email_path", None)
        email = self.get_email_from_email_path(email_path)

        if email:
            parsed_result = self.runnable.invoke(email)
            next_node = "E-mail Checker"
            return {
                "parsed_result": parsed_result,
                "next": next_node,
                "status": f"Parsed email: {os.path.basename(email_path)}"
            }
        else:
            return {"parsed_result": None, "next": END, "status": f"Failed to parse email: {os.path.basename(email_path)}"}
    
    def get_email_from_email_path(self, email_path):
        return extract_text_from_pdf(email_path)

# Email Checker Agent
class EmailCheckerAgent:
    def __init__(self, runnable):
        self.runnable = runnable
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        parsed_email = state.get("parsed_result")
        if parsed_email:
            validated_result = self.runnable.invoke(parsed_email)
            next_node = "Manager"
            return {
                "validated_result": validated_result,
                "next": next_node,
                "status": "Email validated"
            }
        else:
            return {"validated_result": None, "next": END, "status": "Email validation failed"}

# Function to determine if email should be parsed
def should_parse_email(state):
    if state["next"] == "E-mail Parser":
        return "E-mail Parser"
    else:
        return "end"

# Initialize graph builder
builder = StateGraph(State)

# Define nodes
builder.add_node("Manager", Manager())
builder.add_node("E-mail Parser", EmailParserAgent(email_parse_runnable))
builder.add_node("E-mail Checker", EmailCheckerAgent(email_validation_runnable))

# Add edges
builder.add_edge(START, "Manager")
builder.add_conditional_edges(
    "Manager",
    should_parse_email,
    {
        "E-mail Parser": "E-mail Parser",
        "end": END
    }
)
builder.add_edge("E-mail Parser", "E-mail Checker")
builder.add_edge("E-mail Checker", "Manager")

# Compile graph
graph = builder.compile()

# Streamlit UI
def run():

    if "booking_messages" not in st.session_state:
        st.session_state.booking_messages = []
    
    if "parsed_results" not in st.session_state:
        st.session_state.parsed_results = {}
    
    if "validated_results" not in st.session_state:
        st.session_state.validated_results = {}
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Ensure the data_booking directory exists
    os.makedirs("./data_booking/", exist_ok=True)

    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append(uploaded_file)
        
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

    # Display uploaded files
    if st.session_state.uploaded_files:
        st.write("Uploaded files:")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file.name}")

    # Process button
    if st.button("Process Emails"):
        with st.spinner("Processing uploaded files..."):
            for file in st.session_state.uploaded_files:
                file_path = os.path.join("./data_booking", file.name)
                with open(file_path, "wb") as temp_file:
                    temp_file.write(file.getvalue())
            
            thread_id = str(uuid.uuid4())
            config = RunnableConfig(
                configurable={
                    "dir_path": "./data_booking/",
                    "thread_id": thread_id,
                },
                recursion_limit=100
            )
            
            status_placeholder = st.empty()
            events = graph.stream(
                {"messages": "", "status": "", "parsed_result": None, "validated_result": None, "processed_emails": []},
                config=config,
                stream_mode="values"
            )
            for event in events:
                if "status" in event:
                    status_placeholder.write(event["status"])
                if "parsed_result" in event and event["parsed_result"]:
                    email_path = event.get("email_path", "unknown")
                    st.session_state.parsed_results[email_path] = event["parsed_result"]
                if "validated_result" in event and event["validated_result"]:
                    email_path = event.get("email_path", "unknown")
                    st.session_state.validated_results[email_path] = event["validated_result"]
        
        st.success("Processing complete!")

    # Chat interface
    for message in st.session_state.booking_messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}\n\n<div style='font-size:0.8em; color:#888;'>{message['timestamp']}</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask a question about the booking or email content")

    if prompt:
        user_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.booking_messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        
        with st.chat_message("user"):
            st.markdown(f"{prompt}\n\n<div style='font-size:0.8em; color:#888;'>{user_timestamp}</div>", unsafe_allow_html=True)
        
        with st.spinner("Thinking..."):
            try:
                context = "Parsed Emails:\n"
                for idx, (email_path, result) in enumerate(st.session_state.parsed_results.items()):
                    context += f"Email {idx + 1} ({os.path.basename(email_path)}):\n{result}\n\n"
                
                context += "Validation Results:\n"
                for idx, (email_path, result) in enumerate(st.session_state.validated_results.items()):
                    context += f"Email {idx + 1} ({os.path.basename(email_path)}) Validation:\n{result}\n\n"
                
                context += f"User Question: {prompt}"
                
                response = llm.predict(context)
                ai_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
                st.session_state.booking_messages.append({"role": "assistant", "content": response, "timestamp": ai_timestamp})
                with st.chat_message("assistant"):
                    st.markdown(f"{response}\n\n<div style='font-size:0.8em; color:#888;'>{ai_timestamp}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    run()