# Copyright (c) 2024 Tongyang Systems.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import streamlit as st
import datetime
import uuid
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict
from typing_extensions import TypedDict

# Initialize Groq LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)


# Initialize vector store and retriever
def load_documents(sources: List[str]) -> List:
    docs = []
    for source in sources:
        if source.startswith('http'):
            loader = WebBaseLoader(source)
        elif source.endswith('.pdf'):
            loader = PyPDFLoader(source)
        else:
            raise ValueError(f"Unsupported source type: {source}")
        docs.extend(loader.load())
    return docs

def create_or_load_vectorstore(sources: List[str], index_name: str = "index_sanction") -> FAISS:
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Set the index directory dynamically
    index_dir = os.path.join(current_dir, index_name)
    index_path = os.path.join(index_dir, "index")
    
    print(f"Checking for index at: {index_path}/index.faiss")
    if os.path.exists(f"{index_path}/index.faiss"):
        print(f"Loading existing vector store from {index_path}/index.faiss")
        try:
            vectorstore = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Will create a new vector store.")
    else:
        print(f"Vector store not found at {index_path}/index.faiss")
    
    print("Creating new vector store...")
    docs = load_documents(sources)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(doc_splits, OpenAIEmbeddings())
    
    print(f"Saving vector store to {index_path}...")
    try:
        os.makedirs(index_dir, exist_ok=True)  # Ensure the directory exists
        vectorstore.save_local(index_path)
        print("Vector store saved successfully.")
    except Exception as e:
        print(f"Error saving vector store: {e}")
    
    return vectorstore

# List of URLs and PDF files to load documents from
sources = [
    "./index_sanction/cherry_compliance.pdf",
    "./index_sanction/department_of_commerce.pdf",
    "./index_sanction/financial_sanctions_targets_in_uk.pdf"
]

# Create or load vector store
vectorstore = create_or_load_vectorstore(sources)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

# Initialize web search tool
web_search_tool = TavilySearchResults()


# Define prompt templates and chains
rag_prompt = PromptTemplate(
    template="""
    # Your Role
    You are a sanction specialist who has a deep understanding of the sanction policies of the United Nations, European Union, and United Kingdom.
    You are asked to provide detailed information for the questions.
    Use the following pieces of context to answer the users question in details.

    ----------
    # Instruction
    1. Given the following summaries of a long document and a question, create a final answer with references ("SOURCES[number]"), use "SOURCES[number]" in capital letters regardless of the number of sources you use.
    2. For each SOURCES reference, include the page number for PDF documents or the URL for web pages in parentheses. `For example: SOURCES[1] (page 5) or SOURCES[2] (https://example.com)`.
    3. Provide the information in a clear and concise manner in a way that is easy to understand with enriched explanations as much as possible. 
    4. Provide the feedback with bullet points to list the information in organized manner.
    5. If the question is not clear, ask the user to clarify the question.
    6. If the question is asking about sanction information, please answer with reference to relevant regulations, laws, and case law. The feedback should include the relevant sanction provisions, penalties, and enforcement mechanisms.
    7. If the question made by Korean, please answer in Korean.
    8. If the question made by French, please answer in French.
    9. If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    10. At the end of your response, list all the sources used with their full reference information include the page number for PDF documents or the URL for web pages in parentheses. For example: SOURCES[1] (page 5) or SOURCES[2] (https://example.com).
    -----------
    
    # Question: {question}     
    # Documents: {documents} 

    # Answer: 
    """,
    
    input_variables=["question", "documents"],
)

rag_chain = rag_prompt | llm | StrOutputParser()

# Data model for the output
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with tool call
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a teacher grading a quiz. You will be given: 
1/ a QUESTION 
2/ a set of comma separated FACTS provided by the student

You are grading RELEVANCE RECALL:
A score of 1 means that ANY of the FACTS are relevant to the QUESTION. 
A score of 0 means that NONE of the FACTS are relevant to the QUESTION. 
1 is the highest (best) score. 0 is the lowest score you can give. 

Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "FACTS: \n\n {documents} \n\n QUESTION: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    web_results = web_search_tool.invoke({"query": question})
    documents.extend(
        [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]
    )
    return {"documents": documents, "question": question, "steps": steps}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"


# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

def run():
    if "booking_messages" not in st.session_state:
        st.session_state.booking_messages = []

    # Display chat messages
    for message in st.session_state.booking_messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{message['content']}\n\n<div style='font-size:0.8em; color:#888;'>{message['timestamp']}</div>", unsafe_allow_html=True)
            if "steps" in message and message["role"] == "assistant":
                with st.expander("View steps"):
                    st.write(message["steps"])

    # Chat input
    prompt = st.chat_input("Any question about Compliance?")

    if prompt:
        # Add user message
        user_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.booking_messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"{prompt}\n\n<div style='font-size:0.8em; color:#888;'>{user_timestamp}</div>", unsafe_allow_html=True)
        
        # Get AI response
        with st.spinner("Thinking..."):
            try:
                # Using app.invoke() for Compliance-specific responses
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                response = app.invoke(
                    {
                        "question": prompt,
                        "generation": "",
                        "search": "",
                        "documents": [],
                        "steps": []
                    },
                    config
                )
                ai_response = response.get("generation", "No response generated")
                ai_steps = response.get("steps", [])
                ai_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
                # Add and display AI response
                st.session_state.booking_messages.append({"role": "assistant", "content": ai_response, "timestamp": ai_timestamp, "steps": ai_steps})
                with st.chat_message("assistant"):
                    st.markdown(f"{ai_response}\n\n<div style='font-size:0.8em; color:#888;'>{ai_timestamp}</div>", unsafe_allow_html=True)
                    if ai_steps:
                        with st.expander("View steps"):
                            st.write(ai_steps)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    run()