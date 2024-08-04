
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set page config as the first Streamlit command
import streamlit as st
st.set_page_config(page_title="ContainerGenie.ai", page_icon="🚢", layout="wide")


from PIL import Image
import _coastal_schedule_monitor
import _contingency_simulator
import _compliance_checker
import _credit_checker
import _schedule_coordinator
import _vsa_validator
import _special_remark_handler
import _booking_data_receiver



# UI setup
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    [data-testid="column"] {
        border-right: 1px solid #ccc;
        padding-right: 20px;
    }
    [data-testid="column"] + [data-testid="column"] {
        border-right: none;
        padding-left: 20px;
    }
    .medium-header {
        font-size: 22px;
        font-weight: 600;
        color: #555;
    }
    .small-header {
        font-size: 18px;
        font-weight: 600;
        color: #555;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.5); 
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .coming-soon {
        color: gray;
        font-style: italic;
        margin-top: 10px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    # Add image to the top of the sidebar
    image = Image.open('./img/containergenie.png')
    st.image(image, use_column_width=True)
    
    st.markdown("### Story Selection")
    
    menu = st.radio(
        "ContainerGenie MyStory",
        options=[
            "Introduction",
            "📚 Smart Vessel Operation Series > 📖 Schedule Simulation Story",
            "📚 Responsive Customer Series > 📖 Booking Registration Story",
        ]
    )

    # Add dropdown menus conditionally
    if menu == "📚 Smart Vessel Operation Series > 📖 Schedule Simulation Story":
        st.markdown("### Episode Selection")
        vessel_episode = st.selectbox(
            "Schedule for a vessel voyage direction",
            options=["ARONIA 202402W", "VESSEL VOY 202403E", "VESSEL VOY 202404W", "..."]
        )

    elif menu == "📚 Responsive Customer Series > 📖 Booking Registration Story":
        st.markdown("### Episode Selection")
        booking_episode = st.selectbox(
            "Bookings under a vessel voyage direction",
            options=["ARONIA 202402W", "VESSEL VOY 202403E", "VESSEL VOY 202404W", "..."]
        )


st.sidebar.info("This app demonstrates an LLM-LangChain-RAG-Agentic Workflow based containergenie.ai service for Container Liner Operation")



left_column, right_column = st.columns([1, 3])



# Initialize session states
if "current_story" not in st.session_state:
    st.session_state.current_story = None

if "schedule_messages" not in st.session_state:
    st.session_state.schedule_messages = []

if "booking_messages" not in st.session_state:
    st.session_state.booking_messages = []




# Check if story has changed
if st.session_state.current_story != menu:
    st.session_state.current_story = menu


# Contents of each menu
if menu == "Introduction":
    with left_column:
        st.header("Welcome")
        st.write("Explore our stories.")
    
    with right_column:
        st.header("ContainerGenie.ai")
        st.write("Learn more about container logistics through our interactive platform.")

        with st.expander("About ContainerGenie.ai"):
            st.write("""
            ContainerGenie.ai is an innovative agentic workflow platform designed to simplify and enhance container liner operations. 
            Our AI-powered agents and tools provide insights and simulations to help you make informed decisions in your liner operation and logistics processes.
            """)

        with st.expander("Available Stories"):
            st.write("""
            - **Schedule Simulation Story**: Detecting delays and simulate for optimizing schedule recovery.
            - **Booking Registration Story**: Handy booking registration process for container shipments.
            """)

        with st.expander("How to Use"):
            st.write("""
            1. Select a series > story from the sidebar menu.
            2. Select a episode vessel voyage direction for schedule story
               or, select a episode booking for booking story.
            3. Select a chapter in the 2nd level menu.
            4. Follow the prompts and instructions within each story.
            5. Explore different scenarios and learn about container liner operation.
               or, use the insights gained to improve your container liner operations.
            """)

elif menu == "📚 Smart Vessel Operation Series > 📖 Schedule Simulation Story":
    with left_column:
        st.markdown('<p class="small-header">Chapters for My Schedule Simulation Story</p>', unsafe_allow_html=True)
        episode = st.radio(
            "Choose a chapter",
            options=[
                "📜 Chapter 1. Coastal_Schedule_Monitor", 
                "📜 Chapter 2. Contingency_Simulator", 
                "📜 Chapter 3. Schedule_Coordinator", 
                "📜 Chapter 4. VSA_Validator", 
                ],
            key="schedule_sim"
        )

    with right_column:
        st.header(f"Schedule Simulation Story: {episode}")
        if episode == "📜 Chapter 1. Coastal_Schedule_Monitor":
            st.write("Storyboard for Schedule Simulation Story > Chapter 1. Coastal Schedule Monitor")
            _coastal_schedule_monitor.run()
            _coastal_schedule_monitor.run_streamlit()
        elif episode == "📜 Chapter 2. Contingency_Simulator":
            st.write("Storyboard for Schedule Simulation Story > Chapter 2. Contingency Simulator")
            _contingency_simulator.run()
            _contingency_simulator.run_streamlit()
        elif episode == "📜 Chapter 3. Schedule_Coordinator":
            st.write("Storyboard for Schedule Simulation Story > Chapter 3. Schedule_Coordinator")
            _schedule_coordinator.run()
            _schedule_coordinator.run_streamlit()
        elif episode == "📜 Chapter 4. VSA_Validator":
            st.write("Storyboard for Schedule Simulation Story > Chapter 4. VSA_Validator")
            _vsa_validator.run()
            _vsa_validator.run_streamlit()



elif menu == "📚 Responsive Customer Series > 📖 Booking Registration Story":
    with left_column:
        st.markdown('<p class="small-header">Chapters for My Booking Registration Story</p>', unsafe_allow_html=True)
        episode = st.radio(
            "Choose a chapter",
            options=[
                "📜 Chapter 1. Booking_Data_Receiver", 
                "📜 Chapter 2. Compliance_Checker", 
                "📜 Chapter 3. Credit_Checker", 
                "📜 Chapter 4. Special_Remark_Handler", 
                ],
            key="booking_reg"
        )

    with right_column:
        st.header(f"Booking Registration Story: {episode}")
        if episode == "📜 Chapter 1. Booking_Data_Receiver":
            st.write("Storyboard for Booking Registration Story > Chapter 1. ")
            _booking_data_receiver.run()
        elif episode == "📜 Chapter 2. Compliance_Checker":
            st.write("Storyboard for Booking Registration Story > Chapter 2. ")
            _compliance_checker.run()
        elif episode == "📜 Chapter 3. Credit_Checker":
            st.write("Storyboard for Booking Registration Story > Chapter 3. ")
            _credit_checker.run()
        elif episode == "📜 Chapter 4. Special_Remark_Handler":
            st.write("Storyboard for Booking Registration Story > Chapter 4. ")
            _special_remark_handler.run()


elif menu == "📖 IMDG Q&A Story":
    with left_column:
        st.markdown('<p class="small-header">My IMDG Q&A Episode</p>', unsafe_allow_html=True)
        episode = st.radio(
            "Choose an episode",
            options=[
                "📜 Q&A Episode 1", 
                "📜 Q&A Episode 2", 
                "📜 Q&A Episode 3"
                ],
            key="imdg_faiss"
        )

footer = """
<div class="footer">
    <p>© 2024 ContainerGenie.ai. All rights reserved. | 0.15</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)