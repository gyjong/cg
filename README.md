
# 🚢 ContainerGenie.ai: Driving Efficiency with Gemini, LangChain, RAG, and Intelligent Agent.

📊 Summary of ContainerGenie.ai: The advent of Large Language Models (LLM) represents a transformative leap in the realm of artificial intelligence, acting as a 🧠 brain that can process and understand vast amounts of data. When combined with the capabilities of 🔗 LangChain, 🔍 Retrieval-Augmented Generation (RAG), and 🤖 LLM Agents, these technologies can revolutionize the way container liner solutions operate, paving the way for a future with minimal UI and reduced human intervention.

ContainerGenie.ai is an AI-powered platform that revolutionizes container shipping operations by seamlessly integrating Gemini, LangChain, RAG, and intelligent agent collaboration to automate workflows, optimize logistics, enhance decision-making, and drive efficiency for small to medium-sized container liners.

## 🌟 Key Features

- 🧠 Enhanced Decision Making: The LLM-powered system processes vast amounts of data from various sources, providing context-rich insights for better operational decisions.
- 🔄 Adaptive Problem Solving: LangChain facilitates the creation of flexible, goal-oriented workflows that can autonomously adapt to changing conditions in container operations.
- 📚 Contextual Knowledge Integration: RAG technology allows the system to augment its decision-making with the most up-to-date and relevant information from the company's knowledge base and external sources.
- 🤖 Autonomous Operations: The Agentic Workflow enables ContainerGenie.ai to perform complex tasks with minimal human intervention, significantly reducing the need for specialized users.
- 📈 Continuous Learning and Improvement: The system learns from each interaction and outcome, constantly refining its processes and recommendations for optimized container liner operations.
- 🔗 Seamless Process Integration: By leveraging these technologies, ContainerGenie.ai can easily integrate with and enhance existing operational processes, creating a more cohesive and efficient ecosystem.
- 💬 Natural Language Interaction: Users can interact with the system using natural language, making it more accessible and reducing the learning curve for new users.
- 🛡️ Proactive Issue Resolution: The combination of these technologies allows ContainerGenie.ai to anticipate potential problems and suggest preventive measures before issues arise.
- 🎯 Customized Insights: The system can provide tailored analytics and recommendations based on each user's role and specific operational context.
- 🚀 Scalable Intelligence: As the complexity of operations grows, the LLM-LangChain-RAG-Agentic Workflow architecture can scale to handle increasing demands without a proportional increase in human resources.

## 🧩 Structure Explanation

- 🌍 Genre: Broadest operational area (e.g., Smart Vessel Operation)
- 📚 Series: Specific business domain (e.g., Vessel Schedule)
- 📖 Story: Concrete business process (e.g., Schedule Simulation)
- 📑 Chapter: Detailed operational steps (e.g., Schedule Delay Monitoring, Recovery Simulation, etc.)
- 📅 Episode: Actual operational data provided by the Legacy System (e.g., specific voyage data for a particular vessel)


## 🛠️ Technology Stack

- 🧠 LLM (Large Language Model): Natural language processing and understanding
- 🔗 LangChain: Application development framework utilizing LLM
- 🔍 RAG (Retrieval-Augmented Generation): Enhanced information retrieval and generation
- ⚙️ Agentic Workflow System: Automated business process management

## 📂 Directory Structure

* containergenie/
   * 📁 data_booking/       - Directory for booking data storage
   * 📁 data_credit/        - Directory for credit-related data storage
   * 📁 data_schedule/      - Directory for schedule-related data storage
   * 📁 img/                - Directory for image file storage
   * 📁 index_sanction/     - Sanction-related index data
   * 📁 index_vsa/          - VSA (Vessel Sharing Agreement) index data
   * 📁 sample_bookings/    - Sample booking data in pdf format
   * 📄 _booking_data_receiver.py     - Booking data receiver module
   * 📄 _coastal_schedule_monitor.py  - Coastal schedule monitoring module
   * 📄 _compliance_checker.py        - Compliance checking module
   * 📄 _contingency_simulator.py     - Contingency situation simulator module
   * 📄 _credit_checker.py            - Credit checking module
   * 📄 _genie_integrator.py          - Genie integration module
   * 📄 _schedule_coordinator.py      - Schedule coordination module
   * 📄 _vsa_validator.py             - VSA validation module
   * 📄 .env_sample         - Sample environment variables configuration file
   * 📄 .gitignore          - List of files excluded from Git version control
   * 📄 containergenie.py   - ContainerGenie main execution file
   * 📄 email_recipients.csv - List of email recipients
   * 📄 poetry.lock         - Poetry dependency lock file
   * 📄 pyproject.toml      - Project configuration and dependency definition file
   * 📄 README.md           - Project description document


## 🕰️ Version History

| Date       | Version                  | Description                                                       |
|------------|--------------------------|-------------------------------------------------------------------|
| 2024-08-05 | containergenie 0.15      | Initial release for google api competition                        |


## 🏃 Getting Started with ContainerGenie.ai

1. Set up the environment:
   Run `poetry install --no-root`

2. Configure API keys:
   - Rename `.env_sample` to `.env`
   - Open `.env` and input the required API keys:
     GOOGLE_API_KEY="YOUR_API_KEY"
     TAVILY_API_KEY="YOUR_API_KEY"
     OPENAI_API_KEY="YOUR_API_KEY"

3. Launch the application:
   Execute `streamlit run containergenie.py`


## 👩🏻‍💻👩🏻‍💻 Project Team
Product Management, Design Pattern, and Architecture: Kenny Jung

Kenny Jung led the product management and was responsible for designing the architecture and applying design patterns across the application. His expertise ensured that the project aligned with user needs and technical best practices.

Business Flow Engineering: KyoungJong Yoon, YongJae Chae

KyoungJong Yoon and YongJae Chae were instrumental in developing the business flow engineering for the project. Their work focused on optimizing business processes to ensure the smooth integration of the AI components within the overall business strategy.

LLM, LangChain, LangGraph, and RAG Framework Engineering: Jungwon Youn, Kyungpyoo Ham

Jungwon Yoon and Kyoungpyoo Ham specialized in the engineering of the LLM (Large Language Model), LangChain, LangGraph, and Retrieval-Augmented Generation (RAG) frameworks. Their contributions were crucial in developing the Agentic Workflow application, leveraging these technologies to build a highly effective and scalable AI solution.



## Copyright (c) 2024 Tongyang Systems.
All rights reserved. This project and its source code are proprietary and confidential. Unauthorized copying, modification, distribution, or use of this project, via any medium, is strictly prohibited without the express written permission of the copyright holder.
