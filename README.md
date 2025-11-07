# LangGraph Chatbot

A sophisticated chatbot application built using LangGraph, Streamlit, and the Groq LLM platform. The application demonstrates different implementations of chat functionality including basic chat, streaming responses, thread management, and database persistence. It also integrates LangSmith for observability and tracing of the LLM interactions.

## Features

- 🤖 Powered by Groq's Llama 3.1 8B Instant model
- 💬 Multiple chat implementation variants:
  - Basic chat interface
  - Streaming responses
  - Thread management
  - Database-backed persistence
- 🎯 Clean and intuitive Streamlit UI
- 🔄 Conversation history management
- 📝 Thread-based chat organization
- 💾 SQLite database integration for persistent storage
- 📊 LangSmith integration for observability and tracing
- 🔍 Real-time monitoring of LLM interactions
- 📈 Performance tracking and debugging capabilities

## Project Structure

- `langgraph_backend.py` - Core chatbot implementation using LangGraph with in-memory storage
- `langgraph_database_backend.py` - Extended backend implementation with SQLite persistence
- `streamlit_frontend.py` - Basic Streamlit frontend implementation
- `streamlit_frontend_streaming.py` - Frontend with streaming response capability
- `streamlit_frontend_threading.py` - Frontend with thread management
- `streamlit_frontend_database.py` - Frontend with database persistence

## Prerequisites

- Python 3.12+
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amanp27/LangGraph-Chatbot.git
cd LangGraph-Chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install streamlit langgraph langchain-groq langchain-core python-dotenv
```

4. Set up your environment variables:
Create a `.env` file in the project root and add your API keys:
```
GROQ_API_KEY=your_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_langsmith_project_name  # Optional: defaults to 'default'
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # Optional: default endpoint
```

## Usage

You can run any of the frontend implementations using Streamlit:

```bash
# For basic chat interface
streamlit run streamlit_frontend.py

# For streaming responses
streamlit run streamlit_frontend_streaming.py

# For thread management
streamlit run streamlit_frontend_threading.py

# For database-backed chat
streamlit run streamlit_frontend_database.py
```

## Features by Implementation

### Basic Chat (`streamlit_frontend.py`)
- Simple chat interface
- Basic message history
- Single conversation thread

### Streaming Chat (`streamlit_frontend_streaming.py`)
- Real-time streaming of AI responses
- Improved user experience with immediate feedback
- Basic message history

### Threaded Chat (`streamlit_frontend_threading.py`)
- Multiple conversation management
- Thread-based chat organization
- Ability to switch between conversations
- New chat creation

### Database Chat (`streamlit_frontend_database.py`)
- Persistent storage of conversations
- SQLite database backend
- Thread management
- Conversation history preservation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Aman Prajapati

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
