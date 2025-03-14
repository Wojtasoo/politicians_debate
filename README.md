# Debate Simulation Project

## Overview
This project provides an AI-powered debate simulation system that facilitates discussions between two virtual political figures. The system generates topics, retrieves relevant data, and enables dynamic conversations between two debating agents. It leverages OpenAI's GPT models and external APIs for information retrieval and content generation.

## Features
- **Automated Debate Setup**: Creates and initiates AI-driven debates.
- **Real-time Debate Simulation**: AI agents engage in structured debates based on predefined roles.
- **Webhooks Integration**: Triggers debate processing upon external events.
- **Data Retrieval**: Fetches relevant debate content using APIs and a vector database.
- **Customizable AI Agents**: Modify debate participants, system messages, and debate formats.

## Technologies Used
- **Python**: Primary programming language.
- **Flask**: Web framework for handling webhooks.
- **LangChain & OpenAI GPT**: AI model for debate generation and response synthesis.
- **ChromaDB**: Vector database for storing retrieved debate-relevant data.
- **Jina AI API & Google Search API**: Fetches external content for debate context.
- **Dotenv**: Manages environment variables.

## Project Structure
```
.
├── agentsdebate.py     # Core logic for AI debate agents
├── webhook.py          # Flask webhook for handling debate events
├── requirements.txt    # Dependencies required for the project
├── README.md           # Documentation
```

## Installation
1. **Clone the Repository:**
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   Create a `.env` file and configure the required API keys:
   ```
   OPENAI_API_KEY=<your_openai_api_key>
   JINA_API_KEY=<your_jina_api_key>
   SERPER_API_KEY=<your_serper_api_key>
   PORT=6000
   ```

## Usage
### Running the Webhook Server
Start the Flask webhook server by running:
```sh
python webhook.py
```
The server listens for POST requests at `/debate-created`, which initiates the debate process.

### Triggering a Debate
Send a POST request to the webhook with the required payload:
```json
{
    "debateId": "12345",
    "prompt": "Climate change policies",
    "politician1": "John Doe",
    "politician2": "Jane Smith"
}
```
This will trigger a simulated debate between the two AI-generated politicians.

## API Endpoints
### `/debate-created` (POST)
- **Description**: Initiates a new debate.
- **Request Body**:
  ```json
  {
      "debateId": "<unique_id>",
      "prompt": "<debate_topic>",
      "politician1": "<name>",
      "politician2": "<name>"
  }
  ```
- **Response**:
  ```json
  {
      "message": "Debate received and processing initiated"
  }
  ```

## Future Enhancements
- **Live User Interaction**: Enable real-time user inputs to influence debates.
- **Multi-agent Debates**: Support more than two participants in discussions.
- **Data Expansion**: Improve knowledge retrieval using additional sources.
- **Graphical UI**: Develop a front-end for a more interactive experience.

## License
This project is licensed under the MIT License.

