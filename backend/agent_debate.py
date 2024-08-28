import streamlit as st
import os
from dotenv import load_dotenv
from typing import Callable, List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langsmith import traceable

from convex import ConvexClient
from typing_extensions import TypedDict
import requests
import json
import datetime
import chromadb


load_dotenv()
client = ConvexClient(os.getenv("CONVEX_URL"))
db=chromadb.Client()
#db.clear_system_cache()
db.delete_collection("Data_1")
db.delete_collection("Data_2")
collection_1 = db.create_collection(name="Data_1")
collection_2 = db.create_collection(name="Data_2")

# class ReadAgent(TypedDict):
#     #ID: str
#     #status: str
#     #Date: datetime
#     topic: str
#     speaker_1: str
#     speaker_2: str
#     system_message: SystemMessage
#     model: ChatOpenAI
#     message_history: List[str]
class ReadAgent:
    def __init__(self, topic: str, speaker_1: str, speaker_2: str, system_message: SystemMessage, model: ChatOpenAI, message_history: List[str]):
        self.topic = topic
        self.speaker_1 = speaker_1
        self.speaker_2 = speaker_2
        self.system_message = system_message
        self.model = model
        self.message_history = message_history
    
    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")

    def send(self, speaker_idx: int) -> str:
        # Determine which collection to use based on the current speaker
        if speaker_idx % 2 == 0:
            # If speaker index is even, use collection "Data_1"
            collection_name = "Data_1"
        else:
            # If speaker index is odd, use collection "Data_2"
            collection_name = "Data_2"
        
        # Generate a query based on current conversation or topic
        query = f"{self.topic}"
        
        # Retrieve relevant data from the chosen ChromaDB collection
        relevant_data = retrieve_relevant_data(self, query, collection_name)
        
        # Combine relevant data into a single content source
        combined_content = "\n".join(relevant_data)
        
        # Include the combined content in the message
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [combined_content + f"{self.speaker_2}: "])),
            ]
        )
        return message.content
    
    # def reset(self):
    #     self["message_history"] = ["Here is the conversation so far."]
    
    # def receive(self, name: str, message: str) -> None:
    #     self["message_history"].append(f"{name}: {message}")
    
    # def send(self, speaker) -> str:
    #     message = self["model"].invoke(
    #         [
    #             self["system_message"],
    #             HumanMessage(content="\n".join(self["message_history"] + [f"{self[{speaker}]}: "])),
    #         ]
    #     )
    #     return message.content
    
def load_data(ID, status, Date, topic, speaker_1, speaker_2, system_message, model, message_history):
    return ReadAgent(
        ID=ID,
        status=status,
        Date=Date,
        topic=topic,
        speaker_1=speaker_1,
        speaker_2=speaker_2,
        system_message=system_message,
        model=model,
        message_history=message_history,
    )

names = {
    "Supporter",
    "Opposer",
}
global topic 
topic="" #= "The current impact of automation and artificial intelligence on employment"
word_limit = 50  # word limit for task brainstorming

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names)}"""

def generate_topic_and_context(agent: ReadAgent) -> dict:
    #topic=agent["topic"]
    # Get the topic from the agent
    topic = agent.topic
    
    # Define the prompt for generating the context
    topic_specifier_prompt = [
        SystemMessage(content="""You provide json output of the tasks given as in example. Output example:
        {
            "topic": "text",
            "context": "text",
            "serper_query_1": "text",
            "serper_query_2": "text",
            "initial_prompt": "text"
        }"""),
        HumanMessage(
            content=f"""{topic}
            
            You are the moderator.
            1. Please make the topic more specific
            2. Provide short 2-sentence context about the topic.
            3. Based on the topic and context generate two identical search queries about the political views of {agent.speaker_1} and {agent.speaker_2} respectively on the given topic. 
            4. Generate initial prompt for the debate based on the topic of the debate. Speak directly to the {agent.speaker_1} and {agent.speaker_2}.
            Do not add anything else."""
        )
    ]
    
    # Get the response from the model
    response = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content
    
    # Parse the JSON string response into a dictionary
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}, response: {response}")
    
    return response_dict
    #response = agent.model.invoke(topic_specifier_prompt).content


def get_data_discussion(agent: ReadAgent) -> dict:
    response_dict = generate_topic_and_context(agent=agent)
    
    # Access the parsed JSON keys correctly
    query_1 = response_dict["serper_query_1"]
    query_2 = response_dict["serper_query_2"]
    queries = [query_1, query_2]
    
    # Modify the output if needed
    url = "https://google.serper.dev/search"
    
    for j in range(2):
        payload = json.dumps({
            "q": queries[j],
            "gl": "pl"
        })

        headers = {
            'X-API-KEY': 'cd0cc120ee4c7c12049a05c49bd1b9a1d0e6f4c8',
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        # Parse the JSON response
        search_results = response.json()

        # Extract the first 5 unique links from the search results
        links = []
        for result in search_results.get("organic", []):
            link = result.get("link")
            if link and link not in links:  # Ensure uniqueness
                links.append(link)
            if len(links) == 5:  # Stop after 5 links
                break

        headers = {'Accept': 'application/json', 'Authorization': 'Bearer jina_60d6d9ee5b774a548573f4197af013c10mFxfHr5oEiI82i4EWK_ox_O0Z74'}

        # List to store the responses
        responses = []

        # Process each link through the Jina API
        for link in links:
            jina_url = f'https://r.jina.ai/{link}'
            response = requests.get(jina_url, headers=headers)
            if response.status_code == 200:
                response_json = response.json()
                content = response_json.get("data", {}).get("content")  # Extract content field
                if content:
                    responses.append(content)
                else:
                    raise Exception(f"Content not found in response: {response_json}")
            else:
                raise Exception(f"Request failed: {response.status_code}")

        for i, link in enumerate(links):
            if j == 0:
                if check_URL(link, "Data_1"):
                    continue
                else:
                    docs = split_text(responses[i], link)
                    for doc in docs:
                        collection_1.add(
                            documents=[doc["page_content"]],
                            metadatas=[doc["metadata"]],
                            embeddings=[embedding_function.embed_documents([doc["page_content"]])]
                        )
            else:
                if check_URL(link, "Data_2"):
                    continue
                else:
                    docs = split_text(responses[i], link)
                    for doc in docs:
                        collection_2.add(
                            documents=[doc["page_content"]],
                            metadatas=[doc["metadata"]],
                            embeddings=[embedding_function.embed_documents([doc["page_content"]])]
                        )

        links.clear()
        responses.clear()

    return response_dict

def retrieve_relevant_data(agent: ReadAgent, query: str, collection_name: str) -> List[str]:
    # Perform similarity search in the specified ChromaDB collection based on the provided query
    collection = db.get_collection(collection_name)
    
    # Retrieve documents from ChromaDB using a similarity search
    results = collection.query(query_texts=[query], n_results=5)  # Fetch top 5 relevant documents

    # Debugging: Print the type and content of results to check its format
    print("Type of results:", type(results))
    print("Content of results:", results)
    
    # Check if the results are a list and contain the expected structure
    if isinstance(results, list) and all(isinstance(doc, dict) for doc in results):
        documents = [doc["page_content"] for doc in results if "page_content" in doc]
    else:
        raise ValueError(f"Unexpected results format: {results}")

    return documents

    
class DialogueSimulator:
    def __init__(
        self,
        agents: List[ReadAgent],
        selection_function: Callable[[int, List[ReadAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send(speaker_idx)

        # 3. everyone receives the message
        # Check if the message is from speaker_1 or speaker_2
        for receiver in self.agents:
            if speaker_idx % 2 == 0:
                sender = speaker.speaker_1
            else:
                sender = speaker.speaker_2
            receiver.receive(sender, message)

        # 4. increment time
        self._step += 1

        return sender, message

def select_next_speaker(step: int, agents: List[ReadAgent]) -> int:
    idx = (step) % len(agents)
    return idx

def generate_system_message(name, description):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}
Your goal is to persuade your conversation partner of your point of view.

DO look up information from provided database to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

Do not add anything else.

Stop speaking the moment you finish speaking from your perspective.
"""

def generate_agent_description(name):
    agent_specifier_prompt = [
        SystemMessage(
            content="You can add detail to the description of the conversation participant."),
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a creative description of {name}, in {word_limit} words or less. 
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else."""
        ),
    ]
    agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
    return agent_description

agent_descriptions = {name: generate_agent_description(name) for name in names}
for name, description in agent_descriptions.items():
    print(description)

agent_system_messages = {
    name: generate_system_message(name, description)
    for name, description in zip(names, agent_descriptions.values())
}

# Set the layout of the page to wide mode
st.set_page_config(layout="wide")

# Title at the top of the page
st.markdown("""
    <style>
    .title-center {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-center">AI-Powered Debate</div>', unsafe_allow_html=True)

# Define the layout of the page
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    # Left Select Box
    left_option = st.selectbox("Select Politician", ["Donald Tusk", "Szymon Hołownia", "Radosław Sikorski", "Włodzimierz Czarzasty", "Sławomir Mentzen"], key="left_selectbox")
    if left_option=="Donald Tusk":
        st.image("Tusk.jpeg", caption="Donald Tusk", use_column_width=True)
    elif left_option=="Szymon Hołownia":
        st.image("Hołownia.jpeg", caption="Szymon Hołownia", use_column_width=True)
    elif left_option=="Radosław Sikorski":
        st.image("Sikorski.webp", caption="Radosław Sikorski", use_column_width=True)
    elif left_option=="Włodzimierz Czarzasty":
        st.image("Czarzasty.webp", caption="Włodzimierz Czarzasty", use_column_width=True)
    elif left_option=="Sławomir Mentzen":
        st.image("Mentzen.jpg", caption="Sławomir Mentzen", use_column_width=True)

with col3:
    # Right Select Box
    right_option = st.selectbox("Select Politician", ["Andrzej Duda", "Mateusz Morawiecki", "Jarosław Kaczyński", "Mariusz Błaszczak", "Grzegorz Braun"], key="right_selectbox")
    if right_option=="Andrzej Duda":
        st.image("Duda.jpg", caption="Andrzej Duda", use_column_width=True)
    elif right_option=="Mateusz Morawiecki":
        st.image("Morawiecki.jpg", caption="Mateusz Morawiecki", use_column_width=True)
    elif right_option=="Jarosław Kaczyński":
        st.image("Kaczor.jpeg", caption="Jarosław Kaczyński", use_column_width=True)
    elif right_option=="Mariusz Błaszczak":
        st.image("Błaszczak.jpg", caption="Mariusz Błaszczak", use_column_width=True)
    elif right_option=="Grzegorz Braun":
        st.image("Braun.jpeg", caption="Grzegorz Braun", use_column_width=True)


with col2:
    # Chat window below the title and image
    st.subheader("Chat Window")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input box for user to type messages
    user_input = st.text_input("Enter topic:", "")
    
    # Add a submit button
    if st.button("Start Debate"):
        if user_input:
            # Use user_input as the topic
            topic = user_input
            
            # Clear previous chat history
            st.session_state.chat_history = []
            
            # Generate agent descriptions and system messages (reuse your existing functions)
            agents = [
                ReadAgent(
                    #name=name,
                    topic=topic,
                    speaker_1=left_option,
                    speaker_2=right_option,
                    system_message=SystemMessage(content=system_message),
                    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
                    message_history=[]
                )
                for name, system_message in zip(
                    names, agent_system_messages.values()
                )
            ]
            
            simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
            simulator.reset()
            simulator.inject("Moderator", get_data_discussion(agents[0]))
            #st.session_state.chat_history.append(f"(Moderator): {specified_topic}")
            
            # Simulate conversation
            for i in range(6):
                name, message = simulator.step()
                st.session_state.chat_history.append((name, message))
            
            # Display chat history
            for name, message in st.session_state.chat_history:
                st.write(f"**{name}:** {message}")
            
        else:
            st.warning("Please enter a topic before starting the debate.")



############################RAG############################

from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    llm_type = os.getenv("LLM_TYPE", "groq")
    if llm_type == "ollama":
        return ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0)
    elif llm_type == "groq":
        print("Using GROQ")
        return ChatGroq(temperature=0.1, model_name="llama-3.1-70b-versatile")
    else:
        return ChatOpenAI(temperature=0, model="gpt-4o-mini")

def get_embeddings():
    embedding_type = os.getenv("LLM_TYPE", "groq")
    if embedding_type == "ollama":
        return OllamaEmbeddings(model="llama3.1:8b-instruct-q4_0")
    else:
        return OpenAIEmbeddings()


from langchain.schema import Document

embedding_function = get_embeddings()

#path_to_file=input("Enter the path to the directory: ")
#file_name=input("Enter the name of the file: ")


def split_text(text, url):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits_pypdf = splitter.split_documents(text)
    documents = []
    for i, chunk in enumerate(all_splits_pypdf):
        document=[
                {"page_content": chunk.page_content, "metadata": {"source": f"{url}", "chunk_number": i, "page_number": chunk.metadata.get("page", None)}}
        ]
        # document = Document(
        #     page_content=chunk.page_content,
        #     metadata={
        #         "source": f"{url}",   
        #         "chunk_number": i,
        #         "page_number": chunk.metadata.get("page", None)
        #     }
        # )
        documents.append(document)
    return documents

def check_URL(url, collection_name):
    
    collection = db.get_collection(collection_name)
    # Query the collection to check if any document contains the URL in its metadata.
    results = collection.get(
        where={"metadata.source": url}
    )

    # Check if any document matches the query
    if results:
        return True
    return False

#retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 8, 'score_threshold': 0.7})

from typing_extensions import TypedDict


class AgentState(TypedDict):
    prompt: str
    llm_output: str
    documents: list[str]
    classifications: list[str]

def retrieve_docs(state: AgentState):
    prompt = state["prompt"]
    print("Prompt: "+prompt)
    #documents = retriever.invoke(input=prompt)
    #print("RETRIEVED DOCUMENTS:", documents)
    #state["documents"] = [doc.page_content for doc in documents]
    return state

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


def prompt_classifier(state: AgentState):
    system="""You are a grader assesing the type of information in a paper. \n
    <instructions>
    Asses if the information you just read is a technical information or a scientific one.
    </instructions>
    
    <Examples>
    Technical Information -> Methods used, procedures, specifications of some machines, devices, etc.
    Scientific Information -> Theoretical concepts, scenarios and mathematical models, graphs and experiments with conclusions etc.
    </Examples>
    
    If the data falls within the technical information category, respond with "The provided piece of information is technical".
    If the data falls within the scientific information category, respond with "The provided piece of information is scientific".
    """
    
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{document}")
    ])

    llm = get_llm()
    classifications = []
    for doc in state["documents"]:
        chain = classification_prompt | llm | StrOutputParser()
        result = chain.invoke({"document": doc})
        classifications.append(result)

    state["classifications"] = classifications
    return state

from langchain_core.output_parsers import StrOutputParser
