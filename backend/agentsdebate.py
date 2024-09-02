import os
from dotenv import load_dotenv
from typing import Callable, List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langsmith import traceable

import requests
import json
import chromadb
import concurrent.futures
import json

load_dotenv()
db=chromadb.Client()
collection_1 = db.create_collection(name="Data_1")
collection_2 = db.create_collection(name="Data_2")

class ReadAgent:
    def __init__(self, ID: str, prompt: str, speaker_1: str, speaker_2: str, context: str, topic: str, initiate: str, system_message: SystemMessage, model: ChatOpenAI, message_history: List[str], data_cache: dict):
        self.ID = ID
        self.prompt = prompt
        self.speaker_1 = speaker_1
        self.speaker_2 = speaker_2
        self.context = context
        self.topic = topic
        self.initiate = initiate
        self.system_message = system_message
        self.model = model
        self.message_history = message_history
        self.data_cache = data_cache

    def reset(self):
        self.message_history = ["Here is the conversation so far <Conversation history>:"]

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")

    def send(self, step_: int) -> str:
        # Determine which collection to use based on the current speaker
        if step_ % 2 == 0:
            # If speaker index is even, use collection "Data_2"
            relevant_data = self.data_cache["Data_1"]
        else:
            # If speaker index is odd, use collection "Data_1"
            relevant_data = self.data_cache["Data_2"]
            
        combined_content = "\n".join(relevant_data)

        # Include the combined content in the message
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [combined_content])),
            ]
        )
        return message.content

def update_debate_data(debate_id, speaker, message):
    try:
        UPDATE_DEBATE_ENDPOINT = f"https://striped-sockeye-134.convex.site/updateDebateField"
        payload = {
            "debateId": debate_id,
            "newDebateContent": {
                "speaker": speaker,
                "message": message
            }
        }
        
        # Make POST request to Convex update function
        response = requests.post(UPDATE_DEBATE_ENDPOINT, json=payload)

        if response.status_code != 200:
            return False, f"Failed to update debate, status code {response.status_code}"

        return True, 'Debate updated successfully'

    except Exception as e:
        print(f"Error updating debate: {e}")
        return False, str(e)

def load_data(ID, prompt, speaker_1, speaker_2, system_message):
    return ReadAgent(
        ID=ID,
        prompt=prompt,
        speaker_1=speaker_1,
        speaker_2=speaker_2,
        context="",
        topic="",
        initiate="",
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
        message_history=[],
        data_cache={
        "Data_1": [],
        "Data_2": []
        }
    )

word_limit = 50  # word limit for task brainstorming

def generate_topic_and_context(agent: ReadAgent) -> dict:
    prompt = agent.prompt
    
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
            content=f"""{prompt}
            
            You are the moderator.
            1. Please make the topic more specific based on the given prompt
            2. Provide short 2-sentence context about the topic.
            3. Based on the topic and context generate two identical search queries about the political views of {agent.speaker_1} for serper_query_1 and {agent.speaker_2} for serper_query_2 respectively on the given topic.
            4. As a moderator of the debate generate initial introduction for the debate based on the topic of the debate. Speak directly to the {agent.speaker_1} and {agent.speaker_2}. Ask {agent.speaker_1} to proceed with the question first.
            Do not add anything else."""
        )
    ]
    
    # Get the response from the model
    response = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)(topic_specifier_prompt).content
    
    # Parse the JSON string response into a dictionary
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}, response: {response}")
    
    return response_dict

def check_URL(url, collection_name):
    
    collection = db.get_collection(collection_name)
    # Query the collection to check if any document contains the URL in its metadata.
    results = collection.get(
        where={"metadata.source": url}
    )

    print(results)
    # Check if any document matches the query
    if len(results["documents"]) > 0:
        return True
    return False

API_KEYS = [
    'jina_1732664b08a34983b49d52051d6c71b9V2ra2V59ec8vVL2Vz32bnFh2T4Kp',
    'jina_508b6dfd03dd4761b8270ec514b769b5Py9dFogV6lP5XvWMw9FGc3MkxkAs',
    'jina_f88228a368d4411bab039dff868c4f06vauKVTkQfkb-JcpacBYap9ZluF4m',
    'jina_a6a334ac556641d3b42504bcf0c35d140JKE7nGS-K8kKMtNMId0whJHHpqk',
    'jina_69cc95f3f6ef4ef7b19b9a1bf26e82feuN3uaw9yf0vWNddiLl4H6wBfSRZp',
    'jina_1c0850e3d3ed429292b4711168091d3b0o9b-nWoGwacNpKpKrLxJvzzxqZw'
]
call_counter = 0

def get_next_key():
    global call_counter
    key = API_KEYS[call_counter % len(API_KEYS)]
    call_counter += 1
    return key


def call_jina_api(link, token):
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    jina_url = f'https://r.jina.ai/{link}'
    response = requests.get(jina_url, headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("data", {}).get("content")  # Extract content field
    else:
        print(f"Request failed: {response.status_code}")
        return None

@traceable
def get_data_discussion(agent: ReadAgent) -> ReadAgent:
    response_dict = generate_topic_and_context(agent=agent)
    
    # Access the parsed JSON keys correctly
    query_1 = response_dict["serper_query_1"]
    query_2 = response_dict["serper_query_2"]
    queries = [query_1, query_2]
    agent.topic = response_dict["topic"]
    agent.context = response_dict["context"]
    agent.initiate = response_dict["initial_prompt"]
    
    
    update_debate_data(agent.ID, "moderator", agent.initiate)

    # Modify the output if needed
    url = "https://google.serper.dev/search"
    
    for j in range(2):
        payload = json.dumps({
            "q": queries[j],
            "gl": "en"
        })

        headers = {
            'X-API-KEY': 'cd0cc120ee4c7c12049a05c49bd1b9a1d0e6f4c8',
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")

        # Parse the JSON response
        search_results = response.json()

        # Extract the first 5 unique links from the search results
        links = []
        for result in search_results.get("organic", []):
            link = result.get("link")
            if link and link not in links:  # Ensure uniqueness
                links.append(link)
            if len(links) == 6:
                break
        
        responses = []
        # Process each link through the Jina API
        missed_links=0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_link = {
                executor.submit(call_jina_api, link, get_next_key()): link for link in links
            }
            for future in concurrent.futures.as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    content = future.result()
                    if content:
                        responses.append(content)
                    else:
                        missed_links += 1
                except Exception as exc:
                    print(f"Link {link} generated an exception: {exc}")
                    missed_links += 1

        for i in range(len(links)-missed_links):
            link=links[i]
            if j == 0:
                #if check_URL(link, "Data_1")==True:
                #    continue
                #else:
                docs = split_text(responses[i], link)
                #print("Docs_1:",docs)
                a=0
                for doc in docs:
                    for doc_dict in doc:
                        collection_1.add(
                            ids=[str(a)],
                            documents=[doc_dict["page_content"]],
                            metadatas=[doc_dict["metadata"]],
                            #embeddings=[embedding_function.embed_documents([doc_dict["page_content"]])]
                        )
                        a+=1
            else:
                #if check_URL(link, "Data_2")==True:
                #    continue
                #else:
                docs = split_text(responses[i], link)
                #print("Docs_2:",docs)
                a=0
                for doc in docs:
                    for doc_dict in doc:
                        collection_2.add(
                            ids=[str(a)],
                            documents=[doc_dict["page_content"]],
                            metadatas=[doc_dict["metadata"]],
                            #embeddings=[embedding_function.embed_documents([doc_dict["page_content"]])]
                        )
                        a+=1

        links.clear()
        responses.clear()
    
    data_cache = retrieve_relevant_data(agent.topic)
    agent.data_cache= data_cache

    return agent

def retrieve_relevant_data(topic) -> dict:
    data_cache = {
        "Data_1": [],
        "Data_2": []
    }

    # Load data from the "Data_1" collection
    collection_1 = db.get_collection("Data_1")
    results_1 = collection_1.query(query_texts=[topic], n_results=6)  # Fetch all data or a large number
    data_cache["Data_1"] = [doc for doc in results_1['documents'][0]]
    
    # Load data from the "Data_2" collection
    collection_2 = db.get_collection("Data_2")
    results_2 = collection_2.query(query_texts=[topic], n_results=6)  # Fetch all data or a large number
    data_cache["Data_2"] = [doc for doc in results_2['documents'][0]]
    
    print("Data_Cache:",data_cache)

    return data_cache


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
        self._step = 0

    def inject(self, name: str, message: str):
        # Introduce the discussion with no step increment
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1
        #print(f"({name}): {message}\n")

    def step(self) -> tuple[str, str]:
        # 1. Choose the next speaker based on current step
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        
        # 2. Next speaker sends message
        message = speaker.send(self._step)

        # 3. Everyone receives the message
        for receiver in self.agents:
            if self._step % 2 == 0:
                sender = speaker.speaker_2
                politician = "politician2"
            else:
                sender = speaker.speaker_1
                politician = "politician1"
            receiver.receive(sender, message)

        success, result = update_debate_data(speaker.ID, politician, message)
        if not success:
            print(f"Failed to update debate: {result}")

        # 4. Increment the step only if the debate was updated successfully
        #if success:
            self._step += 1

        return sender, message

def select_next_speaker(step: int, agents: List[ReadAgent]) -> int:
    # Ensure that step 0 corresponds to speaker 1 and the alternation logic
    # step 0 -> speaker 1 (index 0)
    # step 1 -> speaker 2 (index 1), and so on
    return step % len(agents)

def generate_system_message(name, agent_description, conversation_description, agent: ReadAgent) -> str:
    if name == agent.speaker_1:
        collected_data = agent.data_cache["Data_1"]
    else:
        collected_data = agent.data_cache["Data_2"]

    return f"""{conversation_description}
    
        You are a speaker in the debate as follows: {agent_description}. 

        <context_data>
        {collected_data} 
        </context_data>

        <goal_of_the_task>
            Prepare response for yourself in the debate topic of {agent.topic}. Based on this briefing you will respond to your opponent by carefully analysing the <conversation_history>. Defend your stance to convince the listeners to your point of view. be very thorough in analysing the <context_data> provided here so you have your strong opinion and stand for the debate. Criticisize the opponents ideas promoting your stance.

            Your response should not exceed 150 words.
        </goal_of_the_task>

        <steps to execute the task>

            1. Your name is {name}. Speak from the first person perspective as {name}.

            2. Your goal is to overthrow point of view of your oppoent.

            3. DO look up information from <context_data> to support your claims and refute your opponent's.
            4. DO cite your sources.
            5. Speak shortly and concisely, no more than 3 sentences. 
            
            6. DO NOT fabricate fake citations.
            7. DO NOT cite any source that you did not look up.
            8. DO NOT provide "Conclusions", or summaries of any kind at the end of your response.
            9. Avoid repeating the same information, if you have already mentioned this argument in: <Conversation history>, do not bring it up again. If you can't find any more relevant information in <context_data> to support your arguments on the topic than those arleady said, inform your opponent that you don't have anything more to add on the subject.
            
        </steps_to_execute_the_task>
        
        Do not add anything else.

        Stop speaking the moment you finish speaking from your perspective.
        """

def generate_agent_description(name, conversation_description, agent: ReadAgent) -> str:
    
    if name==agent.speaker_1:
        collected_data = agent.data_cache["Data_1"]
    else:
        collected_data = agent.data_cache["Data_2"]
    
    agent_specifier_prompt = [
        SystemMessage(
            content=f"Add detail description of the conversation participant (speaker) named {name} indicating the opinions and stance of the speaker for the debate topic."),
        HumanMessage(
            content=f"""{conversation_description}
            Please provide a description of {name}, in {word_limit} words or less.
            The description must be concise, no more than 3 sentences. 
            Description must contain point of view regarding the topic: {agent.topic} using following input data: 
            <input_data>
            {collected_data}.
            </input_data>
            
            Analyse the input data very thorough in order to summarise your stance on the debate topic of the speaker.
            
            Do not add anything else."""
        ),
    ]
    
    agent_description = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)(agent_specifier_prompt).content
    return agent_description

def start_debate(ID, prompt, speaker_1, speaker_2):
    # Generate agent descriptions and system messages
    names = {speaker_1, speaker_2}

    conversation_description = f"""Here is the topic of conversation: {prompt}
    The participants are: {', '.join(names)}"""
    
    shared_agent = get_data_discussion(
        load_data(
            ID=ID, prompt=prompt, speaker_1=speaker_1, speaker_2=speaker_2, system_message=[]
        )
    )

    agent_descriptions = {
        name: generate_agent_description(name, conversation_description, agent=shared_agent) for name in names
    }
    for name, description in agent_descriptions.items():
        print(description)

    agent_system_messages = {
        name: generate_system_message(name, agent_description, conversation_description=conversation_description, agent=shared_agent)
        for name, agent_description in zip(names, agent_descriptions.values())
    }

    # Use the shared data cache to create agents
    message=list(agent_system_messages.values())[0]
    agents = [
        ReadAgent(
            ID=ID,
            prompt=prompt,
            speaker_1=speaker_1,
            speaker_2=speaker_2,
            context=shared_agent.context,
            topic=shared_agent.topic,
            initiate=shared_agent.initiate,
            system_message=message,
            model=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
            message_history=[],
            data_cache=shared_agent.data_cache
        )
        for name, message in zip(names, agent_system_messages.values())
    ]

    simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
    simulator.inject("Moderator", agents[0].initiate)

    message_history = []
    
    def add_message(sender: str, content: str):
        message = {
            "politician": sender,
            "message": content
        }
        message_history.append(message)

    add_message("Moderator", f"{agents[0].initiate}")
    
    # Simulate conversation
    for i in range(6):
        name, message = simulator.step()
        add_message(name, message)
        #print(f"{name}: {message}\n")
    
    def display_history():
        for message in message_history:
            print(f"{message['politician']}: {message['message']} \n")
            
    print("----------------------Chat History-----------------------------------")
    display_history()

############################RAG#############################

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text, url):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits_pypdf = splitter.split_text(text)
    documents = []
    for i, chunk in enumerate(all_splits_pypdf):
        document=[
                {"page_content": chunk, "metadata": {"source": f"{url}", "chunk_number": i}}
        ]
        documents.append(document)
    return documents
