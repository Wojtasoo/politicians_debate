import streamlit as st
import os
from dotenv import load_dotenv
from typing import Callable, List

from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langsmith import traceable

from convex import ConvexClient
import requests
import json

load_dotenv()

client = ConvexClient(os.getenv("CONVEX_URL"))
#print(client.query("debaterequest:getDebaterequest"))

class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI, topic: str) -> None:
        self.data = {
            "name": name,
            "system_message": system_message,
            "model": model,
            "topic": topic,
            "message_history": ["Here is the conversation so far."]
        }

    def reset(self):
        self.data["message_history"] = ["Here is the conversation so far."]

    def send(self) -> str:
        message = self.data["model"].invoke(
            [
                self.data["system_message"],
                HumanMessage(content="\n".join(self.data["message_history"] + [f"{self.data['name']}: "])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        self.data["message_history"].append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
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
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.data["name"], message)

        # 4. increment time
        self._step += 1

        return speaker.data["name"], message

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        topic: str,
        politician: str,
        search_output: List[str],
        #tool_names: List[str],
        **tool_kwargs,
    ) -> None:
        super().__init__(name, system_message, model, topic)
        #self.tools = load_tools(tool_names, **tool_kwargs)
        self.politician = politician
        self.search_output = search_output

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(
            # self.tools,
            self.data["model"],
            self.search_output,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
        )
        
        query = f"Based on the following: {prompt_for_debate_topic(self)}. Generate search query about the political views of {self.politician} on the given topic."
                        
        # Modify the output if needed
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query,
            "gl": "pl"
        })

        headers = {
            'X-API-KEY': 'cd0cc120ee4c7c12049a05c49bd1b9a1d0e6f4c8',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

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
        
        headers = {'Accept': 'application/json','Authorization': 'Bearer jina_60d6d9ee5b774a548573f4197af013c10mFxfHr5oEiI82i4EWK_ox_O0Z74'}

        # List to store the responses
        responses = []

        # Process each link through the Jina API
        for link in links:
            url = f'https://r.jina.ai/{link}'
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                response_json = response.json()
                content = response_json.get("data", {}).get("content")  # Extract content field
                if content:
                    responses.append(content)
                else:
                    raise Exception(f"Content not found in response: {response_json}")
            else:
                raise Exception(f"Request failed: {response.status_code}")


        self.data["search_output"] = responses
        #self.search_output = responses
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.data["system_message"].content] + self.data["message_history"] + [f"{self.data['name']}" + self.data["search_output"]]
                )
            )
        )

        return message.content

names = {
    "Supporter",
    "Opposer",
}
global topic 
topic="" #= "The current impact of automation and artificial intelligence on employment"
word_limit = 50  # word limit for task brainstorming

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names)}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)


def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
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

def generate_system_message(name, description):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}
Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

Do not add anything else.

Stop speaking the moment you finish speaking from your perspective.
"""


agent_system_messages = {
    name: generate_system_message(name, description)
    for name, description in zip(names, agent_descriptions.values())
}

for name, system_message in agent_system_messages.items():
    print(name)
    print(system_message)

def prompt_for_debate_topic(agent: DialogueAgent) -> str:
    topic=agent.data["topic"]
    topic_specifier_prompt = [
        SystemMessage(content="You can make a topic more specific."),
        HumanMessage(
            content=f"""{topic}
            
            You are the moderator.
            Please make the topic more specific and provide short 2 sentance context about the topic.
            Please reply with the specified quest in {word_limit} words or less. 
            Speak directly to the participants: {*names,}.
            Do not add anything else."""
        ),
    ]
    specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

    print(f"Original topic:\n{topic}\n")
    print(f"Detailed topic:\n{specified_topic}\n")
    return specified_topic


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx


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
                DialogueAgentWithTools(
                    name=name,
                    system_message=SystemMessage(content=system_message),
                    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
                    topic=topic,
                    politician=left_option,
                    search_output=[],
                    #tool_names=tools,
                    top_k_results=2,
                )
                for name, system_message in zip(
                    names, agent_system_messages.values()
                )
            ]
            
            extra_agent = DialogueAgentWithTools(
            name=name,  # Unique name for the new agent
            system_message=SystemMessage(content="You are the moderator of the debate."),
            model=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
            topic=topic,
            politician=right_option,  # You can use another politician or any other config
            search_output=[],
            #tool_names=tools,  # Omit if not using tools
            top_k_results=2,
        )

            # Append the new agent to the existing list of agents
            agents.append(extra_agent)
            
            simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
            simulator.reset()
            simulator.inject("Moderator", prompt_for_debate_topic(agents[0]))
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

