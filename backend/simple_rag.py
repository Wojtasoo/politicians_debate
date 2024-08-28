# -*- coding: utf-8 -*-
"""Zadanie1_simple_RAG.ipynb

Original file is located at
    https://colab.research.google.com/drive/1sOZ3AwxBoWPrQ_OnxH8ZOPQlxAmocX5E
"""

from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable
import argparse

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

"""#### Creating VectorDatabase"""

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from chromadb import Client

embedding_function = get_embeddings()

#path_to_file=input("Enter the path to the directory: ")
#file_name=input("Enter the name of the file: ")
parser = argparse.ArgumentParser(description="A script that processes a command-line argument.")

parser.add_argument('text', type=str, help="The text to be processed")
parser.add_argument('URL', type=str, help="URL of the text")

args = parser.parse_args()

def load_pdf(file_name, file_path):
    loader = PyPDFLoader(file_path+file_name)
    pdf_pages = loader.load()
    return pdf_pages


def split_text(text, url):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits_pypdf = splitter.split_documents(text)
    documents = []
    for i, chunk in enumerate(all_splits_pypdf):
        document = Document(
            page_content=chunk.page_content,
            metadata={
                "source": f"{url}",   
                "chunk_number": i,
                "page_number": chunk.metadata.get("page", None)
            }
        )
        documents.append(document)
    return documents

def check_URL(url):
    client = Client()
    
    collection = client.get_collection("Data")
    # Query the collection to check if any document contains the URL in its metadata.
    results = collection.get(
        where={"metadata.source": url}
    )

    # Check if any document matches the query
    if results:
        return True
    return False

db = Chroma.from_documents(
    documents=split_text(args.text, args.URL),
    embedding=embedding_function
)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 8, 'score_threshold': 0.7})

from typing_extensions import TypedDict


class AgentState(TypedDict):
    prompt: str
    llm_output: str
    documents: list[str]
    classifications: list[str]

def retrieve_docs(state: AgentState):
    prompt = state["prompt"]
    print("Prompt: "+prompt)
    documents = retriever.invoke(input=prompt)
    print("RETRIEVED DOCUMENTS:", documents)
    state["documents"] = [doc.page_content for doc in documents]
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


def rewriter(prompt):
    system = """You are a prompt re-writer that converts an input prompt to a better version that is optimized \n
        for keyword document search. Look at the input and try to reason about the underlying semantic intent / meaning.\n
        As an answer simply provide a refined version of the prompt, don't explain or argument your changes."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial prompt: \n\n {prompt} \n Formulate an improved prompt.",
            )
        ]
    )
    llm = get_llm()
    prompt_rewriter = re_write_prompt | llm | StrOutputParser()
    output = prompt_rewriter.invoke({"prompt": prompt})
    prompt = output
    return prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

@traceable
def generate_answer(state: AgentState):
    llm = get_llm()
    prompt = state["prompt"]
    context = "\n\n".join([f"{doc} -> Classified as {cls}" for doc, cls in zip(state["documents"], state["classifications"])])
    
    template = """
    Based on the following context, provide a classification of the information in the given paper.:
    {context}
    """

    prompt = ChatPromptTemplate.from_template(
        template=template,
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"prompt": prompt, "context": context})
    state["llm_output"] = result
    return state

from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("text_classification", prompt_classifier)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge("retrieve_docs", "text_classification")
workflow.add_edge("text_classification", "generate_answer")
workflow.add_edge("generate_answer", END)


workflow.set_entry_point("retrieve_docs")

app = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except:
    pass

user_prompt=input("Enter the prompt: ")
user_prompt=rewriter(user_prompt)
result = app.invoke({"prompt": f"{user_prompt}"})
print(result["llm_output"])
