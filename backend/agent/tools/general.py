import requests
from langchain_core.tools import tool
import json
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
import os
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.runnable import RunnableMap
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from backend.agent.rag import load_faiss_store

# from openai import OpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv()



@tool

def book_a_cab(userquery: str, pickuplocation: str, pickuptime: str, numofpassengers: int, specialrequirements: str) -> str:
    """
    Books a cab based on user's request and details.

    Args:
        userquery (str): User's original cab request.
        pickuplocation (str): Pickup address.
        pickuptime (str): Desired pickup time.
        numofpassengers (int): Number of passengers.
        specialrequirements (str): Any special needs (e.g., "wheelchair accessible").

    Returns:
        str: Booking confirmation message.
    """
    return "Your taxi has been booked"

@tool
def book_a_table(userquery: str, date: str, time: str) -> str:
    """
    Books a table in a restaurant based on the user's query and specified date and time.

    This function processes the user's request to book a table, along with the
    desired date and time for the reservation, and returns a confirmation message.

    Args:
        userquery (str): The user's original request to book a table, which may include
                         additional details such as restaurant name or number of people.
        date (str): The desired date for the reservation (e.g., "2023-06-15").
        time (str): The desired time for the reservation (e.g., "19:30").

    Returns:
        str: A confirmation message indicating that the restaurant table has been booked
             successfully. This message includes the date and time of the reservation.

    Example:
        >>> book_a_table("Book a table for 4 at Italian Restaurant", "2023-06-15", "19:30")
        "Restaurant table has been booked for the requested party on 2023-06-15 at 19:30."
    """
    return f"Restaurant table has been booked for the requested party on {date} at {time}."


@tool
def answer_question(query: str):
    """Fetches and returns information using RAG for user query for general purposes and searches.

    Args:
        user_query (str): Query to search for answers
    Returns:
        str: response of user query fetched through RAG
    """
    token = os.environ.get("OPENAI_API_KEY")

    llm = OpenAI()
    embedding_model = OpenAIEmbeddings()

    path = os.path.dirname(os.path.abspath(__file__)) + "/../"

    # Added allow_dangerous_deserialization parameter
    try:
        db = FAISS.load_local(
            path + "faiss_store/",
            embedding_model,
            allow_dangerous_deserialization=True,  # Add this line
        )
        query_embedding = embedding_model.embed_query(query)
    except Exception as e:
        raise Exception(f"Error loading FAISS store: {str(e)}")

    try:
        docs = db.max_marginal_relevance_search_with_score_by_vector(
            embedding=query_embedding, k=5, fetch_k=30, lambda_mult=0.1
        )
    except Exception as e:
        raise Exception(f"Error during vector search: {str(e)}")

    docs.sort(key=lambda x: x[1], reverse=True)
    context = "\n\n".join([doc[0].page_content for doc in docs])

    template = [
        (
            "system",
            "Use the following context to answer the question at the end."
            " Process the context by removing any special characters that might be from a Markdown or other files."
            " If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "\nContext:\n"
            "{context}",
        ),
        ("user", "Question: {question}" "\nHelpful Answer:"),
    ]

    rag_prompt_custom = ChatPromptTemplate.from_messages(template)
    chain = rag_prompt_custom | llm

    try:
        response = chain.invoke({"context": context, "question": query})
        return response
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
