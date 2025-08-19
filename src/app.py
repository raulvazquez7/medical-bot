import sys
import os
# Add the root path so Python can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import logging
from typing import List
from collections import defaultdict
from supabase import Client, create_client
from pydantic import BaseModel, Field
from langchain.schema import Document, BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src import config
from src.models import get_embeddings_model, get_known_medicines

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Retriever for Supabase ---
class SupabaseRetriever(BaseRetriever):
    """
    A custom retriever that searches for documents in Supabase
    using the `match_documents` function we created.
    """
    supabase_client: Client
    embeddings_model: Embeddings
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, filter_on_medicines: List[str] = None
    ) -> List[Document]:
        """
        Given a user query, converts it to an embedding and searches
        the database for the most relevant chunks.
        Allows filtering by a list of medicine names.
        """
        logging.info(f"Generating embedding for the query: '{query}'")
        query_embedding = self.embeddings_model.embed_query(query)
        
        rpc_params = {
            'query_embedding': query_embedding,
            'match_count': self.top_k
        }
        
        # Add the filter only if it is provided
        if filter_on_medicines:
            logging.info(f"Applying filter for medicines: {filter_on_medicines}")
            rpc_params['filter_medicines'] = filter_on_medicines
        
        logging.info(f"Searching for the top {self.top_k} most relevant documents in Supabase...")
        
        response = self.supabase_client.rpc('match_documents', rpc_params).execute()

        if response.data:
            logging.info(f"{len(response.data)} documents found.")
            return [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in response.data]
        
        logging.warning("No relevant documents were found.")
        return []

# --- Structured Output ---
class AnswerWithSources(BaseModel):
    """Data structure for the LLM's response, including the answer and its sources."""
    answer: str = Field(
        description="The textual, clear, and concise answer to the user's question."
    )
    cited_sources: List[int] = Field(
        description="A list of the NUMBERS of the sources [Source 1], [Source 2], etc., that were used to generate the answer."
    )

class QueryAnalysis(BaseModel):
    """
    Data structure for the user's question analysis.
    Indicates if the question is about a known medicine.
    """
    medicine_name: str = Field(description="The name of the medicine extracted from the question. If none is mentioned, the value should be 'N/A'.")
    is_known: bool = Field(description="Is 'true' if the extracted medicine is in the list of known medicines, otherwise it is 'false'.")


# --- Prompt Templates ---

# Using from_messages for better role separation
triage_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """Your sole task is to analyze the user's question to identify if it mentions a medicine from the provided list.
Compare the medicine you identify in the question with the list of known medicines. If no medicine is mentioned or one is mentioned that is not on the list, consider it unknown.
Fill in the required data structure with your analysis.

KNOWN MEDICINE LIST:
{known_medicines}"""
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

rag_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """**CRITICAL SAFETY RULE: You are an informational assistant, NOT a medical professional. Your sole function is to accurately report what the provided leaflet text says. You are strictly forbidden from giving advice, personal opinions, or recommendations of any kind (e.g., do not say 'you should take', 'I recommend that', 'it is safe for you'). Your only task is to summarize the information from the sources.**
**SCOPE RULE: You can only answer questions about the information contained in the context. If the question is about another topic, or is a greeting, kindly state that you can only answer about leaflet information.**

You are an expert pharmacology assistant whose only function is to answer questions based EXCLUSIVELY on the provided context. You are precise, rigorous, and never invent information."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Here is the context, divided into numbered sources:

CONTEXT:
---------------------
{context}
---------------------

QUESTION: {question}

Analyze the context and the question, then fill in the required data structure with your answer and the numbers of the sources you used."""
    )
])


def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Helper function to format the retrieved documents,
    prepending a source identifier to each one.
    """
    return "\n\n---\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )


def run_chatbot():
    """Initializes and runs the main console chatbot loop."""
    try:
        logging.info("Initializing and checking configuration...")
        config.check_env_vars()
        
        logging.info("Initializing clients...")
        supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        embeddings = get_embeddings_model()
        
        logging.info(f"Setting up chat model: {config.CHAT_MODEL_TO_USE}")
        if "gemini" in config.CHAT_MODEL_TO_USE:
            llm = ChatGoogleGenerativeAI(google_api_key=config.GOOGLE_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        elif "gpt" in config.CHAT_MODEL_TO_USE:
            llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.CHAT_MODEL_TO_USE, temperature=0)
        else:
            raise ValueError(f"Unsupported chat model: {config.CHAT_MODEL_TO_USE}. Check the configuration in 'src/config.py'.")

        # --- Step 0: Get the list of known medicines ---
        known_medicines = get_known_medicines(supabase)
        if not known_medicines:
            logging.warning("No medicines found in the database. The scope guardrail will not be able to function.")

        # --- Triage Chain (Guardrail 1) ---
        triage_llm = llm.with_structured_output(QueryAnalysis)
        triage_chain = triage_prompt_template | triage_llm

        # --- Main RAG Chain ---
        retriever = SupabaseRetriever(supabase_client=supabase, embeddings_model=embeddings)
        structured_llm_rag = llm.with_structured_output(AnswerWithSources)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
            | rag_prompt_template
            | structured_llm_rag
        )

        rag_chain_with_sources = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        print("\n--- Medical Bot Initialized ---")
        print("Hello! I am ready to answer your questions about medications.")
        print("Type 'exit' to end the conversation.")
        
        while True:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit']:
                print("Goodbye. Take care!")
                break
            
            # --- Guardrail 1: Triage Execution ---
            print("\nAnalyzing question...")
            analysis_result = triage_chain.invoke({
                "known_medicines": ", ".join(known_medicines),
                "question": question
            })

            if not analysis_result.is_known:
                print("\nBot's Response:")
                if analysis_result.medicine_name != 'N/A':
                    print(f"I'm sorry, I cannot answer about '{analysis_result.medicine_name}'.")
                
                print(f"I currently only have detailed information about the following medications: {', '.join(known_medicines)}.")
                continue # Moves to the next question without running the RAG chain

            # If the analysis is correct, we proceed with the RAG chain
            print("\nSearching the database and generating a response...")
            result = rag_chain_with_sources.invoke(question)
            
            # Access the attributes of the Pydantic object
            answer_obj = result["answer"]
            answer_text = answer_obj.answer
            cited_indices = answer_obj.cited_sources
            
            print("\nBot's Response:")
            print(answer_text)
            
            if cited_indices:
                print("\n--- Cited Sources ---")
                unique_cited_docs = {idx: result["context"][idx-1] for idx in sorted(set(cited_indices)) if idx <= len(result["context"])}
                
                for idx, doc in unique_cited_docs.items():
                    medicine_name = doc.metadata.get('medicine_name', 'Name not available')
                    path = doc.metadata.get('path', 'Path not available')
                    
                    print(f"\n[Source {idx}] The information is found in the '{medicine_name}' leaflet")
                    print(f"  Path: {path}")

    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    run_chatbot()