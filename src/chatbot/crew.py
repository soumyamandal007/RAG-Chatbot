from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
# from chatbot.src.chatbot.pinecone_vectorstore import VectorStore
from vectorstore import VectorStore
from chunker import DocumentChunker
from datetime import datetime
from pathlib import Path

from crewai_tools import QdrantVectorSearchTool
from transformers import AutoTokenizer, AutoModel
import torch

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from pydantic import PrivateAttr
import os
from dotenv import load_dotenv
load_dotenv()


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def custom_embeddings(text: str) -> list[float]:
    # Tokenize and get model outputs
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    
    # Use mean pooling to get text embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Convert to list of floats and return
    return embeddings[0].tolist()

# Initialize Qdrant search tool
qdrant_tool = QdrantVectorSearchTool(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="document_chunks",
    limit=3,
    custom_embedding_fn=custom_embeddings,  # Pass the custom embedding function
    filter_by=None,  # Add this
    filter_value=None , # Add this
    score_threshold=0.01,
)

@CrewBase
class RAGChatbot:
    """Simplified RAG Chatbot with 2 agents and 2 tasks for basic functionality."""
    vector_store = None
    agents_config = {}  # Add this line to initialize agents_config
    tasks_config = {}   # Add this line to initialize tasks_config

    def __init__(self):
        self.vector_store = VectorStore()
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Research Specialist",
            goal="Retrieve and analyze comprehensive context from the knowledge base with high precision",
            backstory="""Expert information retrieval specialist who:
                - Excels at semantic search and context analysis
                - Understands technical documentation deeply
                - Identifies key concepts and relationships
                - Ensures retrieved information is relevant and complete""",
            tools=[qdrant_tool],
            allow_delegation=False
        )

    @agent
    def response_generator(self) -> Agent:
        return Agent(
            role="Technical Content Synthesizer",
            goal="""Create comprehensive, well-structured responses that:
                - Directly address the user's specific query
                - Synthesize information from all relevant context
                - Maintain technical accuracy and clarity
                - Include practical examples and explanations""",
            backstory="""Senior technical writer and ML expert who:
                - Specializes in explaining complex technical concepts
                - Creates clear, structured documentation
                - Ensures responses are relevant and actionable
                - Maintains coherent narrative flow
                - Balances technical depth with accessibility""",
            tools=[qdrant_tool],
            allow_delegation=False
        )
        
    @task
    def retrieval_task(self) -> Task:
        return Task(
            description="""Analyze the query and retrieve comprehensive context by:
                1. Understanding the core question and related concepts
                2. Performing targeted searches with appropriate keywords
                3. Ensuring all relevant aspects are covered
                4. Validating context completeness and relevance
                5. Organizing retrieved information logically""",
            agent=self.researcher(),
            expected_output="Comprehensive and relevant context information, organized by topic"
        )

    @task
    def response_generation_task(self) -> Task:
        return Task(
            description="""Create a detailed, well-structured response that:
                1. Directly addresses the user's specific question
                2. Synthesizes all relevant context into a coherent narrative
                3. Provides clear explanations with examples
                4. Uses appropriate technical depth
                5. Maintains focus on the query's core topic
                6. Includes practical implications where relevant""",
            agent=self.response_generator(),
            expected_output="Comprehensive, relevant, and well-structured response with clear examples and explanations"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RAG Chatbot crew with retrieval and response generation tasks."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
    
    def index_documents(self, directory="knowledge"):
        """Index documents from a directory into the vector store."""
        try:
            base_path = Path(__file__).parent.parent.parent
            knowledge_dir = base_path / directory
            print(f"Looking for documents in: {knowledge_dir}")
            
            if not knowledge_dir.exists():
                raise Exception(f"Knowledge directory not found at {knowledge_dir}")
            if not knowledge_dir.is_dir():
                raise Exception(f"{knowledge_dir} is not a directory")
                
            chunks = DocumentChunker().chunk_directory(str(knowledge_dir))
            if not chunks:
                raise Exception("No documents found in knowledge directory")
                
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            print("Texts:", texts)
            print("Metadata:", metadatas)
            
            self.vector_store.add_texts(texts, metadatas)
            print(f"Added {len(texts)} texts to vector store")
            return len(texts)
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return 0

    def answer_question(self, user_query):
        """Answer a user's question using the retrieval and response generation tasks."""
        # Retrieve context from the vector store
        context = self.vector_store.search(user_query, top_k=3)
        context_text = "\n".join([result["text"] for result in context])
        
        inputs = {
            'topic': 'Machine Learning',
            'user_query': user_query,
            'retrieved_context': context_text,
            # 'instructions': (
            #     "Generate a clear, structured response using only the information from the provided context. "
            #     "When invoking the 'Search Knowledge Base' tool, please return a JSON object in the following format: "
            #     "{\"query\": \"<your query>\", \"top_k\": 3}. Do not include any extra keys such as 'description' or 'type'."
            # ),
            'instructions': (
            "Generate a clear, structured response using only the provided context. "
            "When calling the 'qdrant_tool' tool, please return a valid JSON object with only the keys 'query' and 'top_k'. "
            "For example: {\"query\": \"What factors affect the performance of a machine learning model?\", \"top_k\": 3}. "
            "Do not include any extra keys such as 'description' or 'type'."
        ),
            'current_year': str(datetime.now().year)
        }
        
        # Create the crew and execute tasks sequentially
        crew_instance = self.crew() 
        response = crew_instance.kickoff(inputs=inputs)
        
        # Add evaluation
        from evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        evaluation_results = evaluator.evaluate_response(
            query=user_query,
            context=context_text,
            response=response
        )
        
        return {
            'user_query': user_query,
            'retrieved_context': context_text,
            'response': response,
            'evaluation': evaluation_results  # Evaluation is omitted for now
        }
