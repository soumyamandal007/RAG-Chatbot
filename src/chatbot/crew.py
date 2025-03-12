from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from vectorstore import VectorStore
from chunker import DocumentChunker
from datetime import datetime
from pathlib import Path

from crewai_tools import QdrantVectorSearchTool
from transformers import AutoTokenizer, AutoModel
import torch

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
    limit=5,
    custom_embedding_fn=custom_embeddings,  # Pass the custom embedding function
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
            allow_delegation=False,
            max_retries=3  # Add retry attempts
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
            allow_delegation=False,
            max_retries=3  # Add retry attempts
        )
        
    @task
    def retrieval_task(self) -> Task:
        return Task(
            description="""IMPORTANT: Search for information about: {user_query}
                Steps:
                1. Use the qdrant_tool to search with the user's query
                2. If needed, perform additional searches with key terms
                3. Review and validate all search results
                4. Return only relevant information""",
            agent=self.researcher(),
            expected_output="Retrieved information that answers the query",
            context_vars=["user_query"]  # Explicitly tell the task which input variables to use
        )

    @task
    def response_generation_task(self) -> Task:
        return Task(
            description="""Create a response for: {user_query}
                Steps:
                1. Use the retrieved information from the previous task
                2. Synthesize a clear and accurate response
                3. Include relevant examples if available
                4. Ensure the response directly answers the question""",
            agent=self.response_generator(),
            expected_output="A comprehensive response that answers the user's query",
            context=[self.retrieval_task()]
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
        try:
            inputs = {
                'user_query': user_query,
                'instructions': (
                    f"Answer this specific question: {user_query}\n\n"
                    "Steps:\n"
                    "1. Search the knowledge base for relevant information\n"
                    "2. Create a clear, structured response\n"
                    "3. Include examples when available\n"
                    "4. Ensure accuracy and completeness"
                ),
                'current_year': str(datetime.now().year)
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    crew_instance = self.crew()
                    response = crew_instance.kickoff(inputs=inputs)
                    # Add evaluation using RAGEvaluator
                    from evaluator import RAGEvaluator
                    evaluator = RAGEvaluator()
                    
                    # Get context for evaluation
                    context = self.vector_store.search(user_query, top_k=5)
                    context_text = "\n".join([result["text"] for result in context])
                    
                    # Perform evaluation
                    evaluation_results = evaluator.evaluate_response(
                        query=user_query,
                        context=context_text,
                        response=response
                    )
                    
                    return {
                        'user_query': user_query,
                        'retrieved_context': context_text,
                        'response': response,
                        'evaluation': evaluation_results
                    }
                except ValueError as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed, retrying...")
                        continue
                    raise
                    
            return {
                'user_query': user_query,
                'response': "I apologize, but I encountered an error processing your query. Please try again.",
                'evaluation': None
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                'user_query': user_query,
                'response': "An error occurred while processing your request. Please try again.",
                'evaluation': None
            }