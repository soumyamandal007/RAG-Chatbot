# # src/chatbot/crew.py
# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from vectorstore import VectorStore
# from chunker import DocumentChunker, index_documents
# from evaluator import RAGEvaluator
# from datetime import datetime

# # Tools for the agents
# from crewai.tools import BaseTool
# from typing import Type, Optional, ClassVar
# from pydantic import BaseModel, Field

# class RetrieveInformationInput(BaseModel):
#     """Input schema for RetrieveInformation."""
#     query: str = Field(..., description="The search query to retrieve information.")
#     top_k: int = Field(5, description="Number of results to return.")

# class RetrieveInformationTool(BaseTool):
#     name: str = "Information Retrieval Tool"
#     description: str = (
#         "Use this tool to retrieve relevant information from the knowledge base."
#     )
#     args_schema: Type[BaseModel] = RetrieveInformationInput
#     vector_store = ClassVar[Optional[VectorStore]] = None

#     def _get_vector_store(self):
#         if self.vector_store is None:
#             self.vector_store = VectorStore()
#         return self.vector_store

#     def _run(self, query: str, top_k: int = 5) -> str:
#         vector_store = self._get_vector_store()
#         results = vector_store.search(query, top_k=top_k)
        
#         # Format results for readability
#         formatted_results = []
#         for i, result in enumerate(results):
#             formatted_results.append(
#                 f"Result {i+1} (Score: {result['score']:.2f}):\n"
#                 f"Source: {result['metadata'].get('source', 'Unknown')}\n"
#                 f"Text: {result['text']}\n"
#             )
        
#         return "\n".join(formatted_results)

# class IndexDocumentsInput(BaseModel):
#     """Input schema for IndexDocuments."""
#     directory: str = Field("knowledge", description="Directory containing documents to index.")

# class IndexDocumentsTool(BaseTool):
#     name: str = "Document Indexing Tool"
#     description: str = (
#         "Use this tool to index documents from a directory into the knowledge base."
#     )
#     args_schema: Type[BaseModel] = IndexDocumentsInput

#     def _run(self, directory: str = "knowledge") -> str:
#         try:
#             num_chunks = index_documents(directory)
#             return f"Successfully indexed documents from {directory}. Created {num_chunks} chunks."
#         except Exception as e:
#             return f"Error indexing documents: {str(e)}"

# class EvaluateResponseInput(BaseModel):
#     """Input schema for EvaluateResponse."""
#     query: str = Field(..., description="The user query.")
#     context: str = Field(..., description="The retrieved context.")
#     response: str = Field(..., description="The generated response.")

# class EvaluateResponseTool(BaseTool):
#     name: str = "Response Evaluation Tool"
#     description: str = (
#         "Use this tool to evaluate the quality of a RAG response."
#     )
#     args_schema: Type[BaseModel] = EvaluateResponseInput

#     def _run(self, query: str, context: str, response: str) -> str:
#         evaluator = RAGEvaluator()
#         results = evaluator.evaluate_response(query, context, response)
        
#         # Format results for readability
#         formatted_results = [
#             f"Overall Score: {results['overall_score']:.2f} (Passed: {results['overall_passed']})"
#         ]
        
#         for metric, result in results.items():
#             if metric not in ["overall_score", "overall_passed"]:
#                 formatted_results.append(
#                     f"{metric.replace('_', ' ').title()}: {result['score']:.2f} "
#                     f"(Passed: {result['passed']})\n"
#                     f"Reasoning: {result['reasoning']}"
#                 )
        
#         return "\n".join(formatted_results)

# @CrewBase
# class RAGChatbot:
#     """RAG Chatbot crew for answering questions based on knowledge base"""

#     agents_config = 'config/agents.yaml'
#     tasks_config = 'config/tasks.yaml'

#     @agent
#     def researcher(self) -> Agent:
#         return Agent(
#             config=self.agents_config['researcher'],
#             tools=[RetrieveInformationTool()],
#             verbose=True
#         )

#     @agent
#     def context_evaluator(self) -> Agent:
#         return Agent(
#             config=self.agents_config['context_evaluator'],
#             tools=[EvaluateResponseTool()],
#             verbose=True
#         )

#     @agent
#     def response_generator(self) -> Agent:
#         return Agent(
#             config=self.agents_config['response_generator'],
#             verbose=True
#         )

#     @agent
#     def knowledge_indexer(self) -> Agent:
#         return Agent(
#             config=self.agents_config['knowledge_indexer'],
#             tools=[IndexDocumentsTool()],
#             verbose=True
#         )

#     @task
#     def index_documents_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['index_documents_task'],
#             agent=self.knowledge_indexer
#         )

#     @task
#     def retrieval_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['retrieval_task'],
#             agent=self.researcher
#         )

#     @task
#     def context_evaluation_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['context_evaluation_task'],
#             agent=self.context_evaluator
#         )

#     @task
#     def response_generation_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['response_generation_task'],
#             agent=self.response_generator
#         )

#     @task
#     def evaluation_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['evaluation_task'],
#             agent=self.context_evaluator
#         )

#     @crew
#     def crew(self) -> Crew:
#         """Creates the RAG Chatbot crew"""
#         return Crew(
#             agents=self.agents,
#             tasks=self.tasks,
#             process=Process.sequential,
#             verbose=True,
#         )

#     def answer_question(self, user_query):
#         """Answer a user's question using the RAG process."""
#         inputs = {
#             'topic': 'General Knowledge',
#             'user_query': user_query,
#             'current_year': str(datetime.now().year)
#         }
        
#         # First, ensure documents are indexed
#         index_result = self.index_documents_task.execute(inputs)
        
#         # Retrieve relevant information
#         retrieved_context = self.retrieval_task.execute(inputs)
        
#         # Evaluate if the context is sufficient
#         inputs['retrieved_context'] = retrieved_context
#         evaluation_result = self.context_evaluation_task.execute(inputs)
        
#         # If the context is insufficient, retrieve more information
#         # This would be more sophisticated in a real implementation
        
#         # Generate a response based on the retrieved context
#         response = self.response_generation_task.execute(inputs)
        
#         # Evaluate the quality of the response
#         inputs['generated_response'] = response
#         evaluation = self.evaluation_task.execute(inputs)
        
#         return {
#             'user_query': user_query,
#             'retrieved_context': retrieved_context,
#             'response': response,
#             'evaluation': evaluation
#         }


from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from chatbot.src.chatbot.pinecone_vectorstore import VectorStore
from chunker import DocumentChunker
from datetime import datetime
from pathlib import Path 

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from pydantic import PrivateAttr

# Tool input schemas
class SearchInput(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(3, description="Number of results to return")

class IndexInput(BaseModel):
    directory: str = Field(..., description="Directory path containing documents to index")

# Custom tools
class SearchKnowledgeBaseTool(BaseTool):
    name: str = "Search Knowledge Base"
    description: str = "Search the knowledge base for relevant information"
    args_schema: Type[BaseModel] = SearchInput
    _vector_store: any = PrivateAttr()  # Changed to sunder name

    def __init__(self, vector_store):
        super().__init__()
        self._vector_store = vector_store

    def _run(self, query: str, top_k: int = 3) -> str:
        results = self._vector_store.search(query, top_k=top_k)
        return "\n".join([
            f"Context: {result['text']}\nScore: {result['score']}" 
            for result in results
        ])

class IndexDocumentsTool(BaseTool):
    name: str = "Index Documents"
    description: str = "Index new documents into the knowledge base"
    args_schema: Type[BaseModel] = IndexInput
    _vector_store: any = PrivateAttr()  # Changed to sunder name

    def __init__(self, vector_store):
        super().__init__()
        self._vector_store = vector_store

    def _run(self, directory: str) -> str:
        try:
            num_chunks = self._vector_store.index_documents(directory)
            return f"Successfully indexed {num_chunks} document chunks"
        except Exception as e:
            return f"Error indexing documents: {str(e)}"


@CrewBase
class RAGChatbot:
    """RAG Chatbot crew for answering questions based on knowledge base"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    vector_store = None

    def __init__(self):
        self.vector_store = VectorStore()
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="ML Research Specialist",
            goal="Find relevant ML information from knowledge base",
            backstory="Expert at finding precise ML information",
            tools=[SearchKnowledgeBaseTool(self.vector_store)],
            verbose=True,
            allow_delegation=False
        )

    @agent
    def response_generator(self) -> Agent:
        return Agent(
            role="ML Response Generator",
            goal="Generate accurate ML responses from context",
            backstory="Expert at explaining ML concepts clearly",
            tools=[SearchKnowledgeBaseTool(self.vector_store)],
            verbose=True,
            allow_delegation=False
        )
        
    # @agent
    # def knowledge_indexer(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['knowledge_indexer'],
    #         tools=[IndexDocumentsTool(self.vector_store)],
    #         verbose=True,
    #         # llm_model="ollama/llama2"
    #     )

    # @agent
    # def researcher(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['researcher'],
    #         tools=[SearchKnowledgeBaseTool(self.vector_store)],
    #         verbose=True,
    #         # llm_model="ollama/llama2"
    #     )

    # @agent
    # def response_generator(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['response_generator'],
    #         tools=[SearchKnowledgeBaseTool(self.vector_store)],
    #         verbose=True,
    #         # llm_model="ollama/llama2"
    #     )
        
    @task
    def index_documents_task(self) -> Task:
        return Task(
            description="Index new documents into the knowledge base",
            agent=self.knowledge_indexer(),  # Call the method
            expected_output="Successfully indexed documents"
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            description="Search and retrieve relevant information from the knowledge base",
            agent=self.researcher(),  # Call the method
            expected_output="Retrieved relevant context"
        )

    @task
    def response_generation_task(self) -> Task:
        return Task(
            description="Generate a response based on the retrieved context",
            agent=self.response_generator(),  # Call the method
            expected_output="Generated response based on context"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RAG Chatbot crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
    def index_documents(self, directory="knowledge"):
        """Index documents from a directory into the vector store."""
        try:
            # Resolve absolute path to knowledge directory
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
                
            # Print chunks for debugging
            # print(f"Found chunks: {chunks}")
            
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            print("This is the text     ", texts)
            print("------------------------------------")
            print("This is the metadata      ", metadatas)
            
            # Add debug print for vector store
            self.vector_store.add_texts(texts, metadatas)
            print(f"Adding {len(texts)} texts to vector store")
            return len(texts)
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return 0

    def answer_question(self, user_query):
        """Answer a user's question using the RAG process."""
        # Search for relevant context using vector store
        context = self.vector_store.search(user_query, top_k=3)
        context_text = "\n".join([result["text"] for result in context])
        
        inputs = {
            'topic': 'Machine Learning',
            'user_query': user_query,
            'context': context_text,
            'current_year': str(datetime.now().year)
        }
        
        # Generate response using the crew
        # response = self.response_generation_task.execute(inputs)
        crew_instance = self.crew() 
        response = crew_instance.kickoff(inputs = inputs)
        
        return {
            'user_query': user_query,
            'retrieved_context': context_text,
            'response': response,
            'evaluation': None  # Removed evaluation for simplicity
        }