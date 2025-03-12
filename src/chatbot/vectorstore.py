import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self):
        # Initialize the encoder for generating embeddings.
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = "document_chunks"  # Name of your Qdrant collection
        
        # Initialize Qdrant using environment variables
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Delete existing collection if it exists, then create a new one
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        print("New Qdrant collection is being created")
        dimension = self.encoder.get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        print("Entering the add_texts function")
        if not texts:
            print("No texts to add to vector store")
            return
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        processed_texts = [str(text.text) if hasattr(text, "text") else str(text) for text in texts]
        # print("This is the processed texts", processed_texts)
        
        try:
            embeddings = self.encoder.encode(processed_texts).tolist()
            points = []
            for i, (text, embedding, metadata) in enumerate(zip(processed_texts, embeddings, metadatas)):
                metadata_copy = metadata.copy()
                metadata_copy["text"] = text
                points.append(
                    PointStruct(
                        id=i,  # Changed from str(i) to i for unsigned integer
                        vector=embedding,
                        payload=metadata_copy
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Added {len(points)} points to Qdrant collection")
        except Exception as e:
            print(f"Error adding texts to vector store: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5):
        """
        Searches the Qdrant collection for texts similar to the query.
        Returns a list of results with text, associated metadata, and a similarity score.
        """
        try:
            # Generate the embedding for the query.
            query_embedding = self.encoder.encode(query).tolist()
            
            # Use query_points instead of search
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            ).points
            
            results = []
            if not search_result:
                return [{"text": "No relevant information found.", "metadata": {}, "score": 0}]
            
            for point in search_result:
                text = point.payload.get("text", "").strip() if point.payload else ""
                if text:
                    similarity = 1 - point.score if hasattr(point, 'score') else 0.0
                    results.append({
                        "text": text,
                        "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                        "score": similarity
                    })
            
            if not results:
                return [{"text": "I apologize, but I don't have any relevant information to answer this question.", "metadata": {}, "score": 0}]
            
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            # print("Search results:", results)
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return [{"text": "Error occurred during search.", "metadata": {}, "score": 0}]
