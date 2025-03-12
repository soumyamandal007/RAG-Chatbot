import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self):
        # Initialize the encoder for generating embeddings.
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_name = "document-chunks"
        
        # Initialize Pinecone using API key and environment from .env
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if the index exists; if not, create it.
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
        print("New Database is being created")
        dimension = self.encoder.get_sentence_embedding_dimension()
        self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    ),
                tags={
                            "environment": "development"
                        }
            )
        
        # Connect to the index.
        self.index = self.pc.Index(self.index_name)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Adds texts and their metadata to the vector store.
        Each text is embedded and upserted into the Pinecone index.
        """
        print("Entering the add_tesxts function")
        if not texts:
            print("No texts to add to vector store ")
            return
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        processed_texts = []
        for text in texts:
            if hasattr(text, "text"):
                processed_texts.append(str(text.text))
            else:
                processed_texts.append(str(text))
        print("This is the processed texts", processed_texts)        
        
        try:
            # Generate embeddings for processed texts
            embeddings = self.encoder.encode(processed_texts).tolist()
            
            # Prepare vectors for upsert
            vectors = []
            for i, (text, embedding, metadata) in enumerate(zip(processed_texts, embeddings, metadatas)):
                metadata_copy = metadata.copy()
                metadata_copy["text"] = text
                vectors.append((str(i), embedding, metadata_copy))
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                print(f"Indexed batch {i//batch_size + 1} of {(len(vectors)-1)//batch_size + 1}")
                
        except Exception as e:
            print(f"Error adding texts to vector store: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5):
        """
        Searches the vector store for texts similar to the query.
        Returns a list of results with the text, associated metadata, and similarity score.
        """
        try:
            # Generate the embedding for the query.
            query_embedding = self.encoder.encode(query).tolist()
            
            # Perform the query on the Pinecone index.
            result = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            matches = result.get("matches", [])
            
            if not matches:
                return [{"text": "No relevant information found.", "metadata": {}, "score": 0}]
            
            # Process and return the results, filtering out empty texts
            results = []
            for match in matches:
                text = match["metadata"].get("text", "").strip()
                if text:  # Only include non-empty texts
                    results.append({
                            "text": text,
                            "metadata": {k: v for k, v in match["metadata"].items() if k != "text"},
                            "score": match["score"]
                        })
                        
            if not results:
                return [{"text": "I apologize, but I don't have any relevant information to answer this question.", 
                        "metadata": {}, 
                        "score": 0}]
            
            # Sort results by score in descending order
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            print("Search results:", results)
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return [{"text": "Error occurred during search.", "metadata": {}, "score": 0}]
