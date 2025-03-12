from chonkie import WordChunker, SemanticChunker
from pathlib import Path
import os

class DocumentChunker:
    def __init__(self, chunk_size=512, chunk_overlap=100, use_semantic=True):
        """
        Initialize a document chunker.
        
        Args:
            chunk_size: The target size of each chunk
            chunk_overlap: The amount of overlap between chunks
            use_semantic: Whether to use semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        
        if use_semantic:
            # Semantic chunker for more intelligent chunking
            self.chunker = SemanticChunker()
        else:
            # Basic chunker for character-based chunking
            self.chunker = WordChunker()
    
    def chunk_text(self, text):
        """Chunk text into smaller pieces."""
        return self.chunker(text)
    
    def chunk_file(self, file_path):
        """Chunk a file into smaller pieces."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        return self.chunk_text(text)
    
    def chunk_directory(self, directory_path):
        """Chunk all text files in a directory."""
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory {directory_path} not found")
        
        results = []
        
        for file_path in directory_path.glob("**/*.txt"):
            chunks = self.chunk_file(file_path)
            file_relative_path = file_path.relative_to(directory_path)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                results.append({
                    "text": chunk,
                    "metadata": {
                        "source": str(file_relative_path),
                        "chunk_index": i
                    }
                })
        
        return results

