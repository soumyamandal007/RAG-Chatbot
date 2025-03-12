import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_list = pc.list_indexes()

print(index_list)