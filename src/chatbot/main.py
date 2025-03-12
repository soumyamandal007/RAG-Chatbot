#!/usr/bin/env python
import sys
import warnings
import json
from datetime import datetime
from pathlib import Path

from crew import RAGChatbot
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the RAG chatbot interactively.
    """
    chatbot = RAGChatbot()
    
    # First, ensure documents are indexed
    print("Indexing documents...")
    num_chunks = chatbot.index_documents()  # Changed from index_documents_task.execute
    print(f"Documents indexed successfully! Created {num_chunks} chunks.")
    
    print("\nRAG Chatbot initialized. Type 'exit' to quit.")
    
    while True:
        user_query = input("\nEnter your question: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        print("\nProcessing your question...")
        result = chatbot.answer_question(user_query)
        
        print("\n" + "="*50)
        print("Response:")
        print(result['response'])
        print("\n" + "="*50)
        
        if result['evaluation']:
            print("\nEvaluation Results:")
            print("-" * 30)
            for metric, data in result['evaluation'].items():
                if metric != "overall":
                    print(f"{metric.replace('_', ' ').title()}:")
                    print(f"  Score: {data['score']:.2f}")
                    print(f"  Passed: {'✓' if data['passed'] else '✗'}")
                    if data.get('suggestions'):
                        print("  Suggestions:")
                        for suggestion in data['suggestions']:
                            print(f"    - {suggestion}")
            
            print("\nOverall Assessment:")
            print(f"Score: {result['evaluation']['overall']['score']:.2f}")
            print("Strengths:", ", ".join(result['evaluation']['overall']['summary']['strengths']))
            if result['evaluation']['overall']['summary']['weaknesses']:
                print("Areas for Improvement:", ", ".join(result['evaluation']['overall']['summary']['weaknesses']))
        
        print("="*50)       

# def run():
#     """
#     Run the RAG chatbot interactively.
#     """
#     chatbot = RAGChatbot()
    
#     # First, ensure documents are indexed
#     print("Indexing documents...")
#     chatbot.index_documents_task.execute({
#         'topic': 'General Knowledge',
#         'current_year': str(datetime.now().year)
#     })
#     print("Documents indexed successfully!")
    
#     print("\nRAG Chatbot initialized. Type 'exit' to quit.")
    
#     while True:
#         user_query = input("\nEnter your question: ")
#         if user_query.lower() in ['exit', 'quit']:
#             break
        
#         print("\nProcessing your question...")
#         result = chatbot.answer_question(user_query)
        
#         print("\n" + "="*50)
#         print("Response:")
#         print(result['response'])
#         print("\n" + "="*50)
#         # print("Evaluation Summary:")
#         # print(result['evaluation'])
        
#         # Save the interaction
#         # save_interaction(result)

# def save_interaction(result):
#     """Save the interaction to a file for later analysis."""
#     interactions_dir = Path("interactions")
#     interactions_dir.mkdir(exist_ok=True)
    
#     # Create a filename based on timestamp and first few words of query
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     query_words = result['user_query'].split()[:5]
#     query_slug = "_".join(query_words).lower()
#     filename = f"{timestamp}_{query_slug}.json"
    
#     filepath = interactions_dir / filename
    
#     with open(filepath, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2)

# def batch_test():
    """
    Run batch testing on a set of predefined questions.
    """
    test_questions = [
        "What are the latest developments in AI agents?",
        "How can I implement RAG in my own application?",
        "What is the difference between CrewAI and LangChain?",
        "What are some best practices for document chunking?",
        "How can I evaluate my RAG system effectively?"
    ]
    
    chatbot = RAGChatbot()
    
    # First, ensure documents are indexed
    print("Indexing documents...")
    chatbot.index_documents_task.execute({
        'topic': 'General Knowledge',
        'current_year': str(datetime.now().year)
    })
    
    results = []
    for question in test_questions:
        print(f"\nTesting question: {question}")
        result = chatbot.answer_question(question)
        results.append(result)
    
    # Save all results
    test_results_dir = Path("test_results")
    test_results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = test_results_dir / f"batch_test_{timestamp}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to {filepath}")

if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "test":
    #     batch_test()
    # else:
    #     run()
    run()