# Agentic RAG Chatbot with CrewAI, Chonkie, and Evaluation

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot with agentic behavior using CrewAI. The system leverages semantic chunking through Chonkie and incorporates comprehensive evaluation metrics.

## Architecture Overview

### Core Components

1. **Agent Framework: CrewAI**

   - Implements two specialized agents:
     - Research Specialist: Handles context retrieval and analysis
     - Technical Content Synthesizer: Generates accurate, well-structured responses
   - Sequential process flow with task delegation and error handling
   - Maximum retry attempts configured for reliability

2. **Vector Database: Qdrant**

   - Chosen for:
     - High performance with semantic search
     - Easy scalability
     - Rich filtering capabilities
     - Cloud and self-hosted options
   - Integrated through custom tools for efficient vector operations

3. **Document Processing: Chonkie**

   - Implements semantic chunking for optimal context retrieval
   - Features:
     - Adaptive chunk sizing based on content semantics
     - Metadata preservation
     - Overlap handling for context continuity
   - Integration with vector store indexing pipeline

4. **Evaluation**
   - Comprehensive metrics implementation:
     - Contextual Precision (0-1 scale)
     - Contextual Recall (0-1 scale)
     - Contextual Relevancy (0-1 scale)
     - Answer Relevancy (0-1 scale)
     - Faithfulness (0-1 scale)
   - Real-time evaluation during response generation
   - Detailed feedback for system improvement

## Key Features

### 1. Agentic Behavior

- Dynamic task management through CrewAI
- Intelligent context retrieval with semantic search
- Adaptive response generation based on context quality
- Error handling and retry mechanisms

### 2. Advanced Chunking Strategy

- Semantic-aware document partitioning
- Optimal chunk size determination
- Metadata enrichment for improved retrieval
- Context overlap management

### 3. RAG Pipeline

- Efficient document indexing
- High-precision context retrieval
- Comprehensive response generation
- Quality evaluation and feedback

## Evaluation Metrics

### Context Retrieval Metrics

1. **Contextual Precision**

   - Measures relevance of retrieved chunks
   - Implemented using semantic similarity scoring

2. **Contextual Recall**

   - Evaluates completeness of retrieved information
   - Uses coverage analysis of key concepts

3. **Contextual Relevancy**
   - Assesses query-context alignment
   - Implements bi-directional relevance scoring

### Content Generation Metrics

1. **Answer Relevancy**

   - Measures response alignment with query
   - Evaluates comprehensiveness and accuracy

2. **Faithfulness**
   - Verifies response grounding in context
   - Checks for hallucination and accuracy

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CrewAI
- Qdrant
- Required API keys (GeminiAPI, Qdrant)

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables for API keys.

```
MODEL=
GEMINI_API_KEY=
QDRANT_API_KEY=
QDRANT_URL=
```

4. Install `crewai-tools`.

```
pip install 'crewai[tools]'
```

5. Run the chatbot.

```
python main.py
```

### Results

`user_query`: What are the key differences between L1 and L2 regularization?

`Response`:
The key differences between L1 and L2 regularization lie in the type of penalty they apply and their effect on the model coefficients.

L1 Regularization (Lasso) adds a penalty term proportional to the _absolute value_ of the coefficients. This encourages sparsity in the model by driving some coefficients to exactly zero. As a result, L1 regularization is often used for feature selection, effectively simplifying the model by excluding irrelevant features.

L2 Regularization (Ridge), on the other hand, adds a penalty term proportional to the _squared value_ of the coefficients. This shrinks the coefficients towards zero but generally does not eliminate them entirely. L2 regularization is commonly used to handle multicollinearity (high correlation between predictor variables) and to prevent overfitting by reducing the magnitude of all coefficients.

`Evaluation Results`:

- Contextual Precision:
  - Score: 0.80
  - Passed: ✓
- Contextual Recall:
  - Score: 1.00
  - Passed: ✓
- Contextual Relevancy:
  - Score: 0.80
  - Passed: ✓
- Answer Relevancy:
  - Score: 1.00
  - Passed: ✓
- Faithfulness:
  - Score: 1.00
  - Passed: ✓

Overall Assessment:
Score: 0.92
Strengths: Strong contextual_precision, Strong contextual_recall, Strong contextual_relevancy, Strong answer_relevancy, Strong faithfulness
