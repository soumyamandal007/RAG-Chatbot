# from deepeval.metrics import (
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
#     ContextualRelevancyMetric,
#     AnswerRelevancyMetric,
#     FaithfulnessMetric
# )
import google.generativeai as genai
import json
from pathlib import Path
from datetime import datetime
import os

class RAGEvaluator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        self.eval_dir = Path("evaluation_results")
        self.eval_dir.mkdir(exist_ok=True)
    
    def evaluate_response(self, query, context, response):
        if isinstance(context, list):
            context = "\n\n".join(context)
        
        results = {}
        metrics = [
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "answer_relevancy",
            "faithfulness"
        ]
        
        for metric in metrics:
            score = self._evaluate_with_gemini(query, context, response, metric)
            results[metric] = {
                "score": score,
                "passed": score >= self.threshold,
                "suggestions": self._get_suggestions(metric, score)
            }
        
        # Calculate overall metrics
        scores = [result["score"] for result in results.values()]
        results["overall"] = {
            "score": sum(scores) / len(scores),
            "passed": all(result["passed"] for result in results.values()),
            "summary": self._generate_summary(results)
        }
        
        return results
    
    def _get_suggestions(self, metric_name, score):
        suggestions = {
            "contextual_precision": [
                "Consider using more specific search queries",
                "Adjust vector similarity threshold",
                "Review chunking strategy"
            ],
            "contextual_recall": [
                "Increase the number of retrieved chunks",
                "Implement semantic search improvements",
                "Review document preprocessing"
            ],
            "contextual_relevancy": [
                "Fine-tune embedding model",
                "Implement query expansion",
                "Add metadata filtering"
            ],
            "answer_relevancy": [
                "Improve prompt engineering",
                "Add structured output format",
                "Implement answer validation"
            ],
            "faithfulness": [
                "Add fact-checking",
                "Implement source attribution",
                "Strengthen context adherence"
            ]
        }
        
        if score < 0.6:
            return suggestions.get(metric_name, [])
        return []
    
    def _generate_summary(self, results):
        strengths = []
        weaknesses = []
        
        for metric, data in results.items():
            if metric != "overall":
                if data["score"] >= 0.8:
                    strengths.append(f"Strong {metric}")
                elif data["score"] < 0.6:
                    weaknesses.append(f"Weak {metric}")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_needed": len(weaknesses) > 0
        }
    
    def _evaluate_with_gemini(self, query, context, response, metric_type):
        prompt_templates = {
            "contextual_precision": """
                Evaluate the precision of the context (0-1):
                1. Does it contain specific, relevant information?
                2. Is the information directly related to the query?
                3. Are technical details accurate and well-explained?
                
                Query: {query}
                Context: {context}
                Only respond with a number between 0 and 1.
            """,
            "contextual_recall": """
                Evaluate the completeness of the response (0-1):
                1. Does it cover all key aspects from the context?
                2. Are important details from the context included?
                3. Is there a logical flow of information?
                
                Query: {query}
                Context: {context}
                Response: {response}
                Only respond with a number between 0 and 1.
            """,
            "contextual_relevancy": """
                Evaluate the relevancy and depth (0-1):
                1. How well does the context align with the query intent?
                2. Is the technical depth appropriate?
                3. Are examples or explanations provided where needed?
                
                Query: {query}
                Context: {context}
                Only respond with a number between 0 and 1.
            """,
            "answer_relevancy": """
                Evaluate response quality (0-1):
                1. Does it directly address the query?
                2. Is it well-structured and clear?
                3. Does it provide sufficient technical depth?
                4. Are concepts explained thoroughly?
                
                Query: {query}
                Response: {response}
                Only respond with a number between 0 and 1.
            """,
            "faithfulness": """
                Evaluate response accuracy (0-1):
                1. Does it strictly use information from the context?
                2. Are technical details accurately represented?
                3. Is there any information not supported by context?
                4. Are concepts explained without distortion?
                
                Context: {context}
                Response: {response}
                Only respond with a number between 0 and 1.
            """
        }
        
        prompt = prompt_templates[metric_type].format(
            query=query,
            context=context,
            response=response
        )
        
        try:
            result = self.model.generate_content(prompt)
            # Extract numerical score from response
            score = float(result.text.strip())
            return min(max(score, 0), 1)  # Ensure score is between 0 and 1
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return 0.5  # Default middle score on error