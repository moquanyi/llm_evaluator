import os
import json
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()

class ModelEvaluator:
    def __init__(
        self,
        test_model: str,
        evaluation_dataset: str,
        output_format: str = "both"  # "text", "json", or "both"
    ):
        self.test_model = test_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.dataset = self._load_dataset(evaluation_dataset)
        self.output_format = output_format

    def _load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load the evaluation dataset."""
        with open(dataset_path, 'r') as f:
            return json.load(f)

    def _generate_model_response(self, prompt: str) -> str:
        """Generate a response from the test model."""
        response = self.client.chat.completions.create(
            model=self.test_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average score, ignoring NA values."""
        total_score = 0.0
        total_weight = 0.0
        
        for aspect, score in scores.items():
            # Skip if score is NA or aspect not in weights
            if score == "NA" or aspect not in weights:
                continue
            weight = weights[aspect]
            total_score += score * weight
            total_weight += weight
        
        # Return weighted average if we have valid scores, otherwise NA
        return total_score / total_weight if total_weight > 0 else "NA"

    def _evaluate_response(
        self,
        prompt: str,
        reference_answer: str,
        model_response: str,
        domain: str
    ) -> Dict[str, Any]:
        """Evaluate the model's response against the reference answer."""
        domain_criteria = self.dataset["evaluation_criteria"][domain]
        
        # Extract weights for each aspect
        weights = {k: v['weight'] for k, v in domain_criteria.items()}
        
        # Create domain-specific evaluation prompt
        criteria_text = "\n".join([
            f"- {criterion}: {details['description']}" 
            for criterion, details in domain_criteria.items()
        ])
        
        evaluation_prompt = f"""
Evaluate this response:
Prompt: {prompt}
Reference Answer: {reference_answer}
Model Response: {model_response}

Evaluate based on these criteria:
{criteria_text}

Provide a detailed evaluation explaining your reasoning.
"""

        score_prompt = f"""
Based on your evaluation, provide scores from 0-1 (or "NA" if not applicable) for each criterion.
Return your response in this exact JSON format:
{{
    "criterion1": score1,
    "criterion2": score2,
    ...
}}
Use only numbers (0-1) or "NA" as values. Do not include any explanation text in this response.

Criteria to score:
{str({k: v['description'] for k, v in domain_criteria.items()})}
"""

        try:
            # Get detailed evaluation
            evaluation_response = self.client.chat.completions.create(
                model=self.test_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Provide detailed evaluation of the response."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0
            )
            evaluation_text = evaluation_response.choices[0].message.content

            # Get numerical scores
            score_response = self.client.chat.completions.create(
                model=self.test_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Provide numerical scores based on the criteria."},
                    {"role": "user", "content": score_prompt}
                ],
                temperature=0
            )
            
            try:
                scores = json.loads(score_response.choices[0].message.content)
            except json.JSONDecodeError:
                print(f"Error parsing score response: {score_response.choices[0].message.content}")
                scores = {k: "NA" for k in domain_criteria.keys()}
            
            # Calculate weighted average explicitly
            overall_score = self._calculate_weighted_score(scores, weights)
            scores['overall_score'] = overall_score
            
            return {
                "text_evaluation": evaluation_text,
                "scores": scores
            }
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "text_evaluation": f"Error during evaluation: {str(e)}",
                "scores": {k: "NA" for k in domain_criteria.keys()}
            }

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the test model on the entire dataset."""
        start_time = time.time()
        
        results = {
            "metadata": {
                "test_model": self.test_model,
                "evaluation_time": datetime.now().isoformat(),
                "dataset_info": self.dataset["metadata"],
                "timing_info": {
                    "total_time": 0,
                    "by_domain": {}
                }
            },
            "evaluations": []
        }

        for item in tqdm(self.dataset["data"], desc="Evaluating responses"):
            domain = item["domain"]
            
            # Initialize domain timing if not exists
            if domain not in results["metadata"]["timing_info"]["by_domain"]:
                results["metadata"]["timing_info"]["by_domain"][domain] = {
                    "total": 0,
                    "samples": []
                }
            
            sample_timing = {}
            
            # Time model response generation
            response_start = time.time()
            model_response = self._generate_model_response(item["prompt"])
            response_end = time.time()
            sample_timing["response_generation"] = response_end - response_start
            
            # Time evaluation process
            eval_start = time.time()
            evaluation = self._evaluate_response(
                item["prompt"],
                item["reference_answer"],
                model_response,
                item["domain"]
            )
            eval_end = time.time()
            sample_timing["evaluation"] = eval_end - eval_start
            
            # Calculate total sample time
            sample_timing["total"] = eval_end - response_start
            
            # Add timing to domain stats
            results["metadata"]["timing_info"]["by_domain"][domain]["samples"].append(sample_timing)
            
            results["evaluations"].append({
                "domain": item["domain"],
                "prompt": item["prompt"],
                "reference_answer": item["reference_answer"],
                "model_response": model_response,
                "evaluation": evaluation,
                "timing": sample_timing
            })

        # Calculate domain totals
        for domain, timing in results["metadata"]["timing_info"]["by_domain"].items():
            timing["total"] = sum(s["total"] for s in timing["samples"])
            timing["avg_response_time"] = sum(s["response_generation"] for s in timing["samples"]) / len(timing["samples"])
            timing["avg_evaluation_time"] = sum(s["evaluation"] for s in timing["samples"]) / len(timing["samples"])

        # Calculate aggregate scores by domain
        results["aggregate_scores"] = {}
        for domain in set(eval["domain"] for eval in results["evaluations"]):
            domain_scores = [eval["evaluation"]["scores"] for eval in results["evaluations"] 
                           if eval["domain"] == domain]
            
            # Get all unique score types in this domain
            score_types = set()
            for scores in domain_scores:
                score_types.update(scores.keys())
            
            # Calculate average for each score type
            results["aggregate_scores"][domain] = {
                score_type: sum(s.get(score_type, 0) for s in domain_scores) / len(domain_scores)
                for score_type in score_types
            }
        
        # Calculate and add total time
        total_time = time.time() - start_time
        results["metadata"]["timing_info"]["total_time"] = total_time

        # Save results
        output_path = f"results/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create/update symbolic link to latest results
        latest_link = "results/evaluation_results_latest.json"
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(output_path), latest_link)
        
        # Print timing summary
        print("\nTiming Summary:")
        print(f"Total evaluation time: {total_time:.2f}s")
        for domain, timing in results["metadata"]["timing_info"]["by_domain"].items():
            print(f"\n{domain.capitalize()} Domain:")
            print(f"  Total time: {timing['total']:.2f}s")
            print(f"  Average response generation: {timing['avg_response_time']:.2f}s")
            print(f"  Average evaluation time: {timing['avg_evaluation_time']:.2f}s")

        return results

    def format_results(self, results: Dict[str, Any], format_type: str = None) -> str:
        """Format the evaluation results as text or JSON."""
        format_type = format_type or self.output_format
        
        if format_type == "json":
            return json.dumps(results, indent=2)
            
        text_output = f"""
Evaluation Results
=================
Test Model: {results['metadata']['test_model']}
Evaluation Time: {results['metadata']['evaluation_time']}

Aggregate Scores by Domain
-------------------------
"""
        for domain, scores in results["aggregate_scores"].items():
            text_output += f"\n{domain.capitalize()} Domain:\n"
            for metric, score in scores.items():
                text_output += f"  {metric}: {score:.2f}\n"

        text_output += f"""
Detailed Evaluations
-------------------
"""
        for eval in results["evaluations"]:
            text_output += f"\nDomain: {eval['domain']}\n"
            text_output += f"Prompt: {eval['prompt']}\n"
            text_output += f"Evaluation:\n{eval['evaluation']['text_evaluation']}\n"
            text_output += f"Scores:\n"
            for metric, score in eval['evaluation']['scores'].items():
                text_output += f"  {metric}: {score}\n"
            text_output += "-" * 80 + "\n"
        
        if format_type == "text":
            return text_output
        else:
            return text_output, results

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(
        test_model="gpt-3.5-turbo",
        evaluation_dataset="datasets/evaluation_dataset_latest.json",
        output_format="both"
    )
    results = evaluator.evaluate()
    formatted_results = evaluator.format_results(results)
