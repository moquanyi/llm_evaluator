import os
import json
from typing import Dict, Any, List, Tuple
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

    def _calculate_weighted_score(self, scores: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Calculate weighted average score, handling NA values."""
        valid_scores = 0
        weighted_sum = 0
        total_weight = 0
        
        for criterion, score in scores.items():
            if criterion == 'overall_score':
                continue
                
            weight = weights.get(criterion, 1.0)
            if score != "NA" and isinstance(score, (int, float)):
                weighted_sum += float(score) * weight
                total_weight += weight
                valid_scores += 1
        
        if valid_scores == 0 or total_weight == 0:
            return 0
            
        return weighted_sum / total_weight

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
Evaluate this response based on the given criteria. For each criterion:
1. Provide a detailed evaluation explaining your reasoning
2. Give a score from 0 to 1 (or "NA" if not applicable)

Response to evaluate:
Prompt: {prompt}
Reference Answer: {reference_answer}
Model Response: {model_response}

Evaluation Criteria:
{criteria_text}

Format your response as follows:
[Criterion Name]
Evaluation: [Your detailed evaluation]
Score: [0-1 or NA]

After evaluating all criteria, provide a JSON object containing just the scores in this format:
{{
    "criterion1": score1,
    "criterion2": score2,
    ...
}}
"""

        try:
            # Get evaluation and scores in a single call
            response = self.client.chat.completions.create(
                model=self.test_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Provide a detailed evaluation with numerical scores for each criterion."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0
            )
            
            full_response = response.choices[0].message.content
            
            # Extract the JSON scores from the end of the response
            try:
                # Find the last occurrence of a JSON-like structure
                json_start = full_response.rindex("{")
                json_end = full_response.rindex("}") + 1
                scores_text = full_response[json_start:json_end]
                scores = json.loads(scores_text)
                
                # Remove the JSON part from the evaluation text
                evaluation_text = full_response[:json_start].strip()
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Error parsing scores from response: {e}")
                scores = {k: "NA" for k in domain_criteria.keys()}
                evaluation_text = full_response
            
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
                "dataset_info": self.dataset.get("metadata", {})
            },
            "evaluations": [],
            "aggregate_scores": {},
            "timing_info": {
                "total_time": 0,
                "by_domain": {}
            }
        }
        
        # Group samples by domain
        domain_samples = {}
        for item in self.dataset.get("data", []):  
            domain = item["domain"]
            if domain not in domain_samples:
                domain_samples[domain] = []
            domain_samples[domain].append(item)
        
        # Process each domain
        for domain, samples in domain_samples.items():
            domain_start_time = time.time()
            domain_timing = {
                "total": 0,
                "samples": [],
                "avg_response_time": 0,
                "avg_evaluation_time": 0
            }
            
            # Process samples within the domain
            for item in tqdm(samples, desc=f"Evaluating {domain} responses"):
                sample_start_time = time.time()
                
                # Generate model response
                response_start_time = time.time()
                model_response = self._generate_model_response(item["prompt"])
                response_time = time.time() - response_start_time
                
                # Evaluate response
                eval_start_time = time.time()
                evaluation = self._evaluate_response(
                    item["prompt"],
                    item["reference_answer"],
                    model_response,
                    item["domain"]
                )
                eval_time = time.time() - eval_start_time
                
                # Record timing for this sample
                sample_timing = {
                    "response_generation": response_time,
                    "evaluation": eval_time,
                    "total": time.time() - sample_start_time
                }
                domain_timing["samples"].append(sample_timing)
                
                # Store evaluation results
                results["evaluations"].append({
                    "domain": item["domain"],
                    "prompt": item["prompt"],
                    "reference_answer": item["reference_answer"],
                    "model_response": model_response,
                    "evaluation": evaluation,
                    "timing": sample_timing
                })
            
            # Calculate domain timing averages
            domain_timing["total"] = time.time() - domain_start_time
            if domain_timing["samples"]:
                domain_timing["avg_response_time"] = sum(s["response_generation"] for s in domain_timing["samples"]) / len(domain_timing["samples"])
                domain_timing["avg_evaluation_time"] = sum(s["evaluation"] for s in domain_timing["samples"]) / len(domain_timing["samples"])
            
            results["timing_info"]["by_domain"][domain] = domain_timing
        
        # Calculate aggregate scores by domain
        domain_scores = {}
        for eval_result in results["evaluations"]:
            domain = eval_result["domain"]
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(eval_result["evaluation"]["scores"])
        
        # Calculate average scores for each domain
        for domain, scores_list in domain_scores.items():
            results["aggregate_scores"][domain] = {}
            # Get all score types (excluding overall_score which we'll calculate)
            score_types = set()
            for scores in scores_list:
                score_types.update(k for k in scores.keys() if k != "overall_score")
            
            # Calculate average for each score type
            for score_type in score_types:
                valid_scores = [
                    float(s[score_type]) 
                    for s in scores_list 
                    if score_type in s and s[score_type] != "NA" and isinstance(s[score_type], (int, float))
                ]
                if valid_scores:
                    results["aggregate_scores"][domain][score_type] = sum(valid_scores) / len(valid_scores)
                else:
                    results["aggregate_scores"][domain][score_type] = 0
            
            # Calculate overall score for domain
            weights = self.dataset["evaluation_criteria"][domain]
            domain_weights = {k: v["weight"] for k, v in weights.items()}
            results["aggregate_scores"][domain]["overall_score"] = self._calculate_weighted_score(
                results["aggregate_scores"][domain],
                domain_weights
            )
        
        # Record total evaluation time
        results["timing_info"]["total_time"] = time.time() - start_time
        
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

Timing Summary
-------------
Total evaluation time: {results['timing_info']['total_time']:.2f}s

"""
        # Add domain timing summaries
        for domain, timing in results['timing_info']['by_domain'].items():
            text_output += f"{domain.capitalize()} Domain:\n"
            text_output += f"  Total time: {timing['total']:.2f}s\n"
            if timing.get('samples'):
                text_output += f"  Average response generation: {timing['avg_response_time']:.2f}s\n"
                text_output += f"  Average evaluation time: {timing['avg_evaluation_time']:.2f}s\n"
            text_output += "\n"
            
        if results["aggregate_scores"]:
            text_output += "Aggregate Scores by Domain\n"
            text_output += "-------------------------\n"
            for domain, scores in results["aggregate_scores"].items():
                text_output += f"\n{domain.capitalize()} Domain:\n"
                for metric, score in scores.items():
                    text_output += f"  {metric}: {score:.2f}\n"
        
        if results["evaluations"]:
            text_output += f"""
Detailed Evaluations
-------------------
"""
            for eval in results["evaluations"]:
                text_output += f"\nDomain: {eval['domain']}\n"
                text_output += f"Prompt: {eval['prompt']}\n"
                text_output += f"Model Response: {eval['model_response']}\n"
                text_output += f"Evaluation:\n{eval['evaluation']['text_evaluation']}\n"
                text_output += f"Scores:\n"
                for metric, score in eval['evaluation']['scores'].items():
                    text_output += f"  {metric}: {score}\n"
                text_output += "-" * 80 + "\n"
        
        if format_type == "text":
            return text_output
        else:
            return text_output, results

    def save_results_to_json(self, results, output_path=None):
        """
        Save evaluation results to a JSON file and create a symbolic link to the latest results.
        
        Args:
            results (dict): Evaluation results dictionary
            output_path (str, optional): Path to save the results. If None, will generate a timestamped path.
        
        Returns:
            str: Path where the results were saved
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("results", f"evaluation_results_{timestamp}.json")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert any non-serializable objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=serialize_datetime)
        
        print(f"Results saved to {output_path}")
        # Create or update symbolic link to latest results
        latest_link = os.path.join("results", "evaluation_results_latest.json")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(output_path), latest_link)
        
        return output_path

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(
        test_model="gpt-3.5-turbo",
        evaluation_dataset="datasets/evaluation_dataset_latest.json",
        output_format="both"
    )
    results = evaluator.evaluate()
    formatted_results = evaluator.format_results(results)
    evaluator.save_results_to_json(formatted_results)
