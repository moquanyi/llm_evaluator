import os
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time
from collections import defaultdict

load_dotenv()

class ModelEvaluator:
    def __init__(
        self,
        test_model: str,
        evaluation_dataset: str,
        output_format: str = "both"  # "text", "json", or "both"
    ):
        self.test_model = test_model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.dataset = self._load_dataset(evaluation_dataset)
        self.output_format = output_format

    def _load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load the evaluation dataset."""
        with open(dataset_path, 'r') as f:
            return json.load(f)

    async def _generate_single_response(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single response using the test model."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": sample["prompt"]
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.test_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            result = {
                "domain": sample["domain"],
                "prompt": sample["prompt"],
                "reference_answer": sample["reference_answer"],
                "model_response": response.choices[0].message.content,
                "timing": {"response_generation": response.usage.total_tokens / 1000.0}
            }
            
            # Preserve original index if it exists
            if 'original_index' in sample:
                result['original_index'] = sample['original_index']
                
            return result
        except Exception as e:
            print(f"Error generating response: {e}")
            result = {
                "domain": sample["domain"],
                "prompt": sample["prompt"],
                "reference_answer": sample["reference_answer"],
                "model_response": f"Error during generation: {str(e)}",
                "timing": {"response_generation": 0}
            }
            
            # Preserve original index if it exists
            if 'original_index' in sample:
                result['original_index'] = sample['original_index']
                
            return result

    async def _evaluate_single_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single response."""
        try:
            domain_criteria = self.dataset["evaluation_criteria"][response["domain"]]
            aspects_text = "\n".join([
                f"- {k}: {v['description']}\n  Aspects to consider: {', '.join(v['aspects'])}"
                for k, v in domain_criteria.items()
            ])
            
            evaluation_prompt = f"""
Evaluate the following response against the reference answer for a {response["domain"]} domain question.

Question: {response["prompt"]}

Model Response:
{response["model_response"]}

Reference Answer:
{response["reference_answer"]}

Evaluate the response based on these criteria:
{aspects_text}

1. Provide a detailed evaluation explaining your reasoning
2. For each criterion, assign a score between 0 and 1 (1 being perfect)
3. Format your response as follows:

[criterion_name]
Evaluation: [Your detailed evaluation]
Score: [0-1]

Repeat for each criterion, then provide a JSON object at the end with all scores.
"""
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Provide a detailed evaluation with numerical scores for each criterion."
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ]
            
            eval_response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            evaluation_text = eval_response.choices[0].message.content
            scores = self.calculate_scores(evaluation_text, response["domain"])
            
            result = {
                **response,  # Preserve all original fields including original_index
                "evaluation": {
                    "text_evaluation": evaluation_text,
                    "scores": scores
                },
                "timing": {
                    **response["timing"],
                    "evaluation": eval_response.usage.total_tokens / 1000.0,
                    "total": response["timing"]["response_generation"] + eval_response.usage.total_tokens / 1000.0
                }
            }
            
            return result
        except Exception as e:
            print(f"Error in evaluation: {e}")
            result = {
                **response,  # Preserve all original fields including original_index
                "evaluation": {
                    "text_evaluation": f"Error during evaluation: {str(e)}",
                    "scores": {}
                },
                "timing": {
                    **response["timing"],
                    "evaluation": 0,
                    "total": response["timing"]["response_generation"]
                }
            }
            
            return result

    async def _batch_generate_responses(self, samples: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """Generate responses for a batch of samples in parallel."""
        # Add original index to each sample
        for i, sample in enumerate(samples):
            sample['original_index'] = i

        # Process in batches
        all_responses = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            tasks = [self._generate_single_response(sample) for sample in batch]
            batch_responses = await asyncio.gather(*tasks)
            all_responses.extend(batch_responses)

        # Sort responses by original index and remove the index
        all_responses.sort(key=lambda x: x['original_index'])
        for response in all_responses:
            del response['original_index']

        return all_responses

    async def _batch_evaluate_responses(self, responses: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """Evaluate a batch of responses concurrently."""
        # Add original index to each response
        for i, response in enumerate(responses):
            response['original_index'] = i

        # Process in batches
        all_evaluations = []
        for i in range(0, len(responses), batch_size):
            batch = responses[i:i + batch_size]
            tasks = [self._evaluate_single_response(response) for response in batch]
            batch_evaluations = await asyncio.gather(*tasks)
            all_evaluations.extend(batch_evaluations)

        # Sort evaluations by original index and remove the index
        all_evaluations.sort(key=lambda x: x['original_index'])
        for evaluation in all_evaluations:
            del evaluation['original_index']

        return all_evaluations

    async def evaluate_async(self, batch_size: int = 5) -> Dict[str, Any]:
        """Evaluate the test model's performance using async/await."""
        start_time = time.time()
        
        results = {
            "metadata": {
                "test_model": self.test_model,
                "evaluation_time": datetime.now().isoformat(),
            },
            "timing_info": {
                "total_time": 0,
                "by_domain": {}
            },
            "evaluations": [],
            "aggregate_scores": {}
        }

        # Group samples by domain
        domain_samples = {}
        for sample in self.dataset["data"]:
            domain = sample["domain"]
            if domain not in domain_samples:
                domain_samples[domain] = []
            domain_samples[domain].append(sample)

        # Process each domain
        for domain, samples in domain_samples.items():
            domain_start = time.time()
            print(f"\nProcessing {domain} domain...")
            
            # Generate responses in batches
            responses = await self._batch_generate_responses(samples, batch_size)
            
            # Evaluate responses in batches
            evaluated_responses = await self._batch_evaluate_responses(responses, batch_size)
            
            # Add to results
            results["evaluations"].extend(evaluated_responses)
            
            # Calculate domain timing
            domain_end = time.time()
            domain_time = domain_end - domain_start
            results["timing_info"]["by_domain"][domain] = {
                "total": domain_time,
                "samples": evaluated_responses
            }

        # Calculate aggregate scores
        self._calculate_aggregate_scores(results)
        
        # Calculate total time
        end_time = time.time()
        results["timing_info"]["total_time"] = end_time - start_time
        
        return results

    def evaluate(self, batch_size: int = 5) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate_async."""
        return asyncio.run(self.evaluate_async(batch_size))

    def calculate_scores(self, evaluation_text: str, domain: str) -> Dict[str, Any]:
        """Calculate scores from the evaluation text."""
        try:
            # Try to find JSON at the end of the text
            json_start = evaluation_text.rindex("{")
            json_end = evaluation_text.rindex("}") + 1
            scores_text = evaluation_text[json_start:json_end]
            scores = json.loads(scores_text)
        except (ValueError, json.JSONDecodeError):
            # If no JSON found, parse the text manually
            scores = {}
            current_criterion = None
            for line in evaluation_text.splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current_criterion = line[1:-1].strip()
                elif line.startswith("Score:") and current_criterion:
                    try:
                        scores[current_criterion] = float(line.split(":")[-1].strip())
                    except ValueError:
                        scores[current_criterion] = 0.0

        # Ensure all criteria have scores
        domain_criteria = self.dataset["evaluation_criteria"][domain]
        for criterion in domain_criteria:
            if criterion not in scores:
                scores[criterion] = 0.0

        # Calculate overall score
        weights = {k: v["weight"] for k, v in domain_criteria.items()}
        weighted_sum = sum(scores[k] * weights[k] for k in domain_criteria)
        total_weight = sum(weights.values())
        scores["overall_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0

        return scores

    def _calculate_aggregate_scores(self, results: Dict[str, Any]) -> None:
        """Calculate aggregate scores by domain."""
        domain_scores = {}
        
        # Initialize domain scores
        for eval_result in results["evaluations"]:
            domain = eval_result["domain"]
            if domain not in domain_scores:
                domain_scores[domain] = {
                    "count": 0,
                    "scores": defaultdict(float)
                }
            
            # Add scores
            scores = eval_result["evaluation"]["scores"]
            domain_scores[domain]["count"] += 1
            for criterion, score in scores.items():
                domain_scores[domain]["scores"][criterion] += score

        # Calculate averages and store in results
        results["aggregate_scores"] = {}
        for domain, data in domain_scores.items():
            count = data["count"]
            if count > 0:
                results["aggregate_scores"][domain] = {
                    criterion: score / count
                    for criterion, score in data["scores"].items()
                }

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
                text_output += f"  Average response generation: {sum(s['timing']['response_generation'] for s in timing['samples']) / len(timing['samples']):.2f}s\n"
                text_output += f"  Average evaluation time: {sum(s['timing']['evaluation'] for s in timing['samples']) / len(timing['samples']):.2f}s\n"
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
