import os
import json
import yaml
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import random
import time

load_dotenv()

class DatasetGenerator:
    def __init__(
        self,
        domains: List[str],
        gold_standard_model: str = "gpt-4-turbo",
        samples_per_domain: int = 100,
        config_path: str = "config/evaluation.yaml"
    ):
        self.domains = domains
        self.gold_standard_model = gold_standard_model
        self.samples_per_domain = samples_per_domain
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation criteria configuration."""
        default_config = {
            "evaluation_criteria": {
                "accuracy": "Measure of factual correctness",
                "coherence": "Logical flow and consistency",
                "relevance": "Appropriateness to the prompt",
                "completeness": "Coverage of all aspects",
            },
            "prompt_templates": {
                "general": {
                    "templates": [
                        "Generate a question about {topic}",
                        "Compare and contrast {topic_1} and {topic_2}",
                        "Explain the concept of {topic} in simple terms"
                    ],
                    "use_model_variables": False,
                    "variable_prompts": {}
                },
                "coding": {
                    "templates": [
                        "Create a programming challenge about {data_structure} and {functionality}",
                        "Optimize the following {code} for {optimization_criteria}",
                        "Solve the problem of {problem} using {data_structure}"
                    ],
                    "use_model_variables": False,
                    "variable_prompts": {}
                },
                "math": {
                    "templates": [
                        "Prove the {theorem} for {conditions}",
                        "Solve the {equation_type} equation: {equation}",
                        "Explain the concept of {concept_1} and {concept_2} in {topic}"
                    ],
                    "use_model_variables": False,
                    "variable_prompts": {}
                }
            },
            "evaluation_instructions": {
                "general": [
                    "Provide a clear and concise answer",
                    "Use simple language and avoid jargon",
                    "Include relevant examples or anecdotes"
                ],
                "coding": [
                    "Write clean and readable code",
                    "Use proper indentation and formatting",
                    "Include comments to explain your thought process"
                ],
                "math": [
                    "Use mathematical notation and symbols correctly",
                    "Provide step-by-step solutions to equations",
                    "Explain the underlying concepts and theories"
                ]
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return default_config

    def _validate_response(self, response: str, prompt: str) -> str:
        """Validate and format the model's response according to prompt constraints."""
        response = response.strip()
        
        # Check for word limits
        if "(1 word only)" in prompt:
            words = response.split()
            if len(words) > 1:
                return words[0]
        
        elif "(1-2 words only)" in prompt:
            words = response.split()
            if len(words) > 2:
                return " ".join(words[:2])
        
        elif "(1-3 words only)" in prompt:
            words = response.split()
            if len(words) > 3:
                return " ".join(words[:3])
        
        elif "(2-4 words only)" in prompt:
            words = response.split()
            if len(words) < 2:
                return response + " example"  # Add a generic word if too short
            if len(words) > 4:
                return " ".join(words[:4])
        
        # Check for character limits
        if "max 30 characters" in prompt and len(response) > 30:
            return response[:30]
        
        if "max 50 characters" in prompt and len(response) > 50:
            return response[:50]
        
        # Special handling for code snippets
        if "def example():" in prompt:
            lines = response.split("\n")
            if len(lines) > 3:
                return "\n".join(lines[:3])
            if not response.startswith("def"):
                return f"def example():\n{response}"
        
        return response

    def _generate_variables_with_model(self, domain: str, variable_prompts: Dict[str, str]) -> Dict[str, str]:
        """Generate variables using the gold standard model with validation."""
        variables = {}
        for var_name, prompt in variable_prompts.items():
            response = self.client.chat.completions.create(
                model=self.gold_standard_model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an expert in {domain}. Generate only the requested output, no explanations. Follow the length/format constraints exactly."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            variables[var_name] = self._validate_response(response_text, prompt)
        return variables

    def _get_default_variables(self, domain: str) -> Dict[str, str]:
        """Get default variables for a domain."""
        config_path = os.path.join("config", "evaluation.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        default_vars = config.get('default_variables', {})
        if domain not in default_vars:
            return {}
        
        domain_vars = default_vars[domain]
        return {
            key: random.choice(value) if isinstance(value, list) else value
            for key, value in domain_vars.items()
        }

    def _generate_prompt(self, domain: str) -> tuple:
        """Generate a prompt for the specified domain."""
        domain_config = self.config["prompt_templates"][domain]
        template = random.choice(domain_config["templates"])
        
        # Store variables and their values
        variables_used = {}
        
        if domain_config.get("use_model_variables", False):
            variables = self._generate_variables_with_model(domain, domain_config["variable_prompts"])
        else:
            variables = self._get_default_variables(domain)
            
        # Record all variables and their values
        variables_used = {
            "template": template,
            "values": variables
        }
        
        # Format the template with the variables
        prompt = template
        for var_name, var_value in variables.items():
            prompt = prompt.replace("{" + var_name + "}", var_value)
        
        return prompt, variables_used

    def _generate_reference_answer(self, prompt: str, domain: str) -> str:
        """Generate a reference answer using the gold standard model with domain-specific instructions."""
        # Get domain-specific evaluation instructions
        instructions = self.config.get("evaluation_instructions", {}).get(domain, [])
        instruction_text = "\n".join(instructions) if instructions else ""

        system_prompt = f"""You are an expert in {domain}. Generate a high-quality response that will serve as a reference answer.
        
Your response should follow these domain-specific guidelines:
{instruction_text}

Ensure your response is:
1. Comprehensive and accurate
2. Well-structured and clear
3. Appropriate for the domain
4. Demonstrating expert-level knowledge"""

        response = self.client.chat.completions.create(
            model=self.gold_standard_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def generate(self) -> Dict[str, Any]:
        """Generate the complete evaluation dataset."""
        start_time = time.time()
        
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "gold_standard_model": self.gold_standard_model,
                "domains": self.domains,
                "samples_per_domain": self.samples_per_domain,
                "timing_info": {
                    "total_time": 0,
                    "by_domain": {}
                }
            },
            "evaluation_criteria": self.config["evaluation"],
            "data": []
        }

        for domain in self.domains:
            domain_start_time = time.time()
            domain_timing = {
                "total": 0,
                "samples": []
            }
            
            for i in tqdm(range(self.samples_per_domain), desc=f"Generating {domain} samples"):
                sample_timing = {}
                
                # Time prompt generation
                prompt_start = time.time()
                prompt, variables_used = self._generate_prompt(domain)
                prompt_end = time.time()
                sample_timing["prompt_generation"] = prompt_end - prompt_start
                
                # Time reference answer generation
                ref_start = time.time()
                reference_answer = self._generate_reference_answer(prompt, domain)
                ref_end = time.time()
                sample_timing["reference_generation"] = ref_end - ref_start
                
                # Calculate total sample time
                sample_total = ref_end - prompt_start
                sample_timing["total"] = sample_total
                
                domain_timing["samples"].append(sample_timing)
                
                dataset["data"].append({
                    "domain": domain,
                    "prompt": prompt,
                    "reference_answer": reference_answer,
                    "timing": sample_timing,
                    "prompt_metadata": variables_used
                })
            
            # Calculate domain total time
            domain_end_time = time.time()
            domain_total = domain_end_time - domain_start_time
            domain_timing["total"] = domain_total
            
            # Add domain timing to metadata
            dataset["metadata"]["timing_info"]["by_domain"][domain] = domain_timing
        
        # Calculate and add total time
        total_time = time.time() - start_time
        dataset["metadata"]["timing_info"]["total_time"] = total_time
        
        # Save dataset with timestamp
        output_path = f"datasets/evaluation_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("datasets", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        # Create/update symbolic link to latest dataset
        latest_link = "datasets/evaluation_dataset_latest.json"
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(output_path), latest_link)
        
        # Print timing summary
        print("\nTiming Summary:")
        print(f"Total time: {total_time:.2f}s")
        for domain, timing in dataset["metadata"]["timing_info"]["by_domain"].items():
            print(f"\n{domain.capitalize()} Domain:")
            print(f"  Total time: {timing['total']:.2f}s")
            avg_prompt = sum(s["prompt_generation"] for s in timing["samples"]) / len(timing["samples"])
            avg_ref = sum(s["reference_generation"] for s in timing["samples"]) / len(timing["samples"])
            print(f"  Average prompt generation: {avg_prompt:.2f}s")
            print(f"  Average reference generation: {avg_ref:.2f}s")

        return dataset

if __name__ == "__main__":
    # Example usage
    generator = DatasetGenerator(
        domains=["general", "coding", "math"],
        samples_per_domain=2,  # Small sample for testing
    )
    dataset = generator.generate()
