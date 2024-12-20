import os
import json
import yaml
from typing import Dict, Any, List, Tuple
from datetime import datetime
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
import random
import time
import statistics
import re

load_dotenv()

class DatasetGenerator:
    def __init__(
        self,
        domains: List[str],
        gold_standard_model: str = "gpt-4-turbo",
        samples_per_domain: int = 100,
        config_path: str = "config/evaluation.yaml",
    ):
        self.domains = domains
        self.gold_standard_model = gold_standard_model
        self.samples_per_domain = samples_per_domain
        self.config_path = config_path
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Ensure default_variables exist for each domain
            if 'default_variables' not in config:
                config['default_variables'] = {}
            return config

    async def _generate_single_variable(self, var_name: str, prompt: str, domain: str) -> Tuple[str, str]:
        """Generate a single variable using the model."""
        try:
            response = await self.client.chat.completions.create(
                model=self.gold_standard_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,  # Higher temperature for more diversity
                max_tokens=50
            )
            return var_name, response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating variable {var_name}: {str(e)}")
            return var_name, None

    async def _generate_variables_with_model(self, variable_prompts: Dict[str, str], domain: str) -> Dict[str, str]:
        """Generate variables using the gold standard model concurrently."""
        variables = {}
        
        # Create tasks for all variables
        tasks = [self._generate_single_variable(var_name, prompt, domain) 
                for var_name, prompt in variable_prompts.items()]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Process results in order
        for var_name, value in results:
            if value is None:
                raise RuntimeError(f"Failed to generate variable {var_name}")
            
            if var_name.endswith('_pairs'):
                # If this is a pair variable, split it into two parts
                try:
                    # Replace common separators with comma for consistent splitting
                    normalized_value = (value.replace('|', ',')
                                          .replace(' and ', ',')
                                          .replace(' vs ', ',')
                                          .replace(' - ', ',')
                                          .replace(' vs. ', ','))
                    parts = [p.strip() for p in normalized_value.split(',')]
                    
                    if len(parts) >= 2:
                        prefix = var_name.rsplit('_', 1)[0]
                        variables[f"{prefix}_1"] = parts[0]
                        variables[f"{prefix}_2"] = parts[1]
                    else:
                        raise ValueError(f"Invalid pair format for {var_name}: {value}")
                except Exception as e:
                    raise RuntimeError(f"Error processing pair variable {var_name}: {e}")
            else:
                variables[var_name] = value
        
        return variables

    async def _generate_reference_answers(self, prompts: List[str], indices: List[int], domain: str) -> List[Tuple[int, str]]:
        """Generate reference answers concurrently."""
        tasks = [self._generate_single_reference(prompt, index, domain) for index, prompt in zip(indices, prompts)]
        return await asyncio.gather(*tasks)

    async def _generate_single_reference(self, prompt: str, index: int, domain: str) -> Tuple[int, str]:
        """Generate a single reference answer."""
        try:
            # Create the messages list
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert in {domain}. Provide detailed, accurate, and comprehensive answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Make the API call
            response = await self.client.chat.completions.create(
                model=self.gold_standard_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            return (index, response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error generating reference answer: {e}")
            return (index, "Error generating reference answer.")

    def _generate_prompt_template(self, domain: str) -> tuple:
        """Generate a prompt template for the specified domain."""
        domain_config = self.config["prompt_templates"][domain]
        template = random.choice(domain_config["templates"])
        
        # Store variables and their values
        variables_used = {
            "template": template,
            "values": {}  # Will be filled by _generate_variables_with_model
        }
        
        return template, variables_used

    async def generate_async(self) -> Dict[str, Any]:
        """Generate the complete evaluation dataset using async/await."""
        start_time = time.time()
        dataset = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "gold_standard_model": self.gold_standard_model,
                "samples_per_domain": self.samples_per_domain,
                "domains": self.domains
            },
            "evaluation_criteria": self.config["evaluation"],
            "data": []
        }

        domain_timing = {}
        dataset_index = 0  # Keep track of the overall dataset index

        for domain in tqdm(self.domains, desc="Generating samples"):
            domain_start = time.time()
            print(f"\nProcessing {domain} domain...")
            
            # Generate all prompt templates first
            prompt_templates = []
            for _ in range(self.samples_per_domain):
                template, variables_used = self._generate_prompt_template(domain)
                prompt_templates.append(template)
            
            # Keep track of used prompts to avoid duplicates
            used_prompts = set()
            prompts_with_indices = []  # List of (index, prompt) tuples
            dataset_entries = []  # Store all entries before filtering
            
            # Process each template and generate variables
            for template_idx, prompt_template in enumerate(prompt_templates):
                prompt_start = time.time()
                
                # Extract variable names from the prompt using regex
                var_names = re.findall(r'\{([^}]+)\}', prompt_template)
                
                # Only generate the variables that are used in the prompt
                variable_prompts = {}
                for var_name in var_names:
                    # Handle special cases for pairs
                    if var_name.startswith(('topic_', 'concept_')):
                        base_name = var_name.rsplit('_', 1)[0] + '_pairs'
                        if base_name in self.config["default_variables"][domain]:
                            variable_prompts[base_name] = self.config["default_variables"][domain][base_name]["prompt"]
                    elif var_name in self.config["default_variables"][domain]:
                        variable_prompts[var_name] = self.config["default_variables"][domain][var_name]["prompt"]
                
                # Generate variables once
                try:
                    variables = await self._generate_variables_with_model(variable_prompts, domain)
                    
                    # Replace variables in the prompt template
                    prompt = prompt_template
                    for var_name, var_value in variables.items():
                        if var_name.endswith('_pairs'):
                            # Handle pairs (topic_pairs -> topic_1, topic_2)
                            prefix = var_name.rsplit('_', 1)[0]
                            prompt = prompt.replace(f"{{{prefix}_1}}", variables[f"{prefix}_1"])
                            prompt = prompt.replace(f"{{{prefix}_2}}", variables[f"{prefix}_2"])
                        else:
                            prompt = prompt.replace("{" + var_name + "}", var_value)
                    
                    # Only add if prompt is unique
                    if prompt not in used_prompts:
                        used_prompts.add(prompt)
                        prompts_with_indices.append((dataset_index, prompt))
                        
                        prompt_time = time.time() - prompt_start
                        
                        # Add to dataset entries
                        dataset_entries.append({
                            "domain": domain,
                            "prompt_template": prompt_template,
                            "prompt": prompt,
                            "variables": variables,
                            "timing": {
                                "prompt_generation": prompt_time,
                                "reference_generation": 0,
                                "total": prompt_time
                            }
                        })
                        dataset_index += 1
                except Exception as e:
                    print(f"Error generating variables for template {template_idx}: {e}")
                    continue
            
            # Only keep up to samples_per_domain unique prompts
            if len(dataset_entries) > self.samples_per_domain:
                dataset_entries = dataset_entries[:self.samples_per_domain]
                prompts_with_indices = prompts_with_indices[:self.samples_per_domain]
            
            # Add filtered entries to dataset
            dataset["data"].extend(dataset_entries)
            
            # Generate reference answers concurrently
            if prompts_with_indices:  # Only if we have valid prompts
                ref_start = time.time()
                prompts = [p[1] for p in prompts_with_indices]  # Extract prompts in order
                indices = [p[0] for p in prompts_with_indices]  # Keep track of original indices
                reference_answers = await self._generate_reference_answers(prompts, indices, domain)
                ref_time = time.time() - ref_start
                
                # Update dataset with reference answers in the correct order
                reference_dict = dict(reference_answers)
                for idx in indices:
                    dataset["data"][idx]["reference_answer"] = reference_dict[idx]
                    dataset["data"][idx]["timing"]["reference_generation"] = ref_time / len(prompts)
                    dataset["data"][idx]["timing"]["total"] += ref_time / len(prompts)
            
            # Calculate domain timing
            domain_end = time.time()
            domain_time = domain_end - domain_start
            domain_timing[domain] = {
                "total": domain_time,
                "avg_prompt_generation": statistics.mean(item["timing"]["prompt_generation"] 
                                                      for item in dataset["data"] 
                                                      if item["domain"] == domain),
                "avg_reference_generation": statistics.mean(item["timing"]["reference_generation"]
                                                         for item in dataset["data"]
                                                         if item["domain"] == domain)
            }

        # Add timing information to metadata
        dataset["metadata"]["timing"] = {
            "total_time": time.time() - start_time,
            "by_domain": domain_timing
        }

        # Save dataset to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("datasets", exist_ok=True)
        output_file = f"datasets/evaluation_dataset_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        # Create/update symbolic link to latest dataset
        latest_link = "datasets/evaluation_dataset_latest.json"
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(output_file), latest_link)

        # Print timing summary
        print("\nTiming Summary:")
        print(f"Total time: {dataset['metadata']['timing']['total_time']:.2f}s\n")
        for domain, timing in domain_timing.items():
            print(f"{domain.capitalize()} Domain:")
            print(f"  Total time: {timing['total']:.2f}s")
            print(f"  Average prompt generation: {timing['avg_prompt_generation']:.2f}s")
            print(f"  Average reference generation: {timing['avg_reference_generation']:.2f}s\n")

        return dataset

    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for generate_async."""
        return asyncio.run(self.generate_async())

if __name__ == "__main__":
    # Example usage
    generator = DatasetGenerator(
        domains=["general", "coding", "math"],
        samples_per_domain=15,  # Generate 10 samples per domain
    )
    dataset = generator.generate()
