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

    async def _get_domain_entities(self, domain: str) -> List[Dict[str, Any]]:
        """Get important entities for a domain."""
        prompt = (
            "You are helping to generate an evaluation dataset for testing large language models.\n"
            "For the domain: " + domain + "\n"
            "Please identify key entities (concepts, topics, or areas) that are representative of this domain. Consider:\n"
            "1. Representativeness - these entities should cover core aspects of the domain\n"
            "2. Diversity - entities should be varied to test different aspects\n"
            "3. Practical relevance - entities should be relevant to real-world applications\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '    "entities": [\n'
            "        {\n"
            '            "name": "Linear Algebra",\n'
            '            "description": "Fundamental branch dealing with linear equations and functions",\n'
            '            "difficulty_level": "intermediate",\n'
            '            "suggested_question_count": 3\n'
            "        }\n"
            "    ]\n"
            "}"
        )

        response = await self.client.chat.completions.create(
            model=self.gold_standard_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)['entities']
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print("Error parsing entities for " + domain + ": " + str(e))
            print("Response content: " + response.choices[0].message.content)
            return []

    async def _get_entity_examples(self, domain: str, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get diverse examples for an entity."""
        prompt = (
            "You are helping to generate diverse examples for testing language models.\n"
            "Domain: " + domain + "\n"
            "Entity: " + entity['name'] + "\n"
            "Description: " + entity['description'] + "\n"
            "Difficulty Level: " + entity['difficulty_level'] + "\n\n"
            "Please generate diverse examples related to this entity. Consider:\n"
            "1. Diversity - examples should cover different aspects or applications\n"
            "2. Practicality - examples should be relevant to real-world scenarios\n"
            "3. Clarity - examples should be clear and well-defined\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '    "examples": [\n'
            "        {\n"
            '            "scenario": "Solving a system of linear equations in a business context",\n'
            '            "key_points": ["Multiple variable handling", "Real-world application", "Optimization"],\n'
            '            "context": "A company needs to optimize resource allocation"\n'
            "        }\n"
            "    ]\n"
            "}"
        )

        response = await self.client.chat.completions.create(
            model=self.gold_standard_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)['examples']
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print("Error parsing examples for " + entity['name'] + ": " + str(e))
            print("Response content: " + response.choices[0].message.content)
            return []

    async def _generate_questions(self, domain: str, entity: Dict[str, Any], example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate questions for an example."""
        prompt = (
            "You are helping to generate evaluation questions for language models.\n"
            "Domain: " + domain + "\n"
            "Entity: " + entity['name'] + "\n"
            "Scenario: " + example['scenario'] + "\n"
            "Key Points: " + ', '.join(example['key_points']) + "\n"
            "Context: " + example['context'] + "\n\n"
            "Please generate questions that:\n"
            "1. Test understanding of the scenario\n"
            "2. Cover the key points\n"
            "3. Are appropriate for the difficulty level: " + entity['difficulty_level'] + "\n"
            "4. Have clear, detailed reference answers\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '    "questions": [\n'
            "        {\n"
            '            "question": "How would you solve this system of equations to optimize resource allocation?",\n'
            '            "reference_answer": "To solve this system...",\n'
            '            "focus_area": "Problem-solving methodology"\n'
            "        }\n"
            "    ]\n"
            "}"
        )

        response = await self.client.chat.completions.create(
            model=self.gold_standard_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)['questions']
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print("Error parsing questions for " + example['scenario'] + ": " + str(e))
            print("Response content: " + response.choices[0].message.content)
            return []

    async def generate_async(self) -> Dict[str, Any]:
        """Generate dataset asynchronously."""
        dataset = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "gold_standard_model": "gpt-4-turbo",
                "samples_per_domain": self.samples_per_domain,
                "domains": self.domains,
                "generation_time": 0,
                "domain_timing": {}
            },
            "evaluation_criteria": self._load_evaluation_criteria(),
            "data": []
        }

        start_time = time.time()
        
        for domain in self.domains:
            domain_start = time.time()
            print(f"\nGenerating {domain} domain examples...")
            
            # Step 1: Get domain entities
            entities = await self._get_domain_entities(domain)
            
            # Create a queue to store examples
            example_queue = asyncio.Queue()
            
            # Create tasks to get examples for each entity
            example_tasks = []
            for entity in entities:
                task = asyncio.create_task(self._process_entity(domain, entity, example_queue))
                example_tasks.append(task)
            
            # Create a task to process examples from the queue
            process_task = asyncio.create_task(self._process_examples(domain, example_queue, dataset))
            
            # Wait for all entity tasks to complete
            await asyncio.gather(*example_tasks)
            
            # Signal that no more examples will be added to the queue
            await example_queue.put(None)
            
            # Wait for the processing task to complete
            await process_task
            
            # Record timing for this domain
            domain_end = time.time()
            domain_time = domain_end - domain_start
            dataset["metadata"]["domain_timing"][domain] = domain_time
            print(f"Finished {domain} domain in {domain_time:.2f}s")

        # Record total generation time
        end_time = time.time()
        dataset["metadata"]["generation_time"] = end_time - start_time
        
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
        os.symlink(os.path.basename(output_file), os.path.join("datasets", os.path.basename(latest_link)))
        
        print(f"\nDataset saved to {output_file}")
        print(f"Symbolic link created at {latest_link}")
        
        return dataset

    def _load_evaluation_criteria(self) -> Dict[str, Any]:
        """Load evaluation criteria from config."""
        return self.config["evaluation"]

    async def _process_entity(self, domain: str, entity: Dict[str, Any], queue: asyncio.Queue) -> None:
        """Process a single entity and add its examples to the queue."""
        try:
            examples = await self._get_entity_examples(domain, entity)
            print(f"Entity: {entity}")
            print(f"Examples: {examples}")
            for example in examples:
                await queue.put((entity, example))
        except Exception as e:
            print(f"Error processing entity {entity}: {e}")

    async def _process_examples(self, domain: str, queue: asyncio.Queue, dataset: Dict[str, Any]) -> None:
        """Process examples from the queue and add them to the dataset."""
        questions_count = 0
        while True:
            item = await queue.get()
            if item is None:  # Signal to stop processing
                break
                
            entity, example = item
            try:
                questions = await self._generate_questions(domain, entity, example)
                for question in questions:
                    if questions_count >= self.samples_per_domain:
                        break
                    dataset["data"].append({
                        "domain": domain,
                        "entity": entity["name"],
                        "difficulty_level": entity["difficulty_level"],
                        "scenario": example["scenario"],
                        "prompt": question["question"],
                        "reference_answer": question["reference_answer"],
                        "focus_area": question["focus_area"],
                        "key_points": example["key_points"],
                        "context": example["context"]
                    })
                    questions_count += 1
                
                if questions_count >= self.samples_per_domain:
                    break
            except Exception as e:
                print(f"Error processing example {example}: {e}")
            finally:
                queue.task_done()

    def generate(self) -> Dict[str, Any]:
        """Synchronous wrapper for generate_async."""
        return asyncio.run(self.generate_async())

if __name__ == "__main__":
    # Example usage
    generator = DatasetGenerator(
        domains=["general", "coding", "math"],
        samples_per_domain=30,  # Generate 10 samples per domain
    )
    dataset = generator.generate()
