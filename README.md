# LLM Evaluator

A comprehensive system for generating evaluation datasets and comparing LLM performances.

## Features

- Generate diverse evaluation datasets across multiple domains
- Configure custom gold standard models for reference answers
- Evaluate LLM outputs against reference answers
- Generate both human-readable and structured JSON evaluation outputs
- Customizable evaluation criteria and scoring metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Evaluation Dataset

```python
from generate_dataset import DatasetGenerator

generator = DatasetGenerator(
    domains=["general", "coding", "math"],
    gold_standard_model="gpt-4",
    samples_per_domain=100
)
dataset = generator.generate()
```

### 2. Evaluate Model

```python
from evaluate_model import ModelEvaluator

evaluator = ModelEvaluator(
    test_model="your-model-name",
    evaluation_dataset="path/to/dataset.json"
)
results = evaluator.evaluate()
```

## Configuration

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

- `generate_dataset.py`: Dataset generation functionality
- `evaluate_model.py`: Model evaluation and comparison tools
- `config/`: Configuration files for evaluation criteria
- `utils/`: Utility functions and helpers
