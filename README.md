# LLM Evaluator

A comprehensive system for generating evaluation datasets and comparing LLM performances across different domains.

## Features

- Generate diverse evaluation datasets across multiple domains (general knowledge, coding, mathematics)
- Parallel processing for efficient dataset generation
- Configurable evaluation criteria with weighted scoring
- Interactive HTML visualization of evaluation results
- Support for multiple models and domains
- Detailed performance metrics and analysis

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd llm_evaluator

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_api_key_here
```

2. Configure evaluation criteria in `config/evaluation.yaml`:
```yaml
evaluation:
  general:
    factual_accuracy:
      weight: 1.0
    clarity:
      weight: 1.0
    ...
```

## Usage

### 1. Generate Evaluation Dataset

```python
from generate_dataset import DatasetGenerator

# Initialize generator with desired parameters
generator = DatasetGenerator(
    domains=["general", "coding", "math"],
    samples_per_domain=30  # Adjust based on your needs
)

# Generate dataset
dataset = generator.generate()
```

The generator will:
- Create diverse examples across specified domains
- Process entities in parallel for efficiency
- Save the dataset to `datasets/evaluation_dataset_[timestamp].json`
- Create a symbolic link `datasets/evaluation_dataset_latest.json`

### 2. Evaluate Model

```python
from evaluate_model import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    test_model="gpt-3.5-turbo",  # Or your preferred model
    evaluation_dataset="datasets/evaluation_dataset_latest.json"
)

# Run evaluation
results = evaluator.evaluate()
```

### 3. View Results

```python
# View evaluation results in browser
python view_eval.py results/evaluation_results_[timestamp].json
```

## Project Structure

- `generate_dataset.py`: Asynchronous dataset generation with parallel processing
- `evaluate_model.py`: Model evaluation with weighted scoring system
- `view_eval.py`: Interactive HTML visualization of evaluation results
- `view_dataset.py`: Dataset viewer and inspector
- `config/`: Configuration files for evaluation criteria and weights
- `datasets/`: Generated evaluation datasets
- `results/`: Evaluation results and analysis

## Key Components

### Dataset Generation
- Parallel processing of entities and examples
- Queue-based example processing
- Configurable samples per domain
- Automatic metadata tracking

### Evaluation System
- Domain-specific evaluation criteria
- Weighted scoring system
- Support for various score formats (percentage, decimal)
- Detailed performance metrics

### Results Visualization
- Interactive HTML reports
- Performance breakdown by domain
- Detailed evaluation metrics
- Timing information

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Your chosen license]

## Acknowledgments

- OpenAI for API access
- Contributors and maintainers
