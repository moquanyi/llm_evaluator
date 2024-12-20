#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import Dict, Any, List
import markdown2
from jinja2 import Environment, FileSystemLoader
import webbrowser
from pathlib import Path
import yaml

class DatasetViewer:
    def __init__(self, dataset_path: str = "datasets/evaluation_dataset_latest.json"):
        """Initialize the dataset viewer."""
        self.dataset_path = dataset_path
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(self.template_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_dataset(self) -> Dict[str, Any]:
        """Load the dataset from JSON file."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    def process_math(self, text: str) -> str:
        """Process math equations in text."""
        # Replace $...$ with \(...\) for inline math
        text = text.replace('$', '\\(', 1)
        while '$' in text:
            text = text.replace('$', '\\)', 1)
            if '$' in text:
                text = text.replace('$', '\\(', 1)
        
        # Replace $$...$$ with \[...\] for display math
        text = text.replace('$$', '\\[', 1)
        while '$$' in text:
            text = text.replace('$$', '\\]', 1)
            if '$$' in text:
                text = text.replace('$$', '\\[', 1)
        
        return text

    def process_content(self, text: str) -> str:
        """Process text content with markdown and math."""
        # First process math equations
        text = self.process_math(text)
        
        # Then convert markdown to HTML
        html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables'])
        return html

    def create_html(self, dataset: Dict[str, Any]) -> str:
        """Create HTML content from the dataset."""
        # Load and process template
        template = self.env.get_template('dataset_view.html')
        
        # Load domain config
        config_path = os.path.join(os.path.dirname(__file__), "config/evaluation.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Process all text content
        processed_data = []
        for entry in dataset.get('data', []):
            processed_entry = entry.copy()
            if isinstance(processed_entry.get('prompt'), str):
                processed_entry['prompt'] = self.process_content(processed_entry['prompt'])
            if isinstance(processed_entry.get('reference_answer'), str):
                processed_entry['reference_answer'] = self.process_content(processed_entry['reference_answer'])
            processed_data.append(processed_entry)
        
        # Group entries by domain
        domains = {}
        for entry in processed_data:
            domain = entry.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = {
                    'entries': [],
                    'config': config.get('domains', {}).get(domain, {}),
                    'timing_stats': {
                        'total_time': 0,
                        'prompt_generation': [],
                        'reference_generation': []
                    }
                }
            domains[domain]['entries'].append(entry)
            
            # Add timing stats
            timing = entry.get('timing', {})
            if isinstance(timing, dict):
                domains[domain]['timing_stats']['prompt_generation'].append(timing.get('prompt_generation', 0))
                domains[domain]['timing_stats']['reference_generation'].append(timing.get('reference_generation', 0))
                domains[domain]['timing_stats']['total_time'] += (timing.get('prompt_generation', 0) + 
                                                                timing.get('reference_generation', 0))
        
        # Calculate averages for each domain
        for domain_data in domains.values():
            entries = domain_data['entries']
            if entries:
                prompt_times = domain_data['timing_stats']['prompt_generation']
                ref_times = domain_data['timing_stats']['reference_generation']
                if prompt_times:
                    domain_data['timing_stats']['avg_prompt_generation'] = sum(prompt_times) / len(prompt_times)
                else:
                    domain_data['timing_stats']['avg_prompt_generation'] = 0
                if ref_times:
                    domain_data['timing_stats']['avg_reference_generation'] = sum(ref_times) / len(ref_times)
                else:
                    domain_data['timing_stats']['avg_reference_generation'] = 0
        
        # Calculate total time
        total_time = sum(d['timing_stats']['total_time'] for d in domains.values())
        
        # Render template
        return template.render(
            dataset=dataset,
            domains=domains,
            metadata=dataset.get('metadata', {}),
            total_time=total_time,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            raw_json=json.dumps(dataset, indent=2)
        )

    def create_template(self):
        """Create the HTML template if it doesn't exist."""
        template_path = os.path.join(self.template_dir, 'dataset_view.html')
        if not os.path.exists(template_path):
            template_content = """<!DOCTYPE html>
<html>
<head>
    <title>Dataset Viewer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metadata, .timing-stats, .raw-json {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .domain-section {
            margin-top: 30px;
        }
        .domain-config {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .entry {
            border: 1px solid #dee2e6;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .entry:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1, h2, h3, h4 {
            color: #333;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            max-height: 400px;
        }
        code {
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9em;
            padding: 2px 4px;
            background-color: #f8f9fa;
            border-radius: 3px;
        }
        .timing {
            font-size: 0.9em;
            color: #666;
            background-color: #fff;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background-color: #fff;
        }
        th, td {
            border: 1px solid #dee2e6;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
        }
        .toggle-btn {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 10px;
        }
        .toggle-btn:hover {
            background-color: #0056b3;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dataset Viewer</h1>
        
        <div class="metadata">
            <h2>Metadata</h2>
            {% for key, value in metadata.items() %}
            <p><strong>{{ key }}:</strong> {{ value }}</p>
            {% endfor %}
        </div>

        {% for domain_name, domain_data in domains.items() %}
        <div class="domain-section">
            <h2>{{ domain_name|title }} Domain</h2>
            
            <div class="domain-config">
                <h3>Configuration</h3>
                <p><strong>Use Model Variables:</strong> {{ domain_data.config.get('use_model_variables', '') }}</p>
                
                <h4>Templates:</h4>
                <ul class="template-list">
                {% for template in domain_data.config.get('templates', []) %}
                    <li>{{ template }}</li>
                {% endfor %}
                </ul>

                <div class="timing-stats">
                    <h4>Domain Timing Statistics</h4>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Time (seconds)</th>
                        </tr>
                        <tr>
                            <td>Total Time</td>
                            <td>{{ "%.2f"|format(domain_data.timing_stats.get('total_time', 0)) }}</td>
                        </tr>
                        <tr>
                            <td>Average Prompt Generation</td>
                            <td>{{ "%.2f"|format(domain_data.timing_stats.get('avg_prompt_generation', 0)) }}</td>
                        </tr>
                        <tr>
                            <td>Average Reference Generation</td>
                            <td>{{ "%.2f"|format(domain_data.timing_stats.get('avg_reference_generation', 0)) }}</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            {% for entry in domain_data.entries %}
            <div class="entry">
                <h3>Entry {{ loop.index }}</h3>
                
                <p><strong>Template:</strong> {{ entry.get('prompt_template', '') }}</p>
                <p><strong>Generated Prompt:</strong></p>
                <div>{{ entry.get('prompt', '') }}</div>
                {% if entry.get('reference_answer') %}
                <p><strong>Reference Answer:</strong></p>
                <div>{{ entry.get('reference_answer', '') }}</div>
                {% endif %}

                <div class="timing">
                    <h4>Timing Details</h4>
                    <table>
                        <tr>
                            <th>Stage</th>
                            <th>Time (seconds)</th>
                        </tr>
                        <tr>
                            <td>Prompt Generation</td>
                            <td>{{ "%.3f"|format(entry.get('timing', {}).get('prompt_generation', 0)) }}</td>
                        </tr>
                        <tr>
                            <td>Reference Generation</td>
                            <td>{{ "%.3f"|format(entry.get('timing', {}).get('reference_generation', 0)) }}</td>
                        </tr>
                        <tr>
                            <td><strong>Total</strong></td>
                            <td><strong>{{ "%.3f"|format(entry.get('timing', {}).get('prompt_generation', 0) + entry.get('timing', {}).get('reference_generation', 0)) }}</strong></td>
                        </tr>
                    </table>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endfor %}

        <div class="raw-json">
            <h2>Raw JSON Data</h2>
            <button class="toggle-btn" onclick="toggleRawJson()">Show/Hide Raw JSON</button>
            <pre id="rawJson" class="hidden"><code>{{ raw_json }}</code></pre>
        </div>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "\\(", right: "\\)", display: false},
                    {left: "\\[", right: "\\]", display: true}
                ],
                throwOnError: false
            });
        });

        function toggleRawJson() {
            const rawJson = document.getElementById('rawJson');
            rawJson.classList.toggle('hidden');
        }
    </script>
</body>
</html>
"""
            with open(template_path, 'w') as f:
                f.write(template_content)

    def render(self) -> str:
        """Render the dataset as HTML and save to file."""
        # Create template if it doesn't exist
        self.create_template()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Create HTML content
        html_content = self.create_html(dataset)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f'dataset_view_{timestamp}.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        # Create/update symbolic link to latest view
        latest_link = os.path.join(self.output_dir, 'dataset_view_latest.html')
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(output_path), latest_link)
        
        # Open in browser
        webbrowser.open('file://' + os.path.abspath(output_path))
        
        return output_path

if __name__ == "__main__":
    viewer = DatasetViewer()
    output_path = viewer.render()
