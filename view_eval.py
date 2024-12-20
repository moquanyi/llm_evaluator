#!/usr/bin/env python3

import json
import os
from typing import Dict, Any, List
import markdown2
from jinja2 import Environment, FileSystemLoader
import webbrowser

class EvalViewer:
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(self.template_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_results(self) -> List[Dict[str, Any]]:
        """Load evaluation results."""
        with open(self.results_path, 'r') as f:
            data = json.load(f)
            # The first element is a string containing the summary
            # The second element contains the actual evaluation data
            if isinstance(data, list) and len(data) > 1:
                if isinstance(data[1], dict) and 'timing_info' in data[1]:
                    timing_info = data[1]['timing_info']
                    if isinstance(timing_info, dict) and 'by_domain' in timing_info:
                        # Extract samples from each domain
                        samples = []
                        for domain_data in timing_info['by_domain'].values():
                            if isinstance(domain_data, dict) and 'samples' in domain_data:
                                samples.extend(domain_data['samples'])
                        return samples
            return []

    def process_math(self, text: str) -> str:
        """Process math equations in text."""
        if not isinstance(text, str):
            return str(text)
        
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
        if not isinstance(text, str):
            return str(text)
        
        # Process math equations first
        text = self.process_math(text)
        
        # Convert markdown to HTML, preserving whitespace
        try:
            text = text.replace('\n', '  \n')  # Preserve line breaks
            html = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables'])
            return html
        except Exception as e:
            print(f"Error processing markdown: {e}")
            return text

    def calculate_domain_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each domain."""
        domain_stats = {}
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            domain = result.get('domain', 'unknown')
            if domain not in domain_stats:
                domain_stats[domain] = {'count': 0, 'scores': {}}
            
            stats = domain_stats[domain]
            stats['count'] += 1
            
            # Aggregate scores
            scores = result.get('evaluation', {}).get('scores', {})
            for metric, score in scores.items():
                if isinstance(score, (int, float)):
                    if metric not in stats['scores']:
                        stats['scores'][metric] = []
                    stats['scores'][metric].append(score)
        
        # Calculate averages and format them
        for domain, stats in domain_stats.items():
            avg_scores = {}
            for metric, scores in stats['scores'].items():
                if scores:
                    avg = sum(scores) / len(scores)
                    avg_scores[metric] = {
                        'value': avg,
                        'color': 'high-score' if avg >= 0.8 else ('low-score' if avg < 0.6 else '')
                    }
            stats['average_scores'] = avg_scores
        
        return domain_stats

    def create_html(self, results: List[Dict[str, Any]]) -> str:
        """Create HTML content."""
        template = self.env.get_template('eval_view.html')
        
        # Process results
        processed_results = []
        for result in results:
            if not isinstance(result, dict):
                continue
            
            # Get scores and determine colors
            scores = result.get('evaluation', {}).get('scores', {})
            formatted_scores = {}
            for metric, score in scores.items():
                if isinstance(score, (int, float)):
                    formatted_scores[metric] = {
                        'value': score,
                        'color': 'high-score' if score >= 0.8 else ('low-score' if score < 0.6 else '')
                    }
            
            processed = {
                'domain': result.get('domain', 'unknown'),
                'prompt': self.process_content(result.get('prompt', '')),
                'response': self.process_content(result.get('model_response', '')),  
                'reference': self.process_content(result.get('reference_answer', '')),
                'evaluation': {
                    'scores': formatted_scores,
                    'text': self.process_content(result.get('evaluation', {}).get('text_evaluation', ''))
                }
            }
            processed_results.append(processed)
        
        # Group by domain
        domains = {}
        for result in processed_results:
            domain = result['domain']
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(result)
        
        # Calculate stats
        domain_stats = self.calculate_domain_stats(results)
        
        return template.render(
            domains=domains,
            domain_stats=domain_stats
        )

    def create_template(self):
        """Create the HTML template."""
        template_path = os.path.join(self.template_dir, 'eval_view.html')
        template_content = """<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Results</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .domain-header {
            background: #f8f9fa;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .stats-table th, .stats-table td {
            padding: 8px;
            text-align: left;
            border: 1px solid #dee2e6;
        }
        .stats-table th {
            background: #f8f9fa;
        }
        .score-cell {
            text-align: right;
            font-weight: bold;
        }
        .high-score { color: #28a745; }
        .low-score { color: #dc3545; }
        .eval-item {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .response, .reference {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .response {
            border-left: 4px solid #007bff;
        }
        .reference {
            border-left: 4px solid #28a745;
        }
        .evaluation {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin-top: 20px;
        }
        .eval-scores {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .score-item {
            background: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-weight: bold;
        }
        h1, h2, h3 {
            color: #333;
            margin-top: 0;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }
        code {
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9em;
        }
        .prompt {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Results</h1>
        
        {% for domain_name, stats in domain_stats.items() %}
        <div class="domain-header">
            <h2>{{ domain_name|title }} Domain</h2>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>Average Score</th>
                </tr>
                {% for metric, score_info in stats.get('average_scores', {}).items() %}
                <tr>
                    <td>{{ metric|title }}</td>
                    <td class="score-cell {{ score_info.color }}">
                        {{ "%.2f"|format(score_info.value) }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            {% for result in domains[domain_name] %}
            <div class="eval-item">
                <div class="prompt">
                    <h3>Prompt</h3>
                    {{ result.prompt|safe }}
                </div>
                
                <div class="comparison">
                    <div class="response">
                        <h3>Model Response</h3>
                        {{ result.response|safe }}
                    </div>
                    <div class="reference">
                        <h3>Reference Answer</h3>
                        {{ result.reference|safe }}
                    </div>
                </div>
                
                <div class="evaluation">
                    <h3>Evaluation</h3>
                    <div class="eval-scores">
                        {% for metric, score_info in result.evaluation.scores.items() %}
                        <div class="score-item {{ score_info.color }}">
                            {{ metric|title }}: {{ "%.2f"|format(score_info.value) }}
                        </div>
                        {% endfor %}
                    </div>
                    {{ result.evaluation.text|safe }}
                </div>
            </div>
            {% endfor %}
        </div>
        {% endfor %}
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
    </script>
</body>
</html>"""
        
        with open(template_path, 'w') as f:
            f.write(template_content)

    def render(self) -> str:
        """Render the evaluation results."""
        self.create_template()
        results = self.load_results()
        html_content = self.create_html(results)
        
        output_path = os.path.join(self.output_dir, 'eval_view.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        webbrowser.open('file://' + os.path.abspath(output_path))
        return output_path

if __name__ == "__main__":
    import sys
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/evaluation_results_latest.json"
    viewer = EvalViewer(results_path)
    viewer.render()
