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
        html_content = """
        <html>
        <head>
            <title>Dataset View</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .entry { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .field { margin: 5px 0; }
                .field-name { font-weight: bold; color: #333; }
                .key-points { list-style-type: disc; margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>Dataset Contents</h1>
        """

        # Add metadata
        html_content += "<h2>Metadata</h2>"
        for key, value in dataset["metadata"].items():
            html_content += f'<div class="field"><span class="field-name">{key}:</span> {value}</div>'

        # Add data entries
        html_content += "<h2>Data Entries</h2>"
        for i, entry in enumerate(dataset["data"], 1):
            html_content += f'<div class="entry"><h3>Entry {i}</h3>'
            for key, value in entry.items():
                if key == "key_points":
                    html_content += f'<div class="field"><span class="field-name">{key}:</span>'
                    html_content += '<ul class="key-points">'
                    for point in value:
                        html_content += f'<li>{point}</li>'
                    html_content += '</ul></div>'
                else:
                    html_content += f'<div class="field"><span class="field-name">{key}:</span> {value}</div>'
            html_content += '</div>'

        html_content += """
        </body>
        </html>
        """
        return html_content

    def render(self) -> str:
        """Render the dataset as HTML and save to file."""
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
