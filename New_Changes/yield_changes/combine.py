#!/usr/bin/env python3
"""
Script to combine multiple Jupyter notebooks into a single notebook.
Preserves all cells, their types, and outputs in the order specified.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def read_notebook(notebook_path: str) -> Dict[str, Any]:
    """
    Read a Jupyter notebook file and return its contents as a dictionary.
    
    Args:
        notebook_path: Path to the .ipynb file
        
    Returns:
        Dictionary containing the notebook structure
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        return notebook
    except FileNotFoundError:
        print(f"Error: Notebook '{notebook_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{notebook_path}' is not a valid JSON file.")
        sys.exit(1)


def create_title_cell(title: str) -> Dict[str, Any]:
    """
    Create a markdown cell with a title/separator.
    
    Args:
        title: Title text for the separator
        
    Returns:
        A markdown cell dictionary
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"---\n",
            f"# {title}\n",
            f"---\n"
        ]
    }


def combine_notebooks(notebook_paths: List[str], output_path: str = "Combined.ipynb") -> None:
    """
    Combine multiple Jupyter notebooks into a single notebook.
    
    Args:
        notebook_paths: List of paths to notebooks to combine
        output_path: Path where the combined notebook will be saved
    """
    # Initialize the combined notebook structure
    combined_notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Add a main title cell
    combined_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Combined Notebook\n",
            "\n",
            "This notebook combines multiple Jupyter notebooks into a single document.\n",
            "Each section represents content from individual notebooks.\n"
        ]
    })

    # Process each notebook
    for idx, notebook_path in enumerate(notebook_paths, 1):
        print(f"Processing [{idx}/{len(notebook_paths)}]: {notebook_path}")
        
        notebook = read_notebook(notebook_path)
        
        # Extract notebook name for section title
        notebook_name = Path(notebook_path).stem
        
        # Add separator cell
        combined_notebook["cells"].append(create_title_cell(f"Section {idx}: {notebook_name}"))
        
        # Add all cells from the notebook
        if "cells" in notebook:
            for cell in notebook["cells"]:
                # Create a copy to avoid modifying original
                cell_copy = {
                    "cell_type": cell.get("cell_type", "code"),
                    "metadata": cell.get("metadata", {}),
                    "source": cell.get("source", [])
                }
                
                # Preserve cell outputs if they exist
                if "outputs" in cell:
                    cell_copy["outputs"] = cell.get("outputs", [])
                
                # Preserve execution count if it exists
                if "execution_count" in cell:
                    cell_copy["execution_count"] = cell.get("execution_count")
                
                combined_notebook["cells"].append(cell_copy)
        
        print(f"  ✓ Added {len(notebook.get('cells', []))} cells")

    # Write the combined notebook to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_notebook, f, indent=1, ensure_ascii=False)
        
        print(f"\n✓ Successfully created combined notebook: {output_path}")
        
        # Print summary statistics
        total_cells = len(combined_notebook["cells"])
        code_cells = sum(1 for cell in combined_notebook["cells"] if cell["cell_type"] == "code")
        markdown_cells = sum(1 for cell in combined_notebook["cells"] if cell["cell_type"] == "markdown")
        
        print(f"\nSummary:")
        print(f"  Total cells: {total_cells}")
        print(f"  Code cells: {code_cells}")
        print(f"  Markdown cells: {markdown_cells}")
        
    except IOError as e:
        print(f"Error: Could not write to '{output_path}': {e}")
        sys.exit(1)


def main():
    """Main function to parse arguments and run the combination."""
    
    # Define the notebooks to combine (in order)
    notebooks_to_combine = [
        "01_Data_Preprocessing.ipynb",
        "02_Model_Training_Minimal.ipynb",
        "03_Model_Evaluation_Minimal.ipynb",
        "04_Climate_Impact.ipynb"
    ]
    
    # Allow command-line specification of output path
    output_file = "Combined.ipynb"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    # Check if all notebooks exist
    missing_notebooks = []
    for notebook in notebooks_to_combine:
        if not Path(notebook).exists():
            missing_notebooks.append(notebook)
    
    if missing_notebooks:
        print("Warning: The following notebooks are not in the current directory:")
        for nb in missing_notebooks:
            print(f"  - {nb}")
        print("\nPlease ensure all notebooks are in the current working directory.")
        print(f"Current directory: {Path.cwd()}\n")
    
    # Combine the notebooks
    print(f"Combining {len(notebooks_to_combine)} notebooks...\n")
    combine_notebooks(notebooks_to_combine, output_file)


if __name__ == "__main__":
    main()