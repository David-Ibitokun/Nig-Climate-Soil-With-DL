import json

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open('scratch/nb_analysis.txt', 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            source = "".join(cell.get('source', []))
            cell_type = cell['cell_type']
            out.write(f"\n--- Cell {i} ({cell_type}) ---\n")
            if len(source) > 100:
                out.write(source[:100] + '...\n')
            else:
                out.write(source + '\n')
            
            if any(k in source for k in ['unique_features =', 'scenarios =', 'Yield_kg_per_ha', 'SHAP', 'T2M_AVG', 'for scenario_name, changes in scenarios.items():', 'Region']):
                out.write(">>> INTERESTING CELL <<<\n")
                out.write(source + '\n')
                out.write("============================\n")

if __name__ == '__main__':
    analyze_notebook('Combined.ipynb')
