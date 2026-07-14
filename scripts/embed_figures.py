import os
import re

curriculum_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'curriculum')

figure_mapping = {
    '00-mathematical-foundations-for-machine-learning.md': '00_vector_matrix.png',
    '01-basic-concepts-of-machine-learning-and-artificial-intelligence.md': '01_gradient_descent.png',
    '02-visualization.md': '02_viz_anatomy.png',
    '03-probability-and-statistics.md': '03_bayes_update.png',
    '04-clustering.md': '04_kmeans.png',
    '07-dimensionality-reduction-and-data-decomposition.md': '07_pca_projection.png',
    '08-regression-analysis.md': '08_regression_fit.png',
    '09-classification.md': '09_supervised_map.png',
    '10-neural-networks-and-deep-learning.md': '10_mlp_architecture.png',
    '16-concepts-and-challenges-of-working-with-data.md': '16_leakage_timeline.png',
    '17-closing-synthesis-senior-practice.md': '17_roc_curve.png'
}

for filename, figname in figure_mapping.items():
    filepath = os.path.join(curriculum_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Don't add if already there
        if figname not in content:
            # We want to embed the figure right after the first heading or opening
            embed_str = f"\n\n![{figname.replace('_', ' ').replace('.png', '').title()}](../assets/figures/{figname})\n\n"
            
            # Simple heuristic: add after the first heading
            match = re.search(r'# .*?\n', content)
            if match:
                insert_pos = match.end()
                new_content = content[:insert_pos] + embed_str + content[insert_pos:]
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Added {figname} to {filename}")
            else:
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(embed_str)
                print(f"Appended {figname} to {filename}")
