import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_blobs

# Setup output directory
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'assets', 'figures')
os.makedirs(out_dir, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
TEAL = '#008080'
GOLD = '#FFD700'
DARK = '#2C3E50'

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches='tight')
    plt.close('all')

# 1. Vector/Matrix Intuition
def plot_vector_matrix():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(0, 0, 1, 2, angles='xy', scale_units='xy', scale=1, color=TEAL, label='Vector [1, 2]')
    ax.quiver(0, 0, 2, 0.5, angles='xy', scale_units='xy', scale=1, color=GOLD, label='Vector [2, 0.5]')
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_title('Vector Space Intuition')
    ax.legend()
    save_fig('00_vector_matrix.png')

# 2. Gradient Descent
def plot_gradient_descent():
    x = np.linspace(-3, 3, 100)
    y = x**2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=DARK, lw=2)
    
    # Steps
    pts = [-2.5, -1.5, -0.6, -0.1, 0]
    for i in range(len(pts)-1):
        x1, y1 = pts[i], pts[i]**2
        x2, y2 = pts[i+1], pts[i+1]**2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
        ax.plot(x1, y1, 'o', color=GOLD, markersize=8)
    ax.plot(0, 0, 'o', color=GOLD, markersize=10)
    ax.set_title('Gradient Descent')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Loss')
    save_fig('01_gradient_descent.png')

# 3. Supervised Map (Decision Boundary)
def plot_supervised_map():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y==0, 0], X[y==0, 1], color=TEAL, label='Class 0', s=50)
    ax.scatter(X[y==1, 0], X[y==1, 1], color=GOLD, label='Class 1', s=50, edgecolors='k')
    
    # Decision boundary line
    ax.plot([-1, 2], [-1.5, 2.5], '--', color=DARK, label='Decision Boundary')
    ax.set_title('Supervised Learning: Classification Map')
    ax.legend()
    save_fig('09_supervised_map.png')

# 4. Viz Anatomy
def plot_viz_anatomy():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), color=TEAL, label='Signal')
    ax.set_title('Title: Anatomy of a Visualization', fontsize=14, color=DARK)
    ax.set_xlabel('X-Axis (Units)', fontsize=12)
    ax.set_ylabel('Y-Axis (Magnitude)', fontsize=12)
    ax.legend(loc='upper right')
    ax.annotate('Data Ink', xy=(5, np.sin(5)), xytext=(6, 0.5),
                arrowprops=dict(facecolor=GOLD, shrink=0.05))
    save_fig('02_viz_anatomy.png')

# 5. Bayes Update
def plot_bayes_update():
    x = np.linspace(0, 1, 100)
    prior = [1]*100
    likelihood = x
    posterior = x * prior
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, prior, '--', color='gray', label='Prior (Uniform)')
    ax.plot(x, posterior / np.max(posterior), color=TEAL, lw=2, label='Posterior')
    ax.set_title('Bayesian Update')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Density')
    ax.legend()
    save_fig('03_bayes_update.png')

# 6. K-means
def plot_kmeans():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=3, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [TEAL, GOLD, DARK]
    for i in range(3):
        ax.scatter(X[y_kmeans==i, 0], X[y_kmeans==i, 1], color=colors[i], s=30, alpha=0.6)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               color='red', s=200, marker='X', label='Centroids')
    ax.set_title('K-Means Clustering')
    ax.legend()
    save_fig('04_kmeans.png')

# 7. Leakage Timeline
def plot_leakage_timeline():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot([0, 10], [0, 0], color='black', lw=2)
    ax.plot([4, 4], [-0.5, 0.5], color='red', lw=2)
    ax.text(4, 0.6, 'Index Time', ha='center', color='red', fontweight='bold')
    
    ax.annotate('Legal Features', xy=(2, 0), xytext=(2, -0.6), ha='center', color=TEAL)
    ax.annotate('Leakage (Future Data)', xy=(8, 0), xytext=(8, -0.6), ha='center', color=GOLD)
    
    ax.scatter([1, 2, 3], [0, 0, 0], color=TEAL, s=100, zorder=5)
    ax.scatter([6, 7, 9], [0, 0, 0], color=GOLD, s=100, zorder=5)
    ax.axis('off')
    ax.set_title('Data Leakage Timeline')
    save_fig('16_leakage_timeline.png')

# 8. PCA Projection
def plot_pca():
    np.random.seed(1)
    X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_new = pca.inverse_transform(X_pca)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, color=DARK, label='Original Data')
    ax.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8, color=TEAL, label='PCA Projection')
    ax.set_title('PCA Dimension Reduction')
    ax.legend()
    save_fig('07_pca_projection.png')

# 9. Regression Fit
def plot_regression():
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 5 + np.random.randn(50) * 3
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color=GOLD, edgecolor='k', label='Data')
    ax.plot(x, y_pred, color=TEAL, lw=2, label='Regression Fit')
    ax.set_title('Linear Regression')
    ax.legend()
    save_fig('08_regression_fit.png')

# 10. ROC / Confusion Matrix
def plot_roc():
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(0, 1, 100)
    y = x**(0.5)
    ax.plot(x, y, color=TEAL, lw=2, label='ROC Curve (AUC ~ 0.83)')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
    ax.set_title('Receiver Operating Characteristic')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    save_fig('17_roc_curve.png')

# 11. MLP
def plot_mlp():
    fig, ax = plt.subplots(figsize=(8, 5))
    layer_sizes = [3, 4, 2]
    v_spacing = 1.0
    h_spacing = 2.0
    
    for i, size in enumerate(layer_sizes):
        layer_top = v_spacing * (size - 1) / 2.
        for j in range(size):
            circle = plt.Circle((i * h_spacing, layer_top - j * v_spacing), 0.2, 
                                color=TEAL if i < 2 else GOLD, ec='k', zorder=4)
            ax.add_artist(circle)
            
    for i, (n1, n2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer1_top = v_spacing * (n1 - 1) / 2.
        layer2_top = v_spacing * (n2 - 1) / 2.
        for j1 in range(n1):
            for j2 in range(n2):
                ax.plot([i * h_spacing, (i + 1) * h_spacing],
                        [layer1_top - j1 * v_spacing, layer2_top - j2 * v_spacing],
                        color='gray', alpha=0.5, zorder=1)
    
    ax.axis('equal')
    ax.axis('off')
    ax.set_title('Multi-Layer Perceptron (MLP) Architecture')
    save_fig('10_mlp_architecture.png')

if __name__ == '__main__':
    plot_vector_matrix()
    plot_gradient_descent()
    plot_supervised_map()
    plot_viz_anatomy()
    plot_bayes_update()
    plot_kmeans()
    plot_leakage_timeline()
    plot_pca()
    plot_regression()
    plot_roc()
    plot_mlp()
    print("Figures generated successfully in docs/assets/figures/")
