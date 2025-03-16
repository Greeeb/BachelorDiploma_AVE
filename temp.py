import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes with labels
nodes = {
    "PHM": "Pure Highway Model\n(DQN trained in Highway Env\n100,000 steps, Seed X)",
    "CCS": "Collect Critical States\n(Run 1000 episodes in Merge Env\nSame Seed X)",
    "PRB": "Pre-fill Replay Buffer\n(PHM predicts action, reward, next state)",
    "HMC": "Highway-Merge Criticals\n(Transfer Learning on Critical States\nTraining Steps = Critical States Found)",
    "HMM": "Highway-Merge Model\n(Transfer Learning on Merge Env\n10,000 steps, Random Replay Buffer)",
    "PMM": "Pure Merge Model\n(DQN trained in Merge Env\n100,000 steps, Seed X)",
    "EXP": "This entire process is repeated\n10 times with different seed numbers"
}

# Add edges
edges = [("PHM", "CCS"), ("CCS", "PRB"), ("PRB", "HMC"), ("PHM", "HMM"), ("HMC", "HMM"), ("HMM", "PMM"), ("PMM", "EXP")]

# Add nodes and edges to graph
for node, label in nodes.items():
    G.add_node(node, label=label)

G.add_edges_from(edges)

# Define positions for a clearer layout
pos = {
    "PHM": (0, 5),
    "CCS": (0, 4),
    "PRB": (0, 3),
    "HMC": (1, 2),
    "HMM": (-1, 2),
    "PMM": (0, 1),
    "EXP": (0, 0)
}

# Draw the graph
plt.figure(figsize=(20, 10))
ax = plt.gca()
ax.set_title("Overview of Model Training and Evaluation Process", fontsize=14, fontweight="bold")

# Draw nodes
node_colors = {
    "PHM": "lightblue",
    "CCS": "lightgreen",
    "PRB": "orange",
    "HMC": "red",
    "HMM": "pink",
    "PMM": "lightgray",
    "EXP": "yellow"
}

for node, color in node_colors.items():
    nx.draw_networkx_nodes(G, pos, nodelist=[node],node_shape="s", node_color=color, node_size=10000, edgecolors='black', linewidths=1.5)

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle="->", arrowsize=15, width=2)

# Draw labels
labels = {node: label for node, label in nodes.items()}
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

plt.axis("off")  # Hide axes
plt.show()
