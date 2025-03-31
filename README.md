Sure! Here's the README in proper GitHub markdown format:  

```markdown
# Leveraging Criticality in Reinforcement Learning for Effective Transfer Learning in Autonomous Driving

This repository contains the code and resources for my bachelor's thesis titled **"Leveraging Criticality in Reinforcement Learning for Effective Transfer Learning in Autonomous Driving."**

## 📌 Overview

This project explores methods to enhance transfer learning in autonomous driving by integrating **criticality measures** into reinforcement learning algorithms. The goal is to improve the **adaptability and safety** of autonomous vehicles when transitioning between different driving environments.

## 📁 Repository Structure

```
BachelorDiploma_AVE/
│── .vscode/                # VS Code configuration files
│── Figures/                # Figures used in the thesis
│── models/                 # Pre-trained models and checkpoints
│── rlkit/                  # Reinforcement learning implementation
│── saveResults/            # Experiment results
│── Functions.py            # Utility functions
│── executor.py             # Main script to execute training
│── trainHighwayEnvDQN.py   # Training script for the highway environment
│── trainMergeEnvDQN.py     # Training script for the merging environment
│── visualiseResultsHighway.py # Visualization script
│── requirements.txt        # List of dependencies
│── README.md               # Project documentation
```

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Greeeb/BachelorDiploma_AVE.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BachelorDiploma_AVE
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏎️ Usage

To train a model in the **highway environment**, run:
```bash
python trainHighwayEnvDQN.py
```

For training in the **merging environment**, run:
```bash
python trainMergeEnvDQN.py
```

After training, visualize the results using:
```bash
python visualiseResultsHighway.py
```

## 📊 Results

The results and key findings are documented in the thesis. Some highlights include:

- **Improved transfer learning efficiency** by incorporating **criticality measures**.
- **Enhanced safety** and **performance** of autonomous driving models across different environments.

## 🤝 Contributing

Contributions are welcome! If you’d like to contribute:

1. **Fork** the repository.
2. **Create a new branch** (`git checkout -b feature-branch`).
3. **Commit your changes** (`git commit -m "Add new feature"`).
4. **Push to the branch** (`git push origin feature-branch`).
5. **Submit a pull request**.

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 🙌 Acknowledgments

A special thanks to my advisor and colleagues for their support and guidance throughout this project.

---

Made with ❤️ for autonomous driving research.
```
