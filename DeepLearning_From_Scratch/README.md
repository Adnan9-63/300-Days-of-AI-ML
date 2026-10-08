# 🧠 Deep Learning from Scratch: Architectures & Autograd

This module is dedicated to understanding the mathematical foundations and low-level engineering of modern neural networks. Instead of relying on high-level APIs like Keras or Scikit-Learn, the projects in this module are built **from scratch** to demonstrate a deep understanding of backpropagation, tensor operations, and model architectures.

## 🎯 Engineering Objectives

- **Autograd Engines:** Developing a scalar-based automatic differentiation engine (similar to PyTorch's `autograd`) from the ground up.
- **Language Modeling:** Progressing from simple n-gram/Bigram probabilistic models to complex Multi-Layer Perceptrons (MLPs).
- **Internal Mechanics:** Manually implementing and calculating gradients, analyzing activation distributions, and engineering Batch Normalization layers.
- **Advanced Architectures:** Implementing Deep Learning architectures like WaveNet and Transformers (GPT).
- **Tokenization:** Building a Byte Pair Encoding (BPE) tokenizer to process raw text for Large Language Models.

## 📂 Implementation Roadmap

1. **`01_Autograd_Engine`**: Implementation of a custom automatic differentiation framework (Micrograd) to understand forward and backward passes using calculus and graph theory.
2. **`02_Bigram_Language_Model`**: A character-level language model demonstrating fundamental probability distributions and loss functions (Cross Entropy).
3. **`03_MLP_Architecture`**: Transitioning from Bigram models to a Multi-Layer Perceptron using embedding layers and hidden layers.
4. **`04_Backprop_BatchNorm`**: Deep dive into the initialization of neural networks, vanishing/exploding gradients, and writing the mathematical implementation of Batch Normalization.
5. **`05_WaveNet_Implementation`**: Constructing a hierarchical CNN-like architecture (WaveNet) for structured sequence generation.
6. **`06_Transformer_GPT_From_Scratch`**: Building a generative pre-trained transformer (GPT), including self-attention mechanisms, multi-head attention, and positional encoding.
7. **`07_BPE_Tokenizer`**: Engineering a custom Byte Pair Encoding tokenizer to train LLMs on custom datasets.

## 💡 Custom Implementations & Experiments
*(Note: As models are built, they are trained on custom, unique datasets rather than standard tutorial datasets to demonstrate independent engineering and experimentation.)*

## 🛠️ Tech Stack
* **Languages:** Python
* **Libraries:** PyTorch (for Tensor operations), NumPy, Matplotlib, Graphviz (for computational graph visualization)
* **Concepts:** Calculus, Linear Algebra, Backpropagation, Attention Mechanisms
