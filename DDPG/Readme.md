# 🧠 DDPG Hyperparameter Tuning on Pendulum-v1

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A complete implementation of DDPG with hyperparameter optimization on Gymnasium environment
> new implementation of DDPG with hyperparameter optimization on Replay Buffer

---

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Hyperparameters](#-hyperparameters)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Debugging](#-debugging)
- [References](#-references)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Introduction

This project implements the **DDPG** algorithm to solve the **Pendulum-v1** environment from Gymnasium. The primary goal is **hyperparameter optimization** to achieve the best possible performance and optimize Replay Buffer


---

## ✨ Features

- ✅ Complete DDPG implementation with Experience Replay and Target Networks
- ✅ Grid Search over network atribite and Replay Buffer optimaization
- ✅ 3 independent runs per configuration (to reduce variance)
- ✅ Visualization of top 5 configurations with mean and standard deviation
- ✅ Clean, modular code for easy extension
- ✅ TensorBoard-ready (optional)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository (or create project folder)
mkdir drl-project
cd drl-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# For Windows:
# venv\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium[all] numpy matplotlib tqdm