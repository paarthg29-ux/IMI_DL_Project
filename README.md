# IMI Deep Learning Project

## 📌 Overview

This project explores **materials property prediction and generation** using two approaches:

### 🔹 Phase 1: Compositional AI

* Random Forest (RF)
* Variational Autoencoder (VAE)
* Based on elemental composition

### 🔹 Phase 2: Structural AI (Advanced)

* Graph Neural Network (GNN - CGConv)
* Conditional VAE (CVAE)
* Based on 3D crystal structures

---

## 🧠 Phase 2 Pipeline

1. Download crystal structures (Materials Project)
2. Convert CIF → graph representation
3. Train GNN for property prediction
4. Extract embeddings
5. Train CVAE conditioned on property
6. Generate new materials

---

## 📂 Project Structure

IMI_DL_Project/
├── Phase_1_Compositional_AI/
├── Phase_2_Structural_GNN/
│   ├── train_gnn.py
│   ├── train_cvae.py
│   ├── generate_candidates.py
├── README.md
├── .gitignore

---

## ⚙️ Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* Pymatgen

---

## 🚀 Key Idea

Transition from **composition-based ML → structure-aware deep learning**

---

## 📈 Future Work

* GPU training
* Larger datasets
* Experimental validation
