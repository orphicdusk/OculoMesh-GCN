#  OculoMesh GCN
**Privacy-Preserving Biometric Authentication via Cryptographically Entangled Graph Neural Networks**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20%2F%20Published-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** Traditional biometric authentication systems rely on pixel-based Convolutional Neural Networks (CNNs), rendering them highly susceptible to environmental noise and catastrophic database breaches. **OculoMesh** proposes a paradigm shift: modeling the human iris as a topological graph rather than a 2D image. By mathematically entangling this biological node mesh with SHA-256 derived cryptographic "chaff" nodes, the system achieves zero-knowledge template security. A shared-weight Siamese Graph Convolutional Network (GCN) processes these entangled graphs to verify identity with state-of-the-art environmental robustness.

---

## 🔬 Key Contributions
* **Topological Biometric Extraction:** Discards pixel-density methods in favor of capturing the macro-spatial relationships of iris minutiae, achieving extreme environmental redundancy.
* **Cryptographic Entanglement (Zero-Knowledge):** Injects user-specific SHA-256 chaff nodes directly into the biological mesh. The original biometric template cannot be reverse-engineered if the database is compromised.
* **Noise-Invariant Siamese Inference:** Employs a Siamese GCN architecture that learns the Euclidean manifold of the entangled graphs, proving resilient to aggressive data dropout and coordinate blur.

## 🧠 System Architecture & Mathematical Foundation

### The OculoMesh Pipeline
```mermaid
graph TD
    classDef secure fill:#e8f4f8,stroke:#2b8cbe,stroke-width:2px,color:#000;
    classDef warning fill:#fee0d2,stroke:#de2d26,stroke-width:2px,color:#000;
    classDef success fill:#e5f5e0,stroke:#31a354,stroke-width:2px,color:#000;
    classDef process fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#000;

    subgraph Phase1 [Phase 1: Biometric Acquisition]
        I1[Raw Anchor Image <br> Database]:::process
        I2[Raw Attempt Image <br> Scanner]:::process
    end

    subgraph Phase2 [Phase 2: OculoMesh Extraction and Entanglement]
        I1 --> TE1[Topological Node Extraction]:::process
        I2 --> TE2[Topological Node Extraction]:::process
        
        TE1 --> C1[Inject SHA-256 Chaff Nodes]:::secure
        TE2 --> C2[Inject SHA-256 Chaff Nodes]:::secure
        
        C1 --> G1[(Anchor Graph <br> Nodes & Edges)]:::process
        C2 --> G2[(Attempt Graph <br> Nodes & Edges)]:::process
    end

    subgraph Phase3 [Phase 3: Siamese GCN Inference - Shared Weights]
        G1 --> GCN
        G2 --> GCN
        
        GCN{Siamese_GCN Network <br> 3x GCNConv + Pooling}:::secure
        
        GCN --> E1[Anchor Embedding <br> 512-dim Vector]:::process
        GCN --> E2[Attempt Embedding <br> 512-dim Vector]:::process
    end

    subgraph Phase4 [Phase 4: Cryptographic Verification]
        E1 --> D((Euclidean <br> Distance)):::process
        E2 --> D
        
        D --> T{Distance < 0.42?}:::process
        
        T -- Yes --> Pass[Authentication Granted]:::success
        T -- No --> Fail[Authentication Denied]:::warning
    end



    ## 🧠 Graph Convolutional Formulation

The network utilizes localized first-order approximations of spectral graph convolutions. For a given node $i$ at layer $l$, the feature propagation is defined as:

$$
x_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\deg(i)\deg(j)}} W^{(l)} x_j^{(l)} \right)
$$

where $\mathcal{N}(i)$ represents the neighborhood of node $i$, $W^{(l)}$ is the weight matrix, and $\sigma$ represents the ReLU activation function. The model optimizes a Triplet Margin Loss function to enforce spatial separation between genuine and imposter graphs in a 512-dimensional embedding space.

## 📊 Results: Environmental Stress Testing

The model was trained on an augmented CASIA-Iris dataset containing a 50% cryptographic noise ratio. To evaluate topological redundancy, the system was subjected to rigorous environmental degradation, simulating physical occlusion (e.g., eyelashes/shadows) via random node dropping, and camera degradation via spatial coordinate Gaussian blur.

| Environmental Stress Level | Data Condition | Accuracy (%) | Drop from Baseline |
| :--- | :--- | :--- | :--- |
| **Pristine (Baseline)** | 0% Blur, 0% Occluded | **98.21%** | - |
| **Level 1** | 2% Blur, 10% Occluded | **93.09%** | -5.12% |
| **Level 2** | 5% Blur, 20% Occluded | **93.06%** | -5.15% |
| **Level 3** | 10% Blur, 30% Occluded | **93.24%** | -4.97% |

**Conclusion:** Even when 30% of the entangled biometric nodes are physically destroyed or severely blurred, the OculoMesh architecture maintains a >93% accuracy rate, proving the exceptional stability of macro-topological graph authentication.

## 📂 Repository Structure

```text
OculoMesh-GCN/
│
├── data/                   # Biometric graph datasets (.pt)
├── models/                 
│   └── hivemind_v2.py      # Core Siamese_GCN PyTorch architecture
├── utils/                  
│   ├── dataset.py          # Triplet dataset generation logic
│   └── degradation.py      # Environmental stress simulator
├── weights/                
│   └── hivemind_v2_85plus.pth # Pre-trained FP32 network weights
│
├── notebooks/              
│   └── exploratory_data_analysis.ipynb # Visualizing the topological meshes
│
├── train.py                # Pipeline for training the model from scratch
├── stress_test.py          # Execution script for the environmental gauntlet
└── requirements.txt        # Environment dependencies




## ⚙️ Reproducibility & Usage

### 1. Environment Setup
Clone the repository and install the strict dependencies (requires PyTorch Geometric):

```bash
git clone https://github.com/orphicdusk/OculoMesh-GCN.git
cd OculoMesh-GCN
pip install -r requirements.txt

### 2. Exploratory Data Analysis
To visualize the pristine biometric graphs and observe the physical effects of simulated cryptographic entanglement, run the Jupyter Notebook:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb

### 3. Running the Stress Test Gauntlet
To evaluate the pre-trained model against Level 1, 2, and 3 environmental degradation:

```bash
python stress_test.py

### 📄 Citation
If you utilize this architecture, code, or methodology in your research, please cite the following paper: