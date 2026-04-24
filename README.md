# VeriShield AI

## Explainable Robust Face Authentication Against Adversarial Attacks

**Author:** Usha S Gowda

---

# Table of Contents

1. Executive Overview  
2. Problem Statement  
3. Why This Project Matters  
4. Dataset Used and Why Chosen  
5. Existing Systems and Limitations  
6. Proposed Solution – VeriShield AI  
7. Key Innovations  
8. Complete System Architecture  
9. Core Methodology  
10. Mathematical Formulation  
11. Adversarial Attack Simulation  
12. Explainable AI Module  
13. Comparative Analysis  
14. Challenges Faced and Mitigations  
15. Tech Stack  
16. Project Structure  
17. Installation Guide  
18. How to Run  
19. Results and Outcomes  
20. Important Technical Insights  
21. Future Scope  
22. Author  

---

# 1. Executive Overview

VeriShield AI is an advanced face authentication security system designed to protect biometric login systems from adversarial attacks, manipulated image inputs, and untrustworthy black-box decisions.

Traditional face verification systems mostly compare similarity between two faces. If the similarity score is high enough, the system accepts the user. However, this creates a major security risk because attackers can slightly modify an image using hidden perturbations and fool the system into making incorrect decisions.

VeriShield AI improves this process by combining:

- Face similarity verification  
- Attention consistency monitoring  
- Adversarial attack detection  
- Explainable heatmap reasoning  
- Multi-factor secure decision engine  
- Trust-based authentication logic  

This project demonstrates that future biometric systems should not only be accurate, but also secure, explainable, reliable, and trustworthy.

---

# 2. Problem Statement

Many face authentication systems suffer from the following serious limitations:

## Adversarial Vulnerability

Small pixel-level changes that are often invisible to humans can trick AI models into accepting fake identities or rejecting genuine users.

## Similarity-Only Decisions

If two embeddings are close enough, the system accepts the identity, even if manipulation exists.

## Lack of Explainability

Most systems output only:

Match / No Match

without explaining:

- Why it matched  
- Which face regions were used  
- Whether suspicious attention shift occurred  
- Whether the decision can be trusted  

## Security Risk

This is dangerous because face recognition is used in:

- Banking login systems  
- Smartphone unlock systems  
- Attendance systems  
- Airport e-gates  
- Enterprise identity access  
- Online KYC verification  

---

# 3. Why This Project Matters

Biometric authentication is growing rapidly across industries.

However, if such systems are vulnerable, consequences may include:

- Identity theft  
- Fraudulent access  
- Unauthorized login  
- Genuine user rejection  
- Loss of trust in AI systems  
- Financial and legal risk  

Therefore, modern authentication systems must move beyond pure accuracy and include security intelligence.

---

# 4. Dataset Used and Why Chosen

For this project, we used the **Labeled Faces in the Wild (LFW)** dataset, specifically the **deepfunneled aligned version**, because it is one of the most trusted benchmark datasets for face verification research.

## About LFW Dataset

LFW contains real-world face images collected under unconstrained conditions such as:

- Different lighting conditions  
- Different camera quality  
- Various face angles  
- Facial expressions  
- Background variations  
- Accessories such as glasses, hats, etc.  

This makes it highly suitable for practical biometric verification research.

## Why We Selected LFW

Our project focuses on comparing **two face images for identity verification**. LFW is ideal because it supports pairwise comparison tasks:

- Genuine Pairs = Same Person  
- Impostor Pairs = Different Persons  

## Why Better Than Controlled Datasets

Many datasets contain studio-quality images with fixed lighting and clean backgrounds. Those datasets are easier but less realistic.

LFW is more challenging and therefore better for testing:

- Robustness under natural variations  
- Adversarial attack resistance  
- Trust-aware authentication systems  
- Explainability behavior  

## Why Better Than Larger Datasets for This Use Case

Although datasets like CelebA or VGGFace2 are larger, LFW remains a gold-standard benchmark specifically for verification experiments.

Therefore, LFW was an ideal and academically credible choice.

---

# 5. Existing Systems and Limitations

## FaceNet

Very powerful face recognition model that converts faces into embeddings.

### Limitation

Excellent for recognition accuracy, but not designed for attack awareness or explainability.

---

## Siamese Networks

Learn similarity between two image pairs.

### Limitation

Still often similarity-focused, limited trust intelligence.

---

## Basic CNN Models

Used for face classification and feature extraction.

### Limitation

Weaker robustness and poor interpretability.

---

## Summary Table

| Limitation | Impact |
|-----------|--------|
| Similarity-only decisions | Easy to fool |
| No attack awareness | Vulnerable |
| No explainability | Low trust |
| No attention consistency | Unsafe |
| Black-box outputs | Hard to justify |

---

# 6. Proposed Solution – VeriShield AI

VeriShield AI introduces a smarter identity verification framework.

Instead of only asking:

Do these two faces look similar?

It asks:

- Are they similar?  
- Did the model focus on stable facial landmarks?  
- Did suspicious embedding drift occur?  
- Did attention shift after perturbation?  
- Is this authentication trustworthy?  

This creates a more secure final decision.

---

# 7. Key Innovations

## 1. Multi-Factor Decision Engine

Uses:

- Similarity score  
- Embedding distance  
- Drift score  
- Attention consistency  
- Suspicion score  

## 2. Explainable Authentication

Heatmaps visually explain model reasoning.

## 3. Security-Aware Verification

Rejects suspicious manipulated images even if similarity is high.

## 4. Human Trust Layer

Provides interpretable outputs instead of black-box decisions.

---

# 8. Complete System Architecture

User Uploads Face A + Face B  
↓  
Input Validation Layer  
↓  
Face Detection using MTCNN  
↓  
Image Alignment and Preprocessing  
↓  
Attention CNN Model  
↓  
Embedding Vector + Attention Map  
↓  
Compute:

- Similarity Distance  
- Drift Score  
- Consistency Score  
- Suspicion Score  

↓  
Decision Intelligence Engine  
↓  
Explainability Dashboard  
↓  
Downloadable Security Report

---

# 9. Core Methodology

## Step 1 – Face Detection

Used **MTCNN** for:

- Face localization  
- Landmark detection  
- Alignment  
- Cropping  

This improves consistency before recognition.

---

## Step 2 – Feature Extraction

Attention CNN outputs:

- 128-dimensional identity embedding  
- Spatial attention map  

Embedding captures identity information.

Attention map captures reasoning focus.

---

## Step 3 – Similarity Matching

Compare embeddings of two images.

Low distance = likely same identity.

---

## Step 4 – Attack Evaluation

Modified version of image tested again.

Used attacks:

- FGSM  
- Blur  
- Brightness shift  

---

## Step 5 – Attention Consistency Check

Compare clean vs attacked attention maps.

Stable focus = trustworthy behavior.

---

## Step 6 – Final Decision

Decision based on multiple signals, not only similarity.

---

# 10. Mathematical Formulation

## Embedding Distance

d = ||e1 - e2||₂

Where:

- e1 = embedding of image A  
- e2 = embedding of image B  

Low distance means strong similarity.

---

## Match Confidence

Confidence = (1 - d / T) × 100

Where:

- d = distance  
- T = threshold  

---

## Drift Score

Drift = |d_attacked - d_clean|

Measures instability after attack.

---

## Attention Consistency Loss

L = MSE(A_clean, A_adv)

Where:

- A_clean = clean attention map  
- A_adv = attacked attention map  

Low loss means stable reasoning.

---

## Suspicion Score

Suspicion = k × Drift

Higher drift means higher suspicion.

---

# 11. Adversarial Attack Simulation

## FGSM Attack

FGSM stands for **Fast Gradient Sign Method**.

Formula:

x_adv = x + ε sign(∇x J)

Where:

- x = original image  
- ε = perturbation strength  
- gradient indicates how to fool model  

## Why FGSM Used

- Fast  
- Effective  
- Standard research benchmark  
- Demonstrates vulnerability clearly  

## Additional Practical Attacks

- Blur attack  
- Brightness manipulation  
- Noise perturbation  

These simulate real-world tampering.

---

# 12. Explainable AI Module

Heatmaps show which facial regions influenced the decision.

Examples:

- Eyes  
- Nose bridge  
- Mouth  
- Cheeks  
- Facial contour  

## Why Important

Without explainability:

Black-box system.

With explainability:

Transparent and trustworthy AI system.

Helps validate whether model focus remained logical.

---

# 13. Comparative Analysis

| Model | Accuracy | Explainable | Attack Aware | Security |
|------|----------|------------|-------------|----------|
| FaceNet | Very High | No | No | Medium |
| Basic CNN | Medium | No | No | Low |
| Attention CNN | Good | Partial | Partial | Medium |
| VeriShield AI | High + Safe | Yes | Yes | High |

---

# 14. Challenges Faced and Mitigations

## Low Accuracy Initially

### Cause

- Few epochs  
- Limited tuning  
- Poor threshold calibration  

### Fix

- Increased epochs  
- Better threshold optimization  
- Improved evaluation pipeline  

---

## Slow Runtime

### Cause

CPU-only execution.

### Fix

CUDA acceleration using NVIDIA RTX 3050 GPU.

---

## Weak Heatmaps Initially

### Cause

Low resolution attention maps.

### Fix

- Better interpolation  
- Smoothing  
- Improved overlays  

---

## Multiprocessing Errors

### Cause

Windows DataLoader issues.

### Fix

Used safe multiprocessing structure.

---

# 15. Tech Stack

## Programming Language

- Python

## Deep Learning

- PyTorch

## Computer Vision

- OpenCV  
- PIL

## Face Detection

- facenet-pytorch (MTCNN)

## Dashboard

- Streamlit

## Visualization

- Matplotlib  
- Pandas

## Hardware

- Dell G15 Laptop  
- NVIDIA RTX 3050 GPU

## Version Control

- Git  
- GitHub

---

# 16. Project Structure

VeriShieldAI/

app/  
 app.py

models/  
 best_attention_model.pth

src/  
 attention_model.py  
 train.py  
 evaluate.py  
 utils.py

outputs/  
 heatmaps/  
 plots/  
 reports/

README.md

---

# 17. Installation Guide

git clone https://github.com/Usha-125/AttentionGuard.git

cd AttentionGuard

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

---

# 18. How to Run

## Launch Dashboard

python -m streamlit run app/app.py

## Train Model

python src/train.py

## Evaluate Model

python src/evaluate.py

---

# 19. Results and Outcomes

## Achievements

- Real-time secure dashboard  
- GPU-enabled inference  
- Explainable heatmaps  
- Attack simulation environment  
- Legacy vs proposed comparison  
- Downloadable security reports  

## Key Insight

Even when similarity appears acceptable, suspicious drift may reveal manipulation.

---

# 20. Important Technical Insights

## What Is Novelty In This Project?

The novelty is not only face recognition.

It combines:

- Authentication  
- Adversarial robustness  
- Attention consistency  
- Explainability  
- Trust-based decisions  

Most systems stop at accuracy. VeriShield AI goes beyond accuracy into secure AI.

---

## Why FaceNet Sometimes Shows Higher Accuracy

FaceNet is optimized mainly for recognition accuracy.

VeriShield AI is optimized for:

- Security  
- Robustness  
- Safer decisions  
- Explainability  

Hence sometimes a pure recognition model may score higher on clean datasets.

---

## Why Heatmaps Are Used

Heatmaps show which facial regions influenced the decision.

This improves:

- Transparency  
- Debugging  
- Trust  
- Security validation  

---

## Why Attention Consistency Matters

A genuine face should retain similar focus patterns.

Sudden shift after perturbation may indicate attack.

---

# 21. Future Scope

- Real-time webcam authentication  
- Deepfake detection  
- Liveness detection  
- Mobile deployment  
- Transformer backbone  
- Multi-modal authentication  
- Cloud API deployment  

---

# 22. Author

## Usha S Gowda

Engineering Student  
AI Research Enthusiast  
Security Systems Builder  
Interested in AI Security and Cybersecurity

---

# Final Statement

VeriShield AI proves that future authentication systems should not only be accurate.

They must be:

Secure + Explainable + Trustworthy + Intelligent