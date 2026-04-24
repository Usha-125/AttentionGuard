# VeriShield AI

## Explainable Robust Face Authentication Against Adversarial Attacks

**Author:** Usha S Gowda

---

# Table of Contents

1. Executive Overview
2. Problem Statement
3. Why This Project Matters
4. Existing Systems & Limitations
5. Proposed Solution – VeriShield AI
6. Key Innovations
7. System Architecture
8. Core Methodology
9. Mathematical Formulation
10. Adversarial Attack Simulation
11. Explainable AI Module
12. Comparative Analysis
13. Challenges Faced & Mitigations
14. Tech Stack
15. Project Structure
16. Installation Guide
17. How to Run
18. Results
19. Important Technical Insights
20. Future Scope
21. Author

---

# 1. Executive Overview

VeriShield AI is an advanced **face authentication security system** designed to protect biometric login systems from adversarial attacks and manipulated image inputs.

Traditional face verification systems mostly compare similarity between two faces. If the similarity is high enough, the system accepts the user. However, this creates a serious security risk because attackers can slightly modify an image using hidden perturbations and fool the system.

VeriShield AI improves this by combining:

* Face similarity verification
* Attention consistency monitoring
* Adversarial attack detection
* Explainable heatmap reasoning
* Multi-factor secure decision engine

This project demonstrates that future biometric systems should not only be accurate, but also secure, explainable, and trustworthy.

---

# 2. Problem Statement

Many face authentication systems suffer from the following issues:

## Adversarial Vulnerability

Small invisible pixel-level changes can trick AI models.

## Similarity-Only Decisions

If two embeddings are close enough, the system accepts, even if manipulation exists.

## Lack of Explainability

Most systems output:

Match / No Match

without explaining:

* Why it matched
* Which face regions were used
* Whether suspicious focus shift happened

## Security Risk

Used in banking, devices, attendance, airports, and secure login systems.

---

# 3. Why This Project Matters

Face recognition is used in:

* Smartphones
* Banking KYC
* Airport gates
* Enterprise login
* Attendance systems
* Identity verification portals

If compromised, consequences may include:

* Fraudulent access
* Identity theft
* Unauthorized verification
* Genuine user rejection
* Loss of trust in AI systems

---

# 4. Existing Systems & Limitations

## Common Existing Models

### FaceNet

Very high face recognition accuracy using embeddings.

### Siamese Networks

Compare similarity between image pairs.

### Basic CNN Models

Extract visual features for classification.

---

## Main Limitations

| Limitation                 | Impact          |
| -------------------------- | --------------- |
| Similarity-only decisions  | Easy to fool    |
| No attack awareness        | Vulnerable      |
| No explainability          | Low trust       |
| No focus consistency check | Unsafe          |
| Black-box outputs          | Hard to justify |

---

# 5. Proposed Solution – VeriShield AI

VeriShield AI uses a smarter verification process.

Instead of only asking:

Do these two faces look similar?

It asks:

* Are they similar?
* Did attention remain stable?
* Did suspicious drift occur?
* Was focus manipulated?
* Is this login trustworthy?

---

# 6. Key Innovations

## Multi-Factor Decision System

Uses:

* Similarity score
* Embedding drift
* Attention consistency
* Suspicion score

## Explainable Authentication

Heatmaps visually show model reasoning.

## Security-Aware Verification

Rejects suspicious manipulated images even if similarity is high.

---

# 7. System Architecture

User Uploads Two Face Images
↓
Face Detection using MTCNN
↓
Preprocessing and Alignment
↓
Attention CNN Model
↓
Embedding Vector + Attention Map
↓
Compute:

* Similarity Distance
* Drift Score
* Consistency Score
* Suspicion Score

↓
Final Decision Engine
↓
Explainability Dashboard

---

# 8. Core Methodology

## Step 1 – Face Detection

Used MTCNN to detect and align faces.

## Step 2 – Feature Extraction

Attention CNN outputs:

* 128-dimensional embedding
* Spatial attention map

## Step 3 – Similarity Matching

Compare embeddings of both images.

## Step 4 – Attack Evaluation

Modified image tested again.

## Step 5 – Attention Consistency Check

Compare clean vs attacked attention.

## Step 6 – Final Decision

Secure authentication result.

---

# 9. Mathematical Formulation

## Embedding Distance

d = ||e1 - e2||₂

Where:

* e1 = embedding of image 1
* e2 = embedding of image 2

Low distance means higher similarity.

---

## Match Confidence

Confidence = (1 - d / T) × 100

Where:

* d = distance
* T = threshold

---

## Drift Score

Drift = |d_attacked - d_clean|

Measures model instability.

---

## Attention Consistency Loss

L = MSE(A_clean, A_adv)

Where:

* A_clean = clean attention map
* A_adv = attacked attention map

Low loss means stable model focus.

---

## Suspicion Score

Suspicion = k × Drift

Higher drift = more suspicious input.

---

# 10. Adversarial Attack Simulation

## FGSM Attack

FGSM means **Fast Gradient Sign Method**.

It creates an adversarial image using:

x_adv = x + ε sign(∇x J)

Where:

* x = original image
* ε = attack strength
* gradient guides how to fool model

FGSM is widely used because it is fast, effective, and standard in adversarial AI research.

---

## Additional Practical Attacks

* Blur attack
* Brightness manipulation
* Noise perturbation

These simulate real-world image tampering.

---

# 11. Explainable AI Module

Heatmaps show where the model focused while verifying identity.

Examples:

* Eyes
* Nose bridge
* Mouth
* Cheeks
* Facial contour

## Why Heatmaps Are Important

Without heatmaps:

Black-box system

With heatmaps:

Transparent AI decision system

They help detect whether attention shifted unnaturally under attack.

---

# 12. Comparative Analysis

| Model         | Accuracy    | Explainable | Attack Aware | Security |
| ------------- | ----------- | ----------- | ------------ | -------- |
| FaceNet       | Very High   | No          | No           | Medium   |
| Basic CNN     | Medium      | No          | No           | Low      |
| Attention CNN | Good        | Partial     | Partial      | Medium   |
| VeriShield AI | High + Safe | Yes         | Yes          | High     |

---

# 13. Challenges Faced & Mitigations

## Low Accuracy Initially

### Cause

* Only few epochs
* Limited tuning
* Small dataset

### Fix

* More training epochs
* Better thresholds
* Improved evaluation

---

## Slow Runtime

### Cause

CPU inference.

### Fix

CUDA GPU acceleration using NVIDIA RTX 3050.

---

## Weak Heatmaps

### Cause

Low resolution attention outputs.

### Fix

* Better interpolation
* Smoothing
* Overlay redesign

---

## Multiprocessing Errors

### Cause

Windows DataLoader issues.

### Fix

Used:

if **name** == "**main**"

---

# 14. Tech Stack

## Languages

* Python

## Deep Learning

* PyTorch

## Computer Vision

* OpenCV
* PIL

## Face Detection

* facenet-pytorch MTCNN

## UI

* Streamlit

## Visualization

* Matplotlib

## Hardware

* Dell G15 Laptop
* NVIDIA RTX 3050 GPU

## Version Control

* Git
* GitHub

---

# 15. Project Structure

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
 plots/
 heatmaps/
 reports/

README.md

---

# 16. Installation Guide

git clone [https://github.com/yourusername/VeriShieldAI.git](https://github.com/yourusername/VeriShieldAI.git)

cd VeriShieldAI

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

---

# 17. How to Run

## Run Dashboard

python -m streamlit run app/app.py

## Train Model

python src/train.py

## Evaluate Model

python src/evaluate.py

---

# 18. Results

## Achievements

* Real-time secure dashboard
* GPU-enabled inference
* Explainable heatmaps
* Attack simulation
* Legacy vs secure comparison
* Downloadable reports

## Key Insight

Even when similarity looks acceptable, suspicious drift may reveal manipulation.

---

# 19. Important Technical Insights

## What Is Novelty In This Project?

The novelty is not only face recognition.

It combines:

* Authentication
* Adversarial robustness
* Attention consistency
* Explainability
* Trust-based decision making

Most models stop at accuracy. VeriShield AI goes beyond accuracy into secure AI.

---

## Why FaceNet Can Sometimes Show Higher Accuracy

FaceNet is highly optimized only for recognition accuracy on clean datasets.

VeriShield AI is optimized for:

* Robustness
* Attack detection
* Safer decisions
* Explainability

So sometimes a pure recognition model may score higher on clean accuracy, but a safer system is better in real-world security.

---

## Why FGSM Is Used

FGSM is a standard adversarial benchmark.

Used because:

* Fast to generate
* Effective
* Research accepted
* Shows vulnerability clearly

---

## Why Heatmaps Are Used

Heatmaps show which facial regions influenced the decision.

This helps:

* Transparency
* Debugging
* Trust
* Security validation

---

## Why Attention Consistency Matters

A genuine face should keep similar focus patterns.

If focus suddenly shifts after minor perturbation:

Possible attack detected.

---

# 20. Future Scope

* Transformer-based biometric models
* Webcam real-time verification
* Liveness detection
* Deepfake detection
* Mobile app deployment
* Cloud API service
* Multi-modal authentication

---

# 21. Author

## Usha S Gowda

Engineering Student
AI Research Enthusiast
Security Systems Builder
Interested in AI Security and Cybersecurity

---

# Final Statement

VeriShield AI proves that future authentication systems should not be only accurate.

They must be:

Secure + Explainable + Trustworthy + Intelligent
