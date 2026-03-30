# DermaSafe: Advanced Research Plan v2.0
### Project Title: A Privacy-First, Federated Edge-AI System for Unbiased Skin Health Monitoring and Community Disease Detection
> **Stack:** Python · Neural Networks · Deep Learning (NN&DL) Only  
> **Version:** 1.0 — Advanced Edition | March 2026

---

## 0. Abstract & Problem Formulation

DermaSafe targets the intersection of three unsolved challenges in clinical AI:

```
PROBLEM SPACE:
┌──────────────────────────────────────────────────────────────────────────┐
│  Challenge 1: PRIVACY                                                    │
│    Standard apps upload raw skin images → HIPAA/GDPR violation risk     │
│    Adversarial reconstruction attacks can recover images from embeddings │
│                                                                          │
│  Challenge 2: BIAS                                                       │
│    ISIC dataset: 83% Fitzpatrick Type I-III → systematic underperformance│
│    Clinical misdiagnosis rates 3.4× higher for dark skin (JAMA 2018)    │
│                                                                          │
│  Challenge 3: EPIDEMIOLOGICAL BLINDNESS                                  │
│    No existing dermatology app performs community outbreak surveillance  │
│    Re-emerging skin infections (Monkeypox, Scabies) lack early warning  │
└──────────────────────────────────────────────────────────────────────────┘

SOLUTION: DermaSafe = Edge AI + Biophysical Decomposition + Federated DP + Surveillance
```

---

## 1. Research Pillars Overview

| Pillar | Title | Core Innovation | Primary Metric |
|---|---|---|---|
| P1 | Biophysical Skin Decomposition | ResNet-18 dual-decoder BDN with physics loss | ΔBalAcc FP V-VI ≥+15% |
| P2 | Efficient On-Device Inference | 4-stage KD→Prune→Quant→NAS pipeline | Latency ≤500ms, Size ≤15MB |
| P3 | Federated DP Learning | DP-SGD + SCAFFOLD + Secure Aggregation | ε≤3.0, ΔAcc≤5% |
| P4 | Syndromic Surveillance | 5-layer CUSUM+Kulldorff+MUVP+LSTM-AE pipeline | Lag<72h, FPR<5% |
| P5 | Fairness & Bias Auditing | Adversarial debiasing + per-round FL monitoring | EqOdds Diff≤0.05 |
| P6 | Explainability & Trust | GradCAM on biophysical maps + uncertainty quantification | Clinical trust score |
| P7 | Adversarial Robustness | Certified defenses against image manipulation attacks | Certified accuracy ≥80% |

---

## 2. PILLAR 1 — Biophysical Skin Decomposition Network (BDN)

### 2.1 Research Objective & Novelty

Standard CNNs learn spurious correlations between skin color (pixel values) and diagnosis labels because training datasets are dominated by Fitzpatrick Types I-III. BDN eliminates this by decomposing images into wavelength-specific biological signal channels before any classification occurs.

**Key novelty:** Previous biophysical decomposition work (Tsumura 1999) was purely optical/non-neural. BDN is the first end-to-end learnable decomposition jointly optimized with a diagnosis objective and a physics reconstruction constraint.

### 2.2 Literature Foundation (Extended)

| Paper | Authors | Venue | Year | Contribution to BDN |
|---|---|---|---|---|
| Dichromatic Reflection Model | Shafer | IUCRJ | 1985 | Physical basis — skin as dichromatic reflector |
| ICA for Skin H/M Separation | Tsumura et al. | JOSA | 1999 | Validated H/M optical separation in vivo |
| Deep Reflectance Decomposition | Li et al. | CVPR | 2018 | Neural intrinsic image decomposition framework |
| Skin Optical Properties | Jacques | PMB | 2013 | Spectrophotometric absorption coefficients |
| Deep Skin Color Analysis | Shen et al. | IEEE TMI | 2020 | CNN for skin chromophore estimation |
| Fitzpatrick17k Dataset | Groh et al. | CVPR | 2021 | First large-scale dataset with FP type labels |
| SkinCon | Daneshjou et al. | NeurIPS | 2022 | Clinician-verified concept annotations |
| DDI Dataset | Daneshjou et al. | Science Trans Med | 2022 | Diverse Dermatology Images — bias evaluation |
| Towards Skin Tone Fairness | Kinyanjui et al. | MICCAI | 2020 | Empirical study of FP bias in deep models |
| IntraDerm | Hasan et al. | ECCV | 2022 | Intrinsic decomposition for dermatological analysis |

### 2.3 Experimental Hypotheses

> **H1 (Primary):** BDN achieves ≥15% higher balanced accuracy on Fitzpatrick V-VI vs. raw RGB EfficientNet-B3 baseline on Fitzpatrick17k.

> **H2 (Secondary):** Physics reconstruction loss (L_phys) acts as implicit regularizer — BDN generalizes better to out-of-distribution skin tones unseen during training (zero-shot FP generalization test on DDI).

> **H3 (Ablation):** Removing either H_map or M_map decoder degrades performance disproportionately on dark skin (H_map ablation expected to hurt inflammation detection; M_map ablation expected to hurt pigmentation-based conditions).

### 2.4 Network Architecture (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BIOPHYSICAL DECOMPOSITION NETWORK                    │
│                                                                         │
│  Input: I_rgb (B, 3, 224, 224)                                         │
│      │                                                                  │
│      ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │           SHARED ENCODER — ResNet-18 (pretrained ImageNet)   │       │
│  │  Layer 1: Conv(3→64, 7×7, stride=2) + BN + ReLU             │       │
│  │  Layer 2-5: Residual blocks → feature map F (B,512,14,14)    │       │
│  │  Feature Pyramid: {F2, F3, F4, F5} for multi-scale decoding  │       │
│  └──────────────────────────────────────────────────────────────┘       │
│           │                          │                                  │
│           ▼                          ▼                                  │
│  ┌──────────────────┐      ┌──────────────────────┐                     │
│  │ HEMOGLOBIN HEAD  │      │   MELANIN HEAD        │                    │
│  │ TransposeConv×5  │      │   TransposeConv×5     │                    │
│  │ Skip connections │      │   Skip connections    │                    │
│  │ from F2,F3,F4    │      │   from F2,F3,F4       │                    │
│  │ Final: sigmoid   │      │   Final: sigmoid      │                    │
│  │ H_map(B,1,224,224│      │   M_map(B,1,224,224)  │                    │
│  └──────────────────┘      └──────────────────────┘                     │
│           │                          │                                  │
│           └──────────┬───────────────┘                                  │
│                      ▼                                                  │
│         ┌──────────────────────────┐                                    │
│         │  PHYSICS RECONSTRUCTOR   │                                    │
│         │  I_hat = αH·H_map        │                                    │
│         │        + αM·M_map        │                                    │
│         │        + αS·scatter_term │                                    │
│         │  (αH, αM, αS learned)    │                                    │
│         └──────────────────────────┘                                    │
│                      │                                                  │
│                   L_phys = ||I_rgb - I_hat||²_F                        │
└─────────────────────────────────────────────────────────────────────────┘

BDN OUTPUT → [H_map, M_map] concat (B, 2, 224, 224)
           → EfficientNet-B3 Diagnosis Head
           → 10-class Softmax
```

### 2.5 Multi-Component Loss Function

```
L_TOTAL = λ₁·L_CE          (Classification cross-entropy)
        + λ₂·L_phys         (Biophysical reconstruction fidelity)
        + λ₃·L_fair         (Equalized odds across FP groups)
        + λ₄·L_smooth       (Spatial smoothness of H/M maps)
        + λ₅·L_sparsity     (Melanin map sparsity — clinical prior)
        + λ₆·L_consistency  (H/M maps consistent across augmentations)

Weights (λ₁..λ₆) = (1.0, 0.5, 0.3, 0.1, 0.05, 0.05)
Tuned via Optuna with 200-trial TPE sampler

L_fair = max(|TPR_FP_A - TPR_FP_B|, |FPR_FP_A - FPR_FP_B|)
         for all pairs (A, B) ∈ {FP_I-II, FP_III-IV, FP_V-VI}

L_consistency = (1/K)·Σ||f_θ(aug_k(x)) - f_θ(aug_j(x))||²
                for K random augmentation pairs
```

### 2.6 Ablation Study Design

```
Experiment 1: Decoder ablation
  A. BDN full (H+M)
  B. H_map only → single-channel input to classifier
  C. M_map only → single-channel input
  D. Raw RGB baseline (no BDN)
  E. BDN without physics loss (L_phys=0)
  F. BDN with ICA initialization (warm-start from classical method)

Experiment 2: Encoder backbone ablation
  ResNet-18 vs ResNet-50 vs EfficientNet-B0 encoder
  (BDN encoder must be lightweight — will be replaced by KD in Phase 2)

Experiment 3: Zero-shot generalization
  Train on ISIC (FP I-III heavy) → test on DDI + PAD-UFES (diverse)
  Compare: BDN vs RGB baseline generalization gap
```

### 2.7 Dataset Registry (Complete)

| Dataset | Size | FP Coverage | Conditions | Access | Priority |
|---|---|---|---|---|---|
| ISIC 2019/2020 | 33,126 | Primarily I-III | MEL, NV, BCC, BKL, DF, VASC, SCC, AK | Public (Kaggle) | HIGH |
| Fitzpatrick17k | 16,577 | ALL 6 (labeled) | 114 conditions grouped into 3 severity | GitHub (Harvard) | CRITICAL |
| PAD-UFES-20 | 2,298 | II-V (Brazilian) | BCC, SCC, ACK, SEK, BOW, MEL | Public | HIGH |
| DDI (Stanford) | 656 | HIGH — curated | 78 conditions, diverse population | Public | HIGH |
| SD-198 | 6,584 | Medium | 198 disease categories | Public | MEDIUM |
| SkinCon | 3,230 | Medium | 48 clinical concepts per image | Public | MEDIUM |
| DermNet NZ | ~23,000 | High clinical diversity | 600+ conditions | Web scraping | MEDIUM |
| Derm7pt | 1,011 | Medium | 7-point melanoma checklist | Public | LOW |
| SKINL2 (new 2024) | 11,000 | HIGH — Africa focused | 20 tropical skin conditions | Public | HIGH |

### 2.8 Evaluation Metrics (Full Suite)

| Metric | Computation | Target | Stratification |
|---|---|---|---|
| Balanced Accuracy | (TPR + TNR) / 2 | ≥85% | Per FP type (6 groups) |
| Macro-F1 | Unweighted avg F1 | ≥80% | Per condition class |
| AUROC | Area under ROC | ≥90% | Per class, per FP group |
| Equalized Odds Diff | max(ΔTPR, ΔFPR) across FP groups | ≤0.05 | All FP pairs |
| Demographic Parity Diff | |P(ŷ=1\|A) - P(ŷ=1\|B)| | ≤0.05 | All FP pairs |
| Reconstruction PSNR | 10·log₁₀(MAX²/MSE) | ≥30 dB | Per image |
| FID of H/M maps | Fréchet distance between real/decomposed | ≤30 | Overall |
| ECE (Calibration) | Expected Calibration Error | ≤0.05 | Overall + per FP |

---

## 3. PILLAR 2 — Efficient On-Device Inference Engine

### 3.1 Research Objective

The BDN + EfficientNet-B3 cloud model (~50MB, ~4B MACs) must be compressed to run in real-time on a mid-range smartphone (2GB RAM, Snapdragon 665/MediaTek Helio G85) without medical accuracy loss. Target: <500ms total inference including BDN decomposition + classification.

### 3.2 Literature Foundation

| Paper | Authors | Venue | Year | Relevance |
|---|---|---|---|---|
| MobileNetV3 | Howard et al. | ICCV | 2019 | SE blocks + hard-swish activation for mobile |
| EfficientNet | Tan & Le | ICML | 2019 | Compound scaling: width×depth×resolution |
| Knowledge Distillation | Hinton et al. | NeurIPS WS | 2015 | Soft label transfer via temperature scaling |
| QAT | Jacob et al. | CVPR | 2018 | Simulate quantization during forward pass |
| ProxylessNAS | Cai et al. | ICLR | 2019 | Direct hardware-aware NAS on target device |
| MCUNet | Lin et al. | NeurIPS | 2020 | TinyML: joint NAS + memory optimization |
| MEAL V2 | Shen et al. | ArXiv | 2021 | Multi-teacher ensemble distillation |
| Once-for-All | Cai et al. | ICLR | 2020 | Train once, deploy on any device spec |
| AdaRound | Nagel et al. | ICML | 2020 | Adaptive rounding for post-training quant |
| SpAtten | Wang et al. | HPCA | 2021 | Sparse attention for efficient inference |

### 3.3 Advanced 4-Stage Compression Pipeline

```
STAGE 1: TEACHER TRAINING (Cloud, GPU)
  Architecture: EfficientNet-B4 + BDN (ResNet-18 encoder)
  Training: 100 epochs, AdamW, cosine LR schedule, mixup augmentation
  Target: ≥92% balanced accuracy (ISIC + Fitzpatrick17k combined)
  Output: teacher_v1.pt (54MB)

STAGE 2: MULTI-TEACHER KNOWLEDGE DISTILLATION (MEAL V2)
  Teachers: [EfficientNet-B4-BDN, EfficientNet-B3-BDN, ResNet-50-BDN]
  Student:  MobileNetV3-Small + Lightweight BDN (MobileNetV2 encoder)
  
  Loss: L_KD = Σ_t α_t · KL(σ(z_s/T), σ(z_t/T))
        L_total = 0.3·L_CE(hard) + 0.7·L_KD
  
  Intermediate Feature Matching:
    L_feat = ||proj(F_student) - F_teacher||²_F
    Applied at 3 intermediate layers (FitNets-style)
  
  Temperature: T=4 (start) → T=1 (cosine anneal over training)
  Output: student_distilled.pt (22MB, ~91% teacher accuracy)

STAGE 3: STRUCTURED PRUNING + FINE-TUNING
  Method: Magnitude-based structured channel pruning
  Sensitivity analysis: compute Hessian trace per layer
  Prune aggressively where Hessian trace is low (insensitive layers)
  Target: 45% parameter reduction
  
  Fine-tune with L2-SP regularization (Elastic Weight Consolidation):
    L_EWC = L_CE + λ·||θ - θ_teacher||²_F·F  (F = Fisher information)
  
  Output: student_pruned.pt (12MB)

STAGE 4: QUANTIZATION (Adaptive)
  Step 4a: Post-Training Quantization (PTQ)
    Method: AdaRound (adaptive rounding via BCD optimization)
    Target: INT8 weights + INT8 activations
    Calibration: 512 representative skin images (all 6 FP types)
    
  Step 4b: Verify accuracy drop
    If drop ≤ 1%: proceed to export
    If drop 1-3%: run QAT (10 epochs, simulated INT8 forward pass)
    If drop > 3%: diagnose sensitive layers, apply mixed-precision (INT8+INT16)
  
  Output: student_int8.tflite (8MB) + student.mlmodel (9MB)

OPTIONAL STAGE 5: Hardware-Aware NAS (If latency target not met)
  Search space: MobileNetV3 micro-architecture variants
  Search method: ProxylessNAS latency-aware reward
  Proxy device: Run on actual target hardware (Snapdragon 665 ADB)
  Budget: 200 GPU hours search
```

### 3.4 BDN-Specific Compression (Critical Path)

```
BDN is the bottleneck: ResNet-18 encoder = 11M params, dual decoders = 8M params
BDN-specific compression strategy:

  Option A: Shared Lightweight Encoder
    Replace ResNet-18 with MobileNetV2 (3.4M params)
    Shared encoder feeds both H/M decoders → saves 60% BDN params
    
  Option B: Factorized Decoder
    Replace 5×TransposeConv with 5×DepthwiseSeparableTransposeConv
    Channel-wise decomposition → 8× fewer decoder params
    
  Option C: Early Exit BDN
    Add quality gate at encoder output:
    If image quality score > threshold: use full decoder
    If below threshold: use fast 1-layer decoder (sacrifices map quality)
    
  Selected: Option A + B combined → BDN compressed to 2.1M params, 0.8GB MACs
```

### 3.5 Evaluation Framework

| Metric | Tool | Target | Notes |
|---|---|---|---|
| Inference Latency P50/P95 | ADB shell + custom benchmark APK | P50≤500ms, P95≤800ms | Cold start vs. warm |
| MACs | ptflops / fvcore | ≤300M MACs | BDN + classifier combined |
| Model Size | ls -lh .tflite | ≤15MB compressed | After tflite optimization |
| Accuracy Drop | Test set eval (all FP types) | ≤2% balanced acc | Stratified by FP type |
| Peak RAM | Android Memory Profiler + ADB | ≤256MB peak RSS | Include tensor buffers |
| Battery (mAh) | Monsoon Power Monitor | ≤5% per session | 10 inferences = 1 session |
| Thermal Throttle | Simulated 15-min continuous use | No accuracy regression | Stress test |

---

## 4. PILLAR 3 — Federated Learning with Differential Privacy

### 4.1 Research Objective

Design a federated learning system that allows DermaSafe's model to improve from millions of real-world skin images without any image leaving the device, with mathematically provable privacy guarantees and robustness to adversarial participants.

### 4.2 Literature Foundation (Extended)

| Paper | Authors | Venue | Year | Contribution |
|---|---|---|---|---|
| FedAvg | McMahan et al. | AISTATS | 2017 | Foundational FL algorithm |
| Deep Learning with DP | Abadi et al. | CCS | 2016 | DP-SGD algorithm |
| Secure Aggregation | Bonawitz et al. | CCS | 2017 | Cryptographic gradient masking |
| FedProx | Li et al. | MLSys | 2020 | Proximal term for non-IID |
| SCAFFOLD | Karimireddy et al. | ICML | 2020 | Variance reduction for client drift |
| FLAME | Nguyen et al. | USENIX | 2022 | Byzantine-robust via clustering |
| DP-FedAvg | Geyer et al. | ArXiv | 2017 | User-level DP in FL |
| FL Medical Imaging | Rieke et al. | npj Digital Med | 2020 | Medical FL survey |
| Gradient Inversion | Zhao et al. | NeurIPS | 2020 | Attack: recover images from gradients |
| R-GAP | Zhu et al. | NeurIPS | 2021 | Recursive gradient attack |
| FedMA | Wang et al. | ICLR | 2020 | Layer-wise matched averaging |
| Ditto | Li et al. | ICML | 2021 | Personalized FL with fairness |
| FedDF | Lin et al. | NeurIPS | 2020 | Ensemble distillation in FL |
| pFedMe | Dinh et al. | NeurIPS | 2020 | Personalized FL via Moreau envelope |

### 4.3 Threat Model (Formal)

```
ADVERSARIES AND DEFENSES:

Threat 1: Honest-But-Curious FL Server
  Attack: Server inspects individual gradients Δw_k to infer user data
  Defense: Secure Aggregation (Bonawitz 2017)
    → Server sees only SUM(Δw_k), never individual gradients
  
Threat 2: Gradient Inversion Attack (Zhao et al. 2020, Zhu et al. 2021)
  Attack: Even aggregated gradients can partially reconstruct images
  Defense: DP-SGD with σ ≥ 1.0 + batch size B ≥ 32
    → Theoretical bound: reconstruction quality degrades as O(σ²/B)
  
Threat 3: Byzantine (Malicious) Clients
  Attack: Clients submit poisoned gradients to corrupt global model
  Defense: FLAME clustering + cosine similarity filtering
    → Reject updates where cos_sim(Δw_k, median) < threshold
  
Threat 4: Membership Inference Attack (MIA)
  Attack: Infer if a specific image was in training set
  Defense: DP-SGD guarantees: Pr[A(M(D))=1] ≤ e^ε · Pr[A(M(D'))=1]
    → Target ε=3.0 limits MIA advantage to ≤28% above random

Privacy Guarantee (formal):
  (ε=3.0, δ=1e-5)-DP over T=200 rounds
  Implies: For any adjacent datasets D, D' differing in one user's data:
  Pr[M(D) ∈ S] ≤ e^3.0 · Pr[M(D') ∈ S] + 1e-5   ∀ S ⊆ Output
```

### 4.4 Advanced FL Protocol

```
FEDERATED SYSTEM PARAMETERS:
  Total clients N: 500 (simulation), target 100,000+ (deployment)
  Clients per round K: 100 (20% participation)
  Communication rounds T: 200
  Local epochs E: 5
  Local batch size B: 32
  Server learning rate η_s: 1.0
  Client learning rate η_c: 0.01 (cosine decay per round)

PRIVACY PARAMETERS:
  Target ε: 3.0
  Target δ: 1e-5
  Gradient clipping C: 1.0
  Noise multiplier σ: auto-calibrated (≈1.1 for K=100, T=200, B=32, N=500)
  Accountant: Rényi DP (RDP) via autodp moments accountant

PER-ROUND PROTOCOL:
  1. Server samples K clients uniformly at random (Poisson subsampling)
  2. Server broadcasts (w_t, control_variate_c) to each client k
  3. Client k runs SCAFFOLD-DP local update:
       Initialize: w_k,0 = w_t
       For local step i = 1..E×(n_k/B):
         Sample mini-batch b_i from local data D_k
         Compute per-sample gradients: {∇ℓ(w; x_i) for x_i ∈ b_i}
         Clip each: g̃_i = g_i / max(1, ||g_i||₂/C)
         SCAFFOLD correction: g̃_i -= (c_k - c)  [client drift correction]
         Aggregate + noise: G_b = (Σg̃_i + N(0, σ²C²I)) / |b_i|
         Update: w_k,i+1 = w_k,i - η_c · G_b
       Compute gradient: Δw_k = w_t - w_k,E (model diff)
       Update client control: c_k+ = c_k - c + (1/Eη_c)(w_t - w_k,E)
  
  4. Client encrypts Δw_k via Secure Aggregation (Shamir Secret Sharing)
  5. Server decrypts only SUM: Σ_k Δw_k (individual values hidden)
  6. FLAME Byzantine filter:
       Cluster {Δw_k} via HDBSCAN on cosine distance
       Compute smoothed aggregation: clip by L2-norm before sum
  7. Server update: w_{t+1} = w_t + (η_s/K)·FLAME_filtered_sum
  8. Server control update: c = c + (K/N)·Σ_k(c_k+ - c_k)
  9. RDP accountant: update ε budget consumed this round

NON-IID HANDLING:
  Real skin data is highly non-IID (tropical vs. temperate climates have
  completely different skin condition distributions)
  
  Strategy 1 (SCAFFOLD): Control variates correct client drift
  Strategy 2 (Clustered FL): Group clients by geographic cluster
    → Sub-model per cluster, shared global backbone
  Strategy 3 (FedDF Distillation): Server distills knowledge from 
    client models using public unlabeled skin images (DermNet NZ)
  
  Non-IID simulation: Dirichlet partition with α ∈ {0.1, 0.5, 1.0, ∞(IID)}
  α=0.1 → extreme non-IID (each client has ~1 condition)
  α=0.5 → moderate non-IID (realistic)
  α=∞   → IID baseline
```

### 4.5 Personalized FL for Skin Twin

```
DITTO PERSONALIZED FL:
  Each client maintains both:
    - Global model w_t (from FL aggregation)
    - Personal model v_k (fine-tuned to individual skin)
  
  Personal model objective:
    v_k* = argmin_v [ L(v; D_k) + (λ/2)||v - w_t||² ]
  
  Benefits:
    - v_k captures user's unique skin microbiome, color, texture
    - w_t provides global dermatological knowledge
    - Privacy: v_k never leaves device (stored in Secure Enclave)
    - λ controls personalization-vs-generalization tradeoff
```

### 4.6 Comprehensive Experiment Grid

```python
# Full experimental hyperparameter search
experiments = {
    'epsilon':        [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float('inf')],
    'noise_mult':     [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    'clients_rnd':    [10, 50, 100, 200, 500],
    'local_epochs':   [1, 3, 5, 10],
    'batch_size':     [16, 32, 64, 128],
    'aggregation':    ['FedAvg', 'FedProx', 'SCAFFOLD', 'FLAME+SCAFFOLD'],
    'non_iid_alpha':  [0.1, 0.5, 1.0, float('inf')],
    'personalized':   [True, False],   # Ditto vs. standard
}

# Primary metric: (accuracy, epsilon, convergence_rounds) Pareto front
# Secondary: fairness maintenance across FL rounds (FP-stratified)
# Tertiary: communication efficiency (MB/round × rounds_to_converge)
```

### 4.7 Evaluation Metrics

| Metric | Measurement | Target | Notes |
|---|---|---|---|
| Privacy Budget ε | RDP accountant (autodp) | ε ≤ 3.0 | δ=1e-5 |
| Global Accuracy Drop | vs. centralized baseline | ≤ 5% | Balanced accuracy |
| Personalized Accuracy | Ditto v_k on local test | ≥ centralized | Per-client |
| Convergence Rounds | Rounds to 90% centralized accuracy | ≤ 200 | — |
| Communication (MB/round) | Gradient size × compression | ≤ 2MB | Per client |
| MIA AUC | Shadow model attack (Shokri 2017) | ≤ 0.55 | Near-random |
| Gradient Inversion PSNR | Zhao 2020 attack on aggregated grad | ≤ 10 dB | Unintelligible |
| Byzantine Robustness | 20% malicious clients | Accuracy ≥ 85% | FLAME defense |

---

## 5. PILLAR 4 — Syndromic Surveillance Engine

### 5.1 Research Objective

Build a privacy-preserving outbreak detection system that transforms individual skin diagnoses into community health signals without ever knowing who the individuals are. The system must have clinical-grade sensitivity while maintaining a false positive rate suitable for public health use.

### 5.2 Literature Foundation

| Paper | Authors | Venue | Year | Contribution |
|---|---|---|---|---|
| SaTScan Spatial Scan | Kulldorff | Statistics in Medicine | 1997 | Likelihood-ratio spatial scan |
| CUSUM Surveillance | Fricker | Wiley Handbook | 2007 | Sequential CUSUM for public health |
| K-Anonymity | Sweeney | IJUFKS | 2002 | Location generalization theory |
| Geo-Indistinguishability | Andrés et al. | CCS | 2013 | Local DP for planar Laplace mechanism |
| Deep Anomaly Detection | Chalapathy & Chawla | ArXiv | 2019 | LSTM-AE for time-series anomaly |
| LSTM Epidemic Forecasting | Chimmula & Zhang | Chaos Solitons | 2020 | LSTM for COVID-19 forecasting |
| Prophet | Taylor & Letham | PeerJ | 2018 | Decomposable time-series anomaly detection |
| Transformer Epidemic | Wu et al. | AAAI | 2021 | Temporal fusion transformer for disease |
| Graph Neural Epi | Deng et al. | NeurIPS | 2021 | Spatial-temporal GNN for outbreak detection |
| BioSense 2.0 | Das et al. | MMWR | 2012 | CDC syndromic surveillance architecture |
| ESSENCE | Lombardo et al. | JAMIA | 2003 | Electronic syndromic surveillance system |

### 5.3 Complete Five-Stage Surveillance Architecture

```
═══════════════════════════════════════════════════════════════
                   SIGNAL INGESTION LAYER
═══════════════════════════════════════════════════════════════

ANONYMIZED SIGNAL PACKET (schema v2.0):
{
  "schema_version": "2.0",
  "condition_icd10": "B35.1",           # ICD-10 code (no free text)
  "condition_severity": 2,              # 1=mild, 2=moderate, 3=severe
  "geohash6": "u4pruyd",               # ~1.2km × 0.6km cell (not GPS)
  "week_iso": "2026-W08",              # ISO week (not date)
  "climate_zone": "tropical_humid",    # Coarse climate (not city)
  "device_salt_hash": "a3f9...",       # sha256(device_id + rotating_salt)
  "app_version": "2.1.4"               # For calibration drift detection
}
NOTE: No user ID, no name, no exact location, no image, no timestamp

═══════════════════════════════════════════════════════════════
           STAGE 1: TEMPORAL CUSUM DETECTION
═══════════════════════════════════════════════════════════════

For each (condition_icd10, geohash6) time series:
  
  Baseline computation (52-week rolling):
    μ_t = EWMA(x_{t-1}...x_{t-52}, α=0.1)  # Exponentially weighted mean
    σ_t = rolling std over 52 weeks
    
  CUSUM statistic (two-sided):
    S_t+ = max(0, S_{t-1}+ + (x_t - μ_t - k·σ_t))  # Upper (outbreak)
    S_t- = max(0, S_{t-1}- - (x_t - μ_t + k·σ_t))  # Lower (underreporting)
    
  Parameters:
    k = 0.5 (allowable slack — half a standard deviation)
    h+ = 5.0 (alert threshold for outbreak)
    h- = 3.0 (alert threshold for underreporting)
    
  Seasonality adjustment:
    μ_seasonal = Fourier regression (annual + semi-annual cycles)
    x_adjusted = x_t / μ_seasonal  # Deseasonalize before CUSUM

═══════════════════════════════════════════════════════════════
         STAGE 2: SPATIAL SCAN (KULLDORFF + GNN)
═══════════════════════════════════════════════════════════════

Classical Kulldorff Scan:
  Grid: all geohash6 cells with ≥1 report in past 4 weeks
  Scan: circular windows of radius r ∈ {50m, 200m, 500m, 1km, 2km, 5km}
  
  For each candidate cluster Z (set of geohash cells in window):
    Likelihood ratio:
      LR(Z) = (c_Z/μ_Z)^c_Z × ((C - c_Z)/(C - μ_Z))^(C-c_Z)
      where c_Z = observed cases in Z
            μ_Z = expected cases (Poisson null)
            C   = total cases
    
  Monte Carlo significance (999 permutations):
    Shuffle condition labels across geohash cells
    Record max LR for each permutation
    p-value = rank(LR_observed) / 1000
    Alert if p < 0.01

Graph Neural Network Enhancement (novel contribution):
  Build spatial graph G = (V, E):
    Nodes V: active geohash6 cells
    Edges E: adjacent cells within 5km + shared climate zone
    Node features: [case_count, week_since_first_case, severity_dist]
    Edge features: [geographic_distance, population_flow_proxy]
  
  GNN: 3-layer GraphSAGE → cluster_risk_score per node
  GNN trained on: synthetic outbreak scenarios (Stage 4 simulation data)
  GNN output: P(node is in outbreak cluster) ∈ [0,1]
  
  Combined score: final_score = 0.6·Kulldorff_LR + 0.4·GNN_risk

═══════════════════════════════════════════════════════════════
    STAGE 3: MULTI-USER VERIFICATION PROTOCOL (MUVP v2)
═══════════════════════════════════════════════════════════════

Anti-Gaming Defense (5 checks, all must pass):

  Check 1: Minimum unique devices
    unique_device_count(cluster, window) ≥ 5

  Check 2: Device diversity score  
    MUVP_score = unique_devices / total_reports ≥ 0.80
    (Catches single-device spam: 1 device × 10 reports = score 0.10 → FAIL)

  Check 3: Single-device dominance
    max_single_device_fraction = max(reports_k / total) ≤ 0.30
    (Prevents coordinated multi-account spoofing)

  Check 4: Temporal spread (anti-burst)
    time_spread = max_timestamp - min_timestamp ≥ 48 hours
    AND report_rate NOT exponential burst (Kolmogorov-Smirnov test vs. Poisson)

  Check 5: Geographic entropy
    H_geo = -Σ p_i·log(p_i)  over geohash cells in cluster
    H_geo ≥ log(3)  (reports from ≥3 distinct sub-cells, not all co-located)

  MUVP_PASS = all 5 checks satisfied

═══════════════════════════════════════════════════════════════
      STAGE 4: LSTM-AUTOENCODER ANOMALY SCORING
═══════════════════════════════════════════════════════════════

Architecture:
  Input: X = time series of shape (52, 8)
    Features: [case_count, severity_mean, device_count, geo_spread,
               new_device_fraction, weekly_delta, climate_index, icd_code_embed]
  
  Encoder LSTM:
    LSTM(128) → hidden h_1 ∈ R^128
    LSTM(64)  → bottleneck h_2 ∈ R^64
    
  Decoder LSTM (mirror):
    LSTM(64)  → LSTM(128) → Dense(8) → X_hat
    
  Training:
    Dataset: 3 years synthetic normal baseline (Poisson with seasonality)
             + 100 augmented known outbreak patterns (historical data)
    Loss: MSE(X, X_hat) + 0.1·KL(h_2, N(0,I))  [VAE-style regularization]
    Optimizer: Adam, LR=1e-3, early stopping
    
  Inference (anomaly scoring):
    anomaly_score = MSE(X, X_hat) over past 4 weeks
    Threshold θ = 95th percentile of training reconstruction errors
    Sub-threshold: normal seasonal fluctuation
    Super-threshold: potential outbreak pattern

  Advanced variant: Temporal Fusion Transformer (TFT)
    - Attention over 52-week history
    - Interpretable temporal importance weights
    - Better long-range outbreak pattern recognition

═══════════════════════════════════════════════════════════════
         STAGE 5: AND-LOGIC ALERT GATE + DISPATCH
═══════════════════════════════════════════════════════════════

Composite Alert Condition:
  TRIGGER if:
    (S_t+ > h+ from CUSUM)                      # Temporal anomaly
    AND (Kulldorff p < 0.01 OR GNN_score > 0.8) # Spatial cluster
    AND (MUVP_PASS = True)                       # Anti-gaming verified
    AND (LSTM_AE_score > θ)                      # Pattern matches outbreak
    AND (cooldown_hours_since_last_alert > 72)   # Rate limiting

  Alert payload (NO individual identifiers):
  {
    "alert_id": "uuid_v4",
    "condition_group": "Fungal Infection",        # ICD chapter, not code
    "geographic_zone": "Southern District, Tirupati", # Zone label, not GPS
    "estimated_count_range": "10-20 cases",       # Range bucket, not exact
    "confidence_score": 0.87,
    "recommended_action": "Community screening",
    "data_window": "2026-W06 to 2026-W08",
    "cluster_radius_estimate": "~2km",
    "alert_expiry": "2026-03-15T00:00:00Z"
  }
  
  Dispatch targets:
    → Public Health Dashboard (REST API)
    → Optional: WHO PHIN webhook
    → Optional: District Health Officer SMS alert

═══════════════════════════════════════════════════════════════
```

### 5.4 Simulation & Validation Protocol

```
SYNTHETIC OUTBREAK GENERATOR:
  Background: 500 geohash6 cells, 2-year baseline, condition-specific Poisson rates
  Seasonal model: μ_t = μ_0 × (1 + A·sin(2πt/52 + φ))
  
  Injected outbreaks (N=100 scenarios):
    Condition types: Fungal (Tinea), Scabies, Impetigo, Molluscum, Monkeypox-like
    Sizes: 5, 10, 20, 50, 100, 200 cases (stratified)
    Duration: 3, 7, 14, 30, 60 days
    Growth models: Exponential, Logistic (SIR), Delayed (incubation)
    Geographic patterns: Point source, Diffuse, Multi-focal
    Reporting delay: 0-7 days (random per case)

EVALUATION METRICS:
  Primary:
    Detection Sensitivity (at FPR=0.05): % outbreaks detected
    False Positive Rate: alerts/1000 person-weeks (target <5)
    Detection Lag: days from outbreak start to first alert (target <72h)
    
  Secondary:
    Geographic Precision: centroid distance error (target <2km)
    Size Estimation Error: predicted vs actual count range overlap
    Re-identification Rate: adversarial attempt to link alert → individual (<0.1%)
    
  Stratified by:
    Condition type, outbreak size, geographic pattern, reporting delay
    Climate zone (tropical vs. temperate conditions differ)
```

---

## 6. PILLAR 5 — Algorithmic Fairness & Bias Auditing

### 6.1 Research Objective

Deploy a multi-layer fairness system that operates at training time, inference time, and continuously during federated learning rounds. No model update should be accepted that degrades performance for any demographic group beyond tolerance.

### 6.2 Literature Foundation

| Paper | Authors | Venue | Year | Contribution |
|---|---|---|---|---|
| Equalized Odds | Hardt et al. | NeurIPS | 2016 | Fundamental group fairness definition |
| Fairness Through Awareness | Dwork et al. | ITCS | 2012 | Individual fairness via similarity metrics |
| No Subgroup Left Behind | Martinez et al. | ICML | 2020 | Minimax group fairness optimization |
| Dermatology AI Bias | Adamson & Smith | JAMA Dermatology | 2018 | Clinical evidence: 3.4× misdiagnosis on dark skin |
| AIF360 | Bellamy et al. | IBM Research | 2018 | Comprehensive fairness toolkit |
| Adversarial Debiasing | Zhang et al. | AIES | 2018 | Adversarial training for fair representations |
| Fair Federated Learning | Du et al. | ArXiv | 2021 | Fairness constraints in FL aggregation |
| FLASHE | Niu et al. | INFOCOM | 2022 | Fair and robust aggregation in FL |
| FairBatch | Roh et al. | ICLR | 2021 | Adaptive batch reweighing for fairness |

### 6.3 Four-Layer Debiasing Architecture

```
LAYER 1: STRUCTURAL (BDN — Pillar 1)
  BDN operates on H/M biological signals, not RGB pixel values
  Inherently removes skin-tone-dependent features at feature extraction
  This is the PRIMARY fairness mechanism

LAYER 2: TRAINING-TIME ADVERSARIAL DEBIASING
  Add Fitzpatrick discriminator D_fp: embedding → {FP_I-II, FP_III-IV, FP_V-VI}
  
  Training objectives:
    Classifier: minimize L_CE(diagnosis) + λ·L_adv (maximize D_fp loss)
    Discriminator: minimize L_CE(FP_group)
  
  Minimax game: C and D_fp trained in alternating steps
  Result: Classifier embedding contains no Fitzpatrick type information
  Verified by: linear probe accuracy on FP prediction ≈ 33% (random)

LAYER 3: LOSS REWEIGHING (AIF360 Reweighing)
  Compute importance weights for each (FP_group, condition, label) triple:
    w(g, c, y) = P(g, y) / (P_reference(g) × P_reference(y))
  
  Apply in loss: L_CE_weighted = Σ_i w(g_i, c_i, y_i) · L_CE(ŷ_i, y_i)
  
  Effect: Upweights underrepresented (dark skin, rare condition) combinations
  Re-computed each epoch as data distribution shifts

LAYER 4: FL-ROUND FAIRNESS MONITOR (Pillar 3 integration)
  After each FL aggregation round t:
    Evaluate w_{t+1} on FP-stratified validation set (held on server)
    Compute: {BalAcc(FP_g), EqOddsDiff, DemParityDiff} for all g
    
    Regression alert conditions:
      Δ BalAcc(any FP_g) > 3% drop in single round → ROLLBACK to w_t
      EqOddsDiff > 0.05 → flag round, investigate contributing clients
      Persistent (>3 round) fairness degradation → trigger debiasing Layer 3
    
    Automatic remediation:
      Identify client cluster contributing to bias (cosine similarity analysis)
      Apply FairBatch adaptive reweighing to that cluster's next update
```

### 6.4 Fairness Metrics Suite

| Metric | Formula | Target | Evaluated |
|---|---|---|---|
| Per-FP Balanced Accuracy | (TPR_g + TNR_g) / 2 | ≥85% all groups | Per FL round |
| Equalized Odds Diff | max(\|ΔTPR\|, \|ΔFPR\|) over FP pairs | ≤0.05 | Per FL round |
| Demographic Parity Diff | max(\|P(ŷ=1\|A) - P(ŷ=1\|B)\|) | ≤0.05 | Per FL round |
| Individual Fairness | Lipschitz: d(ŷ_i, ŷ_j) ≤ L·d(x_i, x_j) | L ≤ 5.0 | Periodic |
| Disparate Impact | min_g P(ŷ=1\|g) / max_g P(ŷ=1\|g) | ≥0.8 | Per condition |
| Calibration by FP Group | ECE per FP group | ≤0.05 all groups | Periodic |

---

## 7. PILLAR 6 — Explainability & Clinical Trust (NEW)

### 7.1 Research Objective

DermaSafe must provide clinician-interpretable explanations for each diagnosis to enable clinical trust and regulatory compliance (EU AI Act Article 13 — transparency).

### 7.2 Literature Foundation

| Paper | Authors | Venue | Year | Relevance |
|---|---|---|---|---|
| GradCAM | Selvaraju et al. | ICCV | 2017 | Gradient-weighted class activation maps |
| SHAP | Lundberg & Lee | NeurIPS | 2017 | SHapley Additive exPlanations |
| Concept Bottleneck Models | Koh et al. | ICML | 2020 | Predict via human-interpretable concepts |
| Monte Carlo Dropout | Gal & Ghahramani | ICML | 2016 | Uncertainty estimation via dropout |
| Deep Ensembles | Lakshminarayanan et al. | NeurIPS | 2017 | Calibrated uncertainty via ensembles |
| ProtoPNet | Chen et al. | NeurIPS | 2019 | Prototype-based interpretable networks |

### 7.3 Explanation System Architecture

```
THREE-LAYER EXPLANATION STACK:

Layer A: Spatial Attribution (GradCAM on Biophysical Maps)
  GradCAM(Hemoglobin Map): highlights inflamed regions
  GradCAM(Melanin Map):    highlights pigmentation changes
  
  Overlay on original image:
    Red heatmap = high hemoglobin activation (inflammation)
    Blue heatmap = high melanin activation (pigmentation change)
  
  Clinician interpretation:
    "Diagnosis driven by inflammation in upper-left lesion region"

Layer B: Concept-Level Explanation (SkinCon integration)
  SkinCon: 48 clinical concepts (e.g., "erythematous", "scaly", "vesicular")
  Train linear probe: BDN embeddings → SkinCon concept scores
  
  Output per image:
    "High confidence: erythematous (0.92), scaly (0.88), circular (0.71)"
  
  Clinician validation: Do detected concepts match diagnosis? (trust audit)

Layer C: Uncertainty Quantification
  Method: MC-Dropout (T=50 forward passes) + Temperature Scaling
  Output: 
    Predictive entropy H[ŷ] → overall confidence
    Aleatoric uncertainty → irreducible image noise
    Epistemic uncertainty → model uncertainty (flag for clinician review)
  
  Safety threshold: If epistemic uncertainty > 0.3 → "Refer to dermatologist"
```

---

## 8. PILLAR 7 — Adversarial Robustness (NEW)

### 8.1 Research Objective

Skin condition images may be manipulated (intentionally or via filters) to cause misdiagnosis. DermaSafe must be robust to adversarial perturbations and common image corruptions.

### 8.2 Literature Foundation

| Paper | Authors | Venue | Year | Relevance |
|---|---|---|---|---|
| PGD Attack | Madry et al. | ICLR | 2018 | Projected Gradient Descent adversarial |
| Certified Defenses | Cohen et al. | ICML | 2019 | Randomized smoothing for L2 robustness |
| Adversarial Training | Shafahi et al. | NeurIPS | 2019 | Free adversarial training |
| Natural Corruptions | Hendrycks & Dietterich | ICLR | 2019 | ImageNet-C corruption benchmark |
| Skin-Specific Attacks | Yoon et al. | MICCAI | 2021 | Adversarial attacks on dermoscopy AI |

### 8.3 Robustness Evaluation Suite

```
Attack categories:
  1. Lp-norm attacks: FGSM (ε=8/255), PGD-20 (ε=8/255), C&W-L2
  2. Natural corruptions: JPEG compression, brightness, contrast, blur
  3. Skin-specific: smartphone filter simulation (beauty mode, skin smoothing)
  4. Semantic attacks: realistic lighting/shadow perturbations

Defense strategy:
  Adversarial training: 50% clean + 50% PGD-10 augmented training data
  Certified defense: Randomized smoothing (σ=0.25) for L2 certification
  BDN robustness: H/M maps are inherently more robust than RGB
    → Evaluated: Does BDN reduce adversarial transferability?

Targets:
  Clean accuracy: ≥85% balanced (no degradation from defense)
  PGD-20 accuracy: ≥70% (certified adversarial robustness)
  Corruption robustness: ≥80% on common corruptions
```

---

## 9. Research Timeline (24-Month Advanced Plan)

```
PHASE 1: FOUNDATION & DATA (Months 1-4)
  M1: Dataset acquisition, preprocessing, Fitzpatrick stratification
      → Deliverable: Unified skin dataset loader with FP metadata
  M2: BDN development, physics loss implementation, ablation study
      → Deliverable: BDN v1.0 with benchmarks vs. RGB baseline
  M3: Full cloud pipeline (BDN + EfficientNet-B3), fairness baseline audit
      → Deliverable: Teacher model checkpoint + bias report
  M4: Explainability system (GradCAM + SkinCon + uncertainty)
      → Deliverable: Clinical interpretation dashboard prototype

PHASE 2: EDGE OPTIMIZATION (Months 5-8)
  M5: Multi-teacher knowledge distillation (MEAL V2)
      → Deliverable: Student model matching 98% teacher accuracy
  M6: Structured pruning + EWC fine-tuning
      → Deliverable: Pruned model <15MB with <2% accuracy drop
  M7: AdaRound quantization, QAT if needed, ONNX/TFLite/CoreML export
      → Deliverable: Platform-specific optimized models
  M8: Hardware benchmarking on Snapdragon/MediaTek/Apple
      → Deliverable: Latency/battery/RAM benchmark report

PHASE 3: FEDERATED LEARNING (Months 9-13)
  M9:  FL simulation framework (Flower), non-IID data splits
       → Deliverable: FL simulation infrastructure
  M10: FedAvg baseline, SCAFFOLD integration, convergence analysis
       → Deliverable: FL convergence curves (IID vs. non-IID)
  M11: DP-SGD + Rényi accountant, privacy-accuracy Pareto study
       → Deliverable: Privacy audit report (ε-δ guarantees)
  M12: Byzantine robustness (FLAME), secure aggregation implementation
       → Deliverable: Robustness under 20% malicious client scenario
  M13: Personalized FL (Ditto), Skin Twin integration
       → Deliverable: Personalized vs. global accuracy comparison

PHASE 4: SURVEILLANCE ENGINE (Months 14-18)
  M14: Synthetic outbreak generator, signal packet schema
       → Deliverable: Synthetic data pipeline (100 outbreak scenarios)
  M15: CUSUM + Kulldorff implementation + MUVP v2
       → Deliverable: Epidemiological pipeline with unit tests
  M16: LSTM-AE + Temporal Fusion Transformer training
       → Deliverable: Anomaly detection model benchmarks
  M17: GNN spatial-temporal integration (GraphSAGE)
       → Deliverable: GNN-enhanced cluster detection
  M18: End-to-end surveillance simulation + evaluation
       → Deliverable: Detection sensitivity/FPR/lag report

PHASE 5: ADVERSARIAL ROBUSTNESS & FAIRNESS (Months 19-21)
  M19: Adversarial training, certification (randomized smoothing)
       → Deliverable: Robustness evaluation report
  M20: Continuous FL fairness monitoring, Ditto fairness evaluation
       → Deliverable: Per-round fairness dashboard
  M21: Full adversarial + fairness integration testing
       → Deliverable: Integrated safety evaluation report

PHASE 6: INTEGRATION & PUBLICATION (Months 22-24)
  M22: Full system integration, regression testing, performance profiling
       → Deliverable: End-to-end system passing all tests
  M23: Clinical expert evaluation (IRB-approved user study)
       → Deliverable: Clinical validation report
  M24: Paper submission (target: Nature Digital Medicine / CVPR),
       open-source release, deployment documentation
       → Deliverable: Published paper + GitHub release + pip package
```

---

## 10. Master Dataset Registry

| Dataset | Size | FP Coverage | Conditions | Access | Pillar Use |
|---|---|---|---|---|---|
| ISIC 2020 | 33,126 | Low (I-III) | MEL, NV, BCC, 5 others | Kaggle | P1,P2 |
| Fitzpatrick17k | 16,577 | ALL 6 types | 114 grouped | GitHub (Harvard) | P1,P5 |
| PAD-UFES-20 | 2,298 | II-V | 6 conditions | Public | P1 |
| DDI Stanford | 656 | HIGH | 78 conditions | Public | P1,P5 |
| SD-198 | 6,584 | Medium | 198 categories | Public | P1 |
| SkinCon | 3,230 | Medium | 48 clinical concepts | Public | P6 |
| DermNet NZ | ~23,000 | High | 600+ conditions | Scraping | P3 (FedDF) |
| SKINL2 (2024) | 11,000 | HIGH (Africa) | 20 tropical | Public | P1,P5 |
| Derm7pt | 1,011 | Medium | 7-point checklist | Public | P6 |
| Synthetic GAN | Unlimited | Configurable | All conditions | Self-gen | P3,P4 |
| Synthetic Outbreaks | Unlimited | N/A | Surveillance signals | Self-gen | P4 |

---

## 11. Baselines & Expected Results

| Experiment | Baseline | DermaSafe Expected | Statistical Test |
|---|---|---|---|
| FP V-VI Classification | ResNet-50 RGB (58% BalAcc) | BDN+EfficientNet (≥73%) | McNemar's test p<0.05 |
| Edge Latency | EfficientNet-B3 (2.3s) | Student INT8 (≤500ms) | Paired t-test |
| FL vs. Centralized | FedAvg no DP (91%) | FedAvg+DP ε=3 (≥86%) | 95% CI |
| Outbreak Detection Lag | Threshold-only (5 days) | Full pipeline (≤72h) | Log-rank test |
| Fairness (EqOdds) | Unmitigated (0.18) | Multi-layer (≤0.05) | Bootstrap CI |
| Adversarial (PGD-20) | Undefended (31%) | AT+Smoothing (≥70%) | Certified bound |

---

## 12. Research Risks & Mitigations (Advanced)

| Risk | Probability | Impact | Detection Signal | Mitigation |
|---|---|---|---|---|
| Fitzpatrick17k too small for BDN | Medium | High | Val loss plateau on FP V-VI | GAN augmentation + SKINL2 addition |
| DP-SGD drops accuracy >5% | Medium | High | ε-accuracy curve inflection | Adaptive clipping (Thakkar 2019), larger K |
| Gradient inversion on BDN | Low | Critical | PSNR > 15 dB on R-GAP attack | Increase σ, batch size, add DP |
| Kulldorff false positives in low-pop cells | High | Medium | FPR >10% in simulation | Minimum population threshold, Bayesian scan |
| FL non-IID catastrophic forgetting | Medium | High | Accuracy collapse on global test | Elastic Weight Consolidation, FedDF |
| MUVP bypass via device spoofing | Low | Medium | MUVP score distribution shift | Rate limiting + device attestation |
| BDN map quality degrades at edge | Medium | Medium | PSNR < 25 dB on compressed BDN | Lightweight BDN design (Section 3.4) |
| IRB approval delay | Low | Medium | N/A | Pre-engage IRB M1, use synthetic data initially |

---

## 13. Full Python Library Stack

```python
# ─── Core Deep Learning ────────────────────────────────────────────
torch==2.2.0                # Training + inference
torchvision==0.17.0         # Pretrained models, transforms
timm==0.9.12                # EfficientNet, ViT, MobileNet zoo
tensorflow==2.15.0          # TFLite export
tensorflow-model-optimization  # TFLite pruning + quant
pytorch-lightning==2.1.0    # Training loop + callbacks
onnx==1.15.0
onnxruntime==1.17.0
ptflops                     # MACs/FLOPs profiling
fvcore                      # Facebook's profiler

# ─── Federated Learning ────────────────────────────────────────────
flwr==1.7.0                 # Flower FL framework
syft==0.8.7                 # PySyft (secure aggregation)

# ─── Differential Privacy ──────────────────────────────────────────
opacus==1.4.0               # DP-SGD
autodp==0.2                 # Rényi DP accountant
prv-accountant==0.1.0       # PRV accountant (tighter than RDP)

# ─── Epidemiology & Surveillance ───────────────────────────────────
pysal==2.6.0                # Spatial statistics
libpysal==4.9.0             # Spatial weights
statsmodels==0.14.0         # CUSUM, time series
prophet==1.1.5              # Seasonality decomposition
python-geohash==0.8.5       # Geohash encoding/decoding
torch-geometric==2.4.0      # GNN (GraphSAGE for spatial clustering)

# ─── Fairness & Bias ───────────────────────────────────────────────
aif360==0.5.0               # IBM AI Fairness 360
fairlearn==0.9.0            # Microsoft Fairlearn
folktables==0.0.12          # Fairness datasets

# ─── Explainability ────────────────────────────────────────────────
captum==0.7.0               # PyTorch interpretability
grad-cam==1.4.8             # GradCAM variants
shap==0.44.0                # SHAP explanations

# ─── Adversarial Robustness ────────────────────────────────────────
adversarial-robustness-toolbox==1.17.0  # ART toolkit
foolbox==3.3.3              # Attack implementations
torchattacks==3.4.0         # PyTorch attacks (FGSM, PGD, C&W)

# ─── Uncertainty Quantification ────────────────────────────────────
torch-uncertainty==0.1.3    # Uncertainty toolkit
netcal==1.3.5               # Calibration methods (Temperature Scaling, ECE)

# ─── Model Compression ─────────────────────────────────────────────
torch.quantization          # Built-in PTQ + QAT
torch.nn.utils.prune        # Structured/unstructured pruning
neural-compressor==2.4.0    # Intel Neural Compressor (AdaRound impl.)

# ─── Experiment Tracking ───────────────────────────────────────────
mlflow==2.9.0               # Experiment registry
wandb==0.16.0               # Real-time tracking
optuna==3.5.0               # Hyperparameter optimization (TPE, CMA-ES)
hydra-core==1.3.2           # Config management
omegaconf==2.3.0            # Structured configs

# ─── Data Processing ───────────────────────────────────────────────
albumentations==1.3.1       # Medical image augmentation
Pillow==10.2.0
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.0
scipy==1.12.0               # Statistical tests
```

---

*DermaSafe Advanced Research Plan v1.0 | Python/NN&DL Only | March 2026*
