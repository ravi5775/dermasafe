# DermaSafe: Full Project Structure Blueprint v2.0
### Tools · Workflow · Architecture · Implementation — Advanced All-in-One Reference
> **Stack:** Python · Neural Networks · Deep Learning Only  
> **Version:** 1.0 — Advanced Edition | March 2026

---

## 0. System Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        DERMASAFE SYSTEM ARCHITECTURE                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  ┌─────────────────────────────┐    ┌──────────────────────────────────────┐   ║
║  │      ZONE 1: EDGE DEVICE    │    │       ZONE 2: FL BRIDGE              │   ║
║  │  ┌───────────────────────┐  │    │  ┌────────────────────────────────┐  │   ║
║  │  │ Secure Enclave        │  │    │  │   Flower FL Server             │  │   ║
║  │  │  ┌──────────────┐     │  │    │  │   - SCAFFOLD Aggregation       │  │   ║
║  │  │  │ BDN v2       │     │  │    │  │   - FLAME Byzantine Filter     │  │   ║
║  │  │  │ H_map,M_map  │     │  │ ─► │  │   - Rényi DP Accountant        │  │   ║
║  │  │  └──────┬───────┘     │  │    │  │   - Fairness Monitor           │  │   ║
║  │  │  ┌──────▼───────┐     │  │    │  │   - Secure Aggregation (crypto) │  │   ║
║  │  │  │ EfficientNet │     │  │    │  └────────────────────────────────┘  │   ║
║  │  │  │ -B3 (INT8)   │     │  │    │              │                        │   ║
║  │  │  └──────┬───────┘     │  │    │   Encrypted Δw_k only (no images)    │   ║
║  │  │  ┌──────▼───────┐     │  │    └──────────────────────────────────────┘   ║
║  │  │  │ Skin Twin    │     │  │                                               ║
║  │  │  │ LSTM (local) │     │  │    ┌──────────────────────────────────────┐   ║
║  │  │  └──────┬───────┘     │  │    │    ZONE 3: SURVEILLANCE CLOUD        │   ║
║  │  │  ┌──────▼───────┐     │  │    │  ┌────────────────────────────────┐  │   ║
║  │  │  │ GradCAM      │     │  │    │  │ Signal Stream (anonymized)     │  │   ║
║  │  │  │ Explanation  │     │  │ ─► │  │ CUSUM → Kulldorff → MUVP v2   │  │   ║
║  │  │  └──────┬───────┘     │  │    │  │ GNN → LSTM-AE → Alert Gate    │  │   ║
║  │  │  ┌──────▼───────┐     │  │    │  │ Public Health Dashboard API    │  │   ║
║  │  │  │ MC Dropout   │     │  │    │  └────────────────────────────────┘  │   ║
║  │  │  │ Uncertainty  │     │  │    └──────────────────────────────────────┘   ║
║  │  │  └─────────────┘     │  │                                               ║
║  │  │  RAW IMAGE DISCARDED  │  │                                               ║
║  │  └───────────────────────┘  │                                               ║
║  └─────────────────────────────┘                                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 1. Complete Repository Structure

```
dermasafe/
│
├── README.md                        # Project overview, quickstart, citation
├── LICENSE                          # MIT License
├── CONTRIBUTING.md                  # Contribution guidelines
├── CHANGELOG.md                     # Version history
├── requirements.txt                 # Production dependencies (pinned)
├── requirements-dev.txt             # Dev/test dependencies
├── setup.py                         # Package setup
├── setup.cfg                        # Package metadata
├── pyproject.toml                   # Build system config (PEP 517)
├── Makefile                         # CLI shortcuts:
│                                    #   make data       — download + preprocess
│                                    #   make train      — cloud training
│                                    #   make compress   — full compression pipeline
│                                    #   make fl         — federated learning sim
│                                    #   make surv       — surveillance simulation
│                                    #   make eval       — full evaluation suite
│                                    #   make test       — pytest suite
│                                    #   make lint       — flake8 + black + mypy
│                                    #   make docs       — sphinx docs
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                   # CI: pytest + lint + coverage on every PR
│   │   ├── fairness_check.yml       # FL fairness regression test on model commit
│   │   └── privacy_audit.yml        # Privacy budget verification
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
│
├── configs/                         # OmegaConf/Hydra YAML configs
│   ├── base.yaml                    # Global defaults (seed, device, logging)
│   ├── data/
│   │   ├── isic.yaml
│   │   ├── fitzpatrick17k.yaml
│   │   ├── federated.yaml           # FL data split parameters
│   │   └── surveillance.yaml        # Signal packet schema params
│   ├── model/
│   │   ├── bdn_v2.yaml              # BDN architecture (encoder, decoder dims)
│   │   ├── classifier.yaml          # EfficientNet-B3 params
│   │   ├── skin_twin.yaml           # LSTM Skin Twin params
│   │   ├── student.yaml             # MobileNetV3-Small params
│   │   └── lstm_ae.yaml             # LSTM-AE for surveillance
│   ├── training/
│   │   ├── cloud.yaml               # Cloud training HP (lr, epochs, sched)
│   │   ├── distillation.yaml        # KD: temperature, alpha, beta
│   │   ├── pruning.yaml             # Pruning ratio, sensitivity analysis
│   │   ├── quantization.yaml        # PTQ/QAT params, calibration
│   │   └── adversarial.yaml         # AT: epsilon, PGD steps, α
│   ├── federated/
│   │   ├── server.yaml              # FL server: rounds, clients/round, aggregation
│   │   ├── client.yaml              # Client: local epochs, batch size
│   │   ├── privacy.yaml             # DP: ε, δ, clipping norm, noise mult
│   │   └── personalized.yaml        # Ditto: lambda, local epochs
│   ├── surveillance/
│   │   ├── cusum.yaml               # k_slack, alert threshold, seasonality
│   │   ├── kulldorff.yaml           # radius range, p-value, MC reps
│   │   ├── muvp.yaml                # MUVP thresholds (all 5 checks)
│   │   ├── anomaly.yaml             # LSTM-AE: seq length, hidden dims, percentile
│   │   └── alert.yaml               # Cooldown, dispatch endpoints
│   └── fairness/
│       ├── metrics.yaml             # Alert thresholds, group definitions
│       └── debiasing.yaml           # Adversarial λ, reweighing method
│
├── data/
│   ├── raw/                         # Original downloads (git-ignored, in .gitignore)
│   │   ├── isic2020/
│   │   ├── fitzpatrick17k/
│   │   ├── pad_ufes_20/
│   │   ├── ddi/
│   │   ├── sd198/
│   │   ├── skincon/
│   │   └── skinl2/                  # NEW: Africa-focused dataset
│   ├── processed/                   # Preprocessed (git-ignored)
│   │   ├── train/                   # {image, label, fitzpatrick_type, source}
│   │   ├── val/                     # Stratified by FP type
│   │   └── test/                    # Held-out, stratified by FP type + dataset
│   ├── federated_splits/            # Non-IID client partitions
│   │   ├── iid/                     # IID baseline splits (client_0..499)
│   │   ├── non_iid_alpha0.1/        # Extreme non-IID
│   │   ├── non_iid_alpha0.5/        # Moderate non-IID (realistic)
│   │   └── clustered/               # Geography-clustered splits
│   ├── synthetic/
│   │   ├── dark_skin_augmented/     # GAN-synthesized FP V-VI images
│   │   └── unlabeled_public/        # DermNet NZ (for FedDF distillation)
│   └── surveillance_stream/
│       ├── synthetic_normal_3yr.csv # 3-year Poisson baseline signal stream
│       ├── outbreak_scenarios/      # 100 injected outbreak scenarios
│       └── signal_packets/          # Simulated anonymized packets (dev/test)
│
├── dermasafe/                       # CORE PYTHON PACKAGE
│   ├── __init__.py                  # Package version, top-level imports
│   │
│   ├── data/                        # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── datasets.py              # SkinDataset, FitzpatrickDataset, MultiDataset
│   │   │                            # → __getitem__ returns {image, label, fp_type, meta}
│   │   ├── transforms.py            # Albumentations pipelines:
│   │   │                            # train_transform: RandCrop, Flip, ColorJitter,
│   │   │                            #   GridDistortion, CoarseDropout, Normalize
│   │   │                            # val_transform: CenterCrop, Normalize
│   │   │                            # bdn_transform: minimal (preserve biophysics)
│   │   ├── federated_split.py       # IID: sklearn StratifiedShuffleSplit
│   │   │                            # Non-IID: Dirichlet(α) allocation
│   │   │                            # Clustered: K-means on {condition, geography}
│   │   ├── synthetic_generator.py   # StyleGAN2-ADA fine-tuned on FP V-VI
│   │   │                            # → Generate N dark-skin samples per condition
│   │   ├── signal_packet.py         # SignalPacket dataclass + validation schema
│   │   │                            # → Pydantic model with strict type checking
│   │   └── outbreak_generator.py   # Synthetic outbreak simulator for surveillance
│   │                                # → Poisson background + epidemic growth injection
│   │
│   ├── models/                      # All neural network architectures
│   │   ├── __init__.py
│   │   │
│   │   ├── bdn.py                   # BiophysicalDecompositionNetwork v2
│   │   │   # class BiophysicalDecompositionNetwork(nn.Module)
│   │   │   #   Encoder: ResNet-18 (pretrained) with Feature Pyramid
│   │   │   #   HemoglobinDecoder: 5x DepthwiseSepTransposeConv + skip + sigmoid
│   │   │   #   MelaninDecoder:    5x DepthwiseSepTransposeConv + skip + sigmoid
│   │   │   #   PhysicsReconstructror: learned αH, αM, αS scalars
│   │   │   #   forward(x) → (h_map, m_map)
│   │   │   #   reconstruct_rgb(h, m) → I_hat
│   │   │   #   quality_score(x) → float (for early-exit decision)
│   │   │
│   │   ├── bdn_mobile.py            # Lightweight BDN for edge
│   │   │   # Encoder: MobileNetV2 (3.4M params, vs ResNet-18 11M)
│   │   │   # Decoders: Factorized DepthwiseSepTransposeConv (8x fewer params)
│   │   │   # Compressed: 2.1M params, 0.8G MACs
│   │   │
│   │   ├── classifier.py            # EfficientNet-B3 diagnosis head
│   │   │   # Input: (B, 2, 224, 224) — concatenated [H_map, M_map]
│   │   │   # Backbone: timm.create_model('efficientnet_b3', in_chans=2)
│   │   │   # Head: GAP → Dropout(0.3) → FC(256) → Dropout(0.2) → FC(num_classes)
│   │   │   # Supports: 10-class ISIC, 114-class Fitzpatrick17k
│   │   │
│   │   ├── skin_twin.py             # LSTM Personalization Module
│   │   │   # Input: (B, T, feature_dim) — history of [condition, confidence, date]
│   │   │   # LSTM(128, bidirectional=False) → FC(64) → FC(num_classes)
│   │   │   # Output: Δlogits adjustment vector (added to classifier output)
│   │   │   # Stored: on-device in encrypted SQLite (never synced)
│   │   │
│   │   ├── student_model.py         # Compressed edge model
│   │   │   # BDN Mobile + MobileNetV3-Small classifier
│   │   │   # Pre-distilled, pruned, quantized
│   │   │   # forward_with_explanation() → (logits, h_map, m_map, gradcam)
│   │   │
│   │   ├── lstm_autoencoder.py      # LSTM-AE for surveillance anomaly detection
│   │   │   # Encoder: LSTM(128) → LSTM(64) → bottleneck h ∈ R^64
│   │   │   # Decoder: LSTM(64) → LSTM(128) → Dense(feature_dim)
│   │   │   # VAE variant: bottleneck h ~ N(μ, σ²) (KL regularization)
│   │   │   # anomaly_score(x) → MSE reconstruction error
│   │   │   # fit_threshold(x_normal) → 95th percentile threshold θ
│   │   │
│   │   ├── temporal_fusion.py       # Temporal Fusion Transformer (TFT) — advanced
│   │   │   # Alternative to LSTM-AE for surveillance (better long-range patterns)
│   │   │   # Interpretable: attention weights show which weeks drove anomaly
│   │   │
│   │   ├── gnn_surveillance.py      # GraphSAGE for spatial cluster risk scoring
│   │   │   # Node features: [case_count, weeks_active, severity, device_count]
│   │   │   # Edge features: [geo_distance, climate_similarity]
│   │   │   # 3-layer GraphSAGE → risk_score ∈ [0,1] per node
│   │   │   # Trained on: synthetic outbreak scenarios
│   │   │
│   │   ├── discriminator.py         # Fitzpatrick type discriminator (for debiasing)
│   │   │   # Input: embedding from BDN or classifier
│   │   │   # FC(256) → FC(128) → FC(3)  {FP_I-II, FP_III-IV, FP_V-VI}
│   │   │
│   │   ├── concept_bottleneck.py    # SkinCon concept bottleneck model
│   │   │   # Linear probe: BDN embedding → 48 SkinCon concepts
│   │   │   # Used for clinician-interpretable explanations
│   │   │
│   │   └── dermasafe_full.py        # Assembled complete pipeline
│   │       # class DermaSafePipeline(nn.Module):
│   │       #   Components: bdn, classifier, skin_twin, uncertainty_head
│   │       #   forward(x, history, mc_dropout_passes=50) →
│   │       #     {logits, h_map, m_map, concepts, uncertainty, explanation_map}
│   │       #   explain(x) → clinical explanation dict
│   │       #   is_confident(x) → bool (refers to clinician if False)
│   │
│   ├── losses/                      # All loss functions
│   │   ├── __init__.py
│   │   ├── biophysical_loss.py      # L_phys = ||I_rgb - I_hat||²_F
│   │   │                            # L_smooth = ||∇H_map||₁ + ||∇M_map||₁
│   │   │                            # L_sparsity = ||M_map||₁ (clinical prior)
│   │   │                            # L_consistency = MSE(f(aug1(x)), f(aug2(x)))
│   │   ├── fairness_loss.py         # L_fair = max(|ΔTPR|, |ΔFPR|) across FP pairs
│   │   │                            # L_adv = -L_CE(discriminator, FP_labels)
│   │   ├── distillation_loss.py     # L_KD = KL(softmax(z_s/T), softmax(z_t/T))
│   │   │                            # L_feat = ||proj(F_s) - F_t||²_F (FitNets)
│   │   │                            # L_MEAL = Σ_t α_t·L_KD_t (multi-teacher)
│   │   └── combined_loss.py         # Weighted L_total with dynamic λ scheduler
│   │                                # LossConfig dataclass for all weights
│   │
│   ├── training/                    # Training orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py               # PyTorch Lightning LightningModule
│   │   │                            # Handles: forward, loss, optimizer, scheduler
│   │   │                            # Auto-logs: all metrics to MLflow + W&B
│   │   ├── federated_trainer.py     # Flower NumPyClient implementation
│   │   │                            # fit(): SCAFFOLD-DP local training
│   │   │                            # evaluate(): local test metrics
│   │   │                            # get_parameters() / set_parameters()
│   │   ├── distillation_trainer.py  # Multi-teacher MEAL V2 distillation
│   │   │                            # Teacher ensemble: [B4, B3, R50]
│   │   │                            # Intermediate feature matching (FitNets)
│   │   ├── adversarial_trainer.py   # PGD adversarial training loop
│   │   │                            # 50% clean + 50% PGD-10 batches
│   │   │                            # Randomized smoothing certification
│   │   └── callbacks.py             # PyTorch Lightning callbacks:
│   │                                # FairnessMonitorCallback (per-epoch)
│   │                                # PrivacyBudgetCallback (abort if ε exceeded)
│   │                                # GradCAMVisualizationCallback
│   │                                # CheckpointCallback (best per FP group)
│   │
│   ├── privacy/                     # Privacy guarantees
│   │   ├── __init__.py
│   │   ├── dp_sgd.py                # DPTrainer: Opacus PrivacyEngine wrapper
│   │   │                            # setup(model, optimizer, loader) → dp_triple
│   │   │                            # get_epsilon() → current ε consumed
│   │   │                            # should_abort() → bool (budget check)
│   │   │                            # privacy_report() → detailed ε-δ breakdown
│   │   ├── dp_accountant.py         # Rényi DP + PRV accountant wrapper
│   │   │                            # track_round(σ, q, steps) → cumulative ε
│   │   │                            # export_privacy_certificate() → JSON
│   │   ├── secure_aggregation.py    # Simulated Secure Aggregation
│   │   │                            # Shamir Secret Sharing (N, K threshold)
│   │   │                            # mask_gradient(Δw_k) → masked_k
│   │   │                            # aggregate(masked_updates) → sum only
│   │   ├── noise_calibrator.py      # Auto-calibrate σ for target (ε, δ)
│   │   │                            # Binary search over σ using autodp
│   │   │                            # calibrate(ε, δ, T, K, N, B, E) → σ
│   │   ├── gradient_compressor.py   # Optional: top-k sparse gradient compression
│   │   │                            # Reduces communication cost by ~10x
│   │   │                            # sparsify(Δw_k, k=0.01) → sparse_Δw_k
│   │   └── mia_auditor.py           # Membership Inference Attack simulator
│   │                                # Shokri shadow model attack
│   │                                # mia_auc(model, train_set, test_set) → AUC
│   │
│   ├── compression/                 # Model compression pipeline
│   │   ├── __init__.py
│   │   ├── knowledge_distillation.py  # MEAL V2 multi-teacher distillation
│   │   │                               # Teachers: list of (model, weight) pairs
│   │   │                               # intermediate_feature_matching(): FitNets
│   │   │                               # distill(teacher_list, student, config) → student
│   │   ├── pruner.py                # Structured channel pruning
│   │   │                            # SensitivityAnalyzer: Hessian trace per layer
│   │   │                            # ChannelPruner: magnitude-based, L2 norm
│   │   │                            # prune(model, ratio, sensitivity) → pruned_model
│   │   │                            # finetune_ewc(pruned, teacher, config) → model
│   │   ├── quantizer.py             # Quantization pipeline
│   │   │                            # ptq_adround(model, calibration_loader) → INT8
│   │   │                            # qat_finetune(model, train_loader, epochs) → INT8
│   │   │                            # mixed_precision(model, sensitive_layers) → mixed
│   │   └── exporter.py              # Multi-platform export
│   │                                # to_onnx(model, path, opset=17)
│   │                                # to_tflite(onnx_path, optimize=True) → .tflite
│   │                                # to_coreml(model, path) → .mlmodel
│   │                                # to_onnxruntime(onnx_path) → ORT session
│   │
│   ├── surveillance/                # Community disease surveillance
│   │   ├── __init__.py
│   │   ├── geofuzzer.py             # GPS → geohash fuzzification
│   │   │                            # fuzz_location(lat, lon, precision=6) → geohash
│   │   │                            # planar_laplace_noise(lat, lon, ε_loc) → fuzzed
│   │   │                            # geohash_to_zone_label(geohash) → "District X"
│   │   ├── cusum_detector.py        # CUSUM temporal detector
│   │   │                            # CUSUMState: S_t+, S_t-, μ_seasonal
│   │   │                            # update(condition, geohash, week, count) → alert
│   │   │                            # reset_state(condition, geohash) → None
│   │   │                            # Seasonality: Fourier regression (annual + semi-annual)
│   │   ├── spatial_scan.py          # Kulldorff + GNN spatial scanner
│   │   │                            # KulldorffScanner: Monte Carlo 999 reps
│   │   │                            # scan(condition, active_geohashes) → ClusterResult
│   │   │                            # ClusterResult: {geohash_set, LR, p_value, expected}
│   │   │                            # GNNScanner: GraphSAGE risk scoring overlay
│   │   ├── muvp.py                  # Multi-User Verification Protocol v2
│   │   │                            # MUVPChecker: all 5 checks
│   │   │                            # check_unique_devices(cluster) → bool
│   │   │                            # check_diversity_score(cluster) → float
│   │   │                            # check_single_dominance(cluster) → bool
│   │   │                            # check_temporal_spread(cluster) → bool
│   │   │                            # check_geo_entropy(cluster) → bool
│   │   │                            # verify(cluster) → MUVPResult
│   │   ├── anomaly_scorer.py        # LSTM-AE + TFT anomaly scoring
│   │   │                            # AnomalyScorer: wraps LSTM-AE or TFT model
│   │   │                            # train(normal_stream) → fitted scorer
│   │   │                            # fit_threshold(normal_stream) → θ
│   │   │                            # score(condition, geohash) → anomaly_score
│   │   ├── alert_engine.py          # Five-stage AND-logic alert gate
│   │   │                            # AlertEngine.process_packet(packet) → Alert|None
│   │   │                            # AlertEngine.batch_process(packets) → [Alert]
│   │   │                            # Alert dataclass: condition, zone, count_range, conf
│   │   ├── dashboard_api.py         # REST API (FastAPI) for public health dashboard
│   │   │                            # GET /alerts → list of active alerts
│   │   │                            # GET /alerts/{id} → alert detail
│   │   │                            # GET /heatmap/{condition} → geojson heatmap
│   │   │                            # POST /acknowledge/{id} → health officer ack
│   │   └── simulation.py            # Synthetic outbreak simulation framework
│   │                                # OutbreakGenerator: Poisson + epidemic growth
│   │                                # inject(stream, scenario) → contaminated stream
│   │                                # evaluate(detector, stream, ground_truth) → metrics
│   │
│   ├── fairness/                    # Fairness monitoring & mitigation
│   │   ├── __init__.py
│   │   ├── metrics.py               # All fairness metrics
│   │   │                            # balanced_accuracy_per_group(y, y_hat, groups)
│   │   │                            # equalized_odds_diff(y, y_hat, groups)
│   │   │                            # demographic_parity_diff(y_hat, groups)
│   │   │                            # individual_fairness_score(model, x_pairs, dist)
│   │   │                            # FairnessReport dataclass: all metrics + timestamp
│   │   ├── monitor.py               # FL fairness regression detection
│   │   │                            # FairnessMonitor: per-round evaluation
│   │   │                            # evaluate_round(model, val_per_group) → Report
│   │   │                            # check_regression(curr, prev) → bool + reason
│   │   │                            # recommend_remediation(regression) → action
│   │   ├── debiaser.py              # Multi-layer debiasing
│   │   │                            # AdversarialDebiaser: minimax discriminator training
│   │   │                            # AIF360Reweigher: importance weight computation
│   │   │                            # FairBatchSampler: adaptive batch reweighing
│   │   │                            # apply_all_layers(model, data) → debiased_model
│   │   └── audit_report.py          # Automated fairness audit report generator
│   │                                # generate_report(model, test_sets) → PDF + JSON
│   │                                # Includes: per-FP confusion matrices, calibration
│   │
│   ├── explainability/              # Clinical interpretability (NEW MODULE)
│   │   ├── __init__.py
│   │   ├── gradcam.py               # GradCAM on BDN outputs
│   │   │                            # BDNGradCAM: separate cams for H_map and M_map
│   │   │                            # explain(model, image) → {h_cam, m_cam, overlay}
│   │   │                            # HiResCAM, GradCAM++ variants supported
│   │   ├── concept_explainer.py     # SkinCon concept-level explanation
│   │   │                            # ConceptExplainer: linear probe on BDN embedding
│   │   │                            # explain(embedding) → {concept: score} dict
│   │   │                            # top_concepts(embedding, n=5) → ranked list
│   │   ├── uncertainty.py           # Uncertainty quantification
│   │   │                            # MCDropoutEstimator: T forward passes
│   │   │                            # entropy(mc_probs) → predictive entropy H
│   │   │                            # aleatoric(mc_probs) → data uncertainty
│   │   │                            # epistemic(mc_probs) → model uncertainty
│   │   │                            # calibrate(model, val_loader) → Temperature T*
│   │   │                            # should_refer(epistemic_unc, threshold=0.3) → bool
│   │   └── clinical_report.py       # Clinical explanation report generator
│   │                                # generate_clinical_report(image, prediction) →
│   │                                #   {"diagnosis", "confidence", "skin_signals",
│   │                                #    "clinical_concepts", "explanation_map",
│   │                                #    "uncertainty", "recommendation"}
│   │
│   ├── robustness/                  # Adversarial robustness (NEW MODULE)
│   │   ├── __init__.py
│   │   ├── attacks.py               # Attack implementations (using torchattacks)
│   │   │                            # FGSM, PGD-20, C&W-L2, DeepFool
│   │   │                            # SkinFilterAttack: beauty mode simulation
│   │   │                            # NaturalCorruption: brightness/blur/JPEG
│   │   ├── defenses.py              # Defense implementations
│   │   │                            # AdversarialTrainer: free AT (Shafahi 2019)
│   │   │                            # RandomizedSmoother: certify L2 robustness
│   │   │                            # certify(model, x, σ, n_smooth) → (pred, radius)
│   │   └── evaluator.py             # Robustness evaluation suite
│   │                                # evaluate_attacks(model, test_loader) → robustness_report
│   │                                # RobustnessReport: clean, adv, certified accuracy
│   │
│   ├── evaluation/                  # Comprehensive evaluation framework
│   │   ├── __init__.py
│   │   ├── evaluator.py             # Full evaluation pipeline
│   │   │                            # evaluate_all(model, test_loaders) → EvalReport
│   │   │                            # Stratified by: FP type, condition, dataset
│   │   ├── benchmarker.py           # Mobile performance benchmarking
│   │   │                            # profile_latency(model_path, n_runs=100)
│   │   │                            # profile_memory(model_path)
│   │   │                            # profile_battery(model_path, n_sessions=10)
│   │   ├── privacy_auditor.py       # Privacy evaluation
│   │   │                            # mia_audit(model) → AUC score
│   │   │                            # gradient_inversion_audit(model) → PSNR
│   │   │                            # reconstruct_attempt(gradients) → PSNR
│   │   └── bias_visualizer.py       # Bias visualization tools
│   │                                # plot_fitzpatrick_confusion_matrices()
│   │                                # plot_equalized_odds_radar()
│   │                                # plot_fl_fairness_over_rounds()
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── config.py                # OmegaConf + Hydra config loader
│       │                            # load_config(path, overrides=[]) → DictConfig
│       ├── logger.py                # Dual MLflow + W&B logger
│       │                            # DualLogger: log_metric, log_artifact, log_model
│       ├── seed.py                  # Full reproducibility
│       │                            # set_seed(42): torch, numpy, random, CUDA
│       ├── device.py                # Device selection + memory management
│       │                            # get_device() → 'cuda'|'mps'|'cpu'
│       │                            # clear_gpu_memory() → None
│       ├── visualization.py         # Visualization utilities
│       │                            # overlay_heatmap(image, cam) → overlaid
│       │                            # render_biophysical_maps(h, m) → figure
│       │                            # plot_privacy_accuracy_pareto(results)
│       └── clinical_vocab.py        # ICD-10 codes, condition names, FP type labels
│                                    # icd10_to_group(code) → condition_group
│                                    # FITZPATRICK_TYPES: type descriptions
│
├── scripts/                         # Entry-point scripts
│   ├── download_datasets.py         # Download all datasets with progress tracking
│   │                                # --datasets [all | isic | fitz | ddi | ...] 
│   ├── prepare_data.py              # Full preprocessing pipeline
│   │                                # Resize to 224×224, normalize, split, metadata
│   ├── train_cloud.py               # Cloud training entry point
│   │                                # --config, --stage [bdn|classifier|full]
│   │                                # --bdn_ckpt (for classifier-only training)
│   ├── train_federated.py           # FL simulation entry point
│   │                                # --config, --rounds, --clients, --epsilon
│   ├── compress_model.py            # Full compression pipeline
│   │                                # --stage [distill|prune|quantize|all]
│   │                                # --teacher, --config
│   ├── run_surveillance_sim.py      # Outbreak simulation evaluation
│   │                                # --outbreaks N, --duration Xyr, --seed
│   ├── generate_fairness_report.py  # Fairness audit report generation
│   │                                # --model, --test_data, --output_dir
│   ├── run_privacy_audit.py         # MIA + gradient inversion audit
│   │                                # --model, --train_data, --test_data
│   └── benchmark_edge.py            # Mobile performance benchmark
│                                    # --model_path (.tflite|.mlmodel)
│                                    # --platform [android|ios|onnxruntime]
│
├── fl_server/                       # FL server (deployed to cloud VM)
│   ├── __init__.py
│   ├── server.py                    # Flower FL server main
│   │                                # Launches server with custom strategy
│   │                                # Handles: round coordination, model broadcasting
│   ├── strategy.py                  # DermaSafeFedAvgStrategy(fl.server.strategy.FedAvg)
│   │                                # Overrides aggregate_fit():
│   │                                #   → FLAME Byzantine filtering
│   │                                #   → Fairness-aware weighted aggregation
│   │                                #   → Post-round fairness evaluation
│   │                                # Overrides configure_fit():
│   │                                #   → Send per-client SCAFFOLD control variates
│   ├── aggregation_hooks.py         # Pre/post aggregation hooks
│   │                                # pre_aggregate: verify MUVP-style device diversity
│   │                                # post_aggregate: fairness check → rollback if needed
│   └── model_registry.py            # Server-side model versioning
│                                    # save_checkpoint(round, model, metrics)
│                                    # rollback(round) → previous model
│                                    # export_production(round) → .tflite
│
├── fl_client/                       # FL client (simulated / real device)
│   ├── __init__.py
│   ├── client.py                    # DermaSafeClient(fl.client.NumPyClient)
│   │                                # fit(params, config) → (params, n_examples, metrics)
│   │                                # Uses: SCAFFOLD-DP local training
│   ├── local_trainer.py             # Client-side DP-SGD training
│   │                                # LocalSCAFFOLDTrainer:
│   │                                #   train(model, data, control_variate, dp_config)
│   │                                #   → (updated_model, delta_w, updated_control)
│   ├── signal_emitter.py            # Anonymized signal packet emitter
│   │                                # on_diagnosis(condition, lat, lon):
│   │                                #   → geofuzz → timestamp_bucket → hash_device_id
│   │                                #   → POST /ingest to surveillance API
│   └── skin_twin_manager.py         # On-device Skin Twin management
│                                    # store_history(diagnosis, confidence, timestamp)
│                                    # get_personalized_adjustment(history) → Δlogits
│                                    # clear_history() → None (privacy control)
│
├── surveillance_server/             # Surveillance backend (separate service)
│   ├── __init__.py
│   ├── app.py                       # FastAPI application factory
│   ├── routers/
│   │   ├── ingest.py                # POST /ingest — receive signal packets
│   │   ├── alerts.py                # GET /alerts, GET /alerts/{id}
│   │   ├── heatmap.py               # GET /heatmap/{condition} → GeoJSON
│   │   └── admin.py                 # Admin: threshold tuning, model refresh
│   ├── pipeline.py                  # Full 5-stage pipeline orchestrator
│   │                                # process_packet(packet) → maybe_alert
│   └── background_tasks.py          # Async: run CUSUM, retrain LSTM-AE, refresh GNN
│
├── tests/                           # Full test suite
│   ├── unit/
│   │   ├── test_bdn.py              # BDN forward shapes, grad flow, physics loss
│   │   ├── test_bdn_mobile.py       # Mobile BDN compression correctness
│   │   ├── test_dp_sgd.py           # Gradient norm clipping, noise statistics
│   │   ├── test_noise_calibrator.py # σ calibration correctness vs manual calc
│   │   ├── test_cusum.py            # CUSUM alert triggers, seasonality, two-sided
│   │   ├── test_kulldorff.py        # LR computation, MC p-value distribution
│   │   ├── test_muvp.py             # All 5 MUVP checks, anti-gaming scenarios
│   │   ├── test_anomaly_scorer.py   # LSTM-AE reconstruction, threshold calibration
│   │   ├── test_gnn_surveillance.py # GraphSAGE forward pass, node features
│   │   ├── test_fairness_metrics.py # Equalized odds, DP diff correctness
│   │   ├── test_gradcam.py          # GradCAM output shapes, overlay
│   │   ├── test_uncertainty.py      # MC Dropout entropy bounds
│   │   └── test_geofuzzer.py        # Geohash precision, planar Laplace noise
│   ├── integration/
│   │   ├── test_full_pipeline.py    # Image → diagnosis → signal packet → alert
│   │   ├── test_fl_round.py         # One complete FL round: server+clients
│   │   ├── test_surveillance_e2e.py # 30-day stream → outbreak detection
│   │   ├── test_fairness_fl.py      # FL round → fairness monitor → rollback
│   │   └── test_compression_chain.py # teacher → KD → prune → quant → accuracy
│   ├── adversarial/
│   │   ├── test_pgd_robustness.py   # PGD attack evaluation
│   │   └── test_mia.py              # MIA AUC evaluation
│   └── conftest.py                  # Shared fixtures:
│                                    # synthetic_skin_batch: (B=4, 3, 224, 224) images
│                                    # synthetic_signal_stream: 52-week Poisson stream
│                                    # mock_fl_clients: list of NumPyClient stubs
│
├── notebooks/                       # Research notebooks
│   ├── 01_data_exploration.ipynb    # Dataset statistics, FP distribution, class imbalance
│   ├── 02_bdn_visualization.ipynb   # H/M map decomposition, physics reconstruction
│   ├── 03_bias_analysis.ipynb       # Per-FP accuracy heatmaps, equalized odds plots
│   ├── 04_fl_convergence.ipynb      # FL training curves, privacy-accuracy tradeoff
│   ├── 05_compression_pareto.ipynb  # Accuracy vs. size/latency Pareto front
│   ├── 06_surveillance_sim.ipynb    # Outbreak simulation: ROC, detection lag CDFs
│   ├── 07_explainability.ipynb      # GradCAM + concept explanations visualization
│   ├── 08_adversarial.ipynb         # Attack success rate, certified radius plots
│   └── 09_clinical_validation.ipynb # Expert evaluation protocol + results
│
├── experiments/                     # Organized experiment results
│   ├── exp001_bdn_ablation/         # H_map vs M_map vs both vs RGB baseline
│   ├── exp002_dp_epsilon_sweep/     # ε ∈ {0.5,1,2,3,5,10,∞} vs accuracy
│   ├── exp003_compression_pareto/   # 16-point accuracy×size×latency grid
│   ├── exp004_muvp_calibration/     # MUVP FPR/TPR threshold sweep
│   ├── exp005_fairness_fl/          # Per-round EqOddsDiff over 200 rounds
│   ├── exp006_non_iid_study/        # α ∈ {0.1,0.5,1.0,∞} convergence comparison
│   ├── exp007_adversarial/          # Attack resistance evaluation grid
│   └── exp008_surveillance_eval/    # 100-scenario detection benchmark
│
└── docs/
    ├── architecture.md              # Full system design document
    ├── privacy_model.md             # Formal threat model, DP proofs, ε certificates
    ├── clinical_validation.md       # IRB protocol, expert evaluation methodology
    ├── api_reference.md             # REST API OpenAPI spec (auto-generated)
    ├── deployment_guide.md          # FL server + surveillance server deployment
    └── fairness_report_template.md  # Standard fairness audit report template
```

---

## 2. Module Deep-Dives: Core Implementations

### 2.1 `dermasafe/models/bdn.py` — BDN v2 Full Architecture

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork

class DepthwiseSepTransposeConv(nn.Module):
    """Factorized transposed convolution: depthwise + pointwise (8× fewer params)"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.dw = nn.ConvTranspose2d(in_channels, in_channels, 3, stride, 1,
                                      groups=in_channels, output_padding=stride-1)
        self.pw = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish()  # MobileNetV3 activation

class BDNDecoder(nn.Module):
    """Single-channel biological map decoder with skip connections from FPN"""
    def __init__(self, fpn_channels=256, dims=[256,128,64,32,16]):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(dims)-1):
            self.blocks.append(DepthwiseSepTransposeConv(dims[i], dims[i+1]))
        self.final = nn.Sequential(
            nn.Conv2d(dims[-1], 1, 1),
            nn.Sigmoid()  # Output in [0,1] range (biological signal normalized)
        )
        # Skip connection projections from FPN levels
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(fpn_channels, dims[i+1], 1) for i in range(3)  # 3 skip levels
        ])

class BiophysicalDecompositionNetwork(nn.Module):
    """
    BDN v2: ResNet-18 FPN encoder + dual DepthwiseSep decoder
    Physics reconstruction via learned α scalars.
    
    Input:  (B, 3, 224, 224) RGB skin image
    Output: h_map (B, 1, 224, 224), m_map (B, 1, 224, 224)
    """
    def __init__(self, encoder_name='resnet18', pretrained=True, fpn_channels=256):
        super().__init__()
        
        # Shared encoder with Feature Pyramid
        backbone = models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64 channels, stride 4
        self.layer2 = backbone.layer2  # 128 channels, stride 8
        self.layer3 = backbone.layer3  # 256 channels, stride 16
        self.layer4 = backbone.layer4  # 512 channels, stride 32
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=fpn_channels
        )
        
        # Dual decoders — independent parameters
        self.hemo_decoder = BDNDecoder(fpn_channels)
        self.melanin_decoder = BDNDecoder(fpn_channels)
        
        # Learnable physics reconstruction scalars
        self.alpha_H = nn.Parameter(torch.ones(1) * 0.5)  # Hemoglobin coefficient
        self.alpha_M = nn.Parameter(torch.ones(1) * 0.3)  # Melanin coefficient
        self.alpha_S = nn.Parameter(torch.ones(1) * 0.2)  # Scattering term
        
    def encode(self, x):
        """Extract multi-scale features via FPN"""
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        fpn_out = self.fpn({'0': f1, '1': f2, '2': f3, '3': f4})
        return fpn_out
    
    def forward(self, x):
        fpn_features = self.encode(x)
        h_map = self.hemo_decoder(fpn_features)    # (B, 1, 224, 224)
        m_map = self.melanin_decoder(fpn_features)  # (B, 1, 224, 224)
        return h_map, m_map
    
    def reconstruct_rgb(self, h_map, m_map):
        """
        Physics-based RGB reconstruction.
        I_hat = αH * broadcast(H_map) + αM * broadcast(M_map) + αS * (1 - H - M)
        Used only during training for L_phys computation.
        """
        scatter_term = 1.0 - h_map - m_map  # Scattering remainder
        scatter_term = scatter_term.clamp(0, 1)
        # Broadcast to 3-channel: each chromophore contributes to all RGB channels
        I_hat = (self.alpha_H * h_map.expand(-1, 3, -1, -1) +
                 self.alpha_M * m_map.expand(-1, 3, -1, -1) +
                 self.alpha_S * scatter_term.expand(-1, 3, -1, -1))
        return I_hat.clamp(0, 1)
    
    def quality_score(self, x):
        """
        Assess image quality for early-exit decision.
        Returns float in [0,1]: <0.5 → use fast 1-layer decoder
        """
        with torch.no_grad():
            # Laplacian variance as proxy for image sharpness
            gray = x.mean(dim=1, keepdim=True)
            laplacian = torch.nn.functional.conv2d(
                gray, 
                torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype,
                              device=x.device).view(1,1,3,3),
                padding=1
            )
            return laplacian.var(dim=[1,2,3]).mean().item()
```

### 2.2 `dermasafe/privacy/dp_sgd.py` — DP-SGD with Auto-Calibration

```python
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from dermasafe.privacy.noise_calibrator import NoiseCalibrator
from dermasafe.privacy.dp_accountant import RenyiDPAccountant

class DPTrainer:
    """
    Production-ready DP-SGD trainer with:
    - Opacus PrivacyEngine for per-sample gradient clipping + Gaussian noise
    - Auto-calibrated σ for target (ε, δ)
    - Rényi DP accountant via autodp
    - Hard abort if ε budget is exceeded
    - Privacy certificate export (JSON) for audit
    
    Example usage:
        trainer = DPTrainer(target_epsilon=3.0, target_delta=1e-5, max_grad_norm=1.0)
        model, optimizer, loader = trainer.setup(model, optimizer, loader, 
                                                  epochs=200, batch_size=32)
        # Training loop:
        for x, y in loader:
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # DP-SGD: clipping + noise inside optimizer
            trainer.log_privacy_budget()
            if trainer.should_abort():
                break
    """
    def __init__(self, target_epsilon: float = 3.0, target_delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = None
        self.accountant = RenyiDPAccountant()
        self._rounds_completed = 0
        
    def setup(self, model, optimizer, data_loader, epochs: int, batch_size: int):
        """
        Attach Opacus PrivacyEngine with auto-calibrated noise.
        Returns (dp_model, dp_optimizer, dp_loader).
        """
        n = len(data_loader.dataset)
        steps_per_epoch = n // batch_size
        total_steps = epochs * steps_per_epoch
        q = batch_size / n  # Sampling probability (Poisson subsampling)
        
        # Auto-calibrate noise multiplier σ for target (ε, δ, T, q)
        calibrator = NoiseCalibrator()
        sigma = calibrator.calibrate(
            epsilon=self.target_epsilon,
            delta=self.target_delta,
            steps=total_steps,
            sampling_probability=q
        )
        
        # Validate and fix model for Opacus compatibility
        model = ModuleValidator.fix(model)
        
        self.privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=sigma,
            max_grad_norm=self.max_grad_norm,
        )
        return dp_model, dp_optimizer, dp_loader
    
    def get_epsilon(self) -> tuple[float, float]:
        """Returns (ε, δ) consumed so far via Rényi accountant."""
        if self.privacy_engine is None:
            return (0.0, 0.0)
        eps = self.privacy_engine.get_epsilon(delta=self.target_delta)
        return (eps, self.target_delta)
    
    def should_abort(self) -> bool:
        """Returns True if ε budget is exceeded — training must stop."""
        current_eps, _ = self.get_epsilon()
        return current_eps > self.target_epsilon * 1.05  # 5% buffer
    
    def export_privacy_certificate(self, path: str):
        """Export formal privacy certificate as JSON for regulatory audit."""
        eps, delta = self.get_epsilon()
        cert = {
            "guarantee": f"({eps:.4f}, {delta})-DP",
            "interpretation": f"For any adjacent datasets, probability ratio ≤ e^{eps:.4f}",
            "mechanism": "DP-SGD with Gaussian noise + Rényi DP accounting",
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "actual_epsilon": eps,
            "max_grad_norm": self.max_grad_norm,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(cert, f, indent=2)
```

### 2.3 `dermasafe/surveillance/alert_engine.py` — Five-Stage Alert Pipeline

```python
from dataclasses import dataclass
from typing import Optional
from dermasafe.surveillance.cusum_detector import CUSUMDetector
from dermasafe.surveillance.spatial_scan import KulldorffScanner, GNNScanner
from dermasafe.surveillance.muvp import MUVPChecker
from dermasafe.surveillance.anomaly_scorer import AnomalyScorer
from dermasafe.data.signal_packet import SignalPacket

@dataclass
class Alert:
    alert_id: str
    condition_group: str        # ICD chapter, NOT specific code (privacy)
    geographic_zone: str        # Zone label, NOT GPS coordinates
    estimated_count_range: str  # e.g. "10-20 cases" (range, NOT exact)
    confidence_score: float
    recommended_action: str
    data_window_weeks: str
    cluster_radius_estimate_km: float
    
    # Zero individual identifiers — all fields are aggregate statistics only

class AlertEngine:
    """
    Five-stage AND-logic alert pipeline.
    Input: anonymized signal packets
    Output: verified community outbreak alerts (no individual identifiers)
    
    Stage 1: CUSUM temporal detection (per condition-geohash time series)
    Stage 2: Kulldorff spatial scan + GNN risk overlay
    Stage 3: MUVP v2 anti-gaming verification (5 checks)
    Stage 4: LSTM-AE anomaly scoring (pattern recognition)
    Stage 5: AND-logic gate with rate limiting
    """
    def __init__(self, cusum: CUSUMDetector, kulldorff: KulldorffScanner,
                 gnn_scanner: GNNScanner, muvp: MUVPChecker,
                 anomaly_scorer: AnomalyScorer, config):
        self.cusum = cusum
        self.kulldorff = kulldorff
        self.gnn_scanner = gnn_scanner
        self.muvp = muvp
        self.anomaly_scorer = anomaly_scorer
        self.config = config
        self._alert_cooldown = {}   # {(condition, zone): last_alert_timestamp}
        
    def process_packet(self, packet: SignalPacket) -> Optional[Alert]:
        """
        Process one anonymized signal packet through 5-stage pipeline.
        Returns Alert if all stages trigger, else None.
        All processing is on aggregate statistics — no individual data stored.
        """
        # STAGE 1: Temporal CUSUM
        cusum_alert = self.cusum.update(
            condition=packet.condition_icd10_chapter,  # ICD chapter, not specific code
            geohash=packet.geohash6,
            week=packet.week_iso,
        )
        if not cusum_alert:
            return None  # Normal seasonal variation — no action
        
        # STAGE 2: Kulldorff spatial scan + GNN
        kulldorff_result = self.kulldorff.scan(packet.condition_icd10_chapter,
                                                 packet.geohash6)
        gnn_score = self.gnn_scanner.risk_score(packet.geohash6)
        
        spatial_triggered = (kulldorff_result.p_value < self.config.kulldorff.p_threshold
                              or gnn_score > self.config.gnn.score_threshold)
        if not spatial_triggered:
            return None  # CUSUM triggered but no spatial cluster — isolated case
        
        # STAGE 3: MUVP anti-gaming verification
        muvp_result = self.muvp.verify(kulldorff_result.cluster_device_hashes)
        if not muvp_result.passed:
            return None  # Failed anti-gaming check — likely reporting artifact
        
        # STAGE 4: LSTM-AE anomaly scoring
        anomaly_score = self.anomaly_scorer.score(
            packet.condition_icd10_chapter, packet.geohash6
        )
        if anomaly_score <= self.anomaly_scorer.threshold:
            return None  # Pattern doesn't match known outbreak signatures
        
        # STAGE 5: Rate limiting + alert generation
        cooldown_key = (packet.condition_icd10_chapter, 
                        self.geohash_to_zone(packet.geohash6))
        if self._is_in_cooldown(cooldown_key):
            return None  # Alert already sent for this area recently
        
        # All 5 gates passed — generate alert with NO individual identifiers
        confidence = self._compute_confidence(kulldorff_result, muvp_result.score,
                                               anomaly_score, gnn_score)
        if confidence < self.config.alert.min_confidence:
            return None
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            condition_group=self._icd_chapter_to_group(packet.condition_icd10_chapter),
            geographic_zone=self.geohash_to_zone(packet.geohash6),
            estimated_count_range=self._count_range(kulldorff_result.expected_cases),
            confidence_score=confidence,
            recommended_action=self._recommend_action(packet.condition_icd10_chapter),
            data_window_weeks=f"past {self.config.cusum.window_weeks} weeks",
            cluster_radius_estimate_km=kulldorff_result.radius_km,
        )
        self._update_cooldown(cooldown_key)
        return alert
```

### 2.4 `fl_server/strategy.py` — Advanced FL Strategy

```python
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from dermasafe.fairness.monitor import FairnessMonitor
from dermasafe.privacy.dp_accountant import RenyiDPAccountant

class DermaSafeFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Custom FL aggregation strategy with:
    - SCAFFOLD variance reduction (control variates)
    - FLAME Byzantine-robust filtering
    - Per-round fairness monitoring with automatic rollback
    - Rényi DP budget tracking across rounds
    - Gradient compression support (top-k sparse)
    """
    def __init__(self, fairness_monitor: FairnessMonitor,
                 dp_accountant: RenyiDPAccountant,
                 byzantine_fraction: float = 0.0,
                 fairness_rollback_threshold: float = 0.03,
                 **kwargs):
        super().__init__(**kwargs)
        self.fairness_monitor = fairness_monitor
        self.dp_accountant = dp_accountant
        self.byzantine_fraction = byzantine_fraction
        self.fairness_rollback_threshold = fairness_rollback_threshold
        self.previous_model_params = None
        self.control_variates = {}  # SCAFFOLD: {client_id: c_k}
        self.server_control = None  # SCAFFOLD: global control variate c
        self.round_history = []
        
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Aggregation with FLAME Byzantine filter + SCAFFOLD + fairness check.
        """
        if not results:
            return None, {}
        
        # FLAME: filter Byzantine clients via cosine similarity clustering
        filtered_results = self._flame_filter(results)
        
        # SCAFFOLD-corrected aggregation
        aggregated_params = self._scaffold_aggregate(server_round, filtered_results)
        
        # Update DP budget accounting
        sigma = self._get_round_sigma(server_round)
        q = len(results) / self.num_total_clients
        self.dp_accountant.track_round(sigma=sigma, q=q, steps=self.local_steps)
        current_eps, _ = self.dp_accountant.get_epsilon(delta=self.target_delta)
        
        # Fairness check on aggregated model
        is_regression, reason = self.fairness_monitor.check_regression(
            aggregated_params, threshold=self.fairness_rollback_threshold
        )
        if is_regression:
            print(f"[Round {server_round}] FAIRNESS REGRESSION: {reason}. Rolling back.")
            aggregated_params = self.previous_model_params  # Rollback
        else:
            self.previous_model_params = aggregated_params
        
        self.round_history.append({
            'round': server_round,
            'n_clients': len(filtered_results),
            'n_rejected': len(results) - len(filtered_results),
            'epsilon_consumed': current_eps,
            'fairness_regression': is_regression,
        })
        
        metrics = {
            'epsilon_consumed': current_eps,
            'n_byzantine_filtered': len(results) - len(filtered_results),
            'fairness_regression': int(is_regression),
        }
        return aggregated_params, metrics
    
    def _flame_filter(self, results):
        """
        FLAME Byzantine-robust filtering.
        Cluster client updates by cosine similarity.
        Reject updates in small clusters (potential coordinated attack).
        Then smooth by L2-norm clipping.
        """
        from sklearn.cluster import HDBSCAN
        import numpy as np
        
        update_vectors = [np.concatenate([p.flatten() for p in r.parameters.tensors])
                          for _, r in results]
        update_matrix = np.stack(update_vectors)
        
        # Normalize for cosine clustering
        norms = np.linalg.norm(update_matrix, axis=1, keepdims=True)
        normalized = update_matrix / (norms + 1e-8)
        
        # HDBSCAN clustering on cosine space
        clusterer = HDBSCAN(min_cluster_size=max(2, len(results) // 5),
                             metric='euclidean')  # on normalized = cosine distance
        labels = clusterer.fit_predict(normalized)
        
        # Keep only majority cluster
        majority_label = np.bincount(labels[labels >= 0]).argmax()
        filtered = [r for (_, r), l in zip(results, labels) if l == majority_label]
        
        # Smooth: clip each update L2-norm to median
        median_norm = np.median([np.linalg.norm(v) for v in update_vectors])
        filtered_smoothed = []
        for client_id, fit_res in filtered:
            params = np.concatenate([p.flatten() for p in fit_res.parameters.tensors])
            clipped = params * min(1.0, median_norm / (np.linalg.norm(params) + 1e-8))
            # Reconstruct FitRes with clipped params
            filtered_smoothed.append((client_id, self._reconstruct_fitres(fit_res, clipped)))
        
        return filtered_smoothed
```

### 2.5 `dermasafe/fairness/monitor.py` — FL Fairness Monitor

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

@dataclass
class GroupFairnessMetrics:
    balanced_accuracy: float
    tpr: float            # True Positive Rate (Sensitivity)
    fpr: float            # False Positive Rate
    auroc: float
    ece: float            # Expected Calibration Error
    group: str
    round_num: int

@dataclass  
class FairnessReport:
    round_num: int
    metrics_per_group: Dict[str, GroupFairnessMetrics]
    equalized_odds_diff: float
    demographic_parity_diff: float
    disparate_impact: float
    worst_group: str
    worst_group_accuracy: float
    passed: bool          # True if all metrics within thresholds

class FairnessMonitor:
    """
    Continuous fairness monitor for federated learning.
    Evaluates after EVERY FL aggregation round.
    Detects regression → triggers automatic rollback.
    
    Groups monitored: Fitzpatrick Type {I-II, III-IV, V-VI}
    Metrics: balanced accuracy, equalized odds, demographic parity, ECE
    """
    def __init__(self, config, alert_threshold: float = 0.03,
                 eq_odds_threshold: float = 0.05):
        self.config = config
        self.alert_threshold = alert_threshold  # Max per-group accuracy drop
        self.eq_odds_threshold = eq_odds_threshold
        self.history: list[FairnessReport] = []
        self.rollback_count = 0
        
    def evaluate_round(self, model: torch.nn.Module,
                        val_loaders: Dict[str, torch.utils.data.DataLoader],
                        round_num: int) -> FairnessReport:
        """
        Evaluate model on per-FP-group validation sets.
        val_loaders: {"FP_I-II": loader1, "FP_III-IV": loader2, "FP_V-VI": loader3}
        """
        model.eval()
        group_metrics = {}
        all_preds, all_labels, all_groups = [], [], []
        
        for group_name, loader in val_loaders.items():
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for x, y in loader:
                    logits, _, _ = model(x)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    y_true.extend(y.tolist())
                    y_pred.extend(preds.tolist())
                    y_prob.extend(probs.tolist())
            
            group_metrics[group_name] = GroupFairnessMetrics(
                balanced_accuracy=self._balanced_accuracy(y_true, y_pred),
                tpr=self._tpr(y_true, y_pred),
                fpr=self._fpr(y_true, y_pred),
                auroc=self._auroc(y_true, y_prob),
                ece=self._ece(y_true, y_prob),
                group=group_name,
                round_num=round_num
            )
        
        eq_odds_diff = self._equalized_odds_diff(group_metrics)
        dp_diff = self._demographic_parity_diff(group_metrics)
        disparate_impact = self._disparate_impact(group_metrics)
        worst = min(group_metrics, key=lambda g: group_metrics[g].balanced_accuracy)
        
        report = FairnessReport(
            round_num=round_num,
            metrics_per_group=group_metrics,
            equalized_odds_diff=eq_odds_diff,
            demographic_parity_diff=dp_diff,
            disparate_impact=disparate_impact,
            worst_group=worst,
            worst_group_accuracy=group_metrics[worst].balanced_accuracy,
            passed=(eq_odds_diff <= self.eq_odds_threshold and
                    all(m.balanced_accuracy >= self.config.min_accuracy
                        for m in group_metrics.values()))
        )
        self.history.append(report)
        return report
    
    def check_regression(self, current_params, threshold: float = None) -> tuple[bool, str]:
        """
        Returns (is_regression, reason_string).
        Compare current report vs. previous; trigger rollback if any group drops.
        """
        threshold = threshold or self.alert_threshold
        if len(self.history) < 2:
            return False, "Insufficient history"
        
        curr = self.history[-1]
        prev = self.history[-2]
        
        for group in curr.metrics_per_group:
            delta = (prev.metrics_per_group[group].balanced_accuracy -
                     curr.metrics_per_group[group].balanced_accuracy)
            if delta > threshold:
                self.rollback_count += 1
                reason = (f"{group} accuracy dropped {delta:.3f} "
                           f"({prev.metrics_per_group[group].balanced_accuracy:.3f} "
                           f"→ {curr.metrics_per_group[group].balanced_accuracy:.3f})")
                return True, reason
        
        if curr.equalized_odds_diff > self.eq_odds_threshold:
            reason = f"EqOddsDiff {curr.equalized_odds_diff:.3f} exceeds {self.eq_odds_threshold}"
            return True, reason
        
        return False, "All fairness checks passed"
```

---

## 3. Training Workflow (Complete, Phase-by-Phase)

### Phase 1: Cloud Pre-Training
```bash
# Step 1: Data Pipeline
python scripts/download_datasets.py --datasets all --output data/raw/
python scripts/prepare_data.py \
  --datasets isic2020,fitzpatrick17k,ddi,pad_ufes,skinl2 \
  --output data/processed/ \
  --val_frac 0.15 --test_frac 0.15 \
  --stratify fitzpatrick_type,condition

# Step 2: BDN v2 Training
python scripts/train_cloud.py \
  --config configs/model/bdn_v2.yaml \
  --stage bdn \
  --epochs 80 \
  --loss_weights "lambda1=1.0,lambda2=0.5,lambda3=0.3,lambda4=0.1,lambda5=0.05"
# → Checkpoint: checkpoints/bdn_v2_best.pt (eval on reconstruction PSNR + classification)

# Step 3: Classifier Training (BDN frozen)
python scripts/train_cloud.py \
  --config configs/model/classifier.yaml \
  --stage classifier \
  --bdn_ckpt checkpoints/bdn_v2_best.pt \
  --freeze_bdn True \
  --epochs 100
# → Checkpoint: checkpoints/classifier_best.pt

# Step 4: Joint Fine-tuning (BDN + Classifier unfrozen)
python scripts/train_cloud.py \
  --config configs/training/cloud.yaml \
  --stage joint \
  --bdn_ckpt checkpoints/bdn_v2_best.pt \
  --classifier_ckpt checkpoints/classifier_best.pt \
  --lr 1e-4 --epochs 30  # Low LR for fine-tuning
# → Teacher model: checkpoints/full_teacher.pt

# Step 5: Adversarial Training
python scripts/train_cloud.py \
  --config configs/training/adversarial.yaml \
  --stage adversarial \
  --base_ckpt checkpoints/full_teacher.pt \
  --pgd_eps 8 --pgd_steps 10 --clean_fraction 0.5
# → Robust teacher: checkpoints/full_teacher_robust.pt

# Step 6: Baseline Fairness Audit
python scripts/generate_fairness_report.py \
  --model checkpoints/full_teacher_robust.pt \
  --test_data data/processed/test/ \
  --output_dir experiments/baseline_audit/
# → Report: experiments/baseline_audit/fairness_report.pdf + .json
```

### Phase 2: Edge Compression
```bash
# Step 7: Multi-Teacher Knowledge Distillation (MEAL V2)
python scripts/compress_model.py \
  --stage distillation \
  --teachers "checkpoints/full_teacher_robust.pt:0.5,\
              checkpoints/classifier_best.pt:0.3,\
              checkpoints/bdn_v2_best.pt:0.2" \
  --student configs/model/student.yaml \
  --config configs/training/distillation.yaml \
  --temperature 4 --alpha 0.3 --beta 0.7 \
  --intermediate_matching True
# → Student: checkpoints/student_kd.pt (~23MB)

# Step 8: Sensitivity-Aware Structured Pruning
python scripts/compress_model.py \
  --stage pruning \
  --model checkpoints/student_kd.pt \
  --prune_ratio 0.45 \
  --sensitivity_method hessian_trace \
  --finetune_epochs 10 \
  --finetune_loss ewc  # Elastic Weight Consolidation
# → Pruned: checkpoints/student_pruned.pt (~13MB)

# Step 9: AdaRound INT8 Quantization
python scripts/compress_model.py \
  --stage quantization \
  --model checkpoints/student_pruned.pt \
  --method adround \
  --calibration_samples 512 \
  --accuracy_drop_threshold 0.01  # If >1% drop: auto-switch to QAT
# → Quantized: checkpoints/student_int8.pt (~7MB)

# Step 10: Multi-Platform Export
python scripts/compress_model.py \
  --stage export \
  --model checkpoints/student_int8.pt \
  --platforms tflite,coreml,onnx
# → models/dermasafe_v2.tflite (~8MB)
# → models/dermasafe_v2.mlmodel (~9MB)
# → models/dermasafe_v2.onnx (~7MB)

# Step 11: Edge Benchmarking
python scripts/benchmark_edge.py \
  --model models/dermasafe_v2.tflite \
  --platform android \
  --n_runs 200 \
  --warmup 20
# → Report: experiments/edge_benchmarks/latency_p50_p95.json
```

### Phase 3: Federated Learning
```bash
# Step 12: Non-IID Data Partitioning
python -m dermasafe.data.federated_split \
  --data data/processed/train/ \
  --strategy non_iid_dirichlet \
  --alpha 0.5 \
  --num_clients 500 \
  --output data/federated_splits/non_iid_alpha05/

# Step 13: Launch FL Server
python fl_server/server.py \
  --config configs/federated/server.yaml \
  --rounds 200 \
  --clients_per_round 100 \
  --aggregation scaffold_flame \
  --fairness_monitor True \
  --dp_epsilon 3.0

# Step 14: Launch FL Clients (simulation)
python fl_server/server.py \
  --simulate \
  --num_clients 500 \
  --split_dir data/federated_splits/non_iid_alpha05/ \
  --config configs/federated/client.yaml \
  --dp_epsilon 3.0 \
  --dp_delta 1e-5 \
  --noise_multiplier auto

# Step 15: Privacy Audit
python scripts/run_privacy_audit.py \
  --model checkpoints/fl_global_round200.pt \
  --train_data data/processed/train/ \
  --test_data data/processed/test/ \
  --audit mia,gradient_inversion
# → Report: experiments/privacy_audit/mia_auc.json + gradient_inversion_psnr.json
```

### Phase 4: Surveillance Engine
```bash
# Step 16: Generate Synthetic Signal Stream
python -m dermasafe.data.outbreak_generator \
  --n_cells 500 \
  --years 3 \
  --outbreaks 100 \
  --conditions fungal,scabies,impetigo,molluscum \
  --output data/surveillance_stream/

# Step 17: Train LSTM-AE + Fit Threshold
python -m dermasafe.surveillance.anomaly_scorer \
  --train \
  --data data/surveillance_stream/synthetic_normal_3yr.csv \
  --epochs 100 \
  --percentile 95
# → Model: models/lstm_ae_v2.pt

# Step 18: Train GraphSAGE Spatial Classifier
python -m dermasafe.models.gnn_surveillance \
  --train \
  --outbreak_data data/surveillance_stream/outbreak_scenarios/ \
  --epochs 50
# → Model: models/gnn_surveillance_v2.pt

# Step 19: End-to-End Surveillance Simulation
python scripts/run_surveillance_sim.py \
  --outbreaks 100 \
  --stream data/surveillance_stream/ \
  --cusum_config configs/surveillance/cusum.yaml \
  --kulldorff_config configs/surveillance/kulldorff.yaml \
  --muvp_config configs/surveillance/muvp.yaml
# → Report: experiments/surveillance_eval/detection_report.json
```

---

## 4. YAML Configuration Reference

### `configs/federated/server.yaml`
```yaml
federated:
  rounds: 200
  clients_per_round: 100
  total_clients: 500
  fraction_fit: 0.2         # 100/500 = 20% participation
  fraction_evaluate: 0.1
  min_fit_clients: 10
  min_evaluate_clients: 5
  
  aggregation:
    method: scaffold_flame   # scaffold_flame | fedavg | fedprox
    scaffold:
      correction_factor: 1.0
    flame:
      min_cluster_fraction: 0.5
      norm_clip_percentile: 50.0
    
  differential_privacy:
    enabled: true
    target_epsilon: 3.0
    target_delta: 1.0e-5
    max_grad_norm: 1.0
    noise_multiplier: auto
    accountant: renyi         # renyi | prv | gdp
    
  secure_aggregation:
    enabled: true
    threshold: 10             # Min clients before aggregation
    dropout_rate: 0.1         # Expected client dropout
    
  fairness:
    monitor_enabled: true
    rollback_threshold: 0.03  # Max per-group accuracy drop before rollback
    eq_odds_threshold: 0.05   # Max equalized odds difference
    
  personalization:
    enabled: true
    method: ditto             # ditto | pFedMe
    ditto_lambda: 0.1         # Trade-off: personalization vs. generalization
    local_epochs_personal: 3
```

### `configs/surveillance/muvp.yaml`
```yaml
muvp:
  check_unique_devices:
    min_count: 5
    
  check_diversity_score:
    min_score: 0.80           # unique_devices / total_reports
    
  check_single_dominance:
    max_fraction: 0.30        # Single device max contribution
    
  check_temporal_spread:
    min_hours: 48             # Min time between first and last report
    burst_test:
      method: ks_test         # Kolmogorov-Smirnov vs Poisson null
      p_value_threshold: 0.05 # Reject burst hypothesis if p < 0.05
      
  check_geographic_entropy:
    min_entropy: 1.099        # log(3) — reports from ≥3 distinct sub-cells
    entropy_base: e
```

### `configs/model/bdn_v2.yaml`
```yaml
bdn:
  encoder:
    backbone: resnet18
    pretrained: true
    fpn_channels: 256
    
  decoder:
    type: depthwise_sep_transpose_conv
    dims: [256, 128, 64, 32, 16]
    skip_connections: [2, 3, 4]   # FPN levels to use for skip
    activation: hardswish
    
  physics:
    init_alpha_H: 0.5
    init_alpha_M: 0.3
    init_alpha_S: 0.2
    learnable_alphas: true
    
  loss_weights:
    lambda_ce: 1.0
    lambda_phys: 0.5
    lambda_fair: 0.3
    lambda_smooth: 0.1
    lambda_sparsity: 0.05
    lambda_consistency: 0.05
    
  quality_gate:
    enabled: true             # Early-exit for low-quality images
    threshold: 0.15           # Laplacian variance threshold
    fast_decoder_layers: 1    # Layers for fast decoder path
```

---

## 5. CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: DermaSafe CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt -r requirements-dev.txt
    
    - name: Lint (black + flake8 + mypy)
      run: |
        black --check dermasafe/ tests/
        flake8 dermasafe/ tests/ --max-line-length 100
        mypy dermasafe/ --ignore-missing-imports
    
    - name: Unit tests with coverage
      run: |
        pytest tests/unit/ -v --cov=dermasafe --cov-report=xml \
               --cov-fail-under=85
    
    - name: Integration tests
      run: |
        pytest tests/integration/ -v -x  # Fail fast
    
    - name: Privacy budget check
      run: |
        python -c "
        from dermasafe.privacy.noise_calibrator import NoiseCalibrator
        cal = NoiseCalibrator()
        sigma = cal.calibrate(epsilon=3.0, delta=1e-5, steps=50000, sampling_probability=0.2)
        assert sigma > 0, 'Calibration failed'
        print(f'Calibrated sigma={sigma:.4f} for ε=3.0')
        "
    
    - name: Fairness threshold check
      run: |
        pytest tests/unit/test_fairness_metrics.py -v --strict-markers
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  fairness_regression:
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.message, '[model-update]')
    steps:
    - name: Download model artifact
      run: echo "Download new model checkpoint from artifact store"
    - name: Run fairness evaluation
      run: |
        python scripts/generate_fairness_report.py \
          --model artifacts/new_model.pt \
          --test_data data/processed/test/ \
          --fail_on_regression True \
          --eq_odds_threshold 0.05
```

---

## 6. Key Design Decisions & Rationale

| Design Decision | Choice | Rationale | Alternative & Why Rejected |
|---|---|---|---|
| Skin feature extraction | BDN v2 (FPN + DWSep decoders) | Physics-grounded, skin-tone invariant, FPN multi-scale | RGB CNN → spurious skin-tone correlations |
| Edge backbone | MobileNetV3 + MEAL V2 KD | Best accuracy/latency/size Pareto | SqueezeNet → too low accuracy; EfficientNet-Lite → too large |
| FL algorithm | SCAFFOLD + FLAME | Corrects non-IID drift + Byzantine robustness | Pure FedAvg → diverges on non-IID; FedProx → no Byzantine defense |
| Privacy mechanism | DP-SGD (Opacus) + RDP + PRV | Formal guarantees, PyTorch-native, tighter accounting | PATE → requires public data; local DP → lower utility |
| Quantization method | AdaRound (BCD optimization) | Better accuracy than round-to-nearest PTQ | Standard PTQ → 3-5% drop; Full QAT → too expensive |
| Outbreak detection | 5-stage: CUSUM+Kulldorff+MUVP+LSTM-AE+GNN | Multi-layer verification: low FPR + high sensitivity | Threshold-only → FPR 20%+; single-model → brittle |
| Location privacy | Geohash precision 6 + Planar Laplace noise | K-anonymity at block level + local DP | Exact GPS → unacceptable privacy risk |
| Fairness mechanism | 4-layer: BDN + adversarial + reweighing + FL monitor | Defense-in-depth; no single point of fairness failure | Post-processing calibration only → doesn't fix representation bias |
| Explainability | GradCAM on H/M maps + SkinCon concepts + MC uncertainty | Clinician-actionable at multiple abstraction levels | SHAP only → expensive per inference on mobile |
| FL personalization | Ditto (Moreau envelope) | Individual model + global model maintained; fairness-aware | pFedMe → more complex; FedAvg only → loses personalization |

---

## 7. Performance Targets Summary

```
╔══════════════════════════════════════════════════════════════════════╗
║                  DERMASAFE v2.0 TARGET SPECIFICATIONS               ║
╠══════════════════════════════════════════════════════════════════════╣
║  CLINICAL ACCURACY                                                   ║
║  ├── Balanced Accuracy: ≥85% across ALL Fitzpatrick types            ║
║  ├── AUROC: ≥90% per condition class                                 ║
║  ├── Sensitivity: ≥80% per condition (clinical minimum)              ║
║  └── ECE (Calibration): ≤0.05 (calibrated uncertainty)              ║
╠══════════════════════════════════════════════════════════════════════╣
║  FAIRNESS                                                            ║
║  ├── Equalized Odds Difference: ≤0.05                                ║
║  ├── Demographic Parity Difference: ≤0.05                            ║
║  ├── Disparate Impact: ≥0.80                                         ║
║  └── Per-FP Balanced Accuracy Gap: ≤5% between best and worst group  ║
╠══════════════════════════════════════════════════════════════════════╣
║  EDGE PERFORMANCE                                                    ║
║  ├── Inference Latency P50: ≤500ms, P95: ≤800ms                     ║
║  ├── Model Size: ≤15MB (.tflite compressed)                          ║
║  ├── MACs: ≤300M                                                     ║
║  ├── Peak RAM: ≤256MB                                                ║
║  └── Battery Per Session: ≤5%                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  PRIVACY                                                             ║
║  ├── DP Budget: (ε=3.0, δ=1e-5) — formal Rényi DP guarantee         ║
║  ├── MIA AUC: ≤0.55 (near-random = strong privacy)                  ║
║  └── Gradient Inversion PSNR: ≤10 dB (unintelligible reconstruction) ║
╠══════════════════════════════════════════════════════════════════════╣
║  SURVEILLANCE                                                        ║
║  ├── Detection Sensitivity: ≥85% (at FPR=0.05)                      ║
║  ├── False Positive Rate: <5% per 1,000 person-weeks                 ║
║  ├── Detection Lag: <72 hours from outbreak start                    ║
║  ├── Geographic Precision: <2km centroid error                       ║
║  └── Re-identification Rate: <0.1% (adversarial attack)              ║
╠══════════════════════════════════════════════════════════════════════╣
║  ADVERSARIAL ROBUSTNESS                                              ║
║  ├── PGD-20 (ε=8/255) Accuracy: ≥70%                                ║
║  ├── Certified L2 Radius (σ=0.25): ≥0.25 for 80% of test samples    ║
║  └── Natural Corruption Accuracy: ≥80% (ImageNet-C style)           ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 8. Deliverables Checklist

### Research Deliverables
- [ ] BDN v2 paper: biophysical decomposition for equitable dermatology AI
- [ ] Fitzpatrick-stratified accuracy tables with confidence intervals
- [ ] Privacy-accuracy Pareto curves (ε vs. balanced accuracy per FP group)
- [ ] FL convergence analysis: IID vs. non-IID (4 Dirichlet α values)
- [ ] Surveillance simulation: ROC, detection lag CDF, geographic precision
- [ ] Adversarial robustness report: clean/PGD/certified accuracy
- [ ] Fairness regression study: per-round EqOddsDiff over 200 FL rounds
- [ ] Clinical expert evaluation (10+ dermatologist ratings)

### Engineering Deliverables
- [ ] `pip install dermasafe` Python package (PyPI)
- [ ] Model zoo: `dermasafe_v2.tflite`, `.mlmodel`, `.onnx` (HuggingFace Hub)
- [ ] FL server Docker image (`ghcr.io/dermasafe/fl-server:v2`)
- [ ] Surveillance server Docker image (`ghcr.io/dermasafe/surveillance:v2`)
- [ ] OpenAPI spec for surveillance dashboard REST API
- [ ] Test suite: >90% coverage on `models/`, `privacy/`, `surveillance/`
- [ ] Privacy certificate generator (per-deployment JSON audit trail)
- [ ] Fairness audit report generator (automated PDF + JSON)

### Documentation Deliverables
- [ ] Privacy threat model + formal DP proof (docs/privacy_model.md)
- [ ] Clinical validation protocol + IRB-approved consent form
- [ ] Deployment guide: FL server + surveillance server + edge SDK
- [ ] Bias audit methodology: per-FP evaluation protocol
- [ ] Regulatory compliance notes: EU AI Act, HIPAA, GDPR alignment

---

*DermaSafe Project Blueprint v1.0 | Python/NN&DL Only | March 2026*
