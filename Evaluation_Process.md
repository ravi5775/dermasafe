# DermaSafe Evaluation Process

## Overview
Evaluation in DermaSafe is done in **two different ways**:

1. **Single Image Prediction Evaluation** → for uploaded user images  
2. **Dataset-Level Model Evaluation** → for validation/testing of the trained model  

These two are different and should not be confused.

---

# 1) Single Image Prediction Evaluation

This is used when a user uploads **one skin image** through the dashboard/API.

## Process
1. User uploads an image
2. Backend loads the trained model checkpoint (`best.pt`)
3. Image is preprocessed:
   - resize
   - normalize
   - convert to tensor
4. Batch dimension is added using `unsqueeze(0)`
5. Image is passed into the model
6. Model outputs **logits**
7. Softmax converts logits into **probabilities**
8. Highest probability class is selected as prediction
9. Risk and recommendation are generated

## Output Returned
For one uploaded image, the system returns:

- `prediction` → predicted class index
- `class_name` → disease label name
- `confidence` → top softmax probability
- `uncertainty` → entropy-based uncertainty
- `likely_affected` → true/false
- `risk_level` → low / medium / high
- `recommendation` → advice text
- `device` → cpu/gpu info
- `checkpoint` → model file used

## Important Note
For a **single uploaded image**, the system **does not calculate**:

- Accuracy
- Precision
- Recall
- F1-score
- Fairness metrics
- Robustness metrics

Because those require **many labeled samples**, not one image.

---

# 2) Dataset-Level Model Evaluation

This is used after training to measure how well the model performs on a **validation/test dataset**.

## Purpose
Dataset-level evaluation helps check:

- Overall model performance
- Class-wise performance
- Bias/fairness across groups
- Robustness against attacks/noise

## Evaluation Process
1. Load trained model checkpoint
2. Load validation/test dataset
3. Run predictions on all samples
4. Compare predictions with true labels
5. Compute performance metrics

## Metrics Calculated
Typical dataset-level evaluation includes:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Macro / Weighted averages**
- **Confusion Matrix**

---

# 3) Fairness Evaluation

Fairness evaluation is done only on a **labeled dataset**, not on uploaded single images.

## Why Fairness Matters
Skin disease datasets may contain different skin tones and groups.  
The model should perform fairly across all groups.

## Fairness Checks
The system may evaluate:

- Performance by **Fitzpatrick skin type**
- Macro recall by subgroup
- Performance gaps between groups

## Result
This helps identify whether the model is biased toward certain skin categories.

---

# 4) Robustness Evaluation

Robustness evaluation checks whether the model remains reliable when the image is slightly disturbed.

## Example
The model is tested on:

- Clean images
- Slightly modified/adversarial images

## Metrics
- `clean_accuracy`
- `fgsm_accuracy`

## Purpose
This checks whether the model is stable and secure in real-world usage.

---

# 5) Risk Evaluation Logic

After prediction, the system maps the predicted class into a risk category.

## Current Rule
- **High Risk Classes**: `0, 1, 4`
- **Medium Risk Class**: `2`
- **Low Risk**: all others

## Why This Is Useful
This helps convert raw model output into a more practical result for users and doctors.

---

# Final Summary

## Single Image Upload
Used for **real-time prediction**:
- disease prediction
- confidence
- uncertainty
- risk level
- recommendation

## Dataset Evaluation
Used for **model validation**:
- accuracy
- recall
- F1-score
- fairness
- robustness

---

