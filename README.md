# SynForceNet: Force-Inspired Anomaly Detection for EV Battery Fault Diagnosis

SynForceNet is a deep anomaly detection framework for electric vehicle (EV) battery fault diagnosis. This repository contains the main implementation of **SynForceNet** together with several representative baseline and ablation methods, including **DVAA-SVDD**, **OCSVM**, and **VAE**-based anomaly detection.

The project is designed for comparative studies on battery anomaly detection under practical data settings, with a particular focus on latent-space modeling, anomaly boundary characterization, and balanced evaluation protocols.

---

## Highlights

- Force-inspired latent representation learning for battery anomaly detection
- A unified repository including the proposed model and multiple baselines
- Easy-to-run single-file experimental scripts
- Suitable for comparative studies on EV battery fault diagnosis
- Designed for extension to physics-informed or temporal anomaly detection settings

---

## Repository Structure

```text
.
├── SynForceNet.py
├── Ablation-(DVAA-SVDD).py
├── Ablation-OCSVM.py
├── Ablation-VAE.py
└── README.md
```

### File Description

- **`SynForceNet.py`**  
  Main implementation of the proposed SynForceNet framework.

- **`Ablation-(DVAA-SVDD).py`**  
  Ablation or baseline implementation based on DVAA-SVDD.

- **`Ablation-OCSVM.py`**  
  One-Class SVM baseline for anomaly detection.

- **`Ablation-VAE.py`**  
  Variational Autoencoder baseline for anomaly detection.

- **`README.md`**  
  Project documentation.

---

## Background

Battery safety is one of the central challenges in electric vehicles. In real-world operating conditions, fault patterns are often subtle, sparse, and highly variable, making anomaly detection difficult. Traditional methods frequently rely on distance-based criteria alone, which may be insufficient for capturing complex interactions and latent structural variations in battery data.

To address this issue, **SynForceNet** explores a force-inspired perspective for representation learning in anomaly detection. Instead of relying only on conventional distance-based compactness, the framework is intended to improve latent-space organization and fault separability, thereby enhancing the detection of abnormal battery behavior.

---

## Method Overview

The repository includes the following categories of methods:

### 1. Proposed Method
- **SynForceNet**: the main model proposed in this project for EV battery anomaly detection.

### 2. Baseline / Ablation Methods
- **DVAA-SVDD**: a deep SVDD-style baseline or ablation reference.
- **OCSVM**: a classical shallow one-class anomaly detection baseline.
- **VAE**: a reconstruction-based deep anomaly detection baseline.

These implementations are intended to support controlled experiments and fair comparisons under the same data setting.

---

## Requirements

The code is written in **Python** and mainly depends on the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `matplotlib`
- `tqdm`

Install the common dependencies with:

```bash
pip install numpy pandas scikit-learn torch matplotlib tqdm
```

If you use **conda**, you may create a separate environment first:

```bash
conda create -n synforcenet python=3.9
conda activate synforcenet
pip install numpy pandas scikit-learn torch matplotlib tqdm
```

---

## Quick Start

### 1. Prepare the dataset

Prepare your EV battery dataset in tabular format (e.g., CSV files). Depending on the script, you may need to provide:

- raw operational data,
- preprocessed feature files,
- train / validation / test splits,
- anomaly labels for evaluation.

Please modify the file paths inside each Python script so that they match your local environment.

### 2. Run the proposed method

```bash
python SynForceNet.py
```

### 3. Run ablation / baseline methods

```bash
python "Ablation-(DVAA-SVDD).py"
python Ablation-OCSVM.py
python Ablation-VAE.py
```

---

## Data Format

Although the exact input format can be adjusted depending on your local preprocessing pipeline, a typical dataset may include:

- voltage-related measurements,
- current-related measurements,
- temperature-related measurements,
- engineered statistical features,
- sequence-derived features,
- labels indicating normal or abnormal states for evaluation only.

For one-class or deep SVDD-style training, only normal samples are typically used during model training, while both normal and abnormal samples are used during evaluation.

---

## Evaluation

Typical evaluation metrics include:

- **AUC**
- **Precision**
- **Recall / TPR**
- **F1-score**
- **Threshold-based detection performance**

This repository is especially suitable for experiments under a **balanced evaluation strategy**, where the evaluation set is constructed with matched or controlled proportions of normal and abnormal samples for fair comparison across methods.

---

## Suggested Workflow

A common experimental workflow is:

1. preprocess raw battery data,
2. construct training and evaluation splits,
3. train the proposed method and baseline models,
4. compute anomaly scores,
5. determine thresholds on the validation or evaluation set,
6. compare metrics across different methods.

To ensure fairness, keep the following consistent across all methods:

- input features,
- preprocessing procedure,
- train / validation / test split,
- evaluation protocol,
- metric definitions.

---

## Notes

- Please update all dataset paths before running the scripts.
- Different scripts may use slightly different preprocessing or scoring logic.
- For reproducible comparison, keep the random seed fixed whenever possible.
- If you use a balanced evaluation set, make sure the same split is used across all compared methods.
- Some implementation details may require adjustment based on your local data organization.

---

## Customization

This repository can be extended in several directions:

- integrating stronger temporal modeling,
- introducing physics-informed constraints,
- adding STDP-inspired or biologically motivated mechanisms,
- incorporating reconstruction or contrastive learning objectives,
- evaluating generalization across different vehicles or battery systems.

---

## Intended Use

This codebase is intended for:

- academic research,
- method comparison,
- ablation studies,
- exploratory experiments in EV battery anomaly detection.

It is not intended as a ready-to-deploy industrial fault diagnosis system without further validation, robustness testing, and engineering adaptation.

---

## Citation

If you find this repository useful in your research, please consider citing the corresponding paper once it becomes publicly available.

```bibtex
@misc{synforcenet2026,
  title={SynForceNet: Force-Inspired Anomaly Detection for EV Battery Fault Diagnosis},
  author={Anonymous},
  year={2026},
  note={GitHub repository}
}
```

If your paper is already available on arXiv or submitted to a journal, you can replace the placeholder citation with the formal bibliographic entry.

---

## License

This repository is currently intended for academic and research use.

If you plan to release it publicly, it is recommended to add a proper license file such as:

- `MIT License`, or
- `Apache License 2.0`

---

## Contact

For questions, suggestions, or academic discussion, please open an issue in this repository or contact the repository author.

---

## Acknowledgment

This repository was developed for research on anomaly detection and fault diagnosis in EV battery systems, with an emphasis on representation learning, latent-space geometry, and comparative evaluation against classical and deep one-class baselines.
