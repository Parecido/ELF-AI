## Machine Learning the Quantum Topology of Chemical Bonds

This repository contains the data processing pipelines, machine-learning workflows, and production model used in the study:

> **Machine learning the quantum topology of chemical bonds**  
> *Michal Michalski, Sławomir Berski*

The project integrates **Electron Localization Function (ELF)** basin populations with **machine learning (ML)** to predict bond electron populations directly from molecular geometry in the **QM9 dataset**.

---

## Background

Chemical bonding is described here using **real-space topology** of the ELF rather than orbital-based descriptors.  
Bonding basins are interpreted as **bond electron populations** and used as regression targets.

Key ideas:
- ELF basin populations encode **bond order, polarity, and delocalization**
- Bond populations correlate strongly with **bond length** for many bond types
- **Local chemical environment** is essential for accurate prediction
- ML enables **scalable bonding analysis** across large chemical datasets

---

## Repository Structure

```text
.
├── data/                       # Processed bond-level datasets
├── dft/                        # DFT wavefunction and topology files (WFN / TOP)
├── model/                      # Trained production model
├── cross_validation.py         # Cross-validation & hyperparameter optimization
├── cross_validation_env.py     # Cross-validation & hyperparameter optimization with environment descriptors
├── prepare_ml_data.py          # Dataset construction from ELF/QM9 outputs
├── production_model_env.py     # Final production ML model
└── README.md
```
---

## Running the scripts

All scripts can be executed directly using Python.  
Make sure you are in the root directory of the repository and that required dependencies are installed.

### Prepare the machine-learning dataset

Parses wavefunction (`.wfn`) and topology (`.top`) files and constructs the bond-level dataset used for ML.

```bash
python prepare_ml_data.py
```

### Cross-validation and hyperparameter optimization

Runs 5-fold cross-validation using bond type and bond length as input features.

```bash
python cross_validation.py
```

### Cross-validation with environment descriptors

Runs cross-validation including explicit local chemical environment descriptors.

```bash
python cross_validation_env.py
```

### Train the production model

Trains the final production model using the full dataset and saves it to the `model/` directory.

```bash
python production_model_env.py
```

## Preprint

A preprint of this work is available on [ChemRxiv](https://doi.org/10.26434/chemrxiv-2025-zljl9).
