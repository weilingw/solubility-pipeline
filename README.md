# Solubility Prediction Pipeline

This repository provides a reproducible **machine learning pipeline for solubility prediction** using multiple descriptor sets (MOE, RDKit, Mordred, Morgan fingerprints) and models (XGBoost, Random Forest, SVM).  
It supports **10-fold CV** and **Leave-One-Solute-Out (LOSO) CV**, hybrid mode with COSMO-RS features, and interpretability via SHAP analysis and Morgan fingerprint visualisation.


Features
Models: XGBoost, Random Forest, SVM

Descriptors: MOE, RDKit, Mordred, Morgan fingerprints

Evaluation: 10-fold CV and Leave-One-Solute-Out (LOSO) CV


Interpretability:

SHAP analysis

Morgan fingerprint bitInfo visualisation

Descriptor heatmaps

Hybrid mode: integrates COSMO-RS features as optional hybrid descriptors


Install the environment

Option 1 (recommended)

Download clean-rdkit-env.tar.gz from the Releases page

Unpack and activate:

mkdir C:\envs\clean-rdkit-env

tar -xzf clean-rdkit-env.tar.gz -C C:\envs\clean-rdkit-env

C:\envs\clean-rdkit-env\Scripts\activate

conda-unpack

Run:

& C:\envs\clean-rdkit-env\python.exe -u Pipeline/main_model.py

Option 2:

Create from explicit spec

conda create -n clean-rdkit-env --file env-explicit.txt

conda activate clean-rdkit-env

Note: Plain env.yml may resolve to newer builds and is not guaranteed to reproduce the exact working environment.

For stability, use the packed tarball or env-explicit.txt.


System Requirements

OS: Windows 11 x64

Python: 3.13.0

Windows runtime: Microsoft Visual C++ 2015–2022 Redistributable


Repository Structure

solubility-pipeline/

├── main_model.py

├── merics.py

├── plots.py

├── r2_scrambling.py

├── combined_y_scrambling_plot.py

├── visualizer.py

├── bit_analysis.py

├── pca_rdkit.py

├── moe_shap_heatmap.py

├── rdkit_shap_heatmap.py

├── mordred_shap_heatmap.py

├── requirements.txt

├── README.md

├── example_data/

│   ├── final_filtered_descriptors.txt   # input fie

├── outputs/

│   ├── predictions/          # generated here during runs

└── └── summary_metrics.csv
