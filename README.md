# Solubility Prediction Pipeline

This repository provides a reproducible **machine learning pipeline for solubility prediction** using multiple descriptor sets (MOE, RDKit, Mordred, Morgan fingerprints) and models (XGBoost, Random Forest, SVM).  
It supports **10-fold CV** and **Leave-One-Solute-Out (LOSO) CV**, hybrid mode with COSMO-RS features, and interpretability via SHAP analysis and Morgan fingerprint visualisation.


## Installation
### Option 1: Use packed environment (recommended)
Download [`clean-rdkit-env.tar.gz`](https://github.com/weilingw/solubility-pipeline/releases) from the Releases page and unpack:

```powershell
mkdir C:\envs\clean-rdkit-env
tar -xzf clean-rdkit-env.tar.gz -C C:\envs\clean-rdkit-env
C:\envs\clean-rdkit-env\Scripts\activate
conda-unpack
````
Run:
````
C:\envs\clean-rdkit-env\python.exe -u Pipeline\main_model.py
````

### Option 2:  
Create from explicit spec:
````
conda create -n clean-rdkit-env --file env-explicit.txt
conda activate clean-rdkit-env
````
> **Note:** Plain `env.yml` may not exactly reproduce the same builds across machines.  
> For stability, prefer the packed tarball or `env-explicit.txt`.

## Repository Structure
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

│ └── final_filtered_descriptors.txt # input file

├── outputs/

│ └── predictions/ # generated here during runs

└── summary_metrics.csv # summary metrics
### Configuration (Global Settings)

Before running, open `Pipeline/main_model.py` and adjust the global settings at the top of the file:

```python
# === Global Settings ===
model_type = 'rf'              # 'rf', 'xgb', 'svm'
descriptor_type = 'moe'        # 'morgan', 'mordred', 'moe', 'rdkit'
use_hybrid_mode = True         # include COSMO features as hybrid input
use_random_search = True       # enable RandomizedSearchCV hyperparameter tuning
use_bit_visualization = False  # only used for Morgan fingerprints
use_saved_models = True        # reuse pre-trained models if available
enable_y_scrambling = True     # perform Y-scrambling for significance testing
````

Outputs (predictions, plots, logs) will appear under the `outputs/` and `predictions/` directories.

Requirements
- **OS**: Windows 11 x64
- **Python**: 3.13.0 (64-bit, Conda recommended)  
- **Conda**: ≥23  
- **Windows runtime**: [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)


## 📜 Citation
If you use this pipeline in academic work, please cite:

[TODO: add the paper reference]
59daeda (Update README with configuration section and usage instructions)
