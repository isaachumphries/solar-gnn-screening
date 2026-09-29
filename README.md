# Climate-Aware Solar Material Screening via Graph Neural Networks

**Team:** Isaac Humphries, Vamsi Marrapu, Abdul Mutaal | CSC 4444

Our Project identifies high-performing alternatives to silicon by predicting two key material properties: band gap and formation energy. We implement and train Graph Neural Networks, then rank candidates by expected solar yield at Baton Rouge, LA using real NASA climate data.

---

## Table of Contents
1. [Setup & Running](#setup--running)
2. [Pipeline Walkthrough](#pipeline-walkthrough)
3. [Choices of Note](#choices-of-note)
4. [Known Limitations](#known-limitations)
5. [Project Structure](#project-structure)

---

## Setup & Running

### 1. Create and activate a virtual environment
```bash
python -m venv venv

either windows:
venv\Scripts\activate

or mac/linux/unix:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Materials Project API key
Copy `env.example.py` to `env.py` and replace the placeholder with your own key from https://next-gen.materialsproject.org/api:
```python
MATERIALS_API_KEY = "your_key_here"
```
`env.py` is gitignored so your key is never committed.

### 4. Open and run the notebook
```bash
jupyter notebook main_workflow.ipynb
```
Run all cells top to bottom. The first run fetches data from the Materials Project API (~10-20 min for the full database, likely 40+ minutes of computation). Subsequent runs load from `data/materials_cache.pkl`, but recomputes GNNs and baseline regressors.

**Large files not in this repo.** The materials cache (372 MB), pre-built graph splits (~1.7 GB), and Random Forest baselines (~310 MB each) exceed GitHub's 100 MB file limit, so they are gitignored:
- `data/materials_cache.pkl`
- `data/{train,val,test}_graphs.pt`
- `models/baselines/rf_{bandgap,formation}_pipeline.joblib`

The GNN weights (`models/*_weights.pt`), training histories, Gradient Boosting baselines, and all figures are included. Running the notebook with `PRETRAINED_MODE = False` regenerates the missing files (API fetch plus graph construction and training, 40+ minutes on the full dataset); `SUBSET_MODE = True` gives a fast end-to-end run on 2,000 materials.

Quick run (~1 minute, requires the large files above): With `PRETRAINED_MODE = True` (the default), Steps 2, 3, and 4 all skip their expensive operations — baselines load from `models/baselines/`, graph splits load from `data/{train,val,test}_graphs.pt`, and GNN weights load from `models/`. The only remaining steps are the materials cache load (Step 1), EDA plots (Step 1b), feature extraction (Step 2 feature cell), the NASA POWER API call (Step 7), and inference + ranking (Steps 5, 6, 8). This path is suitable for a demo run.

### Key config flags (cell 2)
| Variable | Default | Effect |
|---|---|---|
| `SUBSET_MODE` | `False` | Set `True` to run on 2000 materials for fast testing |
| `MAX_SAMPLES` | `2000` | Row count used when `SUBSET_MODE=True` |
| `EPOCHS` | `80` | GNN training epochs (early stopping may trigger sooner) |
| `USE_WANDB` | `False` | Set `True` to log runs to Weights & Biases |
| `PRETRAINED_MODE` | `True` | Load saved weights from `models/` and skip all training |

---

## Pipeline Walkthrough

### Step 1: Data Ingestion
Queries the Materials Project API for all materials with band gap between 0.5-3.5 eV (the useful solar window). Returns `formula`, `band_gap`, `formation_energy_per_atom`, `density`, `nsites`, `volume`, and the full `structure` object for each material.

The cache logic prevents redundant API calls:
- Subset mode: uses cache only if it has greater than or equal to `MAX_SAMPLES` rows; samples randomly down to that count.
- Full mode: uses cache only if it has greater than or equal to 150,000 rows ; otherwise re-fetches.

### Step 1b: Exploratory Data Analysis
Three plots: band gap distribution (with silicon 1.12 eV and S-Q optimum 1.34 eV marked), formation energy distribution (with stability boundary at 0), and a scatter of the two. Establishes the baseline picture of the dataset before any modelling.

### Step 2: Baseline Models
Trains scikit-learn Random Forest and Gradient Boosting regressors on 22 selected composition/structural features (means, stdevs, and ranges of atomic mass, electronegativity, radius, atomic number, row, and group). These baselines set the performance floor the GNNs must beat to be considered viable for regression and future use like generation.

### Step 3: Graph Dataset Construction
Converts each crystal structure into a PyTorch Geometric `Data` object:
- Nodes: one per crystallographic site, 11 chemical features from pymatgen (see GNNExplainer section)
- Edges: all atom pairs within 5 Å (bidirectional), with distance as edge attribute
- Labels: `y = [band_gap, formation_energy_per_atom]` (shape 1x2)

Split: 70% train / 15% validation / 15% test.

With `PRETRAINED_MODE = True`, this step loads `data/train_graphs.pt`, `data/val_graphs.pt`, and `data/test_graphs.pt` directly, skipping the pymatgen structure-to-graph conversion which is the single most time-consuming part of the pipeline.

### Step 4: GNN Training (GCN - GAT - GIN)
Trains three architectures, all minimising the Physics-Informed Loss. Each model outputs two values simultaneously: predicted band gap and predicted formation energy. Early stopping monitors validation loss with patience=15.

With `PRETRAINED_MODE = True`, this step loads saved state dicts from `models/GCN_weights.pt`, `models/GAT_weights.pt`, and `models/GIN_weights.pt` and the corresponding training histories from `models/*_history.pkl`, then runs a single evaluation pass on the test set. The pretrained weights were trained on the full filtered Materials Project dataset (51,162 materials, 80 epochs).

### Step 5: Model Comparison
Reports three metrics per model:
- MAE Band Gap: mean absolute error in eV (interpretable, same units as the target)
- MAE Formation Energy: mean absolute error in eV/atom
- Physics Loss: total validation loss including the stability penalty (the actual training objective)
- Saved Epoch: the epoch whose weights were checkpointed (best validation loss)

The learning curve image (`data/learning_curve_colab.png`) is the exact output from the original Google Colab training run that produced the pretrained weights. It is displayed as a static image rather than regenerated, so the curves faithfully reflect the full GPU training run (~51 k materials, 80 epochs) without requiring you to retrain. When `PRETRAINED_MODE = False`, new curves are generated live from the fresh training history.

### Step 6: GNNExplainer
Runs GNNExplainer on the first test sample using the best-performing GNN (lowest band gap MAE). Produces:
- Per-atom importance ranked table with element names
- Element-type aggregation (mean/max/site count)
- Top-10 most influential bonds (element-pair names)
- Bar charts with element labels on the x-axis (colour-coded green = high, red = low)
- Printed interpretation paragraph

See the inline markdown cell after the code for a full explanation of what node and edge importances mean physically.

### Step 7: Climate Integration (NASA POWER)
Calls the NASA POWER Climatology API for Baton Rouge, LA (30.45°N, 91.15°W) to fetch:
- `ALLSKY_SFC_SW_DWN`: all-sky surface shortwave irradiance (kWh/m²/day), averaged over all months
- `T2M`: 2-metre mean temperature (°C)

Falls back to a regional placeholder (4.5 kWh/m²/day, 20°C) if the API is unavailable.

### Step 8: Solar Yield Ranking vs Silicon
Batch-predicts band gap and formation energy for every test material, then scores each with the yield proxy:

```
spectral_match = max(0, 1 - |band_gap - 1.34| / 1.34)
yield = spectral_match * irr_day * 365   [kWh/m²/year]
yield = 0  if  formation_energy_per_atom > 0  (unstable = unbuildable)
```

Outputs a table of the top 10 stable candidates alongside the silicon baseline, with the "vs Silicon" delta signed.

---

## Choices of Note:

### Why we chose 1.34 eV as the optimal band gap

1.34 eV is the Shockley-Queisser theoretical optimum for converting the AM1.5G solar spectrum, the model of the sun from earth's surface. This is the same value anywhere on earth.

What can change by city is total irradiance (`irr_day`), which NASA POWER provides. This scales the absolute yield for every material by the same constant which means the ranking order is identical across all cities (see [Known Limitations](#known-limitations) below). Phoenix gets higher absolute yields than Seattle for every material, but the top-10 list is the same.

### Why the band gap is clipped to 0.5-3.5 eV

- Below ~0.5 eV: too narrow, the material absorbs photons too easily, thermal excitation at room temperature creates excessive noise current.
- Above ~3.5 eV: too wide, the material misses most of the solar spectrum entirely.
- Silicon sits at 1.12 eV and the S-Q optimum is 1.34 eV. The range 0.5-3.5 eV captures all plausible candidates with margin.

---

## Known Limitations

1. City ranking is not truly location-differentiated; The yield proxy uses `irr_day` as a uniform scalar multiplier, so the top-10 ranking is the same for Baton Rouge as for Phoenix or Seattle with only the absolute kWh numbers differring. A proper location-specific ranking would require:
    - Spectral irradiance data (W/m²/nm vs wavelength) to compute a location-specific optimal band gap
    - A temperature correction: solar panels in Baton Rouge operate at ~50-60°C, which narrows every material's effective band gap by ~0.01-0.05 eV and means that materials with slightly higher room-temperature gaps would benefit more in the Gulf Coast heat. NASA POWER does return mean temperature, which could drive a Varshni-equation correction, but that requires per-material Varshni parameters not available in the Materials Project database.

2. GNNExplainer explains one sample; The explanation in Step 6 describes what the GNN attended to for a single test crystal, and cannot explain all possbilities and what is a true driver.

3. Test set size scales with data fetch; In `SUBSET_MODE=True` (2000 rows), the test set is ~300 materials. The top-10 table is drawn from these 300. Furthermore, our GNNs do not pass our baseline unless at least ~10,000 rows are tested.

4. Data imported and preloaded from Google Colab may have differences in results based off which materials are in the test set. Rerunning the model will offer a fresh result.

---

## Project Structure

```
solar_material_pred/
├── main_workflow.ipynb     # Full pipeline notebook (run this)
├── src/
│   ├── data_utils.py       # API work, graph construction, feature extraction, caching
│   ├── models.py           # GCN, GAT, GIN, PhysicsInformedLoss Function, train_model, evaluate
│   └── baselines.py        # Random Forest & Gradient Boosting baseline training, simple ML
├── models/                 # Pretrained weights (loaded when PRETRAINED_MODE=True)
│   ├── GCN_weights.pt
│   ├── GCN_history.pkl
│   ├── GAT_weights.pt
│   ├── GAT_history.pkl
│   ├── GIN_weights.pt
│   ├── GIN_history.pkl
│   └── baselines/
│       ├── rf_bandgap_pipeline.joblib    # not in repo (too large), regenerated
│       ├── rf_formation_pipeline.joblib  # not in repo (too large), regenerated
│       ├── gb_bandgap_pipeline.joblib
│       └── gb_formation_pipeline.joblib
├── data/
│   ├── materials_cache.pkl # Cached API results (not in repo, regenerated)
│   ├── train_graphs.pt     # Pre-built PyG graph splits (not in repo, regenerated)
│   ├── val_graphs.pt
│   ├── test_graphs.pt
│   ├── eda_distributions.png
│   ├── feature_importance.png
│   ├── learning_curve_colab.png  # original Colab training run (displayed in Step 5)
│   ├── learning_curves.png
│   └── gnnexplainer.png
├── env.example.py          # Template: copy to env.py and add your Materials Project API key
└── requirements.txt
```