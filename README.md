# Fractal Modelling Systems in Nature

A collection of computational models exploring how simple rules can reproduce complex natural structures.

This project forms part of my final-year Computer Science dissertation, investigating the extent to which fractal and stochastic models can approximate real-world patterns.

---

## Example Outputs

### Fractal Tree
![Tree](images/tree_example.png)

### Lightning (DBM)
![Lightning](images/lightning_example.png)

### Lévy vs Brownian Walk
![Levy vs Brownian](images/levy_vs_brownian.png)

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Models

### Fractal Tree Model
- Rule-based branching system inspired by L-systems and biological constraints
- Includes:
  - Apical dominance
  - Golden-angle phyllotaxis
  - Pipe model (radius scaling)
- Outputs 3D structures and silhouette projections

---

### Lightning Model (DBM)
- Dielectric Breakdown Model (DBM)-style simulation
- Uses a Laplace field to guide branching growth
- Produces realistic lightning-like structures
- Includes:
  - Field-driven growth
  - Branching hierarchy
  - Structural metrics (tortuosity, branch lengths)

---

### Lévy Walk Model
- Simulates Lévy flight behaviour vs Brownian motion
- Includes:
  - Power-law step generation
  - Maximum likelihood fitting (MLE)
  - Log-likelihood model comparison
  - Search efficiency experiments

---

## Evaluation Pipelines

Each model includes a reproducible evaluation pipeline:

- Multi-seed experiments
- Quantitative metrics (e.g. fractal dimension, distributions)
- CSV outputs
- Plots for analysis

### Run examples:

```bash
python eval_tree_pipeline.py --n 30 --seed0 7 --out out_tree_eval
python eval_lightning_pipeline.py --n 30 --seed0 7 --out out_lightning_eval
python eval_levy_pipeline.py --n 30 --seed0 7 --out out_levy_eval
```

---

## Analysis Outputs

The evaluation pipelines generate quantitative outputs including:

- Fractal dimension (box-counting)
- Branching statistics (trees, lightning)
- Model comparison (power-law vs exponential)
- Search efficiency (Lévy vs Brownian)

---

## Real Image Sanity Check

For the lightning model, a standalone tool is provided to compare real photographs against the model's fractal dimension. 

### Usage:

```bash
python eval_real_lightning_image.py --image path/to/photo.jpg --out out_real_lightning --thresholds 160 180 200
```

#### Directory Mode:
```bash
python eval_real_lightning_image.py --image_dir path/to/crops --out out_real_all --thresholds 160 180 200
```

### Features:
- **Batch Processing**: Process a single image or an entire folder of photographs.
- **Organizational Structure**: Saves per-image masks and diagnostic plots in dedicated subfolders.
- **Threshold Sensitivity**: Measures fractal dimension across multiple brightness thresholds to ensure stability.
- **Model Comparison**: Automatically compares real lightning $D$ values against the generated model's reference range (mean=1.34).
- **Aggregate Metrics**: Generates a `combined_real_lightning_metrics.csv` and `summary_stats.csv`.
- **Comparative Plots**: 
  - `D_histogram.png`: Distribution of $D$ across the real dataset.
  - `D_vs_generated_model_comparison.png`: Visual comparison against the DBM model.
  - `D_by_image_threshold.png`: Multi-image sensitivity analysis.

> [!NOTE]
> This is intended as a limited sanity check for the fractal dimension $D$, not a full structural validation.
