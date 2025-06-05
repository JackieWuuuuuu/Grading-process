# Grading-process

# DR-Lesion-Analysis

This repository contains a set of Python scripts for analyzing diabetic retinopathy (DR) lesion relationships using prior knowledge, co-occurrence statistics, spatial constraints, and least squares optimization.

## File Overview

### 1. `E1-prior.py`
Extracts **prior knowledge** regarding diabetic retinopathy lesion distribution, such as the typical locations and occurrence frequencies of various lesions. This serves as a baseline for later comparative analyses.

### 2. `E2-coocur.py`
Analyzes **co-occurrence relationships** between different DR lesion types. It calculates how often certain lesions appear together and explores their statistical correlations across the dataset.

### 3. `E3-spatial.py`
Performs **spatial relationship analysis** among lesion types. It quantifies lesion spatial arrangement and relative positioning to enhance understanding of disease patterns.

### 4. `Least_squares.py`
Implements a **least squares optimization** algorithm that integrates prior knowledge, co-occurrence, and spatial constraints to fit an optimal lesion relationship model.

### 5. `Least_squares_no.py`
A baseline version of the least squares model **without prior knowledge** incorporated. Useful for ablation study and evaluating the contribution of prior terms.

## Use Case

These scripts are part of a larger diabetic retinopathy research pipeline that supports lesion detection, segmentation, and grading.

- Useful for **statistical lesion analysis
- Helps model **clinical lesion distribution
- Enables **ablation studies** and comparative experiments

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib (for optional visualizations)

