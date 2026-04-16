# NGSIM-US-101 Car-Following Data Processing Tool

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

This project provides a complete processing pipeline for the NGSIM-US-101 dataset, converting raw trajectory data into machine learning training data suitable for **Car-Following** tasks. In addition to processing NGSIM-US-101, it can also process other parts of the NGSIM dataset.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Processing Methods](#2-data-processing-methods)
3. [Usage](#3-usage)
4. [Data Format](#4-data-format)
5. [Output File Structure](#5-output-file-structure)
6. [Project Structure](#6-project-structure)
7. [Citation](#7-citation)

---

## 1. Project Overview

The NGSIM-US-101 dataset contains vehicle trajectory data from US-101 highway in the United States, including high-precision vehicle motion information for three time periods (7:50-8:05, 8:05-8:20, 8:20-8:35).

This project transforms raw trajectory data into standardized temporal training data through an 8-step processing pipeline, directly usable for deep learning models (such as LSTM, Transformer, etc.).

### Key Features

- **Complete Processing Pipeline**: End-to-end processing from raw data to training sets
- **Savitzky-Golay Smoothing**: Effectively removes positioning noise and ensures physical plausibility
- **Strict Car-Following Pair Selection**: Based on spatiotemporal continuity, filtering out lane-changing behaviors
- **Multi-Version Output**: Provides training data with 10-second and 20-second time windows
- **Model Validation**: Built-in LSTM baseline model to verify data usability

---

## 2. Data Processing Methods

The data processing pipeline includes the following main steps:

(1) **Trajectory Smoothing & Denoising**: Uses Savitzky-Golay filter to eliminate positioning jitter in raw data
(2) **Coordinate Conversion**: Converts data from imperial to metric units and performs outlier cleaning
(3) **Car-Following Pair Extraction**: Matches lead and following vehicles based on Vehicle_ID and Preceeding fields, filtering pairs meeting spatiotemporal continuity requirements
(4) **Feature Engineering**: Calculates relative distance, relative velocity, and follower velocity as input features, with follower acceleration as prediction label
(5) **Data Segmentation & Standardization**: Uses sliding window to segment data, applies standardization, and exports to HDF5 format
(6) **Model Validation**: Uses LSTM model to verify dataset usability

---

## 3. Usage

### 1. Environment Setup

```bash
# Create conda environment
conda create -n py-310 python=3.10
conda activate py-310

# Install dependencies
pip install -r requirement.txt
```

### 2. Data Preparation

Place the NGSIM-US-101 raw data in the `./US-101/` directory:
```
US-101/
├── 0750am-0805am/
│   └── trajectories-0750am-0805am.txt
├── 0805am-0820am/
│   └── trajectories-0805am-0820am.txt
└── 0820am-0835am/
    └── trajectories-0820am-0835am.txt
```

### 3. Run Processing Pipeline

```bash
# Step 1: Trajectory Smoothing & Denoising
python code/step1_denoising.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# Step 2: Coordinate Conversion & Cleaning
python code/step2_coordinate_conversion.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# Step 3: Car-Following Pair Extraction
python code/step3_car_following.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# Step 4: Feature Engineering
python code/step4_feature_engineering.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# Step 5&6: Data Segmentation & HDF5 Export
python code/step5_segmentation.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# Step 8: LSTM Model Validation
python code/step8_lstm_validation.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc
```

---

## 4. Data Format

(1) Raw Data Fields (NGSIM-US-101)

| Field | Description | Unit |
|-------|-------------|------|
| Vehicle_ID | Vehicle unique identifier | - |
| Frame_ID | Frame number | - |
| Global_Time | Global timestamp | ms |
| Local_X | Local lateral coordinate | feet |
| Local_Y | Local longitudinal coordinate | feet |
| v_Length | Vehicle length | feet |
| v_Width | Vehicle width | feet |
| v_Class | Vehicle type (1=motorcycle, 2=car, 3=truck) | - |
| v_Vel | Vehicle velocity | mph |
| v_Acc | Vehicle acceleration | ft/s² |
| Lane_ID | Lane number | - |
| Preceeding | Preceding vehicle ID (0 = no preceding vehicle) | - |
| Space_Headway | Space headway | feet |
| Time_Headway | Time headway | s |

(2) Processed Data Format

**HDF5 Format** (`train_10s.h5` / `train_20s.h5`):

```python
import h5py

with h5py.File('code/output/train_10s.h5', 'r') as f:
    X = f['X'][:]  # shape: (n_samples, 100, 3)
    y = f['y'][:]  # shape: (n_samples, 100)

    print(f"Samples: {f.attrs['n_samples']}")
    print(f"Sequence length: {f.attrs['seq_len']}")
    print(f"Features: {f.attrs['n_features']}")
```

(3) Feature Dimension Description

- **X** (Input Features): `[spacing, v_rel, v_follower]`
  - `spacing`: Net spacing between vehicle bodies (m)
  - `v_rel`: Lead vehicle velocity - follower velocity (m/s), positive value indicates follower is catching up
  - `v_follower`: Follower vehicle velocity (m/s)

- **y** (Label): Follower acceleration `acc_follower` (m/s²)

---

## 5. Output File Structure

```
.
├── code/
│   ├── output/                    # Processed data files
│   │   ├── trajectories_smoothed_{period}.csv
│   │   ├── trajectories_cleaned_{period}.csv
│   │   ├── car_following_pairs_{period}.csv
│   │   ├── features_{period}.csv
│   │   ├── train_10s_{period}.h5
│   │   ├── train_20s_{period}.h5
│   │   └── lstm_model_{period}.pth
│   ├── step1_denoising.py         # Step 1: Trajectory Smoothing
│   ├── step2_coordinate_conversion.py  # Step 2: Coordinate Conversion
│   ├── step3_car_following.py     # Step 3: Car-Following Pair Extraction
│   ├── step4_feature_engineering.py    # Step 4: Feature Engineering
│   ├── step5_segmentation.py      # Step 5&6: Segmentation & Persistence
│   └── step8_lstm_validation.py   # Step 8: Model Validation
├── doc/                           # Analysis reports and visualizations
│   ├── pic/                       # Generated images
│   │   ├── denoising_vehicle_*.png
│   │   ├── coordinate_conversion.png
│   │   ├── car_following_pairs.png
│   │   ├── feature_distribution.png
│   │   ├── segmentation.png
│   │   └── lstm_validation.png
│   ├── denoising_report.md
│   ├── coordinate_conversion_report.md
│   ├── car_following_report.md
│   ├── feature_report.md
│   ├── segmentation_report.md
│   └── model_validation_report.md
└── US-101/                        # Raw data (not uploaded to Git)
    ├── 0750am-0805am/
    ├── 0805am-0820am/
    └── 0820am-0835am/
```

---

## 6. Project Structure

```
.
├── .gitignore             			# Git ignore configuration
├── README.md              		# English documentation
├── README_CN.md           		# Chinese documentation
├── Requirements_document_new.md  	# Detailed requirements document
├── requirement.txt         		# Python dependency package list
├── code/                  		# Code directory
│   ├── *.py               			# Processing scripts
│   └── output/            		# Output data
├── doc/                    		# Documentation and visualization
│   ├── *.md               			# Analysis reports
│   └── pic/               			# Images
└── US-101/                 		# Raw data
```

---

## 7. Citation

If you use data processed by this project in your research, please cite:

1. Original NGSIM dataset literature
2. This project's GitHub repository

---

## License

MIT License

---

## Contact

If you have questions or suggestions, feel free to submit an Issue or Pull Request.