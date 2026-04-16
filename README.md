# VVBPConvNet_for_sparse_view_CT

Official implementation of "VVBPConvNet: A Lightweight Convolutional Network for Sparse-View CT Reconstruction"

## Architecture

![Network Architecture](figures/Figure2.tif)

## Requirements

- Python >= 3.11
- PyTorch >= 2.5.1
- CUDA >= 11.8

# Clone this repository

```bash
git clone https://github.com/coring511/VVBPConvNet_for_sparse_view_CT
cd VVBPConvNet_for_sparse_view_CT-main
```

# Create conda environment

```bash
conda create -n vvbpconvnet python=3.11
conda activate vvbpconvnet
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Dataset Preparation

This project uses the following public datasets for training and testing:

1. **LoDoPaB-CT Dataset**  
   - **Description**:  
     The Low-Dose Parallel Beam Computed Tomography Dataset (LoDoPaB-CT) is a benchmark dataset specifically designed for evaluating and developing image reconstruction algorithms. It contains simulated CT scan data for various levels of dose reduction and noise, making it an ideal choice for low-dose CT research.  
   - **Link**: [LoDoPaB-CT Dataset on Zenodo](https://doi.org/10.5281/zenodo.3384092)  
   - **Usage in this project**:  
     - **Training, validation and testing**: We use LoDoPaB-CT for model training, validation and testing.

2. **2016 Low-Dose CT Grand Challenge Dataset**  
   - **Description**:  
     The Low-Dose CT Grand Challenge dataset is a clinical dataset released during the 2016 Low-Dose CT Grand Challenge. It contains real-world low-dose CT scans of patients, which are valuable for testing and benchmarking algorithms in clinical conditions.  
   - **Link**: [AAPM Low-Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/#instructions)  
   - **Usage in this project**:  
     - **Testing**: We use this dataset as a benchmark for testing clinical performance of our proposed methods.
     
### Data Directory Structure

After downloading the LoDoPaB-CT and 2016 Low-Dose CT Grand Challenge datasets, please ensure that the directory structure is organized as follows:
    
```bash
data/
├── train/
│   ├── input/        # VVBP tensors
│   └── target/       # Ground truth
└── val/
│   ├── input/
│   └── target/
└── test/
    └── sinogram60views_VVBP
```

## Pre-trained Models

| Model                | Description     | Download                                                 |
|----------------------|-----------------|---------------------------------------------------------|
| best_model_win128.pth | Full model      | [Baidu Pan]()       |

### Training

```bash
python train.py
```

### Testing

```bash
python test.py --weight weight/best_model_win128.pth --input data/test/sinogram60views_VVBP
```

---

## Citation and License

If you find this work useful, please cite:

```bash

```

If you use this project, please also consider citing the datasets we used:

1. **LoDoPaB-CT Dataset**:
```bash

```

2. **2016 Low-Dose CT Grand Challenge Dataset**:
```bash

```


