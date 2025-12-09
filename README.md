# PET to CT Image Translation using Pix2Pix

This repository contains a project for translating PET (Positron Emission Tomography) images to CT (Computed Tomography) images using the Pix2Pix Generative Adversarial Network (GAN) model. The implementation is based on TensorFlow and is designed to run in Google Colab.

## Overview

The Pix2Pix model is a conditional GAN that learns to map input images (PET scans) to corresponding output images (CT scans). This project preprocesses DICOM medical images, trains the model, and generates synthetic CT images from PET inputs.

The main notebook, `Copy of pix2pix.ipynb`, includes:
- Data preprocessing: Converting DICOM files to PNG images
- Model architecture: U-Net-based generator and PatchGAN discriminator
- Training loop with adversarial and L1 losses
- Inference and visualization of results

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Google Colab (recommended for GPU access)
- Google Drive account for data storage
- Required libraries: `tensorflow`, `tensorflow_addons`, `matplotlib`, `pydicom`, `Pillow`

## Dataset

The project uses the QIN-BREAST dataset from The Cancer Imaging Archive (TCIA). The dataset includes paired PET and CT images for breast cancer patients.

### Data Structure
```
pix2pix_dataset/
├── PET/          # PET DICOM files
├── CT/           # CT DICOM files
├── train/
│   ├── PET/      # Preprocessed PET PNGs
│   └── CT/       # Preprocessed CT PNGs
└── val/
    ├── PET/      # Validation PET PNGs
    └── CT/       # Validation CT PNGs
```

### Data Preparation
1. Download the QIN-BREAST dataset from TCIA
2. Organize PET and CT DICOM files into separate directories
3. Upload to Google Drive under `My Drive/pix2pix_dataset/`
4. The notebook handles DICOM to PNG conversion and resizing

## Installation

1. Open the `Copy of pix2pix.ipynb` notebook in Google Colab
2. Mount your Google Drive
3. Install required packages (already included in the notebook):
   ```bash
   !pip install tensorflow tensorflow_addons matplotlib pydicom Pillow
   ```

## Usage

1. **Data Preprocessing**:
   - Run the cells for mounting Drive and setting paths
   - Execute the DICOM to PNG conversion script
   - Resize images to 256x256 pixels

2. **Model Training**:
   - Define the Generator and Discriminator architectures
   - Set up the training dataset
   - Run the training loop for 150 epochs (or adjust as needed)
   - Checkpoints are saved every 20 epochs

3. **Inference**:
   - Load a trained model
   - Generate CT images from PET inputs
   - Visualize results

### Key Parameters
- Image size: 256x256
- Batch size: 1
- Learning rate: 2e-4
- Loss: GAN loss + 100 * L1 loss

## Results

After training, the model can generate realistic CT images from PET scans. The notebook includes visualization functions to compare input PET, ground truth CT, and predicted CT images.

Example output shows the model's ability to translate PET intensity distributions to CT anatomical structures.

## Files Description

- `Copy of pix2pix.ipynb`: Main implementation notebook
- `Full paper_Conference_Singapore_2023.pdf`: Related research paper
- `PT to CT/manifest-1542731172463/`: Dataset directory structure
- `PT to CT/Other code_versions/`: Alternative notebook versions

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is for educational and research purposes. Please check the license of the QIN-BREAST dataset and TensorFlow for any commercial use.

## Citation

If you use this code in your research, please cite our paper:
```
@inproceedings{ahmed2023structure,
  title={Structure-Enhanced Translation from PET to CT Modality with Paired GANs},
  author={Ahmed, Tasnim and Munir, Ahnaf and Ahmed, Sabbir and Hasan, Md. Bakhtiar and Reza, Md. Taslim and Kabir, Md. Hasanul},
  booktitle={Proceedings of 6th International Conference on Machine Vision and Applications},
  year={2023},
  organization={ACM},
  address={New York, NY, USA},
  pages={5},
  note={https://doi.org/XXXXXXX.XXXXXXX}
}
```
The paper is attached in this repository

## Contact

For questions or collaborations, please open an issue in this repository.
