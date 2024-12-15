# Deepfake Detection Project

This project implements a deep learning model for detecting deepfake images. It utilizes convolutional neural networks (CNN) to classify images as either real or deepfake. The project includes steps for training, generating synthetic data, visualizing model performance, and deploying the model using a Flask web application.

---
## Setup

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Flask
- Other Python dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:

   git clone https://github.com/yourusername/Deepfake_Detection_Project.git
   cd Deepfake_Detection_Project
   
2. Install dependencies:

   pip install -r requirements.txt

## Dataset

The dataset for this project is stored in the data/sample_data/ folder. It contains two subfolders:

df/ – Deepfake images
real/ – Real images
Since the dataset is confidential, you will need to manually add the real and deepfake images to these folders. If you don't have a dataset, you can use any publicly available dataset for deepfake detection.

## Generating Synthetic Data

The project also includes a script to generate augmented versions of the images in data/sample_data/ using TensorFlow's ImageDataGenerator. This is useful for increasing the dataset size and improving model performance.
