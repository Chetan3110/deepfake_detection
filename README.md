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

   git clone https://github.com/Chetan3110/deepfake_detection.git
   cd deepfake_detection
   
2. Install dependencies:

   pip install -r requirements.txt

## Dataset

The dataset for this project is stored in the data/sample_data/ folder. It contains two subfolders:

df/ – Deepfake images
real/ – Real images
Since the dataset is confidential, you will need to manually add the real and deepfake images to these folders. If you don't have a dataset, you can use any publicly available dataset for deepfake detection.

## Generating Synthetic Data

The project also includes a script to generate augmented versions of the images in data/sample_data/ using TensorFlow's ImageDataGenerator. This is useful for increasing the dataset size and improving model performance.

### How to Generate Synthetic Data

1. Place your sample images in data/sample_data/df/ and data/sample_data/real/
   
2. Run the generate_synthetic_data.py script to generate synthetic data:
   
   python scripts/generate_synthetic_data.py

3. The augmented images will be saved in data/synthetic_data/ under the df/ and real/ folders.

### Augmentation Techniques Used:
Rotation
Width and height shift
Zooming
Horizontal flips

## Training the Model

### Jupyter Notebook

The model is trained using the Model.ipynb Jupyter notebook. This notebook covers the following:

1.	Data Preprocessing: Loads and preprocesses the images for training.

2.	Model Architecture: A CNN model for binary classification (real or deepfake).

3.	Training: The model is trained using the augmented dataset.

4.	Evaluation: Model performance is evaluated using accuracy, AUC, confusion matrix, and ROC curve.

### Steps to Train the Model:

1.	Open Model.ipynb in Jupyter Notebook.

2.	Run the cells to preprocess data, define the model, and train it.

3.	After training, the model will be saved as model.h5.

### Visualizations

The following visualizations are generated during model training to assess performance:

1.	Accuracy Plot: Displays training and validation accuracy over epochs.
![acc.png](visualization\acc.png)

2.	Confusion Matrix: Visualizes the confusion matrix of the predictions.

3.	ROC Curve: Displays the Receiver Operating Characteristic curve.

4.	AUC Curve: Displays the Area Under Curve (AUC) metric.

These plots are saved as PNG images in the visualizations/ folder.


### Flask Web Application

The trained model can be deployed as a web application using Flask. The app allows users to upload an image and classify it as real or deepfake.

### Running the Flask App

1.	Make sure the model (model.h5) is saved in the project root directory.

2.	Run the Flask app:

python app.py

3.	Open a web browser and navigate to http://127.0.0.1:5000/.

4.	Upload an image (either real or deepfake), and the model will classify it.

### Example Output in Flask

Once you upload an image, the model will classify it as either:

•	"This is a Real Image!"

•	"This is a Deepfake Image!"

The result will be shown on the page after the upload.


### Visualizations

The following visualizations are saved and can be found in the visualizations/ folder. These can also be embedded in the Flask app to show model performance.

1.	AUC Curve (auc.png): Displays the Area Under Curve for ROC.

2.	Accuracy Plot (acc.png): Displays training and validation accuracy over epochs.

3.	Confusion Matrix (cm.png): Visualizes the confusion matrix.

4.	ROC Curve (roc.png): Displays the Receiver Operating Characteristic curve.


### Conclusion

This project provides an end-to-end pipeline for detecting deepfake images using deep learning. It includes data augmentation, model training, evaluation, and web deployment using Flask. You can use this project to detect deepfakes in images and further explore deep learning for image classification tasks.



