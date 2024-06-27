# Paddy Classification

This repository contains a PyTorch-based model for classifying paddy diseases. The model uses a deep learning approach to identify various diseases in paddy crops from images.

## Dataset

The dataset used for this project is collected from Kaggle, named "Paddy Disease Classification." It contains images of paddy leaves categorized into different disease classes. The dataset includes:
- **Number of classes**: 10
- **Number of images**: 10,406
- **Image dimensions**: 640x480 pixels
- **Image characteristics**:
  - Maximum pixel value: 255.0
  - Minimum pixel value: 0.0
  - Mean pixel value: 115.9670
  - Standard deviation: 71.6155

The dataset can be accessed from [this link](https://drive.google.com/drive/folders/1wYOcrSzxh8MEXws0f_R0FpCf1kOgmM02?usp=sharing).

## Model

The classification model is based on ResNet-18, a convolutional neural network architecture. The model uses transfer learning with a pre-trained ResNet-18 model, where the final fully connected layer is modified to fit the number of classes in the dataset.

### Model Architecture

- **Base Model**: ResNet-18
- **Preprocessing**: 
  - Images are resized to 224x224 pixels.
  - Normalization is applied using ImageNet mean and standard deviation values.
- **Custom Fully Connected Layer**:
    ```python
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    ```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kunalkushwahatg/paddy_classification.git
    cd paddy_classification
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the dataset by downloading it from the provided link and placing it in the appropriate directory.
2. Run the Jupyter notebook `paddy_disease_identification.ipynb` to train and evaluate the model.

## Results

The model achieves competitive accuracy on the dataset and can effectively classify different paddy diseases. For detailed results, refer to the evaluation section in the notebook.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is provided by Kaggle.
- The ResNet-18 model is developed by Microsoft Research.

## Contact

For any inquiries, please contact [Kunal Kushwaha](mailto:kunalkushwahatg@gmail.com).
