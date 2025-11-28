# Handwritten Digit Classifier

A deep learning application that recognizes handwritten digits using Convolutional Neural Networks (CNN). Users can draw digits on an interactive canvas and get real-time predictions.

## 🎯 Overview

This project implements a handwritten digit classification system trained on the MNIST dataset. It features a user-friendly web interface built with Streamlit that allows users to draw digits and receive instant predictions from a pre-trained neural network model.

## ✨ Features

- **Interactive Drawing Canvas**: Draw digits directly in the web interface
- **Real-time Prediction**: Get instant predictions as you draw
- **Adjustable Stroke Width**: Customize brush size for better accuracy
- **High Accuracy**: CNN model trained on MNIST dataset
- **Easy to Use**: Simple and intuitive user interface

## 📦 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThakurUjjawal7/Handwritten-digit-classifier.git
   cd Handwritten-digit-classifier
   ```

2. **Install dependencies**
   
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using Pipenv:
   ```bash
   pipenv install
   ```

3. **Verify model files**
   
   Ensure the pre-trained models are in the `model/` directory:
   - `handwritten.h5`
   - `mnist_model.h5`

## 💻 Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run App.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

3. **Draw and Predict**
   - Draw a digit (0-9) on the white canvas
   - Adjust the stroke width using the slider if needed
   - Click the "Predict Now" button
   - View the prediction result

## 📁 Project Structure

```
Handwritten-digit-classifier/
├── App.py                      # Main Streamlit application
├── README.md                   # Project documentation
├── Pipfile                     # Pipenv dependencies
├── LICENSE                     # License file
├── model/
│   ├── handwritten.h5          # Pre-trained digit classification model
│   └── mnist_model.h5          # Alternative MNIST model
├── Model_Training/
│   └── Main_Model.ipynb        # Jupyter notebook for model training
└── prediction/                 # Prediction-related files
```

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Training Dataset**: MNIST (60,000 training images)
- **Input Shape**: 28×28 grayscale images
- **Output**: Digit prediction (0-9)
- **Framework**: TensorFlow/Keras

### Model Performance

The CNN model achieves high accuracy on the MNIST test set. The model is pre-trained and ready for inference without requiring additional training.

## 🛠️ Technologies Used

- **TensorFlow**: Deep learning framework for model inference
- **Keras**: High-level neural networks API
- **Streamlit**: Web framework for building interactive ML applications
- **Pillow (PIL)**: Image processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization
- **streamlit-drawable-canvas**: Drawing component for Streamlit

## 🔍 How It Works

1. User draws a digit on the canvas
2. Image is captured and preprocessed:
   - Converted to grayscale
   - Resized to 28×28 pixels (MNIST standard)
   - Normalized to values between 0-1
3. Image is fed to the pre-trained CNN model
4. Model predicts the digit (0-9)
5. Prediction is displayed to the user

## 📝 Model Training

To retrain or modify the model, refer to `Model_Training/Main_Model.ipynb`. This Jupyter notebook contains:
- Data loading and preprocessing
- Model architecture definition
- Training and validation
- Model evaluation

## 🐛 Troubleshooting

**Issue**: Model file not found
- **Solution**: Ensure the `model/` directory contains `handwritten.h5`

**Issue**: Port 8501 already in use
- **Solution**: Run with a different port: `streamlit run App.py --server.port 8502`

**Issue**: Predictions are inaccurate
- **Solution**: Ensure digits are drawn clearly and fill most of the canvas

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 👨‍💻 Author

**Thakur Ujjawal**

GitHub: [@ThakurUjjawal7](https://github.com/ThakurUjjawal7)

## 🙏 Acknowledgments

- MNIST dataset providers
- TensorFlow and Keras communities
- Streamlit framework developers
