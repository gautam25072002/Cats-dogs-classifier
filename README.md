# 🐱 Cats vs Dogs Image Classification

A Deep Learning project that classifies images of cats and dogs using **Transfer Learning with MobileNetV2**, built with TensorFlow and deployed as an interactive web application using Streamlit.

## 🚀 Live Demo

**[Try the Cats vs Dogs Classifier](https://cats-dogs-classifier-qcprtgrwx2ag7oy6mi6yn8.streamlit.app/)**

Upload an image of a cat or dog and get an instant prediction with its confidence score.

## 📌 Project Overview

This project uses a pretrained Convolutional Neural Network (CNN), MobileNetV2, to perform binary image classification.

Instead of training an entire CNN from scratch, Transfer Learning was used to leverage features learned from ImageNet and adapt them to the cat-and-dog classification task.

The model was trained using two approaches:

- **Feature Extraction:** Froze the pretrained MobileNetV2 layers and trained a new classification head.
- **Fine-Tuning:** Unfroze selected later layers of MobileNetV2 and trained them using a smaller learning rate.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning framework |
| Keras | Model development |
| MobileNetV2 | Pretrained CNN architecture |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| Google Colab | Model training environment |

## 📂 Dataset

**Cats and Dogs Filtered Dataset**

The dataset contains images belonging to two classes:

- Cats
- Dogs

| Dataset Split | Number of Images |
|---|---:|
| Training | 2,000 |
| Validation | 1,000 |
| Total | 3,000 |

The validation dataset contains 500 cat images and 500 dog images.

### Data Preprocessing

- Resized images to 224 × 224 pixels.
- Converted images into RGB format.
- Applied MobileNetV2 preprocessing.
- Normalized pixel values to the range [-1, 1].
- Used batches of 32 images.

## 🧠 Model Architecture

**Base Model:** MobileNetV2 (ImageNet pretrained weights)

The original classification layer was removed using `include_top=False`.

A custom classification head was added:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

### Model Configuration

- Input shape: 224 × 224 × 3
- Feature extraction: MobileNetV2
- Pooling: GlobalAveragePooling2D
- Regularization: Dropout (0.2)
- Output layer: Dense (1 neuron)
- Activation: Sigmoid
- Loss function: Binary Crossentropy
- Optimizer: Adam

## 📈 Training Process

### 1. Feature Extraction

- Loaded pretrained MobileNetV2 weights from ImageNet.
- Froze the pretrained base model.
- Trained the newly added classification head.
- Used Adam optimizer with a learning rate of 0.001.
- Trained for 5 epochs.

### 2. Fine-Tuning

- Unfroze selected later layers of MobileNetV2.
- Kept earlier layers and Batch Normalization layers frozen.
- Reduced the learning rate to 0.00001.
- Trained for another 3 epochs.

Fine-Tuning helped the model adapt its pretrained visual features to the new classification task.

## 📊 Model Performance

**Final Validation Accuracy: 99.1%**

### Feature Extraction vs Fine-Tuning

| Metric | Feature Extraction | Fine-Tuning |
|---|---:|---:|
| Training Accuracy | 98.85% | 99.85% |
| Validation Accuracy | 98.50% | 99.10% |
| Training Loss | 0.0415 | 0.0068 |
| Validation Loss | 0.0369 | 0.0238 |

### Classification Report

| Metric | Cat | Dog |
|---|---:|---:|
| Precision | 0.99 | 0.99 |
| Recall | 0.99 | 0.99 |
| F1-score | 0.99 | 0.99 |
| Support | 500 | 500 |

The classification metrics are rounded to two decimal places.

The final model correctly classified 991 out of 1,000 validation images.

## ✨ Application Features

- Upload images in JPG, JPEG, and PNG formats.
- Predict whether an image contains a cat or dog.
- Display prediction confidence.
- Interactive interface built with Streamlit.
- Deployed online using Streamlit Community Cloud.

## 📁 Project Structure

```text
cats-dogs-classifier/
│
├── app.py
├── cats_dogs_mobilenetv2.keras
├── requirements.txt
├── README.md
│
└── notebooks/
    └── cats_dogs_training.ipynb
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd cats-dogs-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

**Note:** Running locally requires a compatible TensorFlow installation. The deployed application runs in the Streamlit Cloud environment.

## 🎯 Key Learnings

Through this project, I gained practical experience with:

- Convolutional Neural Networks (CNNs).
- Transfer Learning and Feature Extraction.
- Fine-Tuning pretrained models.
- Image preprocessing and augmentation concepts.
- Binary classification using TensorFlow/Keras.
- Model evaluation using Precision, Recall, F1-score and Confusion Matrix.
- Building interactive ML applications with Streamlit.
- Deploying Deep Learning applications using GitHub and Streamlit Community Cloud.

## 🔮 Future Improvements

- Evaluate the model on a separate independent test dataset.
- Experiment with additional data augmentation techniques.
- Compare MobileNetV2 with other pretrained CNN architectures.
- Improve model robustness on real-world images.
- Explore model optimization and inference performance.

---

**Developed as a hands-on Deep Learning project using TensorFlow and Keras.**