# 🪼 Jellyfish Detection with CNN

This repository contains a full pipeline to detect and classify jellyfish species using a Convolutional Neural Network (CNN) built in TensorFlow/Keras.

## 📦 Dataset

- Source: [Kaggle - Jellyfish Types](https://www.kaggle.com/datasets/anshtanwar/jellyfish-types)
- Downloaded via Kaggle API inside the notebook.
- Includes multiple classes of jellyfish for image classification.

## 🧠 Model Overview

- **Input Size:** 128x128 RGB images
- **Layers:**
  - Conv2D → ReLU → MaxPooling (x3)
  - Flatten
  - Dense → Dropout → Output
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `adam`
- **Metrics:** `accuracy`

## 🛠️ Setup Instructions

### 1. Install dependencies

```bash
pip install tensorflow numpy matplotlib kaggle
```

### 2. Setup Kaggle API Key

Place your `kaggle.json` file in the root directory and run:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run the Notebook

Open `JellyfishDetection.ipynb` in Jupyter or Colab and run all cells.

## 🧪 Testing Your Own Image

```python
from tensorflow.keras.preprocessing import image

img = image.load_img("path/to/image.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)  # batch dimension
prediction = model.predict(img_array)
```

## 💾 Saving & Loading Model

```python
model.save("jellyfish_model.h5")
# To load later:
model = tf.keras.models.load_model("jellyfish_model.h5")
```

> ❗ Don't use `pickle` for saving models; use `.h5` or `SavedModel` format.

## 📌 Improvements To-Do

- Add data augmentation
- Use Transfer Learning (e.g., MobileNet, ResNet)
- Object detection to localize jellyfish in frames


