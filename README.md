# 🌿 Plant Disease Prediction using Deep Learning

This project is a **Deep Learning-based Plant Disease Detection System** that identifies plant leaf diseases from images using a **Convolutional Neural Network (CNN)** model trained on plant disease datasets.

---

## 🚀 Features
- 🧠 Trained CNN model using **TensorFlow/Keras**
- 🌱 Detects multiple **plant leaf diseases** from images
- 📷 Supports **image upload** for live predictions
- 💻 Built with **Streamlit** for a clean and interactive web interface
- 🗂️ Uses a well-structured dataset with labeled classes
- 🔍 Real-time classification with accurate results

---

## 🧩 Tech Stack
- **Python 3**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **OpenCV**
- **Streamlit**

---

## 📂 Project Structure
plant_disease_prediction/
│
├── plant_disease_prediction_model.h5 # Trained CNN model
├── class_indices.json # Class labels
├── app.py # Streamlit app for prediction
├── requirements.txt # Required dependencies
├── sample_images/ # Sample test images
└── README.md # Project documentation

## Plant Disease Prediction

This project uses a pre-trained deep learning model to predict plant diseases.

### Download the pre-trained model
[Download Model]('/content/plant_disease_prediction_model.h5')

---

## ⚙️ How to Run
1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/plant-disease-prediction.git
   cd plant-disease-prediction


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Upload a leaf image and get instant prediction!

📊 Model Details

Architecture: CNN (Convolutional Neural Network)

Framework: TensorFlow / Keras

Accuracy: ~95% on test data

Dataset: PlantVillage dataset (publicly available on Kaggle)

🖼️ Sample Output

✨ Future Enhancements

📈 Improve accuracy with transfer learning (e.g., ResNet, VGG16)

🌍 Deploy on cloud for live usage

📱 Build a mobile app version for farmers
