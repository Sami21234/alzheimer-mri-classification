# 🧠 Alzheimer's Disease Detection using Deep Learning (ResNet50)

An AI-powered medical imaging system to detect Alzheimer's disease stages from brain MRI scans using a ResNet50 deep learning model trained on the OASIS dataset.

---

## 🚀 Project Overview

This project classifies brain MRI images into four categories:
- Mild Dementia
- Moderate Dementia
- Very Mild Dementia
- Non Demented

A pre-trained **ResNet50** CNN was fine-tuned to achieve high accuracy while handling class imbalance and overfitting.

---

## 🗂 Dataset

- **Dataset:** OASIS MRI Dataset  
- **Classes:** 4  
- **Images:** ~86,000 MRI scans  
- Dataset is not included in this repository due to size constraints.

---

## 🧠 Model Architecture

- Base Model: **ResNet50 (ImageNet weights)**
- Input Size: 224 × 224 × 3
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## 📊 Results

- **Validation Accuracy:** ~90%
- Confusion Matrix & Classification Report generated
- Best model saved using ModelCheckpoint

---

## 🖥️ UI

A **Streamlit web application** will allow users to upload MRI images and receive real-time Alzheimer’s stage predictions.

---

## 📁 Project Structure
alzheimer-detection-resnet50/<br>
├── notebooks/<br>
├── src/<br>
├── models/<br>
├── app/<br>
├── requirements.txt<br>
├── README.md<br>

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It is **not a medical diagnostic tool**.

---

## 👨‍💻 Author

**Mohd Sami**  
Aspiring **Data Scientist & AI / ML Engineer**