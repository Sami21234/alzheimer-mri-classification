# 🧠 Alzheimer's Disease Detection using Deep Learning (ResNet50)

An AI-powered medical imaging system to detect Alzheimer's disease stages from brain MRI scans using a ResNet50 deep learning model trained on the OASIS dataset, along with a Streamlit-based web interface for real-time predictions.

---

## 🚀 Project Overview

This project classifies brain MRI images into four categories:
- Mild Dementia
- Moderate Dementia
- Very Mild Dementia
- Non Demented

A pre-trained **ResNet50** CNN was fine-tuned to achieve high accuracy while handling class imbalance and overfitting.

---

## 🚀 Features
- MRI-based Alzheimer classification (4 classes)
- Transfer Learning with ResNet50
- GPU-accelerated training (Google Colab)
- Model checkpointing & early stopping
- Interactive Streamlit UI for inference
- Clean, production-ready project structure

---

## 🧬 Classes Predicted
- Non Demented
- Very Mild Dementia
- Mild Dementia
- Moderate Dementia

---

## 📁 Project Structure
alzheimer-mri-classification/<br>
│<br>
├── notebooks/<br>
│   └── alzheimer_mri_training.ipynb<br>
│<br>
├── models/<br>
│   ├── best_resnet_model.h5<br>
│   └── alzheimer_resnet50_final.h5<br>
│<br>
├── app.py                # Streamlit UI<br>
├── requirements.txt<br>
├── README.md<br>
├── .gitignore<br>
└── .gitattributes<br>
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
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

---

## 📊 Results

- **Validation Accuracy:** ~90%
- Confusion Matrix & Classification Report generated
- Best model saved using ModelCheckpoint

---

## 🖥️ UI

A **Streamlit web application** will allow users to upload MRI images and receive real-time Alzheimer’s stage predictions.

### Run locally:
```bash
streamlit run app.py
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---


## 🏁 Future Improvements
- Add Grad-CAM visual explanations
- Deploy on Streamlit Cloud / HuggingFace Spaces
- Improve class imbalance handling
---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It is **not a medical diagnostic tool**.

---

## 👨‍💻 Author

**Mohd Sami**  
Aspiring **Data Scientist & AI / ML Engineer**