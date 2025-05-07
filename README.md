# ❤️ Heart-Disease-Prediction-using-Neural-Networks

This project predicts the likelihood of heart disease in patients using a neural network built with Python and Keras. It leverages clinical data from the Cleveland Clinic Foundation to train, validate, and evaluate the model, demonstrating how machine learning can support early medical diagnosis.

---

## 📊 Overview

Used the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease), focusing on 14 clinical features including:

- Age  
- Sex  
- Resting blood pressure  
- Cholesterol level  
- Maximum heart rate achieved  
- Exercise-induced angina  

The objective was to predict whether a patient is likely to have heart disease based on these attributes.

---

## 🧠 Procedure

### 1. Loaded and Inspected the Dataset
- Loaded the dataset using `pandas`  
- Examined structure and data types using `.info()` and `.describe()`  
- Identified missing values and understood overall feature distribution  

✅ **Outcome**: Ensured data quality and readiness for preprocessing.

---

### 2. Preprocessed the Data
- Converted categorical features (`cp`, `thal`, `slope`) into numerical using encoding techniques  
- Normalized continuous features for better training performance  
- Handled missing and noisy values to improve model robustness  

✅ **Outcome**: Prepared a clean and machine-readable dataset for model training.

---

### 3. Built and Trained the Neural Network
- Constructed a multi-layer perceptron using `Keras` with:
  - `ReLU` activation for hidden layers  
  - `Sigmoid` activation for binary classification  
  - `Adam` optimizer and `binary_crossentropy` loss  
- Trained the model over several epochs with batch processing  

✅ **Outcome**: Successfully trained a deep learning model capable of binary classification for heart disease detection.

---

### 4. Evaluated and Visualized Model Performance
- Split the data into training and test sets (80/20)
- Evaluated the model using accuracy, precision, recall, and F1-score  
- Plotted training vs. validation accuracy and loss curves  

✅ **Outcome**: Achieved strong predictive performance with ~85–90% accuracy and reliable generalization.

---

## 🛠️ Technologies Used

- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Keras  
- **Environment**: Jupyter Notebook  

---

## ✅ Results

- Achieved ~85–90% prediction accuracy  
- Neural network outperformed baseline models  
- Demonstrated the feasibility of using deep learning for medical diagnostics  

---

## 📌 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- **Institution**: Cleveland Clinic Foundation  
- **Total Records**: 303  
- **Attributes Used**: 14 key medical features  

---

## 📍 Conclusion

By applying a neural network to patient health records, this project shows how machine learning can support doctors in identifying heart disease risks early. With minimal preprocessing and an efficient model architecture, we were able to deliver accurate predictions that could enhance preventative care in healthcare settings.

