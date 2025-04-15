
# Neural Network Challenge: Student Loan Prediction

This repository contains my solution for the **Module 18 Neural Network Challenge**, where I built and evaluated a neural network model to predict student loan repayment based on borrower data.

---

## Background

The company I work for specializes in student loan refinancing. The goal of this challenge is to create a machine learning model that can predict whether a student loan applicant is likely to repay their loan, which enables more accurate and personalized interest rate offers.

A CSV dataset containing borrower and loan-related information (e.g., GPA ranking, STEM degree score, financial workshop score, etc.) was provided. Using this dataset, I created and trained a deep neural network model using TensorFlow/Keras.

---

## Files

- `student_loans_with_deep_learning.ipynb` – Jupyter Notebook containing preprocessing, model training, and evaluation.
- `student_loans.keras` – Saved Keras model for future use or inference.

---

## Model Workflow

### Part 1: Prepare the Data
- Loaded dataset from: [student-loans.csv](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv)
- Target (`y`) = `credit_ranking`
- Features (`X`) = All other columns
- Split into training/testing sets
- Scaled features using `StandardScaler` from `sklearn.preprocessing`

### Part 2: Compile & Evaluate Neural Network
- Built a deep neural network using TensorFlow/Keras
- Used:
  - `binary_crossentropy` loss
  - `adam` optimizer
  - `accuracy` as the evaluation metric
- Trained the model for 100 epochs
- Evaluated model accuracy and loss on test data

### Part 3: Predict Loan Repayment
- Reloaded the saved `.keras` model
- Made binary predictions on test data
- Rounded probabilities and generated a **classification report**

### Part 4: Recommendation System Design
- Discussed the data needed to recommend student loans
- Chose **context-based filtering** based on student profiles
- Addressed challenges such as:
  - Data privacy & FERPA compliance
  - Model bias and fairness

---

## Model Results

| Metric       | Value |
|--------------|-------|
| **Loss**     | 0.5511 |
| **Accuracy** | 74.75% |

### Classification Report
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 0.72      | 0.75   | 0.74     |
| 1     | 0.77      | 0.75   | 0.76     |

---

## Recommendation System Summary

**Data Needed:**
- Student financial profile
- Academic performance
- Loan product attributes
- Career projections
- Demographic context

**Filtering Type Chosen:** 
- Context-Based Filtering  

**Challenges Considered:**  
- Data privacy and security  
- Algorithmic bias and fairness

---

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/jesseparent/neural-network-challenge-1.git
   ```

2. Open `student_loans_with_deep_learning.ipynb` in Jupyter or Google Colab.

3. Install dependencies:
   ```bash
   pip install tensorflow pandas scikit-learn
   ```

4. Run all cells to reproduce the model.


