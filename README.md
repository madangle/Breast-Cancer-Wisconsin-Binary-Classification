# 🩺 Breast Cancer Diagnosis -- Machine Learning Classification

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Machine
Learning](https://img.shields.io/badge/Machine-Learning-green) ![Binary
Classification](https://img.shields.io/badge/Problem-Binary%20Classification-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

------------------------------------------------------------------------

## 📌 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths
worldwide. Early detection significantly improves survival rates.

This project applies multiple **Machine Learning classification models**
to predict whether a breast tumor is:

-   🔴 **Malignant (Cancerous)**
-   🟢 **Benign (Non-Cancerous)**

------------------------------------------------------------------------

## 🧠 a. Problem Statement

The objective is to build and compare machine learning models that can
accurately classify breast tumors as malignant or benign using numerical
features extracted from digitized images of fine needle aspirates (FNA)
of breast masses.

Since this is a medical diagnosis problem, special importance is given
to:

-   Recall
-   F1-score
-   AUC
-   MCC (Matthews Correlation Coefficient)

------------------------------------------------------------------------

## 📊 b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic Dataset)\
**Type:** Binary Classification

### Dataset Characteristics

-   Total Samples: 569
-   Number of Features: 30 numerical features
-   Classes:
    -   Malignant (M)
    -   Benign (B)
-   Feature Type: Real-valued cell nucleus measurements

------------------------------------------------------------------------

## 🤖 c. Models Used

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbors (kNN)\
4.  Gaussian Naive Bayes\
5.  Random Forest (Ensemble)\
6.  XGBoost (Ensemble)

------------------------------------------------------------------------

## 📈 Model Performance Comparison

  -------------------------------------------------------------------------------
  ML Model Name    Accuracy    AUC      Precision     Recall    F1       MCC
  ---------------- ----------- -------- ------------- --------- -------- --------
  Logistic         0.7826      0.9000   0.7778        0.9333    0.8485   0.5004
  Regression                                                             

  Decision Tree    0.8261      0.7500   0.7895        1.0000    0.8824   0.6283

  KNN              0.7391      0.8000   0.7647        0.8667    0.8125   0.3977

  Naive Bayes      0.8696      0.9417   0.8333        1.0000    0.9091   0.7217

  Random Forest    0.8261      0.9167   0.7895        1.0000    0.8824   0.6283

  XGBoost          0.8261      0.9250   0.7895        1.0000    0.8824   0.6283
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📌 Model Observations

  -----------------------------------------------------------------------
  ML Model Name                           Observation
  --------------------------------------- -------------------------------
  Logistic Regression                     Strong AUC and high recall.
                                          Performs well in detecting
                                          malignant cases but moderate
                                          MCC.

  Decision Tree                           Achieved perfect recall (1.0),
                                          meaning no malignant case was
                                          missed. Lower AUC suggests
                                          weaker ranking performance.

  KNN                                     Lowest overall performance.
                                          Lower accuracy and MCC indicate
                                          weaker generalization.

  Naive Bayes                             Best performing model. Highest
                                          Accuracy, AUC, F1-score, and
                                          MCC. Balanced and highly
                                          effective.

  Random Forest                           Strong ensemble model with
                                          excellent recall and high AUC.
                                          Stable performance.

  XGBoost                                 High AUC and perfect recall.
                                          Performance comparable to
                                          Random Forest.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   Scikit-learn
-   XGBoost
-   Pandas
-   NumPy
-   Matplotlib / Seaborn

------------------------------------------------------------------------

## 🚀 How to Run

``` bash
git clone <repository-url>
cd breast-cancer-classification
pip install -r requirements.txt
python main.py
```

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.
