# ML_Used_Smartphone_Price_Analytics
This project addressed the challenges of predicting an appropriate resale price for used smartphones, determining the factors impacting price, and classifying smartphones into Low and High resale price tiers. 

# Introduction
The project Used Smartphone Price Analytics, is focuses on a descriptive and analytical study of the “used_device_data.csv” dataset. The major problem is to find a way to determine the optimal resale price for used smartphones. In today’s highly competitive market setting, the right price is challenging as it depends on various factors such as brand, model, age, and other device specifications.
The main objective of the project is to find the key factors that affect used smartphone price, build accurate price prediction models, and provide useful insights to both sellers and buyers to help make better decisions. 

# Problem
The used smartphone market is highly competitive since there is no clear way to determine the reasonable price of a device. Similar phones with almost identical features are often sold at different prices, and pricing is usually set manually without proper insights. It creates confusion for both buyers and vendors. If prices are very low or very high, it can create a serious problem, such as businesses losing profit or customers. On the other hand, customers lack confidence while shopping from such businesses because they are unsure whether they are getting a proper deal.

## Business Goal
- Find a way to price used smartphones fairly and accurately, based on their features and condition.
- Predict whether a used phone belongs to the high or low category based on its features and condition.

## Analytical Goal
- Develop the machine learning models that accurately predict the used smartphone price.
- Identify the key factors that influence used smartphone prices.
- Develop the machine learning models that can accurately predict smartphones in high or low categories using their features and
  conditions. 

# Ways to Address
- [x] Data Exploration
- [x] Data Preprocessing
- [x] Predictor Analysis and Relevancy
- [x] Data Dimension and Transformation
- [x] Data Partitioning
- [x] Model selection
- [x] Model Fitting, Validation Accuracy, and Test Accuracy
- [x] Model Evaluation
- [x] Recommendations

# Insights
- Most phones have screen sizes between 12.70 cm and 15.34 cm.
- Real camera resolution ranges from 0.08 MP to 48MP (179 records missing).
- Front camera resolution ranges from 0.00 MP to 32 MP. (0.00 MP represents older phones without a Front camera).
- The middle value (median) is 32 GB, meaning half of all phones have 32 GB of storage or less (4 records missing).
- RAM ranges from 0.020 GB to 12 GB, with a mean of about 4.036 GB (4 records missing).
- The common phone battery is around 2100 to 4000 mAh (6 records missing).
- Most phones weigh between 142 grams and 185 grams (7 records missing).
- Smartphones in this dataset were released between 2013 and 2020.
- Days used range from about 91 days to 1094 days (nearly 3 years).
- Normalized_used_price is our target variable and ranges from 1.537 to 6.619.
- Most of the new phone prices were between 4.790 to 5.674.
- The majority of devices support Android (93.05%), with very minimal devices supporting iOS (1.04%), Windows (1.94%), and Other (3.97%) Operating systems.
- Most devices come from well-known brands, Samsung (9.87%), Huawei (7.27%), LG (5.82%), and Lenovo (4.95%). There are many devices categorized as Others (14.53%).
- The vast majority of devices (88.33%) supported 4 G. Only 14.59% of devices supported 5G.

## Handling Missing
- One record was missing both battery and RAM. Instead of imputing these values, the record was omitted because
  - Only one row is affected, so the impact on the dataset is minimal.
  - Imputing could have introduced bias since both critical features are missing.
  - Removal ensures analysis and modeling are based on complete and reliable data.

## Imputation
The dataset contains numeric features ranging from extreme lows to extreme highs:
- Rear camera: 0.08 - 48.0 MP
- Front camera: 0.00 - 32.0 MP
- Internal memory: 0.01 - 1024.0 GB
- RAM: 0.02 - 12.0 GB
- Battery: 500 - 9720 mAh
- Weight: 69 - 855 grams

Due to these highly skewed distributions and extreme values, the median is preferred for imputing missing data over the mean. Unlike the mean, the median is not influenced by extreme values, so it represents a typical device specification.

## Empty Strings
- No empty strings were found in the dataset.

## Handling for Zero
- Multiple analyses confirmed that the 0 in the front_camera_mp in the Nokia phone is definitely valid. So keeping these is more accurate than treating them as missing because there are actual characteristics of each of the devices represented by the 0.

## Handling Outliers
### RAM
RAM values range from **0.02 GB all the way to 12 GB**.
- **Low-end (0.02–0.5 GB)** <br>
   In very old models (e.g. Nokia 2012/2013), these are found to be technically correct.
- **High-end (8–12 GB)** <br>
   These belong to recent premium smartphones. Discarding them as outliers would incorrectly remove useful information.

#### Why RAM was *not* treated as an outlier
If all RAM outliers were capped or removed: 
- Collapse variation in RAM.
- Correlations become NA (as all values converge at 4GB).
- Models will lose predictive power, especially for high-end devices.

**Conclusion:**  
RAM was not found to be an outlier, as the values provided, from legacy to premium devices, are meaningful as predictors for resale prices. 

### Other Features
Features, such as screen size and weight, contain extreme values in the dataset, which could skew the analysis. To maintain accuracy, these outliers were carefully adjusted. 
- The extremely high or low values were replaced with boundaries that reflect typical ranges for each feature.
- This adjustment ensures that the small number of unusual devices does not overly weigh the prices and comparisons.
- The dataset is now better representative of realistic market variations while also increasing the validity of the analysis and modeling.

### Outlier Treatment Summary

| Feature      | Treatment Applied |
|-------------|------------------|
| RAM         | Retained (no capping) |
| Screen Size | Capped to realistic bounds |
| Weight      | Capped to realistic bounds |


## Defining Price Categories for Classification
A 2-tier classification based on business rules, which provides a clear, actionable framework for pricing, inventory management, and model training.

## Outcome Variables
- Regression Outcome: **normalized_used_price**
- Classification Outcome: **price_category**

## Feature Selection
For feature selection, a correlation matrix was used in combination with input from technical experts to identify features with high predictive power while reducing  multicollinearity and data leakage.

## Data Transformation 
 - Binary Encoding of 4G and 5G Features
 - One Hot Encoding for Categorical Variables
   - Models such as Ridge Regression take numerical inputs only.

## Data Partitioning
Stratified Splitting
| Training Set | Validation Set | Test Set |
|----------|----------|----------|
| 60%  | 20%  | 20%  |
    

## Algorithm Selection
- Regression
    - k-Nearest Neighbors (kNN)
    - Random Forest
    - Ridge Regression
- Classification
    - k-Nearest Neighbors (kNN)
    - Random Forest

## Regression Models Evaluation
| Model                | Dataset | RMSE      | MAE       | R²        |
|----------------------|---------|-----------|-----------|-----------|
| k-Nearest Neighbors  | Test    | 0.3272353 | 0.2474099 | 0.7205367 |
| k-Nearest Neighbors  | Test    | 0.2884241 | 0.2181147 | 0.7828961 |
| Ridge Regression     | Test    | 0.3129025 | 0.2276942 | 0.7444814 |

## Classification Models Evaluation
| Models                         | Dataset | Accuracy | Sensitivity | Specificity |
|--------------------------------|---------|----------|-------------|-------------|
| k-Nearest Neighbors (kNN) Classification | Test    | 0.9394   | 0.858       | 0.9648      |
| Random Forest Classification    | Test    | 0.9577   | 0.8757      | 0.9833      |

## Recommendations
- Random Forest is effective for predicting used smartphone prices and classifying them as Low or High because of its accuracy and reliability.
