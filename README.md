# Data Science Pipeline: Data Collection, Cleaning, EDA, Feature Engineering, Feature Selection

## Project Overview

This project demonstrates a complete end-to-end data science workflow starting from raw data collection and ending with feature selection for machine learning models. The goal is to prepare high-quality data that can be used for building predictive models.

This project covers the full data science pipeline:
- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection

---

## Project Structure

Data Science Project

├── Data_Collection_Cleaning_EDA_Feature_Engineering_Selection.ipynb
├── train_delay_data.csv
├── README.md

---

## Objectives

- Understand dataset structure and features  
- Clean and preprocess raw data  
- Perform exploratory data analysis (EDA)  
- Create meaningful new features  
- Select important features for machine learning  

---

## 1. Data Collection

The dataset is loaded from a CSV file and initially explored to understand its structure, types, and basic statistics.

```python
import pandas as pd

df = pd.read_csv("train_delay_data.csv")
df.head()
