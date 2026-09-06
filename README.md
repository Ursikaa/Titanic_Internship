# Titanic_Internship
Titanic dataset data cleaning and preprocessing
# Titanic Internship

## Overview
This repository contains the work completed during the Titanic Internship project. The project focuses on data cleaning, exploratory data analysis, visualization, and unsupervised learning using Python.

## Week 2 – Exploratory Data Analysis

### Objective
The objective of Week 2 was to perform Exploratory Data Analysis (EDA) on the Titanic dataset and identify important patterns, relationships, and trends.

### Work Completed
- Cleaned and preprocessed the Titanic dataset.
- Handled missing values in important numerical columns.
- Performed statistical analysis of the dataset.
- Analysed passenger age and fare distributions.
- Studied survival patterns based on passenger class and sex.
- Created a correlation heatmap to understand relationships between numerical features.
- Created multiple visualizations to support the analysis.

### Week 2 Visualizations
- age_distribution.png
- fare_distribution.png
- correlation_heatmap.png
- survival_by_class.png
- survival_by_sex.png
- survival_count.png

## Week 3 – Unsupervised Learning and Clustering

### Objective
The objective of Week 3 was to apply unsupervised learning techniques to segment Titanic passengers into meaningful groups using K-Means clustering.

### Methodology
The cleaned Titanic dataset was used for clustering. Relevant numerical features including Age, Pclass, SibSp, Parch, and Fare were considered. Missing values were handled before applying the clustering algorithm.

K-Means clustering was applied with 4 clusters. The resulting clusters were analysed using their passenger counts and average feature values.

### Cluster Analysis

| Cluster | Passengers | Average Age | Average Pclass | Average SibSp | Average Parch | Average Fare |
|--------|------------|-------------|----------------|---------------|---------------|--------------|
| 0 | 520 | 27.32 | 2.79 | 0.23 | 0.09 | 11.96 |
| 1 | 32 | 27.97 | 1.00 | 0.72 | 1.28 | 227.08 |
| 2 | 233 | 40.32 | 1.22 | 0.36 | 0.21 | 51.26 |
| 3 | 106 | 15.72 | 2.73 | 2.26 | 1.93 | 30.81 |

### Cluster Insights
- *Cluster 0:* The largest group, containing 520 passengers, with a relatively low average fare of 11.96.
- *Cluster 1:* A small group of 32 passengers with the highest average fare of 227.08 and an average Pclass of 1.00.
- *Cluster 2:* Contains 233 passengers and has the highest average age of 40.32, with an average fare of 51.26.
- *Cluster 3:* Contains 106 passengers and represents the youngest group, with an average age of 15.72. It also has higher average SibSp and Parch values, indicating more passengers travelling with family members.

### Week 3 Visualizations
- cluster_age_fare.png
- cluster_size.png
- average_age_by_cluster.png
- average_fare_by_cluster.png

## Files

### Python Scripts
- data_cleaning.py
- week3_clustering.py

### Datasets

- cleaned_titanic.csv
- titanic_clustered.csv

### Results
- result.txt

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Conclusion
The project demonstrates the complete workflow of working with the Titanic dataset, from data cleaning and exploratory analysis to unsupervised learning and passenger segmentation using K-Means clustering.
