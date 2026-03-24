AI-Driven Customer Intelligence System
Advanced Customer Segmentation Using Unsupervised Learning

Project Overview:
This project focuses on building an end-to-end customer segmentation system using unsupervised machine learning techniques. The goal is to identify hidden customer patterns and transform them into actionable business insights for strategic decision-making.
Unlike supervised learning, this system works on unlabeled real-world data, making it highly relevant for industry use cases such as marketing optimization, customer retention, and revenue growth.



Objectives:
•	Perform complete machine learning lifecycle 
•	Discover hidden customer segments 
•	Analyze customer behavior and spending patterns 
•	Provide business insights and marketing strategies 
•	Compare multiple clustering algorithms scientifically



Selected Dataset:
Total Records: 7,253 
Total columns: 14 
Dataset Type: Used_Cars_dataset



Machine Learning Pipeline:
1️.Data Preprocessing
•	Handling missing values 
•	Removing duplicates 
•	Outlier detection (IQR method) 
•	Encoding categorical variables 
•	Feature scaling (Standardization) 
2️.Feature Engineering
•	RFM Analysis: 
o	Recency 
o	Frequency 
o	Monetary 
•	Behavioral Features: 
o	Average purchase value 
o	Purchase patterns 
•	Derived Metrics: 
o	Value per day 
o	Customer engagement score 

3️.Exploratory Data Analysis (EDA)
•	Univariate analysis 
•	Bivariate analysis 
•	Correlation heatmaps 
•	Distribution plots 

4️.Clustering Algorithms Implemented
•	K-Means Clustering 
•	Hierarchical Clustering 
•	DBSCAN 
•	Gaussian Mixture Model (GMM) 

5️.Model Evaluation
•	Silhouette Score 
•	Davies-Bouldin Index 
•	Calinski-Harabasz Score 

6️.Dimensionality Reduction
•	PCA (Principal Component Analysis) 
•	t-SNE (optional)



Results
All outputs are stored in the results/ folder:
cluster_plots/
•	Visual representation of clusters for each algorithm 
pca_outputs/
•	2D PCA visualizations 
•	Cluster separation plots 
metrics/
•	Model comparison metrics 
•	Cluster summary statistics 



Cluster Interpretation
Each cluster is analyzed based on:
•	Customer behavior 
•	Spending habits 
•	Engagement level 
