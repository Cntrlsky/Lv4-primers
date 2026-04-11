# Lv4 Primers Project



###### *Code used is Python*

###### *Code Viewer is VS-code*

###### *Code Sharing platform is GitHub*

###### *Dead Line is 25 April 2026*



# **Deliverables**



##### Network Security \& Intuition Detection

&#x20;

**1.Network Intrusion Detection System (NIDS)**

&#x09;

&#x09;Description:

&#x09;	Build a system to detect network intrusions and classify attack types using algorithms

&#x09;Datasets:

&#x09;	https://www.kaggle.com/datasets/hassan06/nslkdd

&#x09;	https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

&#x09;	https://research.unsw.edu.au/projects/unsw-nb15-dataset



**2.DDoS Attack Detection \& Prevention**

&#x09;

&#x09;Description:

&#x09;	Detect Distributed Denial of Service attacks in real-time network traffic

&#x09;Datasets:

&#x09;	https://www.unb.ca/cic/datasets/ddos-2019.html

&#x09;	https://www.kaggle.com/datasets/devendra416/ddos-attacks-dataset



**3. Botnet Detection System**

&#x09;

&#x09;Description:

&#x09;	Identify compromised devices forming botnets through traffic pattern analysis

&#x09;Dataset:

&#x09;	https://mcfp.felk.cvut.cz/publications/datasets/ctu-13/

&#x09;	https://www.kaggle.com/datasets/yashchoudhari/botnet-traffic



**4. Network Anomaly Detection**

&#x09;

&#x09;Description:

&#x09;	Detect unusual network behavior patterns indicating potential threats

&#x09;Dataset:

&#x09;	http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

&#x09;	https://www.kaggle.com/datasets/ymirshabani/nf-unsw-nb15-v2



**5. Zero-Day Attack Detection**

&#x09;

&#x09;Description:

&#x09;	Identify previously unknown attacks using unsupervised/deep learning techniques

&#x09;Dataset:

&#x09;	https://www.kaggle.com/datasets/datasetengineer/zero-day-attack-detection-in-logistics-networks

&#x09;	https://research.unsw.edu.au/projects/unsw-nb15-dataset

## 

# **Required Headings**

#### 

#### 1.Project Documentation



**Executive Summary(1-2) pages**

1. Problem statement and motivation
2. Proposed solution overview
3. Key results and achievements
4. Real-world applications and impact
**Technical Report(15-25) pages**

   1. Introduction and Literature Review
   2. Methodology and Architecture
   3. Data Preprocessing Pipeline
   4. Feature Engineering Process
   5. Model Selection Justification
   6. Experimental Setup
   7. Results and Analysis
   8. Discussion and Limitations
   9. Conclusion and Future Work

**Code Documentation**

1. README file with setup instructions
2. Requirements/dependencies list
3. Inline code comments (minimum 20% of code)
4. API documentation for functions
5. User manual for running the system



#### 2.Minimum Code Components



1. Data loading and preprocessing module
2. Feature extraction/engineering module
3. Model implementation/training script
4. Evaluation and metrics calculation
5. Prediction/inference pipeline
6. Unit tests (minimum coverage: 60%)



#### 3.Data Documentation



1. Source attribution and licensing information
2. Data collection methodology description
3. Data dictionary with feature descriptions
4. Class distribution and imbalance analysis
5. Data quality assessment report



#### Processing artifacts



1. Cleaned dataset (CSV/Parquet format)
2. Train/test/validation split files
3. Feature scaling/normalization parameters
4. Encoders and transformers (saved as pickle/ joblib files)



#### 4.Saved Models



1. Trained model file (joblib/pickle/ONNX/H5 format)
2. Model architecture configuration
3. Hyperparameters used for training
4. Training history logs (loss curves, metrics over epochs)



#### Model Evaluation Results



1. Confusion matrix (visualized)
2. Classification report (precision, recall, F1-score per class)
3. ROC-AUC curves (for binary/multi-class)
4. Cross-validation results
5. Comparison table with baseline models



#### 5.Exploratory Data Analysis (EDA) Plot

## ***(MINIMUM 8 FIGURES)***



1. Feature distributions (histograms/boxplots)
2. Correlation heatmap
3. Class imbalance visualization
4. Missing values analysis



#### Model Performance Visualizations



1. Training/validation loss curves
2. Accuracy/precision/recall curves
3. ROC curves and AUC scores
4. Confusion matrix heatmaps
5. Feature importance plots (for tree-based models)
6. t-SNE/PCA visualizations (for clustering)



#### Comparative Analysis Charts



1. Bar chart comparing different algorithms
2. Performance vs complexity trade-off plot
3. Inference time comparison table



#### 6.Sample Predictions



1. **At least 10 test cases with predictions**
2. **Explanation of predictions (SHAP/LIME values if applicable)**
3. **False positive/negative case analysis**



#### 7.Presentation Materials



&#x20;***Slide Deck (15-20 slides)***



1. **Title slide with team members**
2. **roblem introduction**
3. **Related work/literature**
4. **Methodology overview**
5. **System architecture diagram**
6. **Implementation details**
7. **Experimental results**
8. **Live demo screenshots/video**
9. **Challenges faced and solutions**
10. **Future improvements**
11. **Q\&A preparation**



#### 8\. Additional Deliverables (Bonus Items)



1. Docker container for reproducible environment
2. API documentation (Swagger/OpenAPI)
3. CI/CD pipeline configuration
4. Model deployment on cloud platform
5. Research paper draft (IEEE format)
6. Video tutorial explaining the project





# Evaluation Rubric (Total: 20 Points)



**Category 1: Problem Definition \& Motivation (1.5 points)**



**Checklist:**



1. Problem statement is specific and measurable
2. Real-world cybersecurity application identified
3. Current challenges/gaps explained
4. Target users/stakeholders defined
5. Success criteria clearly stated



**Category 2: Literature Review \& Related Work (1.5 points)**



**Checklist:**

&#x09;

1. Minimum 5 academic papers cited (10 for full marks)
2. Papers from reputable venues (IEEE, ACM, Springer, etc.)
3. Published within last 5 years (with exceptions for foundational work)
4. Critical comparison table of methods included
5. Clear explanation of how project advances state-of-the-art



**Category 3: Dataset Selection \& Preprocessing (2.5 points)**



**Checklist:**



1. Dataset source properly attributed and licensed
2. Dataset size and characteristics documented
3. Exploratory analysis includes: distributions, correlations, missing values, outliers
4. Class imbalance identified and addressed (SMOTE, undersampling, etc.)
5. Feature scaling/normalization applied appropriately
6. Train/validation/test split ratio justified (e.g., 70/15/15)
7. Data leakage prevention measures taken
8. All preprocessing steps reproducible



**Category 4: Feature Engineering (2.5 points)**



**Checklist:**

&#x09;

1. Domain knowledge incorporated into features
2. New features created (not just raw dataset columns)
3. Categorical features properly encoded (one-hot, label, target encoding)
4. Temporal/spatial features extracted if applicable
5. Feature selection method applied (mutual info, chi-square, RFE, etc.)
6. Number of final features documented and justified
7. Feature correlation with target analyzed
8. Ablation study showing impact of feature sets on performance



**Category 5: Model Selection \& Implementation (3 points)**



**Checklist:**



1. Minimum 3 different ML algorithms implemented (5 for excellent)
2. At least one deep learning model (if applicable to problem)
3. Baseline model included (dummy classifier, simple threshold)
4. Hyperparameter search space defined and executed
5. Cross-validation strategy appropriate (stratified k-fold, time-series split)
6. Regularization used (L1/L2, dropout, early stopping)
7. Ensemble methods considered (voting, stacking, boosting)
8. Model complexity vs performance trade-off analyzed



**Category 6:(unnamed)(unknown)**



**Checklist:**

&#x09;

1. Metrics appropriate for dataset (imbalanced? multi-class? multi-label?)
2. Confusion matrix included and interpreted
3. Precision, Recall, F1-score reported per class
4. ROC-AUC curve (binary) or macro/micro averages (multi-class)
5. Cross-validation results with mean and std deviation
6. Statistical significance tested (t-test, Wilcoxon, etc.)
7. Comparison table with baseline and related work
8. False positive/negative cases analyzed qualitatively
9. Training time and inference time benchmarked
10. Memory footprint and computational cost documented



**Category 7: Code Quality \& Reproducibility (2 points)**



**Checklist:**



1. Code runs without errors from clean environment
2. Functions are small, focused, and single-purpose
3. Meaningful variable and function names used
4. Magic numbers replaced with named constants
5. Configuration separated from code (config files, argparse)
6. Error handling and logging implemented
7. Git repository with minimum 10 commits
8. README with installation and usage instructions
9. requirements.txt or environment.yml included
10. Unit tests for critical functions
11. No hardcoded paths or sensitive information



**Category 8: Technical Report \& Documentation (2 points)**



**Checklist:**

&#x09;

1. Report follows provided template/structure
2. Abstract/executive summary included (250-300 words)
3. Introduction provides sufficient background
4. Methodology section detailed enough for reproduction
5. All figures have captions and referenced in text
6. Tables formatted professionally
7. Mathematical notation properly typeset
8. References in consistent citation style (IEEE recommended)
9. Page numbers, headers, footers included
10. Spell-check and grammar-check performed
11. Appendix includes additional results, code snippets



**Category 9: Presentation \& Communication (1.5 points)**



**Checklist:**



1. Slide design professional and consistent
2. Text on slides minimal (bullet points, not paragraphs)
3. High-quality images and diagrams
4. Demo video or live demo prepared
5. Each member presents approximately equal time
6. Practice session completed before presentation
7. Backup plan for technical difficulties
8. Anticipated questions prepared with answers
9. Presentation within time limit (+/- 1 minute)
10. Clear take-home message delivered



**Category 10: Innovation \& Originality (1 point)**



**Checklist:**



1. Novel contribution clearly identified
2. Approach differs from existing solutions in literature
3. Creative problem-solving demonstrated
4. Unique insights or observations shared
5. Potential for future research or real-world deployment discussed



# Submission Checklist \& Timeline



**Technical Report (PDF)**



1. File name format: TeamName\_TechnicalReport.pdf
2. *Page limit: 25 pages maximum (excluding appendix)*



**Source Code Repository (GitHub/GitLab link)**



1. Public or private with instructor access
2. Contains all source code, notebooks, and configurations
3. Includes README with setup instructions



**Presentation Slides (PDF or PowerPoint)**



1. File name format: TeamName\_Presentation.pptx/pdf
2. *15-20 slides maximum*



**Demo Video or Live Demo Link**



1. YouTube (unlisted) or recorded video file
2. 5-10 minutes maximum duration
3. Shows working system with sample predictions



**Trained Model Files**



1. Uploaded to cloud storage or included in repo
2. Include model architecture and training config



**Dataset Documentation**



1. Data dictionary
2. Preprocessing scripts
3. Train/test split files (if dataset not public)



**Team Contribution Statement**



1. Each member's contributions listed
2. Signed by all team members
3. Percentage of effort distribution

