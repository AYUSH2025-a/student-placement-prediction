# Student Placement Prediction Using Machine Learning


## 📌Project Overview -

Student Placement Prediction is a Machine Learning project that aims to predict whether a student is likely to be placed based on academic performance, technical skills, internships, projects, aptitude scores, communication skills, and other relevant factors.

The project uses historical student data to identify patterns that influence placement outcomes. The data is explored and preprocessed before developing machine learning models for classification. The initial model developed in Week 3 is Logistic Regression, which predicts whether a student is Placed or Not Placed.

The project also aims to evaluate model performance using standard classification metrics and identify important factors that may contribute to successful placement outcomes. The system is intended to provide useful insights for students and educational institutions to improve placement readiness and support data-driven decision-making.


## 🎯 Objectives -

1. Develop a machine learning model to predict whether a student is likely to be placed.
2. Analyze academic, technical, internship, project, aptitude, and skill-related factors affecting placement.
3. Clean and preprocess the student placement dataset.
4. Perform Exploratory Data Analysis (EDA) to identify patterns and relationships in the data.
5. Compare machine learning algorithms and identify an effective model.
6. Evaluate the model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
7. Provide useful insights that can help students improve their placement readiness.
8. Support educational institutions in identifying students who may require additional training and guidance.


## 📊 Dataset -

The dataset used in this project is the **Campus Placement Prediction Dataset**, obtained from Kaggle. It contains historical information about students and their placement outcomes. Each row represents an individual student, while the columns represent different academic, skill-related, and professional attributes.

### Key Features

- Gender
- SSC Marks
- HSC Marks
- Degree Percentage
- Work Experience
- Employability Test Score
- MBA Percentage
- Specialization
- Salary
- Placement Status

The dataset contains both numerical and categorical features. During the data exploration and preprocessing stage, the dataset was checked for missing values, duplicate records, and data-type inconsistencies.

The **Placement Status** column is the target variable, with two possible outcomes: **Placed** and **Not Placed**.

The dataset is used for Exploratory Data Analysis (EDA), data preprocessing, feature preparation, machine learning model training, and evaluation.


## 🛠️ Technologies Used -

- **Python** – Programming language used for data analysis and machine learning.
- **Pandas** – Used for data loading, manipulation, and analysis.
- **NumPy** – Used for numerical operations and data processing.
- **Matplotlib** – Used for creating data visualizations during EDA.
- **Scikit-learn** – Used for data preprocessing, Logistic Regression, Dummy Classifier, train-test split, model evaluation, and cross-validation.
- **Jupyter Notebook / Google Colab** – Used for developing and running the project.
- **Git & GitHub** – Used for version control and project documentation.


## 🔄 Project Workflow -

The project follows the following Machine Learning workflow:

Dataset Collection
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Train-Test Split
        ↓
Logistic Regression Model
        ↓
Model Prediction
        ↓
Model Evaluation
        ↓
Dummy Classifier Comparison
        ↓
5-Fold Cross-Validation
        ↓
Future Model Improvement


## 📅 Weekly Progress -

### Week 1 – Project Proposal

- Finalized the project topic: Student Placement Prediction Using Machine Learning.
- Defined the problem statement and project objectives.
- Identified the factors that may influence student placement.
- Defined the project scope and planned methodology.
- Identified the required technologies and dataset.
  

### Week 2 – Data Exploration and Preprocessing

- Explored the Campus Placement Prediction dataset.
- Analyzed the dataset structure and features.
- Analyzed placement-status distribution.
- Analyzed CGPA distribution.
- Performed correlation analysis.
- Checked missing values and duplicate records.
- Verified data types.
- Performed categorical encoding.
- Identified non-predictive features.
- Identified potentially important features for placement prediction.


### Week 3 – Initial Model Development

- Developed the initial Logistic Regression model.
- Created a preprocessing pipeline using `ColumnTransformer` and `Pipeline`.
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features.
- Split the dataset into 80% training and 20% testing data.
- Used `random_state = 42` and stratification.
- Added a Dummy Classifier as a baseline.
- Evaluated the model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Performed 5-Fold Cross-Validation.


## 🤖 Machine Learning Model -

### Logistic Regression

Logistic Regression is the initial machine learning model used in this project. It was selected because Student Placement Prediction is a binary classification problem, where the expected outcome is either **Placed** or **Not Placed**.

The model is trained using the preprocessed student placement data. Numerical features are standardized using `StandardScaler`, while categorical features are converted into numerical representations using `OneHotEncoder`.

A `ColumnTransformer` and `Pipeline` are used to organize the preprocessing and model training steps.

### Baseline Model

A **Dummy Classifier** is also used as a baseline benchmark. Its performance is compared with Logistic Regression to determine whether the Logistic Regression model is learning useful patterns from the student data.

### Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- 5-Fold Cross-Validation



## 📈 Results

## 🚀 Future Work

## 📁 Repository Structure -


student-placement-prediction/
│
├── README.md
│
├── data/
│   └── cleaned_student_placement.csv
│
├── notebooks/
│   ├── Week_2_EDA_Preprocessing.ipynb
│   └── Week_3_Initial_Model.ipynb
│
├── results/
│   ├── placement_distribution.png
│   ├── cgpa_distribution.png
│   ├── correlation_heatmap.png
│   └── confusion_matrix.png
│
├── reports/
│   ├── Week_1_Proposal.pdf
│   ├── Week_2_Report.pdf
│   └── Week_3_Report.pdf
│
└── requirements.txt
