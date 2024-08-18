# ML Katas
These katas range from foundational skills to advanced concepts, covering the entire machine learning lifecycle from data preprocessing to deployment and monitoring. They are designed to help you build expertise in practical ML engineering.

## 1. Data Preprocessing Kata
- Objective: Clean and prepare a raw dataset for modeling.
- Task: Given a dataset with missing values, outliers, and mixed data types, write a pipeline to clean, normalize, and encode the data.
- Tools: Pandas, Scikit-Learn
- Bonus: Implement a method to handle categorical variables with a large number of levels.
## 2. Feature Engineering Kata
- Objective: Create new features from existing data.
- Task: Given a dataset, identify and engineer at least five new features that could improve model performance.
- Tools: Pandas, Scikit-Learn, Featuretools
- Bonus: Implement a feature selection technique to evaluate the importance of the engineered features.
## 3. Model Pipeline Kata
- Objective: Build an end-to-end ML pipeline.
- Task: Develop a pipeline that includes data preprocessing, model training, hyperparameter tuning, and evaluation.
- Tools: Scikit-Learn’s Pipeline, GridSearchCV, and cross-validation techniques
- Bonus: Integrate a feature store into the pipeline.
## 4. Model Evaluation Kata
- Objective: Evaluate and compare different models.
- Task: Train multiple models (e.g., Random Forest, SVM, XGBoost) on a dataset and compare their performance using various metrics like accuracy, precision, recall, and AUC-ROC.
- Tools: Scikit-Learn, Matplotlib/Seaborn for visualization
- Bonus: Implement cross-validation and assess the stability of the models.
## 5. Deployment Kata
- Objective: Deploy a trained model to a production environment.
- Task: Train a model and deploy it using a REST API (e.g., Flask) or serverless functions (e.g., AWS Lambda). Set up logging and monitoring for the deployed model.
- Tools: Flask/FastAPI, Docker, AWS Lambda/Azure Functions
- Bonus: Implement A/B testing to evaluate the model in production.
## 6. Model Optimization Kata
- Objective: Optimize a machine learning model.
- Task: Take a basic model and optimize it for both performance and inference speed. Techniques might include hyperparameter tuning, model pruning, or quantization.
- Tools: Scikit-Learn, TensorFlow/PyTorch for model pruning and quantization
- Bonus: Measure the inference time before and after optimization and document the improvement.
## 7. Data Drift Detection Kata
- Objective: Detect and handle data drift in a production model.
- Task: Simulate data drift by altering the distribution of incoming data and implement a method to detect the drift.
- Tools: Pandas, Scikit-Learn, Alibi Detect
- Bonus: Implement an automated alert system when drift is detected.
## 8. Explainability Kata
- Objective: Make model predictions explainable.
- Task: Use explainability tools like LIME or SHAP to explain the predictions of a complex model (e.g., a deep neural network or ensemble model).
- Tools: LIME, SHAP
- Bonus: Create a dashboard that shows model predictions and their explanations.
## 9. Recommender System Kata
- Objective: Build a recommendation system.
- Task: Create a collaborative filtering or content-based recommender system based on user interaction data.
- Tools: Scikit-Learn, TensorFlow/PyTorch, Surprise library
- Bonus: Implement a hybrid recommender system that combines collaborative filtering with content-based recommendations.
## 10. MLOps Kata
- Objective: Set up a complete MLOps pipeline.
- Task: Implement an end-to-end MLOps pipeline that includes version control, CI/CD, model training, deployment, and monitoring.
- Tools: Git, Jenkins/GitHub Actions, Docker, MLflow/Kubeflow
- Bonus: Implement model versioning and rollback strategies.