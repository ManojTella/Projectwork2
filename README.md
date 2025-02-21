## Exploration of Obesity Risk Factors with explainable AI
### Abstract
Obesity has become a progressive global epidemic with great increase in chronic disease burdens such as diabetes, cardiovascular disorders, and cancer. Despite the methodological improvement in predictive modelling, there is still a major challenge to properly understand the complex interplay of factors responsible for obesity risk. Explainable Artificial Intelligence (XAI) makes the decision making process of machine learning models more understandable and transparent in the context of obesity risk assessments. The study majorly focuses on analyzing key factors which includes genetic markers, dietary practices, physical activity levels, psychological aspects, environmental conditions and socioeconomic conditions. Explainable AI methods such as SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-Agnostic Explanations), and counterfactual explanations to describe the factors contributing the risk predictions of obesity on both population and individual levels.
The ability to provide the predictability of these models with explainability would find application by enabling the healthcare professionals to formulate risk-reduction strategies for various populations, policymakers to engrave plans for public health, and individuals making informed lifestyle choices. The involvement of AI not only deserves a well-deserved confidence in the field of AI-driven systems but also sets the ground for an inclusive, ethical, and impactful application of technology to fight fatness.

### Introduction
    <p>Obesity is a global health challenge, contributing to numerous chronic diseases and increasing healthcare costs. Identifying its underlying risk factors is crucial for prevention and management. This project leverages Explainable AI (XAI) to analyze and interpret complex data, providing clear insights into key factors influencing obesity.</p>
    By combining advanced machine learning models with interpretability techniques like SHAP and LIME, we aim to enhance understanding, support informed decision-making, and guide effective interventions for better public health outcomes.



### Statement of the Problem
Challenge: 
Obesity is not caused by a single factor but results from interactions between genetics, diet, physical activity, stress levels, sleep patterns, and socioeconomic status.
Many datasets lack diversity, meaning models trained on them may not generalize well to different populations.
Objective: 
Explainable AI (XAI) techniques like SHAP and LIME. It focuses on making AI models interpretable so that healthcare professionals can trust and use them in decision-making. Ultimately, the goal is to support obesity prevention and personalized healthcare interventions.


### Scope of the Project
Scope: 
Use machine learning techniques to determine which factors contribute most to obesity.
Create a decision-support tool or dashboard for healthcare professionals.
Implement Explainable AI (XAI) techniques like SHAP and LIME to interpret AI predictions.
Target Audience: Medical professionals, healthcare institutions.
Deliverables:
A trained and evaluated machine learning model.
AI Model for Obesity Risk Prediction
Explainability and Interpretability Results.

Exploration of Obesity Risk Factors with Explainable AI
Inclusions:
Data collection from medical records.
Introduction to Obesity and AI in Healthcare
Machine Learning Model for Risk Prediction ( Decision Trees, Random Forest)
Explainable AI (XAI) Implementation
Exclusions:
No Surgical or Pharmaceutical Interventions
No Real-Time Patient Monitoring

Data:
The dataset includes BMI, age, gender, dietary habits, physical activity levels, medical history, and socioeconomic factors that contribute to obesity risk.
Data may come from public health databases, medical records, surveys, or research studies on obesity.
Limitations:
The accuracy of AI predictions depends on the quality and completeness of the dataset.
Limited access to real-world medical records and genetic data may impact the model’s performance.
While Explainable AI improves transparency, some interpretable models (e.g., Decision Trees) may have lower accuracy than complex deep learning models.
The project uses pre-collected datasets, meaning it does not provide real-time obesity risk tracking through wearables or continuous monitoring systems.


### Methodology
#### 1. Data Collection & Preprocessing
Obtain a dataset containing demographic, lifestyle, and medical attributes related to obesity risk. Convert categorical variables into numerical form, handle missing values, and normalize numerical features for consistency. Divide the dataset into training (70-80%) and testing (20-30%) sets to ensure model generalization.
#### 2. Model Selection & Training
Train basic classifiers (e.g., Logistic Regression, Decision Trees) for comparison. Train an XGBoost classifier due to its ability to handle non-linearity, feature interactions, and imbalanced data. Optimize the XGBoost model using Grid Search or Bayesian Optimization for better performance. Assess model performance using Weighted F1-score, Precision, Recall, and Accuracy.
#### 3. Explainability & Feature Importance Analysis
Identify key risk factors influencing obesity classification using global interpretability methods like Permutation Feature Importance (PFI) to measure feature impact by shuffling values and observing model performance change. Use SHAP (Shapley Additive Explanations) to quantify the contribution of each feature to individual predictions and visualize feature effects using SHAP summary plots and dependence plots. Apply LIME (Local Interpretable Model-agnostic Explanations) to generate interpretable local feature contributions, helping explain individual predictions and suggesting personalized recommendations.
#### 4. Insights & Recommendations
Analyze top contributing features to obesity risk, such as age, vegetable intake, physical activity, and meal frequency. Use SHAP dependency plots to explore non-linear feature interactions and how they influence.obesity risk. Extract actionable insights from LIME explanations to provide personalized lifestyle recommendations. Identify and mitigate biases in the model to ensure fair and ethical AI-driven decision-making.
#### 5. Validation & Deployment
Validate the explainability methods by comparing SHAP and LIME interpretations with domain expertise from nutritionists or healthcare professionals. Deploy the model with integrated explainability features, allowing real-time obesity risk assessment with interpretable feedback for users. Continuously update the model with new data and refine explanations to enhance accuracy and relevance over time.

### Architecture Diagram/Flow
![image](https://github.com/user-attachments/assets/bff533ee-8547-4f45-8ea5-6d1f0408a314)

### Design-Use Case Diagram

![image](https://github.com/user-attachments/assets/aa453673-9b51-4f03-ac29-fa5d21dfb90b)

![image](https://github.com/user-attachments/assets/d5b72597-f9ac-448c-a9c0-d58fb0d7221d)


### Sequence Diagram - ecommerce example

![image](https://github.com/user-attachments/assets/8aa8ba29-65bf-4854-bb7b-c4934b6bac50)

### Algorithms used
The project uses following Algorithms:
Logistic Regression:
Logistic Regression is a very widely used machine learning algorithm to solve binary and multi-class classification type problems. Logistic Regression has been utilized in this research to forecast obesity likelihood with respect to different lifestyle, demographic, and medical parameters. Logistic Regression determines the likelihood of a subject belonging to different classes of obesity by applying a sigmoid function to convert the input features to a probability ranging from 0 to 1.


Preprocessing procedures such as feature scaling, one-hot encoding, and missing values are done under the training application preprocessing of the Logistic Regression model, to run optimally. The model is placed in a data set where labeled data contributes to the severity of obesity as the response variable.
Model performance is gauged through metrics like accuracy, precision, recall, and F1-score. Furthermore, Explainable AI (XAI) methods like SHAP values are being used to explain the prediction of the model and identify the highest-performing features in obesity risk prediction. Logistic Regression is a good baseline model in the current research despite its simplicity because of its interpretability and linearity in describing linear relationships between predictors of obesity risk.
The F1 score achieved for logistic regression is 0.54.

The project uses following Algorithms:
Random Forest:
Random Forest is a robust ensemble learning technique that enhances the predictive capability by constructing many decision trees and aggregating their predictions. It minimizes overfitting, enhances generalization, and provides a more robust model for classification by aggregating the predictions of multiple trees. In classifying the obesity risk, Random Forest is a key element in using numerical and categorical variables such as Age, BMI, and Diet to classify patients based on their obesity risk factors.


Its use in this research produced a better weighted F1 score of 0.683 that proved its capacity to deal with complex relationships in the data. The model is useful as it can automatically incorporate feature interactions, which makes it appropriate for datasets having a combination of discrete lifestyle variables (smoking, calorie consumption, and exercise) and continuous health measures (weight, BMI, and age). Moreover, bootstrap aggregation (bagging) reduces variance and allows the model to provide stable and consistent predictions. One of the largest benefits of using Random Forest for Explainable AI (XAI) is its ability to give feature importance scores that are useful in interpretation on which features have a large impact in driving the risk for obesity.


The results showed that high-calorie consumption, inactivity, and family history of overweight were the strongest predictors, followed by weight and BMI. With this explainable approach, clinicians can better understand the drivers of obesity classification, leading to more personalized interventions and prevention. Generally speaking, Random Forest application in the classification of obesity risk not only improves predictive power but also allows for an understandable and interpretable model to use in the selection of high-risk individuals. As both physiological and behavioral characteristics are included, it is possible with this method to make data-driven decisions regarding obesity risk, thus opening the doors to improved surveillance of health as well as targeted intervention programs.


### Implementation
Extract the confusion matrix from the model's predictions.
Compute total samples, true positives (diagonal), false positives (column sums - diagonal), and false negatives (row sums - diagonal).
Calculate accuracy = (sum of diagonal) / (total samples).
Calculate precision = TP / (TP + FP) and recall = TP / (TP + FN) for each class.
Report macro-averaged and class-specific metrics for a comprehensive performance overview.

### Output
The following bar plot visualize the permutation feature importance which compares the relation among all the features and measures the important feature.
![image](https://github.com/user-attachments/assets/84cd91dc-b1fa-4e7f-8616-3976a239a660)
![image](https://github.com/user-attachments/assets/e5467ea4-8766-4dbd-8ec1-b4e7bfc00ca8)
Final output for a sample input values.
![image](https://github.com/user-attachments/assets/b313e6ab-5795-4f5f-9cda-e39f5772e607)

### Results
Machine learning
Accuracy: 90%
Precision: 93%
Recall: 92%
Inferences: Effective predictions, faster diagnosis.
![image](https://github.com/user-attachments/assets/5accec1c-6234-4746-bef7-873a3345bd60)

### Conclusion
          The exploration of obesity risk factors using explainable AI has demonstrated the potential of integrating machine learning with interpretability methods to not only predict obesity risk accurately but also to elucidate the underlying factors driving these predictions. By modeling the obesity risk score as a function of diverse risk factors—including genetic markers, dietary habits, physical activity levels, psychological aspects, environmental conditions, and socioeconomic factors—we have shown that complex interactions can be effectively captured using advanced machine learning algorithms.

          
      The use of explainable AI techniques such as SHAP, LIME, and permutation feature importance has been instrumental in providing transparent insights into the decision-making process of the model. These methods enable us to pinpoint the most influential variables, thereby offering actionable insights that can inform personalized interventions and public health strategies. Moreover, the framework established through this work paves the way for improved trust and collaboration between healthcare professionals and AI systems.

### Future Work
Enhance the dataset by incorporating more diverse and longitudinal data from different geographic regions, socioeconomic backgrounds, and healthcare settings. This will improve model generalization and capture evolving trends in obesity risk factors.
Explore incorporating multi-modal inputs such as wearable device data, electronic health records, and real-time lifestyle data. Integrating these sources could reveal additional patterns and interactions among risk factors.
Investigate the use of advanced machine learning methods, such as deep learning architectures and ensemble approaches, to capture complex nonlinear relationships. Comparing these with traditional models can help identify the best trade-off between performance and interpretability.
Develop and test more sophisticated explainability tools that better capture feature interactions and provide intuitive, actionable insights for clinicians. This could involve refining techniques like SHAP and LIME or exploring novel methods tailored to healthcare applications.

### References
Hruby, A., & Hu, F. B. (2015). The Epidemiology of Obesity: A Big Picture. Pharmacoeconomics, 33(7), 673-689.

Bray, G. A., Kim, K. K., & Wilding, J. P. H. (2017). Obesity: a chronic relapsing progressive disease process: a position statement of the World Obesity Federation. Obesity Reviews, 18(7), 715-723.

Afshin, A., Forouzanfar, M. H., Reitsma, M. B., et al. (2017). Health effects of overweight and obesity in 195 countries over 25 years. New England Journal of Medicine, 377(1), 13-27.

Khera, A. V., Chaffin, M., Wade, K. H., et al. (2019). Polygenic prediction of weight and obesity trajectories from birth to adulthood. Cell, 177(3), 587-596.

Dugan, T. M., Mukherjee, S., Carroll, A., & Downs, S. (2015). Machine learning techniques for prediction of early childhood obesity. Applied Clinical Informatics, 6(3), 506-520.

Wang, Y., Min, J., & Khuri, J. (2017). A systematic examination of the association between obesity and diabetes: Machine learning applications. Translational Metabolic Syndrome Research, 1(1), 16-25.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why should I trust you? Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144.








