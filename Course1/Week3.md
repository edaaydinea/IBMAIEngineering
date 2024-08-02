# Introduction to Classification

1. **Definition of Classification**
    - Classification is a supervised learning approach in machine learning used to categorize unknown items into a discrete set of classes.
2. **Goal of Classification**
    - It attempts to establish the relationship between a set of feature variables and a categorical target variable with discrete values.
3. **How Classification Works**
    - Given training data points and corresponding target labels, classification assigns a class label to an unlabeled test case.
4. **Example of Classification: Loan Default Prediction**
    - A bank can use previous loan default data (including information like age, income, and education) to predict the likelihood of new customers defaulting on a loan.
    - The model categorizes customers as "defaulter" or "non-defaulter" (binary classification, represented as 0 or 1).
5. **Binary vs. Multi-class Classification**
    - Binary classification deals with two classes, e.g., loan defaulters vs. non-defaulters.
    - Multi-class classification involves more than two classes, e.g., predicting which of three medications a patient will respond to based on treatment data.
6. **Business Use Cases for Classification**
    - Predicting customer categories, detecting churn (whether a customer will switch providers), and determining customer response to advertising campaigns.
7. **Applications of Classification**
    - Email filtering, speech recognition, handwriting recognition, biometric identification, and document classification.
8. **Types of Classification Algorithms in Machine Learning**
    - Decision Trees
    - Naive Bayes
    - Linear Discriminant Analysis
    - K-Nearest Neighbors
    - Logistic Regression
    - Neural Networks
    - Support Vector Machines (SVM)
9. **Scope of the Course**
    - The course will cover only a few classification algorithms from the list.

# K-Nearest Neighbors

1. **Introduction to K-Nearest Neighbors (KNN)**
    - KNN is a classification algorithm that predicts the class of a new or unknown case based on its similarity to existing labeled cases.
2. **Example Scenario**
    - A telecommunications provider uses demographic data (e.g., region, age, marital status) to predict customer group membership. The target field, `custcat`, has four possible values: Basic Service, E Service, Plus Service, and Total Service.
3. **Prediction Process**
    - For a new customer (e.g., record number eight), the algorithm finds the nearest known cases and assigns the class label based on these neighbors.
4. **Determining Class Label**
    - The class label can be determined by the closest single neighbor (first nearest neighbor) or by taking a majority vote among multiple neighbors (e.g., five nearest neighbors).
5. **K in K-Nearest Neighbors**
    - The parameter K represents the number of nearest neighbors considered when making a prediction.
6. **Definition of K-Nearest Neighbors Algorithm**
    - KNN takes labeled points and uses them to label other points. It classifies cases based on their similarity to other cases, with nearby points considered "neighbors."
7. **Distance and Similarity Measurement**
    - The algorithm measures the distance or dissimilarity between data points. One common method is the Euclidean distance.
8. **Working of KNN Algorithm**
    - The steps are:
        1. Pick a value for K.
        2. Calculate the distance between the new case and each case in the dataset.
        3. Identify the K nearest observations in the training data.
        4. Predict the response based on the most common response among the K neighbors.
9. **Selecting the Correct K**
    - Choosing a low K (e.g., K=1) can lead to overfitting and sensitivity to noise, resulting in a complex model that may not generalize well. A high K value can lead to an overly generalized model. The best K is determined by testing the model's accuracy with different K values.
10. **Application Beyond Classification**
    - KNN can also be used for regression tasks, predicting continuous values by averaging the target values of the nearest neighbors. For example, predicting the price of a house based on features like the number of rooms, square footage, and the year it was built. The predicted price can be the median of the nearest neighbors' prices.
11. **Conclusion**
    - KNN is versatile and can be used for both classification and regression tasks, depending on the nature of the target variable.

# Evaluation Metrics in Classification

1. **Introduction to Evaluation Metrics**
    - Evaluation metrics assess the performance of a model. They provide insights into areas that might require improvement.
2. **Scenario: Customer Churn Prediction**
    - A historical dataset is used to predict customer churn for a telecommunications company. The model's accuracy is calculated using the test set.
3. **Key Evaluation Metrics**
    - **Jaccard Index**: Measures the similarity between the true labels (`y`) and the predicted labels (`y-hat`). Defined as the size of the intersection divided by the size of the union of the two label sets.
    - **F1 Score**: The harmonic mean of precision and recall. It balances both measures, reaching its best value at 1 (perfect precision and recall) and worst at 0.
    - **Log Loss**: Measures the performance of a classifier where the predicted output is a probability value between 0 and 1. It penalizes wrong predictions with higher probabilities more heavily.
4. **Jaccard Index**
    - For a test set of size 10 with 8 correct predictions, the Jaccard index would be 0.66. If the predicted labels perfectly match the true labels, the accuracy is 1.0; otherwise, it is 0.0.
5. **Confusion Matrix**
    - A table showing the correct and incorrect predictions made by a classifier compared to the actual labels.
    - For a binary classification:
        - **True Positives (TP)**: Correctly predicted positive cases.
        - **False Negatives (FN)**: Incorrectly predicted as negative when they are positive.
        - **True Negatives (TN)**: Correctly predicted negative cases.
        - **False Positives (FP)**: Incorrectly predicted as positive when they are negative.
6. **Precision and Recall**
    - **Precision**: Accuracy of the positive predictions. Defined as TP / (TP + FP).
    - **Recall**: True positive rate, defined as TP / (TP + FN).
7. **F1 Score Calculation**
    - Based on the precision and recall of each label. For example, if the precision and recall are balanced, the F1 score will reflect that balance.
8. **Logarithmic Loss (Log Loss)**
    - Used when the classifier outputs probabilities instead of discrete labels. It measures how far the predicted probabilities are from the actual labels. A lower log loss indicates better accuracy.
9. **Average Accuracy and F1 Score**
    - The overall accuracy of a classifier can be summarized by the average F1 score across all labels. For instance, if the F1 scores for Class 0 and Class 1 are 0.83 and 0.55, respectively, the average accuracy is 0.69.
10. **Applicability**
- Both Jaccard index and F1 score can be used for multiclass classifiers, though this application is beyond the scope of this summary.

These metrics are critical for evaluating and improving machine learning models, providing comprehensive insights into their performance and guiding further refinement.

# Introduction to Decision Trees

- **What is a Decision Tree?**
    - A decision tree is a graphical representation of possible solutions to a decision based on certain conditions. It is used for classification and regression tasks.
    - In classification, the tree is used to classify data into distinct categories.
- **How Decision Trees Help in Classification**
    - Decision trees classify data by making a series of decisions based on the values of different attributes.
    - Each internal node of the tree represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.
- **Example Scenario**
    - **Context**: A medical researcher needs to determine which drug (Drug A or Drug B) is appropriate for future patients based on their attributes (age, gender, blood pressure, cholesterol).
    - **Objective**: Build a model to predict which drug to prescribe.
- **Building a Decision Tree**
    - **Starting Point**: Begin with the entire dataset.
    - **Splitting Criteria**: Split the dataset into subsets based on the values of an attribute.
    - **Example**:
        - **Age**: Split into categories like young, middle-aged, and senior.
        - **Decision**: If the patient is middle-aged, prescribe Drug B. For young or senior patients, further splitting is needed based on additional attributes.
        - **Gender**: Further split the dataset by gender, recommending Drug A for females and Drug B for males.
- **Steps to Construct a Decision Tree**
    - **Choose an Attribute**: Select an attribute to test for splitting the data.
    - **Calculate Significance**: Measure the attribute’s importance in separating the data (how well it divides the data into distinct classes).
    - **Split Data**: Based on the best attribute, split the dataset into branches.
    - **Recursive Process**: Repeat the process for each branch with the remaining attributes until all data is classified or no further splitting is necessary.
- **Using the Tree**
    - Once the decision tree is built, it can be used to classify new data points by following the branches of the tree according to the attribute values of the new data.

# Building Decision Trees

1. **Introduction**
    - **Objective**: Learn how to build a decision tree using a dataset.
    - **Example Dataset**: Patient data with attributes such as age, cholesterol, and gender, and a target variable indicating the effectiveness of Drug A or Drug B.
2. **Decision Tree Construction**
    - **Recursive Partitioning**: Decision trees are built using a process called recursive partitioning to classify data.
    - **Goal**: Choose the most predictive feature to split the data, aiming to create the most "pure" nodes (i.e., nodes where data is mostly of one class).
3. **Choosing Attributes for Splitting**
    - **Initial Test**: Start with attributes like cholesterol or gender.
        - **Cholesterol**: May not provide a clear distinction (e.g., high cholesterol doesn’t definitively indicate the best drug).
        - **Gender**: Provides clearer separation (e.g., females might respond better to Drug A, while males might respond better to Drug B).
4. **Attribute Evaluation**
    - **Purity of Nodes**: An attribute is better if it results in nodes that are more pure (i.e., nodes where the data is predominantly of one class).
    - **Entropy**: Measure of impurity or randomness in the data. Entropy is zero if all data in a node belong to one class and one if data is evenly split.
        - **Entropy Calculation**: Use the frequency of each class to calculate entropy using the formula:
        \[
        \text{Entropy} = -\sum (P_i \cdot \log_2(P_i))
        \]
        where \( P_i \) is the proportion of class \( i \) in the node.
5. **Entropy Calculation Example**
    - **Before Splitting**: Calculate entropy for the entire dataset.
    - **After Splitting**: Calculate the weighted average entropy for each split based on the attribute values.
6. **Information Gain**
    - **Definition**: The reduction in entropy (or increase in certainty) after splitting the data using an attribute.
    - **Calculation**:
    \[
    \text{Information Gain} = \text{Entropy (before split)} - \text{Weighted Entropy (after split)}
    \]
    - **Choosing Best Attribute**: The attribute with the highest information gain is chosen for splitting.
7. **Building the Tree**
    - **First Split**: Based on the attribute with the highest information gain (e.g., gender).
    - **Subsequent Splits**: For each branch, continue testing other attributes and split based on the one that provides the highest information gain.
8. **Example Calculation**
    - **Sex Attribute**:
        - Entropy for female patients: 0.98
        - Entropy for male patients: 0.59
        - Information Gain for sex: 0.151
    - **Cholesterol Attribute**:
        - Entropy for normal cholesterol: 0.80
        - Entropy for high cholesterol: 1.0
        - Information Gain for cholesterol: 0.48
    - **Decision**: Choose the attribute with the higher information gain (cholesterol in this example) as the first splitter.
9. **Recursive Process**
    - **Continue Splitting**: Repeat the process for each branch of the tree using remaining attributes.
    - **Build Pure Leaves**: Aim to create leaves where the classification is clear and pure.
10. **Conclusion**
    - Building a decision tree involves selecting the best attributes to split the data based on entropy and information gain, resulting in a tree that classifies data with high accuracy.