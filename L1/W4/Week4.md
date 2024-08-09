# Logistic Regression

## Intro to Logistic Regression

**Introduction:**

*   Logistic Regression: A statistical and machine learning technique used for classifying records in a dataset based on the values of input fields.
*   Main Focus: Predicting a categorical (binary or multi-class) target variable, unlike linear regression, which predicts a continuous variable.

**Key Concepts:**

*   **Independent Variables (X):** Used to predict the outcome. These should be continuous, but categorical variables can be used if dummy or indicator coded.
*   **Dependent Variable (Y):** The outcome variable, typically binary (e.g., churn/no churn, yes/no, 0/1).

**Applications of Logistic Regression:**

*   Predicting customer churn in a telecommunications dataset.
*   Predicting medical outcomes (e.g., heart attack probability, disease presence).
*   Predicting marketing outcomes (e.g., likelihood of a purchase).
*   Predicting the failure probability of a process or system.
*   Predicting mortgage default probability.

**Characteristics of Logistic Regression:**

*   **Binary Classification:** Logistic regression is commonly used for binary outcomes but can handle multi-class classification.
*   **Probability Estimation:** Provides a probability score between 0 and 1, representing the likelihood of the outcome.
*   **Decision Boundary:** The decision boundary is linear (a line, plane, or hyperplane). Logistic regression can handle non-linear boundaries with polynomial features (out of scope).
*   **Feature Impact:** Coefficients (Theta values) show the impact of each feature on the prediction. Features with coefficients close to zero have less impact.

**Mathematical Formulation:**

*   **Dataset (X):** m x n matrix where m is the number of features, and n is the number of records.
*   **Target (Y):** The binary outcome to predict, coded as 0 or 1.
*   **Model Prediction (Ŷ):** Predicts the probability of Y being 1, where the probability of Y being 0 is calculated as 1 - P(Y=1).

**When to Use Logistic Regression:**

1.  **Categorical/Binary Target Variable:** Especially useful when the target is binary.
2.  **Need for Probability Estimates:** When the probability of an outcome is required.
3.  **Linearly Separable Data:** Works well when the data can be separated by a linear boundary.
4.  **Understanding Feature Impact:** Ideal for understanding the influence of independent variables on the outcome.

**Summary of Important Equations:**

*   **Logistic Function:** Converts the linear regression output to a probability (between 0 and 1).
*   **Decision Rule:** Classify samples based on whether the predicted probability is above or below a threshold (commonly 0.5).

## Logistic Regression vs Linear Regression

#### **Introduction:**
- **Purpose:** Understand the differences between linear regression and logistic regression, and why linear regression is not suitable for binary classification.
- **Key Concepts:** Linear regression predicts continuous outcomes, while logistic regression is used for binary classification by estimating probabilities.

---

#### **Linear Regression Recap:**
- **Use Case:** Predicting continuous values, e.g., predicting income based on age.
- **Equation:** \( y = a + bx_1 \) or \( \Theta^T X = \Theta_0 + \Theta_1 x_1 \)
- **Model:** Fits a line through data points to predict continuous outcomes.
- **Example:** Predicting income using age as the feature.

---

#### **Challenges with Linear Regression for Classification:**
- **Binary Classification Problem:** Predicting a categorical outcome, e.g., customer churn (yes/no).
- **Data Representation:** Age (\( x_1 \)) as the independent variable, churn (0 or 1) as the dependent variable.
- **Linear Regression Limitation:**
  - Fits a line through data, but outputs continuous values.
  - Requires a threshold (e.g., 0.5) to classify outcomes, which isn't ideal.
  - Can output values outside [0, 1], making it unsuitable for probability estimation.

---

#### **Introduction to Logistic Regression:**
- **Goal:** Predict the probability of an outcome belonging to a specific class (e.g., churn = 1).
- **Key Concept:** Uses the **sigmoid function** to map predictions to probabilities between 0 and 1.
- **Sigmoid Function:**
  - **Formula:** \( \text{Sigmoid}(\Theta^T X) = \frac{1}{1 + e^{-\Theta^T X}} \)
  - **Behavior:** 
    - \( \Theta^T X \) large → Sigmoid close to 1.
    - \( \Theta^T X \) small → Sigmoid close to 0.
    - **Output:** Always between 0 and 1, representing probability.

---

#### **Logistic Regression Model:**
- **Probability Interpretation:**
  - **For Class 1:** \( P(y=1|X) = \text{Sigmoid}(\Theta^T X) \)
  - **For Class 0:** \( P(y=0|X) = 1 - \text{Sigmoid}(\Theta^T X) \)
  - **Example:** Predicting the probability of a customer churning based on income and age.

---

#### **Training the Logistic Regression Model:**
1. **Initialize Parameters (\( \Theta \)):** Start with random values.
2. **Model Output:** Calculate \( \text{Sigmoid}(\Theta^T X) \) for each sample.
3. **Calculate Error:** Compare model prediction \( \hat{y} \) with actual label \( y \).
4. **Compute Cost:** Sum errors across all samples to get the model's cost.
   - **Cost Function:** Represents the total error; lower cost indicates a better model.
5. **Optimize Parameters (\( \Theta \)):** Adjust to minimize cost (using methods like **Gradient Descent**).
6. **Repeat Iterations:** Continue until the cost is sufficiently low and model accuracy is satisfactory.

---

#### **Key Concepts to Remember:**
- **Linear Regression:**
  - Best for predicting continuous outcomes.
  - Not suitable for binary classification due to output outside the [0, 1] range.
- **Logistic Regression:**
  - Uses the sigmoid function to model probabilities for binary classification.
  - Effective for predicting binary outcomes and estimating class probabilities.

---

This summary should serve as a comprehensive review of the key points in the video, highlighting the distinctions between linear and logistic regression and the training process of logistic regression.

## Logistic Regression Training

1. **Objective of Logistic Regression:**
   - The goal is to adjust the parameters of the model to best estimate the labels (outcomes) of the samples in the dataset. 

2. **Cost Function:**
   - The cost function measures the difference between the actual values (y) and the predicted values (ŷ) from the model.
   - The general form of the cost function involves the sigmoid function applied to the linear combination of input features and model parameters (θ).
   - For logistic regression, the cost function is derived from the log-likelihood, specifically using the negative log function to handle binary classification (y = 0 or 1).

3. **Optimization with Gradient Descent:**
   - The main challenge is to find the parameters (θ) that minimize the cost function, which in turn minimizes the error in predictions.
   - Gradient descent is an iterative optimization method used to find the minimum of the cost function. 
   - By calculating the gradient (slope) of the cost function, the algorithm updates the parameters in the opposite direction of the gradient to reduce the cost.

4. **Gradient Descent Steps:**
   - **Initialize Parameters:** Start with random values for the parameters.
   - **Compute Cost:** Calculate the cost function with the current parameters.
   - **Calculate Gradients:** Determine the partial derivatives of the cost function with respect to each parameter.
   - **Update Parameters:** Adjust the parameters by subtracting the product of the gradient and the learning rate (μ).
   - **Iterate:** Repeat the process until the cost function reaches a minimum or a set number of iterations is completed.

5. **Learning Rate (μ):**
   - The learning rate controls the size of the steps taken during gradient descent. A larger learning rate results in bigger steps, while a smaller learning rate leads to smaller steps.

6. **Convergence:**
   - The algorithm continues to update the parameters iteratively until the cost function converges to an acceptable minimum value.


# Support Vector Machine

**Introduction to SVM**
- **Definition**: A Support Vector Machine (SVM) is a supervised learning algorithm used for classification tasks. It finds a hyperplane that best separates data into distinct classes.
- **Goal**: To categorize data by finding an optimal separator (hyperplane) that maximizes the margin between different classes.

**How SVM Works**
1. **Feature Mapping**
   - **Transformation**: Maps data to a higher-dimensional space where a linear separator can be used.
   - **Kernelling**: The process of transforming data using a kernel function to make it linearly separable. Common kernel functions include:
     - **Linear**
     - **Polynomial**
     - **Radial Basis Function (RBF)**
     - **Sigmoid**
   - **Choice of Kernel**: Requires experimentation; libraries typically implement various kernel functions.

2. **Finding the Optimal Hyperplane**
   - **Hyperplane**: A decision boundary that separates data points of different classes.
   - **Margin**: The distance between the hyperplane and the closest data points from each class (support vectors). The optimal hyperplane maximizes this margin.
   - **Support Vectors**: Data points closest to the hyperplane. They are crucial for determining the position of the hyperplane.
   - **Mathematics**: Involves optimization procedures that can be solved using gradient descent (not covered in detail).

**Advantages of SVM**
- **Accuracy**: Effective in high-dimensional spaces.
- **Memory Efficiency**: Uses only a subset of training points (support vectors) for the decision function.

**Disadvantages of SVM**
- **Overfitting**: Can occur if the number of features is much larger than the number of samples.
- **No Direct Probability Estimates**: SVM does not provide probability estimates, which are often desired in classification problems.
- **Computational Efficiency**: Can be inefficient for very large datasets (e.g., more than 1,000 rows).

**Applications of SVM**
- **Image Analysis**: Classification of images, handwritten digit recognition.
- **Text Mining**: Spam detection, text classification, sentiment analysis.
- **Gene Expression**: Classification tasks in high-dimensional gene expression data.
- **Other Applications**: Regression, outlier detection, and clustering.

**Summary**
SVM is a powerful classification algorithm that leverages high-dimensional feature space mapping and optimization to find the best separator for data. It is well-suited for tasks with complex and high-dimensional data but may struggle with very large datasets and lack direct probability estimates.
