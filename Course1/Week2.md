# Introduction to Regression

1. **Regression Overview**: It's a method used to predict a continuous dependent variable (Y) based on one or more independent variables (X).
2. **Types of Regression**:
    - **Simple Regression**: Uses one independent variable to predict the dependent variable. Example: Predicting CO2 emissions from engine size.
    - **Multiple Regression**: Uses multiple independent variables to predict the dependent variable. Example: Predicting CO2 emissions from engine size and the number of cylinders.
3. **Applications**: Regression can be applied in various fields such as:
    - **Sales Forecasting**: Predicting sales based on variables like age, education, and experience.
    - **Psychology**: Estimating individual satisfaction from demographic and psychological factors.
    - **Real Estate**: Predicting house prices based on size, number of bedrooms, etc.
    - **Employment Income**: Estimating income from hours worked, education, occupation, etc.
4. **Regression Algorithms**: Different algorithms are suited to different types of regression problems, whether linear or non-linear.

This foundation will be useful as you dive deeper into specific regression techniques and their applications!

# Simple Linear Regression

### **Linear Regression Overview**

1. **Purpose**:
    - Predict a continuous value (e.g., CO2 emissions) using one or more independent variables (e.g., engine size).
2. **Types**:
    - **Simple Linear Regression**: Uses one independent variable.
    - **Multiple Linear Regression**: Uses more than one independent variable.
3. **Process**:
    - **Data Plotting**: Plot independent vs. dependent variables to visualize their relationship.
    - **Fitting a Line**: Use a line to model the relationship between variables. This line is defined by the equation \( y = \theta_0 + \theta_1 x_1 \), where \( \theta_0 \) is the intercept and \( \theta_1 \) is the slope.
4. **Model Fitting**:
    - **Objective**: Minimize the Mean Squared Error (MSE) between predicted and actual values.
    - **Finding Parameters**: Use mathematical or optimization methods to estimate \( \theta_0 \) and \( \theta_1 \) to minimize the error.
5. **Prediction**:
    - Once parameters are estimated, use the model to predict new values by plugging inputs into the equation.
6. **Advantages**:
    - Simple and fast.
    - Easy to understand and interpret.
    - No need for extensive parameter tuning.

### **Example Calculation**

- For a car with an engine size of 2.4, the prediction formula is:

$\text{CO2Emission} = 92.94 + 43.98 \times \text{EngineSize}$
- Substituting EngineSize = 2.4:
$\text{CO2Emission} = 92.94 + 43.98 \times 2.4 = 198.492$

This video is an excellent starting point for understanding and applying linear regression to predict continuous variables.

# Model Evaluation in Regression Models

1. **Purpose**:
    - Assess the accuracy and reliability of a regression model to ensure it performs well on unknown data.
2. **Evaluation Approaches**:
    - **Train and Test on the Same Dataset**:
        - **Process**: Use the entire dataset for training and then test on a portion of it.
        - **Pros**: Simple to implement.
        - **Cons**: High risk of overfitting, where the model performs well on known data but poorly on new, unseen data.
        - **Training Accuracy**: Measures correct predictions on the training dataset.
        - **Out-of-Sample Accuracy**: Measures correct predictions on unseen data. This method usually has low out-of-sample accuracy due to overfitting.
    - **Train/Test Split**:
        - **Process**: Split the dataset into separate training and testing sets. Train the model on the training set and evaluate on the testing set.
        - **Pros**: More realistic evaluation as the test data is not seen by the model during training.
        - **Cons**: Results can vary depending on how the dataset is split. May still suffer from high variability in evaluations.
3. **K-Fold Cross-Validation**:
    - **Process**: Divide the dataset into $K$  folds. Train and test the model $K$  times, each time using a different fold for testing and the remaining $K-1$ folds for training. Average the results to obtain a more consistent measure of accuracy.
    - **Pros**: Reduces variability in the evaluation and provides a more reliable measure of model performance.
    - **Cons**: More complex to implement and computationally intensive.

# Evaluation metrics in Regression Models

### **Accuracy Metrics for Model Evaluation**

1. **Error Definition**:
    - The error in a regression model is the difference between actual data points and the values predicted by the model.
2. **Evaluation Metrics**:
    - **Mean Absolute Error (MAE)**:
        - **Definition**: Average of the absolute values of the errors.
        - **Interpretation**: Provides a straightforward measure of average error without giving extra weight to larger errors.
        - 
        
        $$
        \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
        $$
        
        - where:
            - $y_i$ is the actual value,
            - $\hat{y}_i$ is the predicted value,
            - $n$ is the number of observations.
    - **Mean Squared Error (MSE)**:
        - **Definition**: Average of the squared errors.
        - **Interpretation**: Emphasizes larger errors due to the squaring of the error terms, making it more sensitive to outliers.
        - 
        
        $$
        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        $$
        
        - where:
            - $y_i$ is the actual value,
            - $\hat{y}_i$ is the predicted value,
            - $n$ is the number of observations.
    - **Root Mean Squared Error (RMSE)**:
        - **Definition**: Square root of the Mean Squared Error.
        - **Interpretation**: Provides error measurement in the same units as the response variable, making it easier to interpret.
        - 
        
        $$
        \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
        $$
        
        - where:
            - $y_i$ is the actual value,
            - $\hat{y}_i$ is the predicted value,
            - $n$ is the number of observations.
    - **Relative Absolute Error (RAE)**:
        - **Definition**: Ratio of the total absolute error to the total absolute error of a simple predictor (e.g., mean value of Y).
        - **Interpretation**: Normalizes the error, providing a measure of the model’s performance relative to a simple baseline.
        - 
        
        $$
        \text{RAE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}i|}{\sum{i=1}^{n} |y_i - \bar{y}|} 
        $$
        
        - where
            - $y_i$  is the actual value,
            - $\hat{y}_i$ is the predicted value,
            - $\bar{y}$  is the mean of the actual values,
            - $n$ is the number of observations.
    - **Relative Squared Error (RSE)**:
        - **Definition**: Similar to RAE but focuses on squared errors.
        - **Interpretation**: Often used in calculating $R^2$, which indicates how well the model fits the data.
        - 
        
        $$
         \text{RSE} = \frac{\sum_{i=1}^{n} (y_i - \hat{y}i)^2}{\sum{i=1}^{n} (y_i - \bar{y})^2}
        $$
        
        - where
            - $y_i$  is the actual value,
            - $\hat{y}_i$ is the predicted value,
            - $\bar{y}$  is the mean of the actual values,
            - $n$ is the number of observations.
    - **R-Squared (**$R^2$**)**:
        - **Definition**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
        - **Interpretation**: A higher $R^2$ indicates a better fit of the model to the data.
        
        $$
        R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}i)^2}{\sum{i=1}^{n} (y_i - \bar{y})^2}
        $$
        
        - where
            - $y_i$  is the actual value,
            - $\hat{y}_i$ is the predicted value,
            - $\bar{y}$  is the mean of the actual values,
            - $n$ is the number of observations.

### **Choosing Metrics**

- The choice of metric depends on:
    - The type of model.
    - The nature of the data.
    - Domain-specific requirements.

### **Summary**

- **MAE**: Easy to understand, measures average magnitude of errors.
- **MSE**: Highlights larger errors, less robust to outliers.
- **RMSE**: Provides error in the same units as the response variable.
- **RAE** and **RSE**: Normalize errors to compare against simple models.
- **\( R^2 \)**: Indicates the proportion of variance explained by the model.

Each metric provides different insights into the model's performance and should be chosen based on the context of the problem.

# Multiple Linear Regression

### Types of Linear Regression Models

There are two main types of linear regression models:

1. **Simple Linear Regression**: Uses one independent variable to predict a dependent variable. Example: Predicting CO₂ emissions using engine size.
2. **Multiple Linear Regression**: Uses multiple independent variables to predict a dependent variable. Example: Predicting CO₂ emissions using engine size and the number of cylinders.

### Introduction to Multiple Linear Regression

Multiple linear regression extends simple linear regression by incorporating multiple independent variables to predict a dependent variable. It's essential for situations where multiple factors influence the outcome. For instance:

- **Example**: Predicting CO₂ emissions using engine size, number of cylinders, and fuel consumption.

### Applications of Multiple Linear Regression

Multiple linear regression can address two key types of problems:

1. **Identifying the Strength of Effects**: Determining how various factors (like revision time, test anxiety, lecture attendance, and gender) affect student exam performance.
2. **Predicting Changes**: Understanding how changes in independent variables affect the dependent variable. For example, predicting changes in blood pressure with variations in BMI.

### Model Representation

In multiple linear regression, the target variable \( Y \) is a linear combination of independent variables  $X :
 \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$ 
This can also be represented in vector form as  $\theta^T X$ , where $\theta$ is the parameters vector and $X$ is the feature set vector.

### Estimating Parameters

To make accurate predictions, we need to find the optimal parameters $\theta$. The process involves:

1. **Calculating Errors**: For each prediction, calculate the error as the difference between the predicted and actual values.
2. **Minimizing Mean Squared Error (MSE)**: The goal is to minimize the MSE, which is the mean of the squared errors.

### Methods to Estimate Parameters

Two common methods for parameter estimation are:

1. **Ordinary Least Squares (OLS)**: Uses linear algebra to minimize MSE, suitable for datasets with fewer than 10,000 rows.
2. **Optimization Algorithms**: Techniques like gradient descent, which iteratively minimize error, are suitable for larger datasets.

### Prediction Phase

Once the parameters are estimated, predictions can be made by plugging in the independent variable values into the model equation. For instance:
$\text{CO}_2 \text{ emission} = 125 + 6.2 \times \text{engine size} + 14 \times \text{cylinders} + \ldots$ 

### Addressing Common Concerns

1. **Number of Independent Variables**: Using too many variables without justification can lead to overfitting, making the model less generalizable.
2. **Incorporating Categorical Variables**: Categorical variables can be included by converting them into numerical values (e.g., dummy coding).
3. **Linearity Requirement**: Multiple linear regression requires a linear relationship between the dependent and each independent variable. Scatter plots can help check for linearity.

### Conclusion

Multiple linear regression is a robust method for predicting continuous variables using multiple predictors. By carefully selecting and evaluating predictors, you can build effective models for complex real-world problems.