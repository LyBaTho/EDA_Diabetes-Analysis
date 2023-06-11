"""
| Field                         | Description                                                                           |
|-------------------------------|---------------------------------------------------------------------------------------|
| Pregnancies                   | Whether a patient is pregnant or not                                                  |
| Glucose                       | Refers to elevated blood glucose levels                                               |
| BloodPressure                 | A common comorbidity in diabetes                                                      |
| SkinThickness                 | Certain complications of diabetes can be spotted through skin thickness               |
| Insulin                       | Hormone produced by the pancreas that helps regulate blood sugar levels               |
| BMI                           | BMI metrics of a patient                                                              |
| DiabetesPedigreeFunction      | The likelihood of an individual developing diabetes based on the family history       |
| Age                           | Age of patient                                                                        |

"""

import pandas as pd
import numpy as np
import warnings
import scipy
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings(action = 'ignore')

# Read files

data = pd.read_csv('Diabetes.csv')
data.shape

# Identify variables' data type and convert them if necessary

data.info()

data['Outcome'] = data['Outcome'].astype('category')
data.describe()

"""
Input:
    * Pregnancies: Categorical variable (Ordinal)
    * Glucose: Numeric variable (Discrete)
    * BloodPressure: Numeric variable (Discrete)
    * SkinThickness: Numeric variable (Discrete)
    * Insulin: Numeric variable (Discrete)
    * BMI: Numeric variable (Continuous)
    * DiabetesPedigreeFunction: Numeric variable (Continuous)
    * Age: Numeric variable (Discrete)
  
Output:
    * Outcome: Categorical variable (Ordinal)
"""

# Univariate Analysis - Numerical Variables

num_cols = data.select_dtypes('number').columns.tolist()
print(num_cols)

for column in num_cols:
  print('\n* Column:', column)
  print(len(data[column].unique()), 'unique values')

# Univariate analysis - Continuous variable
def univariate_analysis_continuous_variable(df, feature):
    print("Describe:")
    print(feature.describe(include='all'))
    print("Mode:", feature.mode())
    print("Range:", feature.values.ptp())
    print("IQR:", scipy.stats.iqr(feature))
    print("Var:", feature.var())
    print("Std:", feature.std())
    print("Skew:", feature.skew())
    print("Kurtosis:", feature.kurtosis())

# Number of upper, lower outliers
def check_outlier(df, feature):
    plt.boxplot(feature)
    plt.show()
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    n_O_upper = df[feature > (Q3 + 1.5*scipy.stats.iqr(feature))].shape[0]
    print("Number of upper outliers:", n_O_upper)
    n_O_lower = df[feature < (Q1 - 1.5*scipy.stats.iqr(feature))].shape[0]
    print("Number of lower outliers:", n_O_lower)
    # Percentage of ouliers
    outliers_per = (n_O_lower + n_O_upper)/df.shape[0]
    print("Percentage of ouliers:", outliers_per)
    return Q1, Q3, n_O_upper, n_O_lower, outliers_per

def univariate_visualization_analysis_continuous_variable_new(feature):
    # Histogram
    feature.plot.kde()
    plt.show()      
    feature.plot.hist()
    plt.show()

for con in num_cols:
  print('Variable: ', con)
  univariate_analysis_continuous_variable(data, data[con])
  check_outlier(data, data[con])
  univariate_visualization_analysis_continuous_variable_new(data[con])
  print()

"""
Comment:

    * Pregnancies: No outliers, the most common value is 7, with a negative kurtosis indicating a flat peak and right skewness.
    * Glucose: Has 5 lower outliers, the most common value is 99, with a positive kurtosis indicating a sharp peak and right skewness. The data is concentrated around the range of 100-125.
    * BloodPressure: Has quite a few outliers (~45 outliers), with positive kurtosis and left skewness. The data is concentrated around the value of 80.
    * SkinThickness: Only one outlier, with a negative kurtosis and right skewness. The data is mostly concentrated at the value of 0.
    * Insulin: Has many outliers (~34 outliers), with positive kurtosis and right skewness. The data is primarily centered around 0.
    * BMI: Not too many outliers (~19 outliers), the most common value is 32 (average level), with a distribution close to normal, positive kurtosis, and left skewness.
    * DiabetesPedigreeFunction: Has many outliers (~29 outliers), the most common value is 0.254, with the data concentrated around the range of 0-0.5. Positive kurtosis and right skewness.
    * Age: Relatively few outliers (~9 outliers), mostly falling within the range of 20-30 years old, with positive kurtosis and right skewness.
"""

# Univariate Analysis - Categorical Variables

cat_cols = data.select_dtypes('category').columns.tolist()
print(cat_cols)

for column in cat_cols:
  print('\n* Column:', column)
  print(len(data[column].unique()), 'unique values')

def univariate_analysis_categorical_variable(df, group_by_col):        
    print(df[group_by_col].value_counts())
    df[group_by_col].value_counts().plot.bar(figsize=(5, 6))
    plt.show()

for cat in cat_cols:
  print('Variable: ', cat)
  univariate_analysis_categorical_variable(data, cat)
  print()

"""
Comment:
    
    * Outcome: The number of people without the disease is significantly higher, nearly double (~0.5 times), compared to the number of people with the disease.
"""

# Bivariate Analysis - Numerical - Categorical
    ## Input - Output

# ANOVA 

import statsmodels.api as sm
from statsmodels.formula.api import ols


from statsmodels.stats.weightstats import ztest as ztest

def variables_cont_cat(df, col1, col2):
    
    df_sub = df[[col1, col2]]
    plt.figure(figsize=(5,6))
    sns.boxplot(x=col1, y=col2, data=df_sub, palette="Set3")
    plt.show()
    chuoi = str(col2)+' ~ '+str(col1)
    model = ols(chuoi, data=df_sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA table: ', anova_table)

col1 = 'Outcome'
alpha = 0.05
for i in range(0, len(num_cols)):
    col2 = num_cols[i]
    print('2 variables:', col1, 'and', col2)
    variables_cont_cat(data, col1, col2)
    print()

"""
Comment:

    * Outcome and Pregnancies: have an influence with a P-value (3.495614e-09) < 0.05
    * Outcome and Glucose: have a significant influence with a P-value (8.935432e-43) < 0.05
    * Outcome and BloodPressure: do not have a significant influence with a P-value (0.071514) > 0.05
    * Outcome and SkinThickness: have an influence with a P-value (0.038348) < 0.05
    * Outcome and Insulin: have an influence with a P-value (0.000286) < 0.05
    * Outcome and BMI: have a significant influence with a P-value (1.229807e-16) < 0.05
    * Outcome and DiabetesPedigreeFunction: have a significant influence with a P-value (0.000001) < 0.05
    * Outcome and Age: have a significant influence with a P-value (2.209975e-11) < 0.05

Explanation: The P-value obtained from the ANOVA analysis is significant (P < 0.05), indicating that there is a difference between the groups (Pregnancies, Glucose, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and the types of Outcome (non-disease and disease). 
Except for BloodPressure, where the P-value is > 0.05, suggesting no significant difference.
"""

# Z-TEST (sample size > 30)

'''
Consider the respective Numerical variables in the two groups: 1: Disease & 0: Non-disease of the Outcome.

    * H0: There is no significant difference in the mean of the respective Numerical variables between 1: Disease & 0: Non-disease.
    * H1: There is a significant difference in the mean of the respective Numerical variables between 1: Disease & 0: Non-disease.
'''
from statsmodels.stats.weightstats import ztest

def z_test_loop(data, group_column, value_columns, alpha):

    results = {}
    for column in value_columns:
        group1_data = data[data[group_column] == 0][column]
        group2_data = data[data[group_column] == 1][column]
        z_score, p_value = ztest(group1_data, group2_data, value=group1_data.mean())
        if p_value > alpha:
            result = "Accept the null hypothesis that the means are equal."
        else:
            result = "Reject the null hypothesis that the means are equal."
        results[column] = result
    return results

group_column = 'Outcome'
alpha = 0.05
num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

for i in range(len(num_cols)):
    value_columns = [num_cols[i]]
    results = z_test_loop(data, group_column, value_columns, alpha)
    for column, result in results.items():
        print("Column: {}".format(column))
        print(result)
        print()

"""
Comment:

Because the p-values are smaller than 0.05, we have enough evidence to reject the null hypothesis H0.
"""

# Bivariate Analysis - Numerical - Numerical
    ## Input - Output

for i in range(0, len(num_cols)):
    col1 = num_cols[i]
    for j in range(i+1, len(num_cols)):
        col2 = num_cols[j]
        print('Correlation between 2 variables:', col1, 'and', col2)
        print(data[[col1, col2]].corr())
        print('Pearson Correlation between 2 variables:', col1, 'and', col2)
        print(stats.pearsonr(data[col1], data[col2]))
        print('pearman Correlation between 2 variables:', col1, 'and', col2)
        print(stats.spearmanr(data[col1], data[col2]))
        sns.pairplot(data[[col1, col2]])
        plt.show()        
        print()

"""
Comment:

    * Pregnancies and Glucose: have a correlation because the p-value is < 0.05.
    * Pregnancies and BloodPressure: have a correlation because the p-value is < 0.05.
    * Pregnancies and SkinThickness: have a correlation because the p-value is < 0.05.
    * Pregnancies and Insulin: have a correlation because the p-value is < 0.05.
    * Pregnancies and BMI: do not have a correlation because the p-value is > 0.05.
    * Pregnancies and DiabetesPedigreeFunction: have a correlation because the p-value is < 0.05.
    * Pregnancies and Age: do not have a correlation because the p-value is > 0.05.
---
    * Glucose and BloodPressure: have a correlation because the p-value is < 0.05.
    * Glucose and SkinThickness: have a correlation because the p-value is < 0.05.
    * Glucose and Insulin: have a correlation because the p-value is < 0.05.
    * Glucose and BMI: have a correlation because the p-value is < 0.05.
    * Glucose and DiabetesPedigreeFunction: have a correlation because the p-value is < 0.05.
    * Glucose and Age: have a correlation because the p-value is < 0.05.
---
    * BloodPressure and SkinThickness: have a correlation because the p-value is < 0.05.
    * BloodPressure and Insulin: do not have a correlation because the p-value is > 0.05.
    * BloodPressure and BMI: have a correlation because the p-value is < 0.05.
    * BloodPressure and DiabetesPedigreeFunction: do not have a correlation because the p-value is > 0.05.
    * BloodPressure and Age: have a correlation because the p-value is < 0.05.
---
    * SkinThickness and Insulin: have a correlation because the p-value is < 0.05.
    * SkinThickness and BMI: have a correlation because the p-value is < 0.05.
    * SkinThickness and DiabetesPedigreeFunction: have a correlation because the p-value is < 0.05.
    * SkinThickness and Age: have a correlation because the p-value is < 0.05.
---
    * Insulin and BMI: have a correlation because the p-value is < 0.05.
    * Insulin and DiabetesPedigreeFunction: have a correlation because the p-value is < 0.05.
    * Insulin and Age: have a correlation because the p-value is < 0.05.
---
    * BMI and DiabetesPedigreeFunction: have a correlation because the p-value is < 0.05.
    * BMI and Age: have a correlation because the p-value is < 0.05.
---
    * DiabetesPedigreeFunction and Age: do not have a correlation because the p-value is > 0.05.
---

Comment:

Due to the presence of many outliers in BloodPressure + No correlation with the output (Disease and Non-disease) from the 
ANOVA analysis + weak correlation with other input variables (such as Insulin or DiabetesPedigreeFunction) 
    => Consider removing the BloodPressure variable before including it in the model.
"""

# Using KNN algorithm with varying k (3,5,7,...) to define which k has the best accuracy score 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

X = data[['Pregnancies', 'Glucose', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] 
y = data['Outcome']

def knn_with_varying_k(X, y):
    k_values = [3, 5, 7, 9, 11, 13, 15]
    results = {}

    for k in k_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[k] = accuracy
    return results

accuracy_results = knn_with_varying_k(X, y)

for k, accuracy in accuracy_results.items():
    print("Accuracy for k =", k, ":", accuracy)

"""
Comment:

Choose k = 5 because of its highest accuracy score
"""

# Using Logistic Regression with Confidence Interval 95% to calculate the median accuracy score with looping 10 times

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import KFold
from scipy.stats import sem
from math import sqrt

X = data[['Pregnancies', 'Glucose', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] 
y = data['Outcome']

def logistic_regression_with_kfold(X, y, num_iterations, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    error = sem(accuracy_scores) * 1.96 
    return mean_accuracy, error

num_iterations = 10
k = 5  
accuracy, error = logistic_regression_with_kfold(X, y, num_iterations, k)

print("The average accuracy of the Logistic Regression algorithm with 10 iterations:", accuracy)
print("95% Confidence Interval:", accuracy - error, "-", accuracy + error)
