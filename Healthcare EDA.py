import pandas as pd

# Load the dataset
csv_file_path = 'srilakshmianbarasan/medicine_data.csv'
medicine_data = pd.read_csv(csv_file_path)

'''
Data Cleaning
'''

# Standardize column names
medicine_data.columns = medicine_data.columns.str.strip().str.replace(' ', '_').str.lower()

# Check for missing values
missing_values = medicine_data.isnull().sum()

# Display the cleaned column names and missing values
medicine_data.columns, missing_values

'''
Exploratory Data Analysis
'''

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of review percentages
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(medicine_data['excellent_review_%'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Excellent Review %')

sns.histplot(medicine_data['average_review_%'], bins=20, kde=True, ax=axes[1])
axes[1].set_title('Distribution of Average Review %')

sns.histplot(medicine_data['poor_review_%'], bins=20, kde=True, ax=axes[2])
axes[2].set_title('Distribution of Poor Review %')

plt.show()

# Analysis of side effects
# Splitting the side_effects column into individual side effects and counting their occurrences
from collections import Counter
side_effects_list = medicine_data['side_effects'].str.split(' ')
side_effects_flat = [item.strip() for sublist in side_effects_list for item in sublist if item]
side_effects_count = Counter(side_effects_flat).most_common(10)

# Visualize the most common side effects
side_effects_df = pd.DataFrame(side_effects_count, columns=['Side_Effect', 'Count'])
sns.barplot(x='Count', y='Side_Effect', data=side_effects_df)
plt.title('Top 10 Most Common Side Effects')
plt.show()

'''
Feature Engineering
'''
# Create a new feature for the total positive review percentage
medicine_data['positive_review_%'] = medicine_data['excellent_review_%'] + medicine_data['average_review_%']

# Create a weighted average rating
medicine_data['weighted_rating'] = (
    medicine_data['excellent_review_%'] * 3 +
    medicine_data['average_review_%'] * 2 +
    medicine_data['poor_review_%'] * 1
) / 6


'''
Modeling
'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Select features and target variable
features = ['excellent_review_%', 'average_review_%', 'poor_review_%']
target = 'weighted_rating'

X = medicine_data[features]
y = medicine_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

rmse


'''
Visualization and Reporting
'''

# Plot actual vs. predicted ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Weighted Rating')
plt.ylabel('Predicted Weighted Rating')
plt.title('Actual vs. Predicted Weighted Rating')
plt.show()

# Summary of findings for the presentation
summary = {
    'rmse': rmse,
    'top_side_effects': side_effects_count
}

summary
