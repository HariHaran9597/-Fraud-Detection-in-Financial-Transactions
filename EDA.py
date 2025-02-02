import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 3.1: Load the Dataset

df = pd.read_csv(r'Dataset\creditcard.csv')

print("Distribution of the target variable (Class):")
print(df['Class'].value_counts())

# Plot the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, hue='Class', palette='Set2', legend=False)
plt.title('Distribution of Fraudulent vs. Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()

# Step 4.2: Visualize Transaction Amounts
# Plot the distribution of transaction amounts for fraudulent and non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label='Non-Fraudulent', kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraudulent', kde=True)
plt.title('Distribution of Transaction Amounts by Class')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Step 4.3: Time-Based Analysis
# Plot the distribution of transactions over time
plt.figure(figsize=(10, 6))
sns.histplot(df['Time'], bins=50, color='green', kde=True)
plt.title('Distribution of Transactions Over Time')
plt.xlabel('Time (in seconds)')
plt.ylabel('Frequency')
plt.show()

# Step 4.4: Correlation Analysis
# Plot a correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 4.5: Outlier Detection
# Plot boxplots for the 'Amount' column to detect outliers
plt.figure(figsize=(6, 4))
sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette='Set2', legend=False)
plt.title('Boxplot of Transaction Amounts by Class')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Transaction Amount')
plt.show()