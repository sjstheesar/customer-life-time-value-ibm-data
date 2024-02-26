import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Function to load data
def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to preprocess data
def preprocess_data(data):
    """Preprocess the data by dropping unnecessary columns and handling missing values."""
    if 'Country' in data.columns:
        data.dropna(subset=['Country'], inplace=True)
    data = data.drop(columns=[1, 2, 3, 7, 8, 9])
    return data

# Function to split data into customer and non-customer datasets
def split_customer_noncustomer(data):
    """Split the dataset into customer and non-customer datasets."""
    data_cust = data[data['CancellationDate'].isnull()]
    data_non_cust = data[data['CancellationDate'].notnull()]
    return data_cust, data_non_cust

# Function to plot value counts for categorical variables
def plot_value_counts(data, column, title):
    """Plot the value counts of a categorical variable."""
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=data)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation='vertical')
    plt.show()

# Function to plot distributions for numerical variables
def plot_distributions(data):
    """Plot the distribution of numerical variables."""
    numeric_cols = data.select_dtypes(exclude='object').columns
    for col in numeric_cols:
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Function to plot boxplots by categorical variable
def plot_boxplots(data, y_var):
    """Plot boxplots for a numerical variable grouped by categorical variables."""
    cat_cols = data.select_dtypes(include='object').columns
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    for i, ax in enumerate(axes.flatten()):
        if i < len(cat_cols):
            sns.boxplot(x=cat_cols[i], y=y_var, data=data, ax=ax)
            ax.set_title(f'{y_var} by {cat_cols[i]}')
            ax.set_xlabel(cat_cols[i])
            ax.set_ylabel(y_var)
    plt.tight_layout()
    plt.show()

# Function to plot cross-tabulations with bar plots
def plot_crosstabs(data, cat_vars):
    """Plot cross-tabulations for categorical variables."""
    for var in cat_vars:
        plt.figure(figsize=(8, 5))
        pd.crosstab(data['Province or State'], data[var], values=data['Customer Lifetime Value'], aggfunc="mean").plot(kind="barh")
        plt.title(f"Rata-Rata Nilai Customer LifeTime IBM berdasarkan {var}")
        plt.xlabel('Mean Customer Lifetime Value')
        plt.ylabel(var)
        plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1))
        plt.show()

# Function to plot scatter plots for categorical variables vs continuous variables
def plot_scatter_plots(data, x_var):
    """Plot scatter plots for a continuous variable grouped by categorical variables."""
    cat_cols = data.select_dtypes(include='object').columns
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    for i, ax in enumerate(axes.flatten()):
        if i < len(cat_cols):
            sns.scatterplot(x=x_var, y=data['Customer Lifetime Value'], hue=cat_cols[i], data=data, ax=ax)
            ax.set_title(f'{x_var} vs Customer Lifetime Value by {cat_cols[i]}')
            ax.set_xlabel(x_var)
            ax.set_ylabel('Customer Lifetime Value')
    plt.tight_layout()
    plt.show()

# Function to plot KDE plots for categorical variables vs continuous variables
def plot_kde_plots(data, x_var):
    """Plot KDE plots for a continuous variable grouped by categorical variables."""
    cat_cols = data.select_dtypes(include='object').columns
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    for i, ax in enumerate(axes.flatten()):
        if i < len(cat_cols):
            sns.displot(data=data, x=x_var, hue=cat_cols[i], kind='kde', fill=True, ax=ax)
            ax.set_title(f'{x_var} by {cat_cols[i]}')
            ax.set_xlabel(x_var)
            ax.set_ylabel('Density')
    plt.tight_layout()
    plt.show()

# Main script execution
if __name__ == "__main__":
    # Load data
    file_path = '/content/drive/MyDrive/project/PORTODATA/Loyalty_card_customers.csv/Loyalty_card_customers.csv'
    data = load_data(file_path)
    
    if data is not None:
        # Preprocess data
        data1 = preprocess_data(data)
        
        # Split data into customer and non-customer datasets
        data_cust, data_non_cust = split_customer_noncustomer(data1)
        
        # Plot value counts for categorical variables
        plot_value_counts(data_cust, 'Province or State', "Total Pelanggan IBM Kanada by Province")
        plot_value_counts(data_cust, 'City', "Total Pelanggan IBM Kanada by City")
        
        # Plot distributions for numerical variables
        plot_distributions(data_cust)
        
        # Plot boxplots for categorical variables grouped by customer lifetime value
        plot_boxplots(data_cust, ['Gender', 'Education', 'Marital Status', 'LoyaltyStatus'])
        plot_boxplots(data_non_cust, ['Gender', 'Education', 'Marital Status', 'LoyaltyStatus'])
        
        # Plot cross-tabulations with bar plots for categorical variables
        cat_vars = ['Education', 'Marital Status', 'Gender', 'LoyaltyStatus']
        plot_crosstabs(data_cust, cat_vars)
        
        # Plot scatter plots for categorical variables vs customer lifetime value
        plot_scatter_plots(data_cust, 'Customer Lifetime Value')
        plot_scatter_plots(data_non_cust, 'Customer Lifetime Value')
        
        # Plot KDE plots for categorical variables vs customer lifetime value
        plot_kde_plots(data_cust, 'Customer Lifetime Value')
        plot_kde_plots(data_non_cust, 'Customer Lifetime Value')
