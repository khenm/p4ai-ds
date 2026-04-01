import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from theme import set_theme

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("tabular_context")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def run_tabular_eda():
    logger.info("Loading tabular data...")
    df = pd.read_csv(DATA_CSV)
    colors = set_theme()
    
    # Decode categorical variables
    df_plot = df.copy()
    df_plot['Type'] = df_plot['Type'].map({1: 'Dog', 2: 'Cat'})
    df_plot['Gender'] = df_plot['Gender'].map({1: 'Male', 2: 'Female', 3: 'Mixed'})
    df_plot['Vaccinated'] = df_plot['Vaccinated'].map({1: 'Yes', 2: 'No', 3: 'Not Sure'})

    # 1. Adoption Speed Distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df_plot, x='AdoptionSpeed', palette=colors['sequential'])
    plt.title('Adoption Speed Distribution')
    plt.savefig(os.path.join(OUT_DIR, 'tab_adoption_dist.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Key Categorical Demographics (Mixed Chart Types)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Pie for Animal Type
    type_counts = df_plot['Type'].value_counts()
    axes[0].pie(type_counts, labels=type_counts.index, colors=colors['qualitative'][0:2], autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Animal Type')
    
    # Pie for Gender
    gender_counts = df_plot['Gender'].value_counts()
    axes[1].pie(gender_counts, labels=gender_counts.index, colors=colors['qualitative'][2:5], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Gender')
    
    # Bar (Countplot) for Vaccinated Status
    sns.countplot(data=df_plot, x='Vaccinated', ax=axes[2], palette=colors['qualitative'][3:6])
    axes[2].set_title('Vaccinated Status')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'tab_demographics.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Numeric Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    # Using coolwarm still for correlation, or maybe a manual blend from diverging palette
    sns.heatmap(numeric_df.corr(), annot=False, cmap=sns.blend_palette(colors['diverging'], as_cmap=True), vmin=-1, vmax=1)
    plt.title('Numeric Features Correlation Matrix')
    plt.savefig(os.path.join(OUT_DIR, 'tab_corr_matrix.png'), bbox_inches='tight')
    plt.close()
    
    # 4. Cross Tabulation: Health vs AdoptionSpeed
    plt.figure(figsize=(8, 5))
    ct = pd.crosstab(df_plot['Health'], df_plot['AdoptionSpeed'], normalize='index')
    # Health mapping if we want to decode it, 1: Healthy, 2: Minor Injury, 3: Serious Injury
    ct.index = ct.index.map({1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury', 0: 'Not Specified'})
    ct.plot(kind='bar', stacked=True, color=colors['sequential'], figsize=(8,5))
    plt.title('Health Status vs Adoption Speed (Proportional)')
    plt.legend(title='AdoptionSpeed', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'tab_health_vs_adoption.png'), bbox_inches='tight')
    plt.close()
    
    logger.info("Tabular EDA context generated successfully.")

if __name__ == "__main__":
    run_tabular_eda()
