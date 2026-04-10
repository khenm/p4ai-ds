import os
import pandas as pd
import plotly.express as px

def run_salary_eda():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    data_path = os.path.join(project_root, 'data', 'jobsalary', 'job_salary_prediction_dataset.csv')
    print(f"Reading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    out_dir = os.path.join(project_root, 'ui', 'assets', 'data', 'jobsalary')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Ensuring output directory exists: {out_dir}")
    
    import json
    import numpy as np
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    overview = {
        "total_listings": int(df.shape[0]),
        "feature_count": int(df.shape[1]),
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
        "columns": [],
        "sample_rows": df.head(5).fillna("").to_dict(orient="records"),
        "display_columns": df.columns.tolist()
    }
    
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        missing_pct = round((missing / len(df)) * 100, 2)
        
        feature_type = "Target" if col == 'salary' else ("Numerical" if col in num_cols else "Categorical")
        
        overview["columns"].append({
            "name": col,
            "feature_type": feature_type,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notnull().sum()),
            "missing": missing,
            "missing_pct": missing_pct
        })
    
    overview_out = os.path.join(out_dir, 'dataset_overview.json')
    with open(overview_out, 'w', encoding='utf-8') as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print(f"Saved: {overview_out}")
    
    custom_colors = [
        '#C26A2E', '#5E8A5C', '#6886A5', '#C7727A', '#B8942F', 
        '#9E5420', '#2D4A6B', '#2D5A2B', '#6B3434', '#7A4A1E', 
        '#D98A4E', '#6B5A1E'
    ]
    layout_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # Histogram for Salary
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    fig_sns = plt.gcf()
    fig_sns.patch.set_alpha(0)
    ax_sns = plt.gca()
    ax_sns.patch.set_alpha(0)
    
    sns.histplot(df['salary'], kde=True, color=custom_colors[0], ax=ax_sns)
    plt.title('Distribution of Salary (with KDE)')
    plt.xlabel('Salary')
    plt.ylabel('Density')
    
    ax_sns.tick_params(colors='#333333')
    ax_sns.xaxis.label.set_color('#333333')
    ax_sns.yaxis.label.set_color('#333333')
    ax_sns.title.set_color('#333333')
    
    ax_sns.spines['top'].set_visible(False)
    ax_sns.spines['right'].set_visible(False)
    ax_sns.spines['left'].set_color('#CCCCCC')
    ax_sns.spines['bottom'].set_color('#CCCCCC')
    
    hist_out = os.path.join(out_dir, 'salary_distribution.png')
    plt.tight_layout()
    plt.savefig(hist_out, transparent=True, dpi=300)
    plt.close()
    print(f"Saved: {hist_out}")
    
    # Numerical Correlation Matrix
    num_cols = ['experience_years', 'skills_count', 'certifications', 'salary']
    corr_matrix = df[num_cols].corr()
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        title='Numerical Correlation Matrix',
        color_continuous_scale=['#FAF5EF', '#F2DCC8', '#D98A4E', '#C26A2E', '#9E5420'],
        aspect='auto'
    )
    fig_corr.update_layout(**layout_theme)
    corr_out = os.path.join(out_dir, 'numerical_correlation.json')
    fig_corr.write_json(corr_out)
    print(f"Saved: {corr_out}")
    
    # Salary by Education Level (Boxplot)
    edu_order = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
    fig_box = px.box(
        df, 
        x='education_level', 
        y='salary', 
        title='Salary by Education Level',
        category_orders={'education_level': edu_order},
        labels={'education_level': 'Education Level', 'salary': 'Salary'},
        color='education_level',
        color_discrete_sequence=custom_colors
    )
    fig_box.update_layout(showlegend=False, **layout_theme)
    box_out = os.path.join(out_dir, 'education_boxplot.json')
    fig_box.write_json(box_out)
    print(f"Saved: {box_out}")

    # Experience Years vs Salary (Scatter Plot)
    fig_scatter = px.scatter(
        df,
        x='experience_years',
        y='salary',
        title='Experience Years vs Salary',
        labels={'experience_years': 'Years of Experience', 'salary': 'Salary'},
        opacity=0.6,
        color='experience_years',
        color_continuous_scale=['#C26A2E', '#9E5420', '#5E8A5C', '#2D5A2B']
    )
    fig_scatter.update_layout(**layout_theme)
    scatter_out = os.path.join(out_dir, 'experience_salary_scatter.json')
    fig_scatter.write_json(scatter_out)
    print(f"Saved: {scatter_out}")

    # Company Size Boxplot
    size_order = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
    fig_size = px.box(
        df, x='company_size', y='salary', 
        title='Salary by Company Size',
        category_orders={'company_size': size_order},
        labels={'company_size': 'Company Size', 'salary': 'Salary'},
        color='company_size',
        color_discrete_sequence=custom_colors
    )
    fig_size.update_layout(showlegend=False, **layout_theme)
    fig_size.write_json(os.path.join(out_dir, 'company_size_boxplot.json'))
    print("Saved: company_size_boxplot.json")

    # Job Title Boxplot (Sorted by Median Salary)
    median_jobs = df.groupby('job_title')['salary'].median().sort_values(ascending=False).index.tolist()
    fig_job = px.box(
        df, x='job_title', y='salary',
        title='Salary by Job Title',
        category_orders={'job_title': median_jobs},
        labels={'job_title': 'Job Title', 'salary': 'Salary'},
        color='job_title',
        color_discrete_sequence=custom_colors
    )
    fig_job.update_layout(showlegend=False, xaxis={'categoryorder': 'array', 'categoryarray': median_jobs}, **layout_theme)
    fig_job.write_json(os.path.join(out_dir, 'job_title_boxplot.json'))
    print("Saved: job_title_boxplot.json")

    # Industry Boxplot (Sorted by Median Salary)
    median_industry = df.groupby('industry')['salary'].median().sort_values(ascending=False).index.tolist()
    fig_industry = px.box(
        df, x='industry', y='salary',
        title='Salary by Industry',
        category_orders={'industry': median_industry},
        labels={'industry': 'Industry', 'salary': 'Salary'},
        color='industry',
        color_discrete_sequence=custom_colors
    )
    fig_industry.update_layout(showlegend=False, xaxis={'categoryorder': 'array', 'categoryarray': median_industry}, **layout_theme)
    fig_industry.write_json(os.path.join(out_dir, 'industry_boxplot.json'))
    print("Saved: industry_boxplot.json")

    # Remote Work Boxplot
    remote_order = ['No', 'Hybrid', 'Yes']
    fig_remote = px.box(
        df, x='remote_work', y='salary',
        title='Salary by Remote Work Status',
        category_orders={'remote_work': remote_order},
        labels={'remote_work': 'Remote Work', 'salary': 'Salary'},
        color='remote_work',
        color_discrete_sequence=custom_colors
    )
    fig_remote.update_layout(showlegend=False, **layout_theme)
    fig_remote.write_json(os.path.join(out_dir, 'remote_work_boxplot.json'))
    print("Saved: remote_work_boxplot.json")

    # Location Boxplot (Sorted by Median Salary)
    median_location = df.groupby('location')['salary'].median().sort_values(ascending=False).index.tolist()
    fig_location = px.box(
        df, x='location', y='salary',
        title='Salary by Location',
        category_orders={'location': median_location},
        labels={'location': 'Location', 'salary': 'Salary'},
        color='location',
        color_discrete_sequence=custom_colors
    )
    fig_location.update_layout(showlegend=False, xaxis={'categoryorder': 'array', 'categoryarray': median_location}, **layout_theme)
    fig_location.write_json(os.path.join(out_dir, 'location_boxplot.json'))
    print("Saved: location_boxplot.json")

    # 4. ADVANCED EDA COMPONENTS
    # Feature Interaction: Education vs Salary grouped by Company Size
    import plotly.graph_objects as go
    
    fig_interaction = px.box(
        df, x='education_level', y='salary', color='company_size',
        title='Education Level vs Salary by Company Size',
        category_orders={'education_level': edu_order, 'company_size': size_order},
        labels={'education_level': 'Education Level', 'salary': 'Salary', 'company_size': 'Company Size'},
        color_discrete_sequence=custom_colors
    )
    fig_interaction.update_layout(**layout_theme)
    interaction_out = os.path.join(out_dir, 'education_company_size_boxplot.json')
    fig_interaction.write_json(interaction_out)
    print(f"Saved: {interaction_out}")

    # Class Imbalance & Mean Salary Trend: Job Title Count vs Median Salary
    job_counts = df['job_title'].value_counts()
    job_medians = df.groupby('job_title')['salary'].median()
    merged_job = pd.DataFrame({'count': job_counts, 'median_salary': job_medians}).sort_values('count', ascending=False)
    
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Bar(
        x=merged_job.index, y=merged_job['count'],
        name='Frequency', marker_color=custom_colors[2], yaxis='y1'
    ))
    fig_freq.add_trace(go.Scatter(
        x=merged_job.index, y=merged_job['median_salary'],
        name='Median Salary', mode='lines+markers', line=dict(color=custom_colors[0], width=3), yaxis='y2'
    ))
    fig_freq.update_layout(
        title='Job Title Frequency vs Median Salary',
        yaxis=dict(title='Frequency', side='left'),
        yaxis2=dict(title='Median Salary', side='right', overlaying='y', showgrid=False),
        showlegend=True,
        legend=dict(x=0.01, y=1.1, orientation="h"),
        **layout_theme
    )
    freq_out = os.path.join(out_dir, 'job_title_freq_salary.json')
    fig_freq.write_json(freq_out)
    print(f"Saved: {freq_out}")

    # Cramér's V Heatmap for Categorical Redundancy Check
    import scipy.stats as ss
    import numpy as np

    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    cat_cols_for_cramer = ['job_title', 'industry', 'location', 'remote_work', 'company_size', 'education_level']
    cramer_matrix = pd.DataFrame(index=cat_cols_for_cramer, columns=cat_cols_for_cramer)
    for c1 in cat_cols_for_cramer:
        for c2 in cat_cols_for_cramer:
            if c1 == c2:
                cramer_matrix.loc[c1, c2] = 1.0
            else:
                cramer_matrix.loc[c1, c2] = cramers_v(df[c1], df[c2])
    cramer_matrix = cramer_matrix.astype(float)
    
    fig_cramer = px.imshow(
        cramer_matrix,
        text_auto=".2f",
        title='Categorical Correlation (Cramér\'s V)',
        color_continuous_scale=['#FAF5EF', '#F2DCC8', '#D98A4E', '#C26A2E', '#9E5420'],
        aspect='auto'
    )
    fig_cramer.update_layout(**layout_theme)
    cramer_out = os.path.join(out_dir, 'categorical_correlation.json')
    fig_cramer.write_json(cramer_out)
    print(f"Saved: {cramer_out}")



if __name__ == '__main__':
    run_salary_eda()
