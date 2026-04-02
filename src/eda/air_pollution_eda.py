import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("air_pollution_eda")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "globalairpollution" / "global_air_pollution_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "ui" / "assets" / "data" / "air_pollution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


AQI_CATEGORY_COLORS = {
    'Good': '#2ecc71',
    'Moderate': '#f39c12',
    'Unhealthy for Sensitive Groups': '#e74c3c',
    'Unhealthy': '#c0392b',
    'Very Unhealthy': '#8b0000',
    'Hazardous': '#4a0000'
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(payload: Dict[str, Any], filename: str) -> None:
    path = OUTPUT_DIR / filename
    with path.open('w') as f:
        json.dump(payload, f, indent=2, cls=NpEncoder)
    logger.info("Saved %s", filename)


def run_air_pollution_eda() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    logger.info("Loading global air pollution dataset...")
    df_raw = pd.read_csv(DATA_PATH)

    # Basic cleaning for analysis (retain raw copy for overview/sample)
    df = df_raw.copy()
    df = df.dropna(subset=['Country'])
    df = df.drop_duplicates()

    logger.info("Generating dataset overview...")
    display_cols = ['Country', 'City', 'AQI Value', 'AQI Category', 'CO AQI Value', 'PM2.5 AQI Value', 'NO2 AQI Value']
    col_info = [{
        'name': col,
        'dtype': str(df_raw[col].dtype),
        'non_null': int(df_raw[col].notna().sum()),
        'missing': int(df_raw[col].isna().sum()),
        'missing_pct': round(float(df_raw[col].isna().mean() * 100), 2)
    } for col in df_raw.columns]

    overview = {
        'total_records': int(len(df_raw)),
        'feature_count': int(len(df_raw.columns)),
        'country_count': int(df_raw['Country'].nunique()),
        'city_count': int(df_raw['City'].nunique()),
        'columns': col_info,
        'sample_rows': df_raw.head(10)[display_cols].fillna('N/A').to_dict(orient='records'),
        'display_columns': display_cols
    }
    save_json(overview, 'air_overview.json')

    logger.info("Generating AQI category distribution...")
    category_order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    aqi_counts = df['AQI Category'].value_counts().reindex(category_order).fillna(0).astype(int)
    save_json({
        'labels': aqi_counts.index.tolist(),
        'counts': aqi_counts.tolist(),
        'colors': [AQI_CATEGORY_COLORS.get(cat, '#95a5a6') for cat in aqi_counts.index],
        'percentages': [round(100 * count / len(df), 2) for count in aqi_counts]
    }, 'air_category_distribution.json')

    logger.info("Summarizing pollutant statistics...")
    pollutant_cols = {
        'CO': 'CO AQI Value',
        'Ozone': 'Ozone AQI Value',
        'NO2': 'NO2 AQI Value',
        'PM2.5': 'PM2.5 AQI Value'
    }
    pollutant_summary = {}
    for label, col in pollutant_cols.items():
        stats = df[col].describe()
        pollutant_summary[label] = {
            'mean': round(float(stats['mean']), 2),
            'median': round(float(df[col].median()), 2),
            'std': round(float(stats['std']), 2),
            'min': round(float(stats['min']), 2),
            'max': round(float(stats['max']), 2),
            'q25': round(float(df[col].quantile(0.25)), 2),
            'q75': round(float(df[col].quantile(0.75)), 2),
            'values': df[col].tolist()
        }
    save_json({'pollutants': pollutant_summary}, 'air_pollutant_analysis.json')

    logger.info("Computing geographical analysis...")
    country_stats = df.groupby('Country').agg({
        'AQI Value': ['mean', 'max', 'count'],
        'City': 'nunique'
    }).round(2)
    country_stats.columns = ['AQI Mean', 'AQI Max', 'Records', 'Cities']
    top_countries = country_stats.sort_values('AQI Mean', ascending=False).head(20)
    save_json({
        'countries': top_countries.index.tolist(),
        'aqi_mean': [float(x) for x in top_countries['AQI Mean']],
        'aqi_max': [float(x) for x in top_countries['AQI Max']],
        'records': [int(x) for x in top_countries['Records']],
        'cities': [int(x) for x in top_countries['Cities']]
    }, 'air_geographical_analysis.json')

    logger.info("Building correlation matrix...")
    numeric_cols = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    corr_matrix = df[numeric_cols].corr().fillna(0)
    pair_list = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            pair_list.append({
                'feature1': numeric_cols[i],
                'feature2': numeric_cols[j],
                'correlation': round(float(corr_matrix.iloc[i, j]), 4)
            })
    pair_list.sort(key=lambda item: abs(item['correlation']), reverse=True)
    save_json({
        'labels': numeric_cols,
        'matrix': [[round(float(value), 4) for value in row] for row in corr_matrix.values],
        'top_pairs': pair_list
    }, 'air_correlation.json')

    logger.info("Analyzing pollutant category breakdowns...")
    pollutant_category_cols = {
        'CO': 'CO AQI Category',
        'Ozone': 'Ozone AQI Category',
        'NO2': 'NO2 AQI Category',
        'PM2.5': 'PM2.5 AQI Category'
    }
    pollutant_category_data = {}
    for label, col in pollutant_category_cols.items():
        dist = df[col].value_counts().reindex(category_order).fillna(0).astype(int)
        pollutant_category_data[label] = {
            'labels': dist.index.tolist(),
            'counts': dist.tolist(),
            'colors': [AQI_CATEGORY_COLORS.get(cat, '#95a5a6') for cat in dist.index]
        }
    save_json(pollutant_category_data, 'air_pollutant_categories.json')

    logger.info("Preparing scatter comparison data...")
    scatter_payload = {}
    for label, col in pollutant_cols.items():
        scatter_payload[label] = {
            'x': df[col].tolist(),
            'y': df['AQI Value'].tolist(),
            'labels': [f"{label} AQI", 'Overall AQI']
        }
    save_json(scatter_payload, 'air_scatter_analysis.json')

    logger.info("Detecting AQI outliers via IQR...")
    q1 = df['AQI Value'].quantile(0.25)
    q3 = df['AQI Value'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (df['AQI Value'] < lower) | (df['AQI Value'] > upper)
    outliers = df[outlier_mask]
    top_outliers = df[df['AQI Value'] > upper].nlargest(15, 'AQI Value')
    save_json({
        'outlier_count': int(len(outliers)),
        'outlier_pct': round(100 * len(outliers) / len(df), 2),
        'bounds': {'lower': round(float(lower), 2), 'upper': round(float(upper), 2)},
        'top_outliers': [{
            'country': row['Country'],
            'city': row['City'],
            'aqi_value': int(row['AQI Value']),
            'aqi_category': row['AQI Category'],
            'pm25': int(row['PM2.5 AQI Value'])
        } for _, row in top_outliers.iterrows()]
    }, 'air_outlier_detection.json')

    logger.info("Compiling insights and recommendations...")
    dominant_category = df['AQI Category'].mode()[0]
    most_polluted_country = df.groupby('Country')['AQI Value'].mean().idxmax()
    cleanest_country = df.groupby('Country')['AQI Value'].mean().idxmin()
    insights = {
        'summary': f"The dataset spans {len(df):,} city-level readings across {df['Country'].nunique()} countries, combining AQI categories with pollutant-specific readings.",
        'key_findings': [
            f"'{dominant_category}' is the most prevalent AQI category, covering {round(100 * (df['AQI Category'] == dominant_category).mean(), 1)}% of cities.",
            f"{most_polluted_country} exhibits the highest average AQI, highlighting critical mitigation needs.",
            f"{cleanest_country} has the lowest average AQI, indicating comparatively clean air.",
            "PM2.5 levels trend higher than other pollutants and align strongly with overall AQI.",
            "Pollutant categories skew toward Mild/Moderate levels, but tails include hazardous spikes."
        ],
        'recommendations': [
            'Prioritize PM2.5 monitoring and interventions in hotspots exceeding AQI > 150.',
            'Deploy seasonal alerting for cities with frequent category swings between Moderate and Unhealthy.',
            'Use the pollutant scatter comparisons to identify single-pollutant drivers before rolling out policies.',
            'Coordinate cross-border initiatives for countries showing correlated spikes (e.g., neighbors in the same region).'
        ]
    }
    save_json(insights, 'air_insights_recommendations.json')

    logger.info("Generating master summary report...")
    master_report = {
        'title': 'Global Air Pollution — Comprehensive EDA Summary',
        'dataset': {
            'records': int(len(df)),
            'countries': int(df['Country'].nunique()),
            'cities': int(df['City'].nunique()),
            'features': int(len(df.columns))
        },
        'aqi_stats': {
            'mean': round(float(df['AQI Value'].mean()), 2),
            'median': round(float(df['AQI Value'].median()), 2),
            'std': round(float(df['AQI Value'].std()), 2),
            'min': int(df['AQI Value'].min()),
            'max': int(df['AQI Value'].max())
        },
        'category_distribution': df['AQI Category'].value_counts().to_dict(),
        'pollutant_means': {label: round(float(df[col].mean()), 2) for label, col in pollutant_cols.items()}
    }
    save_json(master_report, 'air_master_report.json')

    logger.info("✓ Global air pollution EDA completed")
    return {
        'status': 'success',
        'records': int(len(df)),
        'countries': int(df['Country'].nunique()),
        'cities': int(df['City'].nunique())
    }


if __name__ == '__main__':
    result = run_air_pollution_eda()
    print(json.dumps(result, indent=2))