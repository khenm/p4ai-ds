import os
import json
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("tabular_context")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "train", "train.csv")
STATE_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "StateLabels.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "data")
os.makedirs(OUT_DIR, exist_ok=True)


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


def save_json(data, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NpEncoder)
    logger.info(f"Saved {filename}")


TYPE_MAP = {1: 'Dog', 2: 'Cat'}
GENDER_MAP = {1: 'Male', 2: 'Female', 3: 'Mixed'}
VACC_MAP = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
STERILIZED_MAP = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
DEWORMED_MAP = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
HEALTH_MAP = {1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury'}
MATURITY_MAP = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large', 0: 'Not Specified'}
FUR_MAP = {1: 'Short', 2: 'Medium', 3: 'Long', 0: 'Not Specified'}
SPEED_NAMES = ['Same Day', 'First Week', 'First Month', '2nd-3rd Month', 'No Adoption']


def _get_dist(series, mapping=None):
    mapped = series.map(mapping).fillna('Unknown') if mapping else series
    counts = mapped.value_counts()
    return {
        'labels': counts.index.tolist(),
        'counts': [int(x) for x in counts.values.tolist()]
    }


def run_tabular_eda():
    logger.info("Loading tabular data...")
    df = pd.read_csv(DATA_CSV)
    states = pd.read_csv(STATE_CSV)
    state_map = dict(zip(states['StateID'], states['StateName']))

    # 1 — Dataset Overview
    display_cols = ['PetID', 'Type', 'Name', 'Age', 'Gender', 'Fee', 'PhotoAmt', 'AdoptionSpeed']
    sample = df.head(10).copy()
    sample['Type'] = sample['Type'].map(TYPE_MAP)
    sample['Gender'] = sample['Gender'].map(GENDER_MAP)

    col_info = [{
        'name': c,
        'dtype': str(df[c].dtype),
        'non_null': int(df[c].notna().sum()),
        'missing': int(df[c].isna().sum()),
        'missing_pct': round(float(df[c].isna().sum() / len(df) * 100), 2)
    } for c in df.columns]

    save_json({
        'total_listings': int(len(df)),
        'feature_count': int(len(df.columns)),
        'dog_count': int((df['Type'] == 1).sum()),
        'cat_count': int((df['Type'] == 2).sum()),
        'columns': col_info,
        'sample_rows': sample[display_cols].fillna('N/A').to_dict(orient='records'),
        'display_columns': display_cols
    }, 'tabular_overview.json')

    # 2 — Adoption Speed Distribution
    sc = df['AdoptionSpeed'].value_counts().sort_index()
    save_json({
        'labels': [int(x) for x in sc.index],
        'counts': [int(x) for x in sc.values],
        'speed_names': SPEED_NAMES,
        'max_count': int(sc.max()),
        'min_count': int(sc.min()),
        'imbalance_ratio': round(float(sc.max() / sc.min()), 2) if sc.min() > 0 else None,
        'type_counts': {'Dog': int((df['Type'] == 1).sum()), 'Cat': int((df['Type'] == 2).sum())}
    }, 'tabular_adoption_dist.json')

    # 3 — Demographics
    save_json({
        'type': _get_dist(df['Type'], TYPE_MAP),
        'gender': _get_dist(df['Gender'], GENDER_MAP),
        'vaccinated': _get_dist(df['Vaccinated'], VACC_MAP),
        'sterilized': _get_dist(df['Sterilized'], STERILIZED_MAP),
        'dewormed': _get_dist(df['Dewormed'], DEWORMED_MAP),
        'fur_length': _get_dist(df['FurLength'], FUR_MAP),
        'maturity_size': _get_dist(df['MaturitySize'], MATURITY_MAP),
    }, 'tabular_demographics.json')

    # 4 — Age & Fee
    age_fee = {
        'age_dog': df[df['Type'] == 1]['Age'].dropna().tolist(),
        'age_cat': df[df['Type'] == 2]['Age'].dropna().tolist(),
        'fee_all': df['Fee'].dropna().tolist(),
        'fee_by_speed': {str(s): df[df['AdoptionSpeed'] == s]['Fee'].dropna().tolist() for s in range(5)}
    }
    save_json(age_fee, 'tabular_age_fee.json')

    # 5 — State Distribution
    df_s = df.copy()
    df_s['StateName'] = df_s['State'].map(state_map).fillna('Unknown')
    sc2 = df_s['StateName'].value_counts().head(15)
    save_json({
        'labels': sc2.index.tolist(),
        'counts': [int(x) for x in sc2.values]
    }, 'tabular_state_dist.json')

    # 6 — Correlation
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'State']
    corr = df[num_cols].corr().fillna(0)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                'feature1': corr.columns[i],
                'feature2': corr.columns[j],
                'correlation': round(float(corr.iloc[i, j]), 4)
            })
    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    save_json({
        'labels': corr.columns.tolist(),
        'matrix': [[round(float(v), 4) for v in row] for row in corr.values],
        'top_pairs': pairs[:15]
    }, 'tabular_correlation.json')

    # 7 — Health vs Adoption Speed
    ct = pd.crosstab(df['Health'], df['AdoptionSpeed'], normalize='index')
    save_json({
        'health_labels': [HEALTH_MAP.get(int(h), 'Unknown') for h in ct.index],
        'speed_labels': [int(x) for x in ct.columns],
        'proportions': [[round(float(v), 4) for v in row] for row in ct.values]
    }, 'tabular_health.json')

    # 8 — Vaccination crosstab
    df_v = df.copy()
    df_v['Vaccinated'] = df_v['Vaccinated'].map(VACC_MAP)
    vct = pd.crosstab(df_v['Vaccinated'], df_v['AdoptionSpeed'], normalize='index')
    save_json({
        'vacc_labels': vct.index.tolist(),
        'speed_labels': [int(x) for x in vct.columns],
        'proportions': [[round(float(v), 4) for v in row] for row in vct.values]
    }, 'tabular_vaccination.json')

    logger.info("Tabular EDA data extraction complete.")


if __name__ == "__main__":
    run_tabular_eda()
