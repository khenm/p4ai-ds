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
BREED_CSV = os.path.join(PROJECT_ROOT, "data", "petfinder", "BreedLabels.csv")
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

    # 9 — Breed vs Adoption Speed
    breeds_df = pd.read_csv(BREED_CSV)
    breed_map_local = dict(zip(breeds_df['BreedID'], breeds_df['BreedName']))
    df_b = df.copy()
    df_b['BreedName'] = df_b['Breed1'].map(breed_map_local).fillna('Mixed Breed')
    df_b['TypeName'] = df_b['Type'].map(TYPE_MAP)

    breed_speed = {}
    for type_name in ['Dog', 'Cat']:
        type_df = df_b[df_b['TypeName'] == type_name]
        top_breeds = type_df['BreedName'].value_counts().head(15).index.tolist()
        type_top = type_df[type_df['BreedName'].isin(top_breeds)]
        ct = pd.crosstab(type_top['BreedName'], type_top['AdoptionSpeed'], normalize='index')
        ct = ct.reindex(top_breeds).fillna(0)
        breed_speed[type_name] = {
            'breeds': top_breeds,
            'speed_proportions': {
                str(s): [round(float(v), 4) for v in (ct[s] if s in ct.columns else [0] * len(top_breeds))]
                for s in range(5)
            },
            'counts': [int((type_df['BreedName'] == b).sum()) for b in top_breeds],
        }
    save_json(breed_speed, 'tabular_breed_speed.json')

    # 8 — Vaccination crosstab
    df_v = df.copy()
    df_v['Vaccinated'] = df_v['Vaccinated'].map(VACC_MAP)
    vct = pd.crosstab(df_v['Vaccinated'], df_v['AdoptionSpeed'], normalize='index')
    save_json({
        'vacc_labels': vct.index.tolist(),
        'speed_labels': [int(x) for x in vct.columns],
        'proportions': [[round(float(v), 4) for v in row] for row in vct.values]
    }, 'tabular_vaccination.json')

    # ── Helper: build proportional stacked-bar data ──────────────────────────
    def _prop_crosstab(series, label_map=None):
        """Return {labels, proportions: {speed: [values]}} for a categorical series."""
        mapped = series.map(label_map).fillna('Unknown') if label_map else series.astype(str)
        ct = pd.crosstab(mapped, df['AdoptionSpeed'], normalize='index')
        labels = ct.index.tolist()
        props = {
            str(s): [round(float(ct[s][lbl]) if s in ct.columns and lbl in ct.index else 0, 4)
                     for lbl in labels]
            for s in range(5)
        }
        counts = [int((mapped == lbl).sum()) for lbl in labels]
        return {'labels': labels, 'proportions': props, 'counts': counts}

    # ── T7: State vs Adoption Speed ───────────────────────────────────────────
    df_s2 = df.copy()
    df_s2['StateName'] = df_s2['State'].map(state_map).fillna('Unknown')
    top10_states = df_s2['StateName'].value_counts().head(10).index.tolist()
    df_s2 = df_s2[df_s2['StateName'].isin(top10_states)]
    ct_state = pd.crosstab(df_s2['StateName'], df_s2['AdoptionSpeed'], normalize='index')
    ct_state = ct_state.reindex(top10_states).fillna(0)
    save_json({
        'states': top10_states,
        'speed_proportions': {
            str(s): [round(float(ct_state[s][st]) if s in ct_state.columns else 0, 4)
                     for st in top10_states]
            for s in range(5)
        },
        'counts': [int((df_s2['StateName'] == st).sum()) for st in top10_states],
    }, 'tabular_state_speed.json')

    # ── T8: Pet Profile (HasName, Type, PureBreed) vs Adoption Speed ──────────
    df_p = df.copy()
    df_p['HasName'] = df_p['Name'].isna().map({True: 'No Name', False: 'Has Name'})
    df_p['TypeName'] = df_p['Type'].map(TYPE_MAP)
    df_p['PureBreed'] = np.where(df_p['Breed2'] == 0, 'Pure Breed', 'Not Pure Breed')
    save_json({
        'hasname': _prop_crosstab(df_p['HasName']),
        'type_speed': _prop_crosstab(df_p['TypeName']),
        'purebreed': _prop_crosstab(df_p['PureBreed']),
    }, 'tabular_profile_speed.json')

    # ── T9: Age vs Adoption Speed by Type ─────────────────────────────────────
    MAX_PER_BUCKET = 500
    age_speed = {}
    for type_key, type_val in [(1, 'dog'), (2, 'cat')]:
        buckets = {}
        for s in range(5):
            vals = df[(df['Type'] == type_key) & (df['AdoptionSpeed'] == s)]['Age'].dropna()
            if len(vals) > MAX_PER_BUCKET:
                vals = vals.sample(MAX_PER_BUCKET, random_state=42)
            buckets[str(s)] = [float(v) for v in vals.tolist()]
        age_speed[type_val] = buckets
    save_json(age_speed, 'tabular_age_speed.json')

    # ── T10: Quantity, MaturitySize, Gender vs Adoption Speed ─────────────────
    df_g = df.copy()
    top7_qty = df_g['Quantity'].value_counts().head(7).index.tolist()
    df_g['QuantityGroup'] = df_g['Quantity'].apply(
        lambda x: str(int(x)) if x in top7_qty else 'Other')
    qty_order = [str(q) for q in sorted([q for q in top7_qty], key=int)] + ['Other']

    # Quantity proportional
    ct_qty = pd.crosstab(df_g['QuantityGroup'], df_g['AdoptionSpeed'], normalize='index')
    ct_qty = ct_qty.reindex([q for q in qty_order if q in ct_qty.index]).fillna(0)
    qty_labels = ct_qty.index.tolist()

    # MaturitySize by type
    def _maturity_by_type(type_val):
        sub = df_g[df_g['Type'] == type_val].copy()
        sub['MaturityName'] = sub['MaturitySize'].map(MATURITY_MAP).fillna('Unknown')
        order = ['Small', 'Medium', 'Large', 'Extra Large', 'Not Specified']
        ct_m = pd.crosstab(sub['MaturityName'], sub['AdoptionSpeed'], normalize='index')
        ct_m = ct_m.reindex([o for o in order if o in ct_m.index]).fillna(0)
        labels = ct_m.index.tolist()
        return {
            'labels': labels,
            'proportions': {
                str(s): [round(float(ct_m[s][l]) if s in ct_m.columns else 0, 4) for l in labels]
                for s in range(5)
            },
            'counts': [int((sub['MaturityName'] == l).sum()) for l in labels],
        }

    save_json({
        'quantity': {
            'labels': qty_labels,
            'proportions': {
                str(s): [round(float(ct_qty[s][l]) if s in ct_qty.columns else 0, 4)
                         for l in qty_labels]
                for s in range(5)
            },
            'counts': [int((df_g['QuantityGroup'] == l).sum()) for l in qty_labels],
        },
        'maturity_dog': _maturity_by_type(1),
        'maturity_cat': _maturity_by_type(2),
        'gender': _prop_crosstab(df['Gender'], GENDER_MAP),
    }, 'tabular_group_speed.json')

    # ── T11: Care Status (Dewormed, Sterilized) vs Adoption Speed ─────────────
    save_json({
        'dewormed': _prop_crosstab(df['Dewormed'], DEWORMED_MAP),
        'sterilized': _prop_crosstab(df['Sterilized'], STERILIZED_MAP),
    }, 'tabular_care_speed.json')

    # ── T12: Fee & Description Length vs Adoption Speed ───────────────────────
    df_f = df.copy()
    top10_fees = df_f['Fee'].value_counts().head(10).index.tolist()
    df_f['FeeGroup'] = df_f['Fee'].apply(
        lambda x: str(int(x)) if x in top10_fees else 'Other')
    fee_order = ['0'] + [str(int(f)) for f in sorted(top10_fees) if f != 0] + ['Other']
    ct_fee = pd.crosstab(df_f['FeeGroup'], df_f['AdoptionSpeed'], normalize='index')
    ct_fee = ct_fee.reindex([f for f in fee_order if f in ct_fee.index]).fillna(0)
    fee_labels = ct_fee.index.tolist()

    df_f['DescLength'] = df_f['Description'].fillna('').apply(len)
    desc_by_speed = {}
    for s in range(5):
        vals = df_f[df_f['AdoptionSpeed'] == s]['DescLength'].dropna()
        if len(vals) > MAX_PER_BUCKET:
            vals = vals.sample(MAX_PER_BUCKET, random_state=42)
        desc_by_speed[str(s)] = [float(v) for v in vals.tolist()]

    save_json({
        'fee': {
            'labels': fee_labels,
            'proportions': {
                str(s): [round(float(ct_fee[s][l]) if s in ct_fee.columns else 0, 4)
                         for l in fee_labels]
                for s in range(5)
            },
            'counts': [int((df_f['FeeGroup'] == l).sum()) for l in fee_labels],
        },
        'desc_length': desc_by_speed,
    }, 'tabular_fee_desc_speed.json')

    logger.info("Tabular EDA data extraction complete.")


if __name__ == "__main__":
    run_tabular_eda()
