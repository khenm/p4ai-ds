/* tabular.js — Tabular EDA Chart Rendering */

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [overview, adoption, demographics, ageFee, stateDist, correlation, health, vaccination] = await Promise.all([
            loadJSON('tabular_overview.json'),
            loadJSON('tabular_adoption_dist.json'),
            loadJSON('tabular_demographics.json'),
            loadJSON('tabular_age_fee.json'),
            loadJSON('tabular_state_dist.json'),
            loadJSON('tabular_correlation.json'),
            loadJSON('tabular_health.json'),
            loadJSON('tabular_vaccination.json'),
        ]);

        renderOverview(overview);
        renderAdoption(adoption);
        renderDemographics(demographics);
        renderAgeFee(ageFee);
        renderState(stateDist);
        renderCorrelation(correlation);
        renderHealth(health);
        renderVaccination(vaccination);
    } catch (e) {
        console.error('Failed to load tabular data:', e);
    }
});

function renderOverview(data) {
    document.getElementById('overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Total Listings</div><div class="stat-value">${data.total_listings.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Features</div><div class="stat-value">${data.feature_count}</div><div class="stat-sub">${data.feature_count} columns</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Dog Listings</div><div class="stat-value">${data.dog_count.toLocaleString()}</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Cat Listings</div><div class="stat-value">${data.cat_count.toLocaleString()}</div></div>`;

    document.getElementById('overview-desc').innerHTML = `The PetFinder.my dataset contains <strong>${data.total_listings.toLocaleString()}</strong> pet adoption listings from Malaysia. Each listing has <strong>${data.feature_count}</strong> features including demographics (Type, Age, Gender, Breed), health info (Vaccinated, Dewormed, Sterilized), listing metadata (Fee, PhotoAmt, VideoAmt), and the target variable <strong>AdoptionSpeed</strong> (0–4).`;

    // Sample rows table
    const cols = data.display_columns;
    let html = '<thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    data.sample_rows.forEach(row => {
        html += '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>';
    });
    html += '</tbody>';
    document.getElementById('sample-table').innerHTML = html;

    // Dtypes table
    let dhtml = '<thead><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>';
    data.columns.forEach(c => {
        dhtml += `<tr><td>${c.name}</td><td>${c.dtype}</td><td class="num">${c.non_null.toLocaleString()}</td><td class="num">${c.missing}</td><td class="num">${c.missing_pct}%</td></tr>`;
    });
    dhtml += '</tbody>';
    document.getElementById('dtypes-table').innerHTML = dhtml;
}

function renderAdoption(data) {
    const traces = data.labels.map((l, i) => ({
        x: [`Speed ${l}`], y: [data.counts[i]], type: 'bar',
        name: `Speed ${l} — ${data.speed_names[i]}`,
        marker: { color: COLORS.speedColors[i] },
        text: [data.counts[i].toLocaleString()], textposition: 'outside',
        showlegend: true,
    }));
    Plotly.newPlot('chart-adoption-bar', traces, plotlyLayout('Adoption Speed Distribution', {
        xaxis: { title: 'Adoption Speed' },
        yaxis: { title: 'Count' },
        legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
    }), PLOTLY_CONFIG);

    Plotly.newPlot('chart-type-pie', [{
        labels: Object.keys(data.type_counts),
        values: Object.values(data.type_counts),
        type: 'pie', hole: 0.45,
        marker: { colors: [COLORS.qualitative[0], COLORS.qualitative[2]] },
        textinfo: 'label+percent', textfont: { size: 13 },
    }], plotlyLayout('Dog vs Cat Proportion', { margin: { t: 40, b: 10 } }), PLOTLY_CONFIG);

    // Balance cards
    document.getElementById('balance-cards').innerHTML = `
        <div class="balance-card"><div class="label">Max Count</div><div class="value">${data.max_count.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Min Count</div><div class="value">${data.min_count.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Imbalance Ratio</div><div class="value warning">${data.imbalance_ratio}x</div></div>
        <div class="balance-card"><div class="label">Balance Level</div><div class="value warning">Imbalanced</div></div>`;

    document.getElementById('adoption-insights').innerHTML = `
        <li>Speed 4 (no adoption after 100+ days) is the most common outcome</li>
        <li>Class imbalance ratio: ${data.imbalance_ratio}x between largest and smallest classes</li>
        <li>Dogs outnumber cats (${data.type_counts.Dog.toLocaleString()} vs ${data.type_counts.Cat.toLocaleString()})</li>
        <li>Consider resampling or class weights for model training</li>`;
}

function renderDemographics(data) {
    const container = document.getElementById('demographics-charts');
    const categories = ['gender', 'vaccinated', 'sterilized', 'dewormed'];
    const titles = ['Gender Distribution', 'Vaccinated', 'Sterilized', 'Dewormed'];

    categories.forEach((cat, i) => {
        const div = document.createElement('div');
        div.className = 'chart-container';
        div.innerHTML = `<div class="chart-box" id="chart-demo-${cat}"></div><p class="chart-description">${titles[i]} distribution across all listings (decoded labels).</p>`;
        container.appendChild(div);

        Plotly.newPlot(`chart-demo-${cat}`, [{
            x: data[cat].labels, y: data[cat].counts, type: 'bar',
            marker: { color: COLORS.qualitative.slice(0, data[cat].labels.length) },
            text: data[cat].counts.map(c => c.toLocaleString()), textposition: 'outside',
        }], plotlyLayout(titles[i], {
            xaxis: { title: '' }, yaxis: { title: 'Count' }
        }), PLOTLY_CONFIG);
    });
}

function renderAgeFee(data) {
    const container = document.getElementById('age-fee-charts');

    // Age histogram by type
    const ageDiv = document.createElement('div');
    ageDiv.className = 'chart-container';
    ageDiv.innerHTML = '<div class="chart-box" id="chart-age-hist"></div><p class="chart-description">Age distribution (in months) by animal type. Dogs and cats show different age profiles.</p>';
    container.appendChild(ageDiv);

    Plotly.newPlot('chart-age-hist', [
        { x: data.age_dog, type: 'histogram', name: 'Dog', opacity: 0.7, marker: { color: COLORS.qualitative[0] }, nbinsx: 40 },
        { x: data.age_cat, type: 'histogram', name: 'Cat', opacity: 0.7, marker: { color: COLORS.qualitative[2] }, nbinsx: 40 },
    ], plotlyLayout('Age Distribution by Type', {
        barmode: 'overlay', xaxis: { title: 'Age (months)', range: [0, 100] }, yaxis: { title: 'Count' }
    }), PLOTLY_CONFIG);

    // Fee vs Adoption Speed
    const feeDiv = document.createElement('div');
    feeDiv.className = 'chart-container';
    feeDiv.innerHTML = '<div class="chart-box" id="chart-fee-speed"></div><p class="chart-description">Fee distribution by adoption speed. Most listings have zero or low fees.</p>';
    container.appendChild(feeDiv);

    const feeTraces = [];
    for (let s = 0; s < 5; s++) {
        feeTraces.push({
            y: data.fee_by_speed[String(s)].filter(f => f <= 500),
            type: 'box', name: `Speed ${s}`,
            marker: { color: COLORS.speedColors[s] },
        });
    }
    Plotly.newPlot('chart-fee-speed', feeTraces, plotlyLayout('Fee vs Adoption Speed', {
        yaxis: { title: 'Fee (RM)', range: [0, 500] }, xaxis: { title: 'Adoption Speed' }
    }), PLOTLY_CONFIG);
}

function renderState(data) {
    Plotly.newPlot('chart-state', [{
        y: data.labels.slice().reverse(), x: data.counts.slice().reverse(),
        type: 'bar', orientation: 'h',
        marker: { color: COLORS.sequential[2] },
    }], plotlyLayout('Top 15 States by Listing Count', {
        xaxis: { title: 'Count' }, yaxis: { title: '' },
        margin: { l: 140 }, height: 450,
    }), PLOTLY_CONFIG);
}

function renderCorrelation(data) {
    // Mask upper triangle
    const masked = data.matrix.map((row, i) => row.map((v, j) => j > i ? null : v));

    Plotly.newPlot('chart-corr', [{
        z: masked, x: data.labels, y: data.labels,
        type: 'heatmap',
        colorscale: [[0, '#2A9D8F'], [0.5, '#E9C46A'], [1, '#E76F51']],
        zmin: -1, zmax: 1,
        hoverongaps: false,
    }], plotlyLayout('Numeric Features Correlation Matrix', {
        height: 550, margin: { l: 100, b: 100 },
        xaxis: { tickangle: -45 }, yaxis: {},
    }), PLOTLY_CONFIG);

    // Top pairs table
    let html = '<thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr></thead><tbody>';
    data.top_pairs.slice(0, 10).forEach(p => {
        const color = p.correlation > 0 ? '#00B894' : '#E17055';
        html += `<tr><td>${p.feature1}</td><td>${p.feature2}</td><td class="num" style="color:${color}">${p.correlation.toFixed(4)}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('corr-pairs-table').innerHTML = html;

    const top3 = data.top_pairs.slice(0, 3);
    document.getElementById('corr-insights').innerHTML = `
        <li>Strongest correlation: <strong>${top3[0].feature1} ↔ ${top3[0].feature2}</strong> (r=${top3[0].correlation.toFixed(3)})</li>
        <li>Vaccinated, Dewormed, Sterilized are highly correlated — responsible pet owners tend to do all three</li>
        <li>Fee and Age show weak positive correlation with AdoptionSpeed</li>
        <li>PhotoAmt shows negative correlation with AdoptionSpeed — more photos → faster adoption</li>`;
}

function renderHealth(data) {
    const traces = [];
    data.speed_labels.forEach((speed, i) => {
        traces.push({
            x: data.health_labels,
            y: data.proportions.map(row => row[i]),
            name: `Speed ${speed}`, type: 'bar',
            marker: { color: COLORS.speedColors[i] },
        });
    });
    Plotly.newPlot('chart-health', traces, plotlyLayout('Health Status vs Adoption Speed', {
        barmode: 'stack', xaxis: { title: 'Health Status' }, yaxis: { title: 'Proportion' },
    }), PLOTLY_CONFIG);
}

function renderVaccination(data) {
    const traces = [];
    data.speed_labels.forEach((speed, i) => {
        traces.push({
            x: data.vacc_labels,
            y: data.proportions.map(row => row[i]),
            name: `Speed ${speed}`, type: 'bar',
            marker: { color: COLORS.speedColors[i] },
        });
    });
    Plotly.newPlot('chart-vacc', traces, plotlyLayout('Vaccination × Adoption Speed', {
        barmode: 'stack', xaxis: { title: 'Vaccination Status' }, yaxis: { title: 'Proportion' },
    }), PLOTLY_CONFIG);
}

// Code snippets for modals
CODE_SNIPPETS['overview'] = `import pandas as pd
df = pd.read_csv("data/petfinder/train/train.csv")
print(f"Shape: {df.shape}")
print(f"Dogs: {(df['Type']==1).sum()}, Cats: {(df['Type']==2).sum()}")
print(df.head(10))
print(df.dtypes)`;

CODE_SNIPPETS['adoption'] = `speed_counts = df['AdoptionSpeed'].value_counts().sort_index()
print("Imbalance ratio:", speed_counts.max() / speed_counts.min())`;

CODE_SNIPPETS['correlation'] = `numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
# Top pairs
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        print(f"{corr.columns[i]} <-> {corr.columns[j]}: {corr.iloc[i,j]:.4f}")`;

TUTORIAL_TEXTS['overview'] = '<p>The Dataset Overview provides a high-level view of the PetFinder training data. It shows the total number of listings, feature count, type distribution, and a sample of the raw data. The data types table helps identify which columns need encoding or transformation.</p>';
TUTORIAL_TEXTS['adoption'] = '<p>Adoption Speed is the target variable (0-4). Speed 0 means the pet was adopted the same day the listing went live. Speed 4 means the pet was not adopted after 100+ days. Understanding the class distribution is critical for model selection — imbalanced classes require special handling.</p>';
TUTORIAL_TEXTS['correlation'] = '<p>The correlation matrix reveals linear relationships between numeric features. Strong correlations between Vaccinated/Dewormed/Sterilized suggest multicollinearity. The masked upper triangle avoids redundant information.</p>';
