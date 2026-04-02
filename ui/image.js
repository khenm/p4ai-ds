/* image.js — Combined EDA Chart Rendering (Tabular + Image) */

/* Gallery state */
let GALLERY_DATA = null;
let GALLERY_TYPE = 'Dog';
let GALLERY_BREED = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [
            overview, adoption, correlation, health, vaccination,
            ageSpeed, feeDescSpeed,
            imgOverview, dims, photoCount, qualTable, composite, bestWorst, dominantColors, pca, tsne, cross, gallery,
            breedClusters,
        ] = await Promise.all([
            loadJSON('tabular_overview.json'),
            loadJSON('tabular_adoption_dist.json'),
            loadJSON('tabular_correlation.json'),
            loadJSON('tabular_health.json'),
            loadJSON('tabular_vaccination.json'),
            loadJSON('tabular_age_speed.json').catch(() => null),
            loadJSON('tabular_fee_desc_speed.json').catch(() => null),
            loadJSON('image_overview.json'),
            loadJSON('image_dimensions.json'),
            loadJSON('image_photo_count.json'),
            loadJSON('image_quality_table.json'),
            loadJSON('image_quality_composite.json'),
            loadJSON('image_best_worst.json'),
            loadJSON('image_dominant_colors.json'),
            loadJSON('image_pca.json'),
            loadJSON('image_tsne.json'),
            loadJSON('image_cross_modality.json'),
            loadJSON('image_gallery.json'),
            loadJSON('image_breed_clusters.json').catch(() => null),
        ]);

        // Tabular
        renderOverview(overview);
        renderAdoption(adoption);
        renderCorrelation(correlation);
        renderHealth(health);
        renderVaccination(vaccination);
        if (ageSpeed) renderAgeSpeed(ageSpeed);
        if (feeDescSpeed) renderFeeDescSpeed(feeDescSpeed);

        // Image
        if (breedClusters) renderBreedClusters(breedClusters);
        renderImgOverview(imgOverview);
        renderDimensions(dims);
        renderPhotoCount(photoCount);
        renderQualityTable(qualTable);
        renderComposite(composite);
        renderBestWorst(bestWorst);
        renderDominantColors(dominantColors);
        renderPCA(pca);
        renderTSNE(tsne);
        renderCrossModality(cross);

        // Gallery
        GALLERY_DATA = gallery;
        initGallery();
    } catch (e) {
        console.error('Failed to load data:', e);
    }
});

/* ═══════════════════════════════════════════════════════ */
/*  TABULAR RENDERERS                                     */
/* ═══════════════════════════════════════════════════════ */

function renderOverview(data) {
    document.getElementById('overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Total Listings</div><div class="stat-value">${data.total_listings.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Features</div><div class="stat-value">${data.feature_count}</div><div class="stat-sub">${data.feature_count} columns</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Dog Listings</div><div class="stat-value">${data.dog_count.toLocaleString()}</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Cat Listings</div><div class="stat-value">${data.cat_count.toLocaleString()}</div></div>`;

    document.getElementById('overview-desc').innerHTML = `The PetFinder.my dataset contains <strong>${data.total_listings.toLocaleString()}</strong> pet adoption listings from Malaysia. Each listing has <strong>${data.feature_count}</strong> features including demographics (Type, Age, Gender, Breed), health info (Vaccinated, Dewormed, Sterilized), listing metadata (Fee, PhotoAmt, VideoAmt), and the target variable <strong>AdoptionSpeed</strong> (0 &ndash; 4).`;

    const cols = data.display_columns;
    let html = '<thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    data.sample_rows.forEach(row => {
        html += '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>';
    });
    html += '</tbody>';
    document.getElementById('sample-table').innerHTML = html;

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
    }], plotlyLayout('Dog vs Cat Proportion', { margin: { t: 50, b: 10 } }), PLOTLY_CONFIG);

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
            xaxis: { title: '' }, yaxis: { title: 'Count' },
        }), PLOTLY_CONFIG);
    });
}

function renderAgeFee(data) {
    const container = document.getElementById('age-fee-charts');

    const ageDiv = document.createElement('div');
    ageDiv.className = 'chart-container';
    ageDiv.innerHTML = '<div class="chart-box" id="chart-age-hist"></div><p class="chart-description">Age distribution (in months) by animal type.</p>';
    container.appendChild(ageDiv);

    Plotly.newPlot('chart-age-hist', [
        { x: data.age_dog, type: 'histogram', name: 'Dog', opacity: 0.7, marker: { color: COLORS.qualitative[0] }, nbinsx: 40 },
        { x: data.age_cat, type: 'histogram', name: 'Cat', opacity: 0.7, marker: { color: COLORS.qualitative[2] }, nbinsx: 40 },
    ], plotlyLayout('Age Distribution by Type', {
        barmode: 'overlay', xaxis: { title: 'Age (months)', range: [0, 100] }, yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);

    const feeDiv = document.createElement('div');
    feeDiv.className = 'chart-container';
    feeDiv.innerHTML = '<div class="chart-box" id="chart-fee-speed"></div><p class="chart-description">Fee distribution by adoption speed (capped at 500 RM).</p>';
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
        yaxis: { title: 'Fee (RM)', range: [0, 500] }, xaxis: { title: 'Adoption Speed' },
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
    const masked = data.matrix.map((row, i) => row.map((v, j) => j > i ? null : v));

    Plotly.newPlot('chart-corr', [{
        z: masked, x: data.labels, y: data.labels,
        type: 'heatmap',
        colorscale: [[0, '#6886A5'], [0.5, '#E9C46A'], [1, '#C26A2E']],
        zmin: -1, zmax: 1,
        hoverongaps: false,
    }], plotlyLayout('Numeric Features Correlation Matrix', {
        height: 550, margin: { l: 100, b: 100, t: 50 },
        xaxis: { tickangle: -45 }, yaxis: {},
    }), PLOTLY_CONFIG);

    let html = '<thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr></thead><tbody>';
    data.top_pairs.slice(0, 10).forEach(p => {
        const color = p.correlation > 0 ? '#5E8A5C' : '#C7727A';
        html += `<tr><td>${p.feature1}</td><td>${p.feature2}</td><td class="num" style="color:${color}">${p.correlation.toFixed(4)}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('corr-pairs-table').innerHTML = html;

    const top3 = data.top_pairs.slice(0, 3);
    document.getElementById('corr-insights').innerHTML = `
        <li>Strongest correlation: <strong>${top3[0].feature1} &harr; ${top3[0].feature2}</strong> (r=${top3[0].correlation.toFixed(3)})</li>
        <li>Vaccinated, Dewormed, Sterilized are highly correlated — responsible pet owners tend to do all three</li>
        <li>Fee and Age show weak positive correlation with AdoptionSpeed</li>
        <li>PhotoAmt shows negative correlation with AdoptionSpeed — more photos lead to faster adoption</li>`;
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
    Plotly.newPlot('chart-vacc', traces, plotlyLayout('Vaccination x Adoption Speed', {
        barmode: 'stack', xaxis: { title: 'Vaccination Status' }, yaxis: { title: 'Proportion' },
    }), PLOTLY_CONFIG);
}

function renderBreedSpeed(data) {
    ['Dog', 'Cat'].forEach(type => {
        const d = data[type];
        const chartId = `chart-breed-${type.toLowerCase()}`;
        const traces = [];
        for (let s = 0; s < 5; s++) {
            traces.push({
                y: d.breeds,
                x: d.speed_proportions[String(s)].map(v => +(v * 100).toFixed(1)),
                type: 'bar',
                orientation: 'h',
                name: `Speed ${s} — ${COLORS.speedNames[s]}`,
                marker: { color: COLORS.speedColors[s] },
                hovertemplate: '%{y}: %{x:.1f}%<extra>Speed ' + s + '</extra>',
            });
        }
        Plotly.newPlot(chartId, traces, plotlyLayout(`${type} — Top 15 Breeds by Adoption Speed`, {
            barmode: 'stack',
            xaxis: { title: 'Proportion (%)', range: [0, 100] },
            yaxis: { title: '', automargin: true, tickfont: { size: 11 } },
            margin: { l: 160, r: 20, t: 50, b: 50 },
            height: 480,
            legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' },
        }), PLOTLY_CONFIG);
    });

    // Derive simple insights from the Dog data
    const dog = data['Dog'];
    const fastestBreed = dog.breeds.reduce((best, breed, i) => {
        const fast = (dog.speed_proportions['0'][i] || 0) + (dog.speed_proportions['1'][i] || 0);
        return fast > best.val ? { name: breed, val: fast } : best;
    }, { name: '', val: 0 });
    const slowestBreed = dog.breeds.reduce((best, breed, i) => {
        const slow = dog.speed_proportions['4'][i] || 0;
        return slow > best.val ? { name: breed, val: slow } : best;
    }, { name: '', val: 0 });
    document.getElementById('breed-speed-insights').innerHTML = `
        <li>Fastest-adopted dog breed: <strong>${fastestBreed.name}</strong> (${(fastestBreed.val * 100).toFixed(1)}% adopted within first week)</li>
        <li>Highest "no adoption" rate among dogs: <strong>${slowestBreed.name}</strong> (${(slowestBreed.val * 100).toFixed(1)}% Speed 4)</li>
        <li>Mixed Breed dogs and cats typically show average adoption rates, reflecting the dataset's diversity</li>
        <li>Breed alone is a weak predictor — photo count remains stronger across all breeds</li>`;
}

/* ── Shared helper: proportional horizontal stacked bar ── */
function _propHBar(divId, title, labels, propsObj, height) {
    const traces = [];
    for (let s = 0; s < 5; s++) {
        traces.push({
            y: labels,
            x: (propsObj[String(s)] || []).map(v => +(v * 100).toFixed(1)),
            type: 'bar', orientation: 'h',
            name: `Speed ${s} — ${COLORS.speedNames[s]}`,
            marker: { color: COLORS.speedColors[s] },
            hovertemplate: '%{y}: %{x:.1f}%<extra>Speed ' + s + '</extra>',
        });
    }
    Plotly.newPlot(divId, traces, plotlyLayout(title, {
        barmode: 'stack',
        xaxis: { title: 'Proportion (%)', range: [0, 100] },
        yaxis: { title: '', automargin: true, tickfont: { size: 11 } },
        margin: { l: 160, r: 20, t: 50, b: 50 },
        height: height || 320,
        legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
    }), PLOTLY_CONFIG);
}

/* ── Shared helper: proportional vertical stacked bar ── */
function _propVBar(divId, title, labels, propsObj) {
    const traces = [];
    for (let s = 0; s < 5; s++) {
        traces.push({
            x: labels,
            y: (propsObj[String(s)] || []).map(v => +(v * 100).toFixed(1)),
            type: 'bar',
            name: `Speed ${s} — ${COLORS.speedNames[s]}`,
            marker: { color: COLORS.speedColors[s] },
            hovertemplate: '%{x}: %{y:.1f}%<extra>Speed ' + s + '</extra>',
        });
    }
    Plotly.newPlot(divId, traces, plotlyLayout(title, {
        barmode: 'stack',
        xaxis: { title: '' },
        yaxis: { title: 'Proportion (%)' },
        legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
    }), PLOTLY_CONFIG);
}

function renderStateSpeed(data) {
    _propHBar('chart-state-speed', 'State vs Adoption Speed (Top 10)',
        data.states.slice().reverse(),
        Object.fromEntries(Object.entries(data.speed_proportions)
            .map(([k, v]) => [k, v.slice().reverse()])), 420);

    const fastest = data.states.reduce((best, s, i) => {
        const fast = (data.speed_proportions['0'][i] || 0) + (data.speed_proportions['1'][i] || 0);
        return fast > best.val ? { name: s, val: fast } : best;
    }, { name: '', val: 0 });
    const slowest = data.states.reduce((best, s, i) => {
        const slow = data.speed_proportions['4'][i] || 0;
        return slow > best.val ? { name: s, val: slow } : best;
    }, { name: '', val: 0 });
    document.getElementById('state-speed-insights').innerHTML = `
        <li>Fastest-adopting state: <strong>${fastest.name}</strong> (${(fastest.val * 100).toFixed(1)}% adopted within first week)</li>
        <li>Highest "no adoption" rate: <strong>${slowest.name}</strong> (${(slowest.val * 100).toFixed(1)}% Speed 4)</li>
        <li>Selangor and Kuala Lumpur account for the largest share of listings — urban areas dominate</li>`;
}

function renderProfileSpeed(data) {
    _propVBar('chart-hasname-speed', 'Has Name vs Adoption Speed',
        data.hasname.labels, data.hasname.proportions);
    _propVBar('chart-type-speed', 'Pet Type vs Adoption Speed',
        data.type_speed.labels, data.type_speed.proportions);
    _propVBar('chart-purebreed-speed', 'Pure Breed vs Adoption Speed',
        data.purebreed.labels, data.purebreed.proportions);
}

function renderAgeSpeed(data) {
    ['dog', 'cat'].forEach(type => {
        const divId = `chart-age-speed-${type}`;
        const traces = [];
        for (let s = 0; s < 5; s++) {
            const vals = (data[type][String(s)] || []).filter(v => v <= 120);
            traces.push({
                y: vals, type: 'box',
                name: `Speed ${s}`,
                marker: { color: COLORS.speedColors[s] },
                boxpoints: false,
            });
        }
        Plotly.newPlot(divId, traces, plotlyLayout(
            `Age vs Adoption Speed — ${type.charAt(0).toUpperCase() + type.slice(1)}`, {
            yaxis: { title: 'Age (months)', range: [0, 60] },
            xaxis: { title: 'Adoption Speed' },
        }), PLOTLY_CONFIG);
    });
}

function renderGroupSpeed(data) {
    _propHBar('chart-quantity-speed', 'Quantity Group vs Adoption Speed',
        data.quantity.labels, data.quantity.proportions, 360);
    _propVBar('chart-gender-speed', 'Gender vs Adoption Speed',
        data.gender.labels, data.gender.proportions);
    _propHBar('chart-maturity-dog', 'Maturity Size vs Adoption Speed — Dog',
        data.maturity_dog.labels, data.maturity_dog.proportions, 300);
    _propHBar('chart-maturity-cat', 'Maturity Size vs Adoption Speed — Cat',
        data.maturity_cat.labels, data.maturity_cat.proportions, 300);
}

function renderCareSpeed(data) {
    _propVBar('chart-dewormed-speed', 'Dewormed vs Adoption Speed',
        data.dewormed.labels, data.dewormed.proportions);
    _propVBar('chart-sterilized-speed', 'Sterilized vs Adoption Speed',
        data.sterilized.labels, data.sterilized.proportions);
}

function renderFeeDescSpeed(data) {
    _propHBar('chart-fee-speed2', 'Fee Group vs Adoption Speed',
        data.fee.labels, data.fee.proportions, 360);
}

function renderBreedClusters(data) {
    const CLUSTER_COLORS = ['#E76F51', '#2A9D8F', '#E9C46A', '#264653', '#A8DADC'];

    ['Dog', 'Cat'].forEach(type => {
        const d = data[type];
        if (!d) return;
        const tl = type.toLowerCase();
        const { breeds, similarity, cluster_labels, n_clusters, cluster_breeds,
                cluster_speed, breed_speed, feature_names, cluster_profiles } = d;

        // ── 1. Similarity heatmap ─────────────────────────────────────────
        // Sort rows/cols by cluster label so similar breeds appear adjacent
        const order = breeds.map((_, i) => i).sort((a, b) => cluster_labels[a] - cluster_labels[b]);
        const ordBreeds = order.map(i => breeds[i]);
        const ordSim = order.map(ri => order.map(ci => similarity[ri][ci]));

        Plotly.newPlot(`chart-breed-sim-${tl}`, [{
            z: ordSim.slice().reverse(),
            x: ordBreeds,
            y: ordBreeds.slice().reverse(),
            type: 'heatmap',
            colorscale: [[0, '#264653'], [0.5, '#E9C46A'], [1, '#E76F51']],
            zmin: -1, zmax: 1,
            hoverongaps: false,
            hovertemplate: '%{y} × %{x}: %{z:.3f}<extra></extra>',
        }], plotlyLayout(`${type} Breed Image Similarity`, {
            height: 460,
            margin: { l: 150, b: 140, t: 50, r: 20 },
            xaxis: { tickangle: -45, tickfont: { size: 10 } },
            yaxis: { tickfont: { size: 10 } },
        }), PLOTLY_CONFIG);

        // ── 2. Cluster vs Adoption Speed ─────────────────────────────────
        const clusterNames = Array.from({ length: n_clusters }, (_, c) => `Cluster ${c + 1}`);
        const clTraces = [];
        for (let s = 0; s < 5; s++) {
            clTraces.push({
                x: clusterNames,
                y: Array.from({ length: n_clusters }, (_, c) =>
                    +((cluster_speed[String(c)]?.[s] || 0) * 100).toFixed(1)),
                type: 'bar',
                name: `Speed ${s} — ${COLORS.speedNames[s]}`,
                marker: { color: COLORS.speedColors[s] },
                hovertemplate: '%{x}: %{y:.1f}%<extra>Speed ' + s + '</extra>',
            });
        }
        Plotly.newPlot(`chart-cluster-speed-${tl}`, clTraces, plotlyLayout(
            `${type} Visual Clusters vs Adoption Speed`, {
            barmode: 'stack',
            xaxis: { title: 'Visual Cluster' },
            yaxis: { title: 'Proportion (%)' },
            legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
        }), PLOTLY_CONFIG);

        // ── 3. Cluster composition table ──────────────────────────────────
        let html = `<table class="data-table"><thead><tr><th>Cluster</th><th>Breeds</th>`;
        feature_names.forEach(f => { html += `<th>${f}</th>`; });
        html += `</tr></thead><tbody>`;
        for (let c = 0; c < n_clusters; c++) {
            const breedList = (cluster_breeds[String(c)] || []).join(', ');
            const prof = cluster_profiles[String(c)] || {};
            const dot = `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${CLUSTER_COLORS[c]};margin-right:6px;"></span>`;
            html += `<tr><td>${dot}Cluster ${c + 1}</td><td>${breedList}</td>`;
            feature_names.forEach(f => {
                const v = prof[f];
                html += `<td class="num">${v != null ? v.toFixed(2) : '—'}</td>`;
            });
            html += `</tr>`;
        }
        html += `</tbody></table>`;
        document.getElementById(`cluster-table-${tl}`).innerHTML = html;

        // ── 4. Breed vs adoption speed (image-sampled) ────────────────────
        const bsTraces = [];
        for (let s = 0; s < 5; s++) {
            bsTraces.push({
                y: breeds,
                x: (breed_speed[String(s)] || []).map(v => +(v * 100).toFixed(1)),
                type: 'bar', orientation: 'h',
                name: `Speed ${s} — ${COLORS.speedNames[s]}`,
                marker: { color: COLORS.speedColors[s] },
                hovertemplate: '%{y}: %{x:.1f}%<extra>Speed ' + s + '</extra>',
            });
        }
        Plotly.newPlot(`chart-breed-img-speed-${tl}`, bsTraces, plotlyLayout(
            `${type} Breed vs Adoption Speed (Image Sample)`, {
            barmode: 'stack',
            xaxis: { title: 'Proportion (%)', range: [0, 100] },
            yaxis: { automargin: true, tickfont: { size: 11 } },
            margin: { l: 160, r: 20, t: 50, b: 50 },
            height: 480,
            legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' },
        }), PLOTLY_CONFIG);
    });

    // ── Cross-correlation heatmap: Dog × Cat ─────────────────────────────
    const cross = data.cross_similarity;
    if (cross) {
        Plotly.newPlot('chart-cross-sim', [{
            z: cross.similarity.slice().reverse(),
            x: cross.cat_breeds,
            y: cross.dog_breeds.slice().reverse(),
            type: 'heatmap',
            colorscale: [[0, '#264653'], [0.5, '#E9C46A'], [1, '#E76F51']],
            zmin: -1, zmax: 1,
            hoverongaps: false,
            hovertemplate: 'Dog: %{y}<br>Cat: %{x}<br>Similarity: %{z:.3f}<extra></extra>',
        }], plotlyLayout('Cross-Correlation: Dog Breeds × Cat Breeds', {
            height: 500,
            margin: { l: 160, b: 160, t: 50, r: 20 },
            xaxis: { tickangle: -45, tickfont: { size: 10 }, title: 'Cat Breeds' },
            yaxis: { tickfont: { size: 10 }, title: 'Dog Breeds' },
        }), PLOTLY_CONFIG);
    }

    // ── Combined clustering chart ─────────────────────────────────────────
    const comb = data.combined;
    if (comb) {
        const { breeds, types, n_clusters, cluster_breeds, cluster_speed } = comb;

        // Cluster vs adoption speed
        const clNames = Array.from({ length: n_clusters }, (_, c) => `Cluster ${c + 1}`);
        const combSpeedTraces = [];
        for (let s = 0; s < 5; s++) {
            combSpeedTraces.push({
                x: clNames,
                y: Array.from({ length: n_clusters }, (_, c) =>
                    +((cluster_speed[String(c)]?.[s] || 0) * 100).toFixed(1)),
                type: 'bar',
                name: `Speed ${s} — ${COLORS.speedNames[s]}`,
                marker: { color: COLORS.speedColors[s] },
            });
        }
        Plotly.newPlot('chart-combined-speed', combSpeedTraces,
            plotlyLayout('Combined Clusters vs Adoption Speed', {
                barmode: 'stack',
                xaxis: { title: 'Cluster' },
                yaxis: { title: 'Proportion (%)' },
                legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
            }), PLOTLY_CONFIG);

        // Cluster composition table
        let html = `<table class="data-table"><thead><tr><th>Cluster</th><th>Dogs</th><th>Cats</th></tr></thead><tbody>`;
        for (let c = 0; c < n_clusters; c++) {
            const members = cluster_breeds[String(c)] || [];
            const dogs = members.filter(m => m.type === 'Dog').map(m => m.breed).join(', ') || '—';
            const cats = members.filter(m => m.type === 'Cat').map(m => m.breed).join(', ') || '—';
            html += `<tr><td><strong>Cluster ${c + 1}</strong></td>
                <td><span style="color:#4A90D9">${dogs}</span></td>
                <td><span style="color:#E76F51">${cats}</span></td></tr>`;
        }
        html += `</tbody></table>`;
        document.getElementById('combined-cluster-table').innerHTML = html;
    }

    // ── Insights ────────────────────────────────────────────────────────
    const dog = data['Dog'];
    if (dog) {
        const fastestCluster = comb
            ? Object.entries(comb.cluster_speed).reduce((best, [c, speeds]) => {
                const fast = (speeds[0] || 0) + (speeds[1] || 0);
                return fast > best.val ? { c: Number(c) + 1, val: fast } : best;
            }, { c: 1, val: 0 }) : { c: 1, val: 0 };
        const mixedClusters = comb
            ? Object.entries(comb.cluster_breeds).filter(([, members]) =>
                members.some(m => m.type === 'Dog') && members.some(m => m.type === 'Cat')).length
            : 0;
        document.getElementById('breed-cluster-insights').innerHTML = `
            <li>Clusters reflect shared image quality profiles (brightness, sharpness, contrast) — not physical appearance</li>
            <li><strong>${mixedClusters}</strong> of ${comb?.n_clusters || 0} combined clusters contain both dog and cat breeds, showing cross-species visual overlap</li>
            <li>Combined Cluster ${fastestCluster.c} has the fastest adoption profile (Speed 0–1: ${(fastestCluster.val * 100).toFixed(1)}%)</li>
            <li>Cross-correlation map reveals which dog and cat breeds share a similar photographic style</li>
            <li>Cosine similarity near 1 = nearly identical image quality profiles across species</li>`;
    }
}

/* ═══════════════════════════════════════════════════════ */
/*  IMAGE RENDERERS                                       */
/* ═══════════════════════════════════════════════════════ */

function renderImgOverview(d) {
    document.getElementById('img-overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Total Images</div><div class="stat-value">${d.total_images.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Avg Photos/Pet</div><div class="stat-value">${d.avg_photos_per_pet}</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Median Resolution</div><div class="stat-value">${d.median_resolution}</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Total Size</div><div class="stat-value">${d.total_size_gb} GB</div></div>`;
    document.getElementById('img-split-stats').innerHTML = `
        <div class="stat-card stat-green"><div class="stat-label">Train Photos</div><div class="stat-value">${d.train_images.toLocaleString()}</div><div class="stat-sub">${d.train_listings.toLocaleString()} listings</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Test Photos</div><div class="stat-value">${d.test_images.toLocaleString()}</div><div class="stat-sub">${d.test_listings.toLocaleString()} listings</div></div>`;
    document.getElementById('img-overview-desc').innerHTML = `The PetFinder image dataset contains <strong>${d.total_images.toLocaleString()}</strong> pet photos across <strong>${d.total_listings.toLocaleString()}</strong> listings (train + test). On average, each pet has <strong>${d.avg_photos_per_pet}</strong> photos. The median resolution is <strong>${d.median_resolution}</strong> pixels with an average file size of <strong>${d.avg_file_size_kb} KB</strong>.`;
}

function renderDimensions(d) {
    const dogIdx = d.types.map((t, i) => t === 'Dog' ? i : -1).filter(i => i >= 0);
    const catIdx = d.types.map((t, i) => t === 'Cat' ? i : -1).filter(i => i >= 0);

    Plotly.newPlot('chart-size-scatter', [
        { x: dogIdx.map(i => d.widths[i]), y: dogIdx.map(i => d.heights[i]), mode: 'markers', type: 'scatter', name: 'Dog', marker: { color: COLORS.qualitative[0], size: 3, opacity: 0.4 } },
        { x: catIdx.map(i => d.widths[i]), y: catIdx.map(i => d.heights[i]), mode: 'markers', type: 'scatter', name: 'Cat', marker: { color: COLORS.qualitative[2], size: 3, opacity: 0.4 } },
    ], plotlyLayout('Image Size: Width vs Height', {
        xaxis: { title: 'Width (px)' }, yaxis: { title: 'Height (px)' }, height: 450,
    }), PLOTLY_CONFIG);

    Plotly.newPlot('chart-filesize', [
        { x: dogIdx.map(i => d.file_sizes[i]), type: 'histogram', name: 'Dog', opacity: 0.7, marker: { color: COLORS.qualitative[0] }, nbinsx: 50 },
        { x: catIdx.map(i => d.file_sizes[i]), type: 'histogram', name: 'Cat', opacity: 0.7, marker: { color: COLORS.qualitative[2] }, nbinsx: 50 },
    ], plotlyLayout('File Size Distribution', {
        barmode: 'overlay', xaxis: { title: 'File Size (KB)' }, yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);

    const shapes = d.reference_lines.map(r => ({
        type: 'line', x0: r.ratio, x1: r.ratio, y0: 0, y1: 1, yref: 'paper',
        line: { color: COLORS.diverging[2], width: 1, dash: 'dash' },
    }));
    const annotations = d.reference_lines.map(r => ({
        x: r.ratio, y: 1, yref: 'paper', text: r.label,
        showarrow: false, font: { size: 10, color: COLORS.diverging[2] }, yanchor: 'bottom',
    }));

    Plotly.newPlot('chart-aspect', [{
        x: d.aspect_ratios, type: 'histogram', nbinsx: 60,
        marker: { color: COLORS.qualitative[3] }, opacity: 0.8,
    }], plotlyLayout('Aspect Ratio Distribution', {
        xaxis: { title: 'Aspect Ratio (W/H)', range: [0.3, 2.5] }, yaxis: { title: 'Count' },
        shapes, annotations,
    }), PLOTLY_CONFIG);
}

function renderPhotoCount(d) {
    Plotly.newPlot('chart-photo-hist', [{
        x: d.photo_amounts, type: 'histogram', nbinsx: 30,
        marker: { color: COLORS.sequential[2] },
    }], plotlyLayout('Photos per Pet Distribution', {
        xaxis: { title: 'Number of Photos', range: [0, 30] }, yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);

    const boxTraces = [];
    for (let s = 0; s < 5; s++) {
        boxTraces.push({
            y: d.speed_stats[String(s)].values,
            type: 'box', name: `Speed ${s} — ${COLORS.speedNames[s]}`,
            marker: { color: COLORS.speedColors[s] },
        });
    }
    Plotly.newPlot('chart-photo-speed', boxTraces, plotlyLayout('Photo Count vs Adoption Speed', {
        yaxis: { title: 'Number of Photos', range: [0, 15] }, xaxis: { title: '' },
    }), PLOTLY_CONFIG);

    document.getElementById('photo-insights').innerHTML = `
        <li>Photo count can show a general trend — pets with more images receive more views, which increases the chance of being noticed and adopted</li>
        <li>Pets with <strong>no adoption (Speed 4)</strong> tend to have fewer photos, suggesting that limited visibility may contribute to rejection</li>
        <li>More photos likely means more exposure rather than a direct causal factor in adoption outcome</li>`;
}

function renderQualityTable(d) {
    let html = '<thead><tr><th>Metric</th>';
    for (let s = 0; s < 5; s++) html += `<th class="num">Speed ${s}</th>`;
    html += '</tr></thead><tbody>';
    d.metrics.forEach(m => {
        html += `<tr><td>${m}</td>`;
        for (let s = 0; s < 5; s++) html += `<td class="num">${d.table[m][String(s)]}</td>`;
        html += '</tr>';
    });
    html += '</tbody>';
    document.getElementById('quality-table').innerHTML = html;
}

function renderComposite(d) {
    const traces = [];
    for (let s = 0; s < 5; s++) {
        traces.push({
            y: d.scores_by_speed[String(s)], type: 'violin', name: `Speed ${s}`,
            box: { visible: true }, meanline: { visible: true },
            marker: { color: COLORS.speedColors[s] },
        });
    }
    Plotly.newPlot('chart-composite', traces, plotlyLayout('Composite Quality Score vs Adoption Speed', {
        yaxis: { title: 'Quality Score (Normalized)' },
    }), PLOTLY_CONFIG);
}

function renderDominantColors(d) {
    const speedContainer = document.getElementById('color-swatches-speed');
    for (let s = 0; s < 5; s++) {
        const palettes = d.by_speed[String(s)] || [];
        const label = `Speed ${s} — ${COLORS.speedNames[s]}`;
        let html = `<div class="speed-row-label">${label}</div><div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">`;
        palettes.forEach(p => {
            html += '<div style="display:flex; gap:2px;">';
            p.colors.forEach(c => {
                html += `<div class="color-swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})" title="rgb(${c.join(',')})"></div>`;
            });
            html += '</div>';
        });
        html += '</div>';
        speedContainer.innerHTML += html;
    }

    const typeContainer = document.getElementById('color-swatches-type');
    ['Dog', 'Cat'].forEach(t => {
        const palettes = d.by_type[t] || [];
        let html = `<div class="speed-row-label">${t}</div><div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">`;
        palettes.forEach(p => {
            html += '<div style="display:flex; gap:2px;">';
            p.colors.forEach(c => {
                html += `<div class="color-swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})" title="rgb(${c.join(',')})"></div>`;
            });
            html += '</div>';
        });
        html += '</div>';
        typeContainer.innerHTML += html;
    });
}

function renderBestWorst(d) {
    const bestGrid = document.getElementById('best-grid');
    d.best.forEach(img => {
        bestGrid.innerHTML += `<div><img src="${img.path}" alt="Best ${img.pet_id}" loading="lazy"><div class="img-label">Score: ${img.score} | ${img.type} | Speed ${img.speed}</div></div>`;
    });

    const worstGrid = document.getElementById('worst-grid');
    d.worst.forEach(img => {
        worstGrid.innerHTML += `<div><img src="${img.path}" alt="Worst ${img.pet_id}" loading="lazy"><div class="img-label">Score: ${img.score} | ${img.type} | Speed ${img.speed}</div></div>`;
    });
}

function renderPCA(d) {
    Plotly.newPlot('chart-pca', [
        { x: d.components.map((_, i) => `PC${i + 1}`), y: d.explained_variance, type: 'bar', name: 'Individual', marker: { color: COLORS.sequential[2] } },
        { x: d.components.map((_, i) => `PC${i + 1}`), y: d.cumulative_variance, type: 'scatter', mode: 'lines+markers', name: 'Cumulative', marker: { color: COLORS.diverging[2] }, line: { width: 2 } },
    ], plotlyLayout('PCA Explained Variance', {
        yaxis: { title: 'Variance Ratio', range: [0, 1.05] }, xaxis: { title: 'Component' },
    }), PLOTLY_CONFIG);
}

function renderTSNE(d) {
    const traces = [];
    for (let s = 0; s < 5; s++) {
        const idx = d.speeds.map((sp, i) => sp === s ? i : -1).filter(i => i >= 0);
        traces.push({
            x: idx.map(i => d.x[i]), y: idx.map(i => d.y[i]),
            mode: 'markers', type: 'scatter', name: `Speed ${s}`,
            marker: { color: COLORS.speedColors[s], size: 3, opacity: 0.5 },
        });
    }
    Plotly.newPlot('chart-tsne', traces, plotlyLayout('t-SNE Projection of Quality Features', {
        xaxis: { title: 't-SNE 1' }, yaxis: { title: 't-SNE 2' }, height: 450,
    }), PLOTLY_CONFIG);
}

function renderCrossModality(d) {
    Plotly.newPlot('chart-heatmap', [{
        z: d.heatmap.values, x: d.heatmap.x_labels, y: d.heatmap.y_labels,
        type: 'heatmap',
        colorscale: [[0, '#5E8A5C'], [0.5, '#E9C46A'], [1, '#C7727A']],
        colorbar: { title: 'Mean Speed' },
        hoverongaps: false,
    }], plotlyLayout('Photo Count x Quality Score — Mean Adoption Speed', {
        xaxis: { title: 'Quality Bin' }, yaxis: { title: 'Photo Count Bin' },
        margin: { l: 80, t: 50 },
    }), PLOTLY_CONFIG);

    if (d.type_quality) {
        const metrics = ['Brightness', 'Contrast', 'Blurriness', 'Colorfulness', 'Saturation'];
        const dogVals = metrics.map(m => d.type_quality.Dog[m]);
        const catVals = metrics.map(m => d.type_quality.Cat[m]);

        Plotly.newPlot('chart-type-quality', [
            { x: metrics, y: dogVals, type: 'bar', name: 'Dog', marker: { color: COLORS.qualitative[0] } },
            { x: metrics, y: catVals, type: 'bar', name: 'Cat', marker: { color: COLORS.qualitative[2] } },
        ], plotlyLayout('Quality Metrics: Dog vs Cat', {
            barmode: 'group', yaxis: { title: 'Median Value' },
        }), PLOTLY_CONFIG);
    }
}

/* ═══════════════════════════════════════════════════════ */
/*  GALLERY                                               */
/* ═══════════════════════════════════════════════════════ */

function initGallery() {
    if (!GALLERY_DATA) return;
    const breeds = Object.keys(GALLERY_DATA[GALLERY_TYPE] || {});
    GALLERY_BREED = breeds[0] || null;
    buildBreedTabs();
    renderGallery();
}

// eslint-disable-next-line no-unused-vars
function switchType(btn) {
    document.querySelectorAll('.type-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    GALLERY_TYPE = btn.dataset.type;
    const breeds = Object.keys(GALLERY_DATA[GALLERY_TYPE] || {});
    GALLERY_BREED = breeds[0] || null;
    buildBreedTabs();
    renderGallery();
}

function buildBreedTabs() {
    const container = document.getElementById('breed-tabs');
    const breeds = Object.keys(GALLERY_DATA[GALLERY_TYPE] || {});
    container.innerHTML = breeds.map(b => {
        const count = (GALLERY_DATA[GALLERY_TYPE][b] || []).length;
        const active = b === GALLERY_BREED ? ' active' : '';
        return `<button class="gallery-tab${active}" data-breed="${b}" onclick="switchBreed(this)">${b} <span style="opacity:0.6;font-size:0.75rem;">(${count})</span></button>`;
    }).join('');
}

// eslint-disable-next-line no-unused-vars
function switchBreed(btn) {
    document.querySelectorAll('#breed-tabs .gallery-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    GALLERY_BREED = btn.dataset.breed;
    renderGallery();
}

// eslint-disable-next-line no-unused-vars
function renderGallery() {
    const grid = document.getElementById('gallery-grid');
    if (!GALLERY_DATA || !GALLERY_BREED) {
        grid.innerHTML = '<p style="color:var(--text-muted);">No images available.</p>';
        return;
    }

    const limit = parseInt(document.getElementById('gallery-count').value, 10);
    const images = (GALLERY_DATA[GALLERY_TYPE][GALLERY_BREED] || []).slice(0, limit);

    grid.innerHTML = images.map(img => {
        const hasBbox = !!img.bbox_pct;
        const bboxData = hasBbox ? `data-bbox='${JSON.stringify(img.bbox_pct)}' data-bbox-label="${img.bbox_label || ''}"` : '';
        return `
        <div class="gallery-item-wrapper">
            <div class="image-area" ${bboxData}>
                <img src="${img.path}" alt="${img.name || img.pet_id}" loading="lazy">
            </div>
            <span class="speed-badge speed-${img.speed}">${img.speed}</span>
            <div class="img-label">${img.name || img.pet_id} &middot; ${img.age}mo</div>
        </div>
        `;
    }).join('');

    grid.querySelectorAll('.image-area[data-bbox]').forEach(area => {
        const imgEl = area.querySelector('img');
        const applyBbox = () => {
            const bboxPct = JSON.parse(area.dataset.bbox);
            const label = area.dataset.bboxLabel || '';
            const [bx, by, bw, bh] = bboxPct;

            const natW = imgEl.naturalWidth;
            const natH = imgEl.naturalHeight;
            const rendW = imgEl.offsetWidth;
            const rendH = imgEl.offsetHeight;

            if (!natW || !natH || !rendW || !rendH) return;

            // Compute the actual rendered image size inside the contain box
            const scale = Math.min(rendW / natW, rendH / natH);
            const rendImgW = natW * scale;
            const rendImgH = natH * scale;
            const offsetX = (rendW - rendImgW) / 2;
            const offsetY = (rendH - rendImgH) / 2;

            // Map bbox percentages to pixel coords within the rendered image
            const pxX = (bx / 100) * rendImgW + offsetX;
            const pxY = (by / 100) * rendImgH + offsetY;
            const pxW = (bw / 100) * rendImgW;
            const pxH = (bh / 100) * rendImgH;

            // Clamp to the actual image area (not the letterbox padding)
            const clampedLeft = Math.max(offsetX, pxX);
            const clampedTop = Math.max(offsetY, pxY);
            const clampedRight = Math.min(offsetX + rendImgW, pxX + pxW);
            const clampedBottom = Math.min(offsetY + rendImgH, pxY + pxH);
            const clampedW = clampedRight - clampedLeft;
            const clampedH = clampedBottom - clampedTop;

            if (clampedW <= 0 || clampedH <= 0) return;

            area.querySelector('.image-bbox')?.remove();
            const div = document.createElement('div');
            div.className = 'image-bbox';
            div.title = label;
            div.style.cssText = `left:${clampedLeft}px; top:${clampedTop}px; width:${clampedW}px; height:${clampedH}px;`;
            if (label) {
                div.innerHTML = `<span class="bbox-label">${label}</span>`;
            }
            area.appendChild(div);
        };

        if (imgEl.complete && imgEl.naturalWidth) {
            requestAnimationFrame(applyBbox);
        } else {
            imgEl.addEventListener('load', applyBbox, { once: true });
        }
    });
}
