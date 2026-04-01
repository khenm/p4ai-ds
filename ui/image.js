/* image.js — Combined EDA Chart Rendering (Tabular + Image) */

/* Gallery state */
let GALLERY_DATA = null;
let GALLERY_TYPE = 'Dog';
let GALLERY_BREED = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [
            overview, adoption, demographics, ageFee, stateDist, correlation, health, vaccination,
            imgOverview, dims, photoCount, scatter, colorSpace, qualTable, composite, bestWorst, dominantColors, pca, tsne, cross, gallery,
        ] = await Promise.all([
            loadJSON('tabular_overview.json'),
            loadJSON('tabular_adoption_dist.json'),
            loadJSON('tabular_demographics.json'),
            loadJSON('tabular_age_fee.json'),
            loadJSON('tabular_state_dist.json'),
            loadJSON('tabular_correlation.json'),
            loadJSON('tabular_health.json'),
            loadJSON('tabular_vaccination.json'),
            loadJSON('image_overview.json'),
            loadJSON('image_dimensions.json'),
            loadJSON('image_photo_count.json'),
            loadJSON('image_quality_scatter.json'),
            loadJSON('image_color_space.json'),
            loadJSON('image_quality_table.json'),
            loadJSON('image_quality_composite.json'),
            loadJSON('image_best_worst.json'),
            loadJSON('image_dominant_colors.json'),
            loadJSON('image_pca.json'),
            loadJSON('image_tsne.json'),
            loadJSON('image_cross_modality.json'),
            loadJSON('image_gallery.json'),
        ]);

        // Tabular
        renderOverview(overview);
        renderAdoption(adoption);
        renderDemographics(demographics);
        renderAgeFee(ageFee);
        renderState(stateDist);
        renderCorrelation(correlation);
        renderHealth(health);
        renderVaccination(vaccination);

        // Image
        renderImgOverview(imgOverview);
        renderDimensions(dims);
        renderPhotoCount(photoCount);
        renderQualityScatter(scatter);
        renderRGBScatter(colorSpace);
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
    const labels = data.labels.map((l, i) => `Speed ${l} — ${data.speed_names[i]}`);
    Plotly.newPlot('chart-adoption-bar', [{
        x: labels, y: data.counts, type: 'bar',
        marker: { color: COLORS.speedColors },
        text: data.counts.map(c => c.toLocaleString()), textposition: 'outside',
    }], plotlyLayout('Adoption Speed Distribution', {
        xaxis: { title: 'Adoption Speed', tickangle: -15 },
        yaxis: { title: 'Count' },
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

/* ═══════════════════════════════════════════════════════ */
/*  IMAGE RENDERERS                                       */
/* ═══════════════════════════════════════════════════════ */

function renderImgOverview(d) {
    document.getElementById('img-overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Total Images</div><div class="stat-value">${d.total_images.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Avg Photos/Pet</div><div class="stat-value">${d.avg_photos_per_pet}</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Median Resolution</div><div class="stat-value">${d.median_resolution}</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Total Size</div><div class="stat-value">${d.total_size_gb} GB</div></div>`;
    document.getElementById('img-overview-desc').innerHTML = `The PetFinder image dataset contains <strong>${d.total_images.toLocaleString()}</strong> pet photos across all listings. On average, each pet has <strong>${d.avg_photos_per_pet}</strong> photos. The median resolution is <strong>${d.median_resolution}</strong> pixels with an average file size of <strong>${d.avg_file_size_kb} KB</strong>.`;
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

    document.getElementById('photo-stats').innerHTML = `
        <div class="stat-card stat-green"><div class="stat-label">Avg Photos (Fast Adopted)</div><div class="stat-value">${d.fast_adopted_avg}</div><div class="stat-sub">Speed 0 &ndash; 1</div></div>
        <div class="stat-card stat-red"><div class="stat-label">Avg Photos (Slow/Not Adopted)</div><div class="stat-value">${d.slow_adopted_avg}</div><div class="stat-sub">Speed 3 &ndash; 4</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Correlation (r)</div><div class="stat-value">${d.correlation}</div><div class="stat-sub">PhotoAmt vs AdoptionSpeed</div></div>`;

    document.getElementById('photo-insights').innerHTML = `
        <li><strong>Pets with more photos get adopted faster</strong> — fast-adopted avg: ${d.fast_adopted_avg} vs slow: ${d.slow_adopted_avg}</li>
        <li>Correlation coefficient: r = ${d.correlation} (negative = more photos lead to faster adoption)</li>
        <li>Photo count is the <strong>strongest visual predictor</strong> from this dataset</li>
        <li>Encourage shelters to upload multiple high-quality photos per listing</li>`;
}

function renderQualityScatter(d) {
    const dogIdx = d.types.map((t, i) => t === 'Dog' ? i : -1).filter(i => i >= 0);
    const catIdx = d.types.map((t, i) => t === 'Cat' ? i : -1).filter(i => i >= 0);

    Plotly.newPlot('chart-quality-scatter', [
        { x: dogIdx.map(i => d.sharpness[i]), y: dogIdx.map(i => d.contrast[i]), mode: 'markers', name: 'Dog', marker: { color: COLORS.qualitative[0], size: 3, opacity: 0.4 } },
        { x: catIdx.map(i => d.sharpness[i]), y: catIdx.map(i => d.contrast[i]), mode: 'markers', name: 'Cat', marker: { color: COLORS.qualitative[2], size: 3, opacity: 0.4 } },
    ], plotlyLayout('Image Quality: Sharpness vs Contrast', {
        xaxis: { title: 'Sharpness (Laplacian Var)' }, yaxis: { title: 'Contrast (Std Dev)' },
    }), PLOTLY_CONFIG);
}

function renderRGBScatter(d) {
    Plotly.newPlot('chart-rgb-scatter', [{
        x: d.r, y: d.g, z: d.b, mode: 'markers', type: 'scatter3d',
        marker: { size: 2, color: d.brightness, colorscale: 'Viridis', opacity: 0.6, colorbar: { title: 'Brightness' } },
    }], plotlyLayout('RGB Color Space Distribution', {
        scene: { xaxis: { title: 'Red' }, yaxis: { title: 'Green' }, zaxis: { title: 'Blue' } },
        margin: { l: 0, r: 0, t: 50, b: 0 }, height: 450,
    }), PLOTLY_CONFIG);
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

    grid.innerHTML = images.map(img => `
        <div>
            <img src="${img.path}" alt="${img.name || img.pet_id}" loading="lazy">
            <span class="speed-badge speed-${img.speed}">${img.speed}</span>
            <div class="img-label">${img.name || img.pet_id} &middot; ${img.age}mo</div>
        </div>
    `).join('');
}
