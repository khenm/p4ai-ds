let SCATTER_DATA = {};
let ACTIVE_POLLUTANT = 'CO';

window.addEventListener('DOMContentLoaded', async () => {
    try {
        const [
            overview,
            categoryDist,
            pollutantStats,
            geo,
            correlation,
            pollutantCategories,
            scatter,
            outliers,
            insights,
        ] = await Promise.all([
            loadJSON('air_pollution/air_overview.json'),
            loadJSON('air_pollution/air_category_distribution.json'),
            loadJSON('air_pollution/air_pollutant_analysis.json'),
            loadJSON('air_pollution/air_geographical_analysis.json'),
            loadJSON('air_pollution/air_correlation.json'),
            loadJSON('air_pollution/air_pollutant_categories.json'),
            loadJSON('air_pollution/air_scatter_analysis.json'),
            loadJSON('air_pollution/air_outlier_detection.json'),
            loadJSON('air_pollution/air_insights_recommendations.json'),
        ]);

        renderOverview(overview);
        renderAQICategories(categoryDist);
        renderPollutantStats(pollutantStats);
        renderGeographical(geo);
        renderCorrelation(correlation);
        renderPollutantCategories(pollutantCategories);
        SCATTER_DATA = scatter;
        plotScatter('CO');
        renderOutliers(outliers);
        renderInsights(insights);
    } catch (error) {
        console.error('Failed to load air pollution data', error);
        showToast('Failed to load air pollution datasets. Check console for details.');
    }
});

function renderOverview(data) {
    document.getElementById('overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Total Records</div><div class="stat-value">${data.total_records.toLocaleString()}</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Countries</div><div class="stat-value">${data.country_count}</div></div>
        <div class="stat-card stat-blue"><div class="stat-label">Cities</div><div class="stat-value">${data.city_count.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Features</div><div class="stat-value">${data.feature_count}</div></div>`;

    document.getElementById('overview-desc').innerHTML = `This dataset captures <strong>${data.total_records.toLocaleString()}</strong> city-level AQI readings spanning <strong>${data.country_count}</strong> countries. Each record combines category-grade AQI labels alongside pollutant-specific AQI values for CO, Ozone, NO<sub>2</sub>, and PM2.5. Missingness is minimal except for country fields in a handful of entries.`;

    let columnTable = '<thead><tr><th>Column</th><th>Type</th><th>Non-null</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>';
    data.columns.forEach((col) => {
        columnTable += `<tr><td>${col.name}</td><td>${col.dtype}</td><td class="num">${col.non_null.toLocaleString()}</td><td class="num">${col.missing.toLocaleString()}</td><td class="num">${col.missing_pct}%</td></tr>`;
    });
    columnTable += '</tbody>';
    document.getElementById('columns-table').innerHTML = columnTable;

    const cols = data.display_columns;
    let sampleTable = '<thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    data.sample_rows.forEach(row => {
        sampleTable += '<tr>' + cols.map(c => `<td>${row[c]}</td>`).join('') + '</tr>';
    });
    sampleTable += '</tbody>';
    document.getElementById('sample-table').innerHTML = sampleTable;
}

function renderAQICategories(data) {
    Plotly.newPlot('chart-category', [{
        labels: data.labels,
        values: data.counts,
        type: 'pie',
        hole: 0.45,
        marker: { colors: data.colors },
        textinfo: 'label+percent',
    }], plotlyLayout('AQI Category Mix', { margin: { t: 40, b: 40 } }), PLOTLY_CONFIG);

    Plotly.newPlot('chart-category-bar', [{
        x: data.labels,
        y: data.counts,
        type: 'bar',
        marker: { color: data.colors },
        text: data.counts.map(c => c.toLocaleString()),
        textposition: 'outside',
    }], plotlyLayout('AQI Category Counts', { yaxis: { title: 'Cities' } }), PLOTLY_CONFIG);

    const dominantIdx = data.counts.indexOf(Math.max(...data.counts));
    const dominantLabel = data.labels[dominantIdx];
    const dominantPct = data.percentages[dominantIdx].toFixed(1);
    const goodShare = data.percentages[0].toFixed(1);
    const highRiskShare = data.percentages.slice(3).reduce((a, b) => a + b, 0).toFixed(1);
    document.getElementById('category-insights').innerHTML = `
        <h4><i class="fa-solid fa-lightbulb"></i> Key Observations</h4>
        <ul>
            <li><strong>${dominantLabel}</strong> dominates with ${dominantPct}% of all readings — most cities sit in a moderate-risk range.</li>
            <li>Only ${goodShare}% of readings are "Good", underscoring the rarity of pristine air.</li>
            <li>High-risk categories (Unhealthy and above) still account for ${highRiskShare}% — concentrated in specific regions.</li>
            <li>Watch the "Unhealthy for Sensitive Groups" band as an early-warning signal before hazardous spikes.</li>
        </ul>`;
}

function renderPollutantStats(data) {
    const statCards = Object.entries(data.pollutants).map(([name, stats]) => `
        <div class="stat-card stat-amber">
            <div class="stat-label">${name} Mean AQI</div>
            <div class="stat-value">${stats.mean}</div>
            <div class="stat-sub">Median ${stats.median} | Std ${stats.std}</div>
        </div>`).join('');
    document.getElementById('pollutant-stats').innerHTML = statCards;

    const baseColors = ['#6886A5', '#C7727A', '#5E8A5C', '#9B7DB8'];
    const pollutantEntries = Object.entries(data.pollutants);
    const traces = pollutantEntries.map(([name, stats], idx) => ({
        y: stats.values,
        type: 'box',
        name,
        boxpoints: 'outliers',
        marker: { color: baseColors[idx % baseColors.length] },
    }));
    Plotly.newPlot('chart-pollutant-box', traces, plotlyLayout('Pollutant AQI Distribution', { yaxis: { title: 'AQI' } }), PLOTLY_CONFIG);

    const means = pollutantEntries.map(([name, stats]) => ({ name, value: stats.mean }));
    Plotly.newPlot('chart-pollutant-mean', [{
        x: means.map(m => m.name),
        y: means.map(m => m.value),
        type: 'bar',
        marker: { color: COLORS.sequential.slice(1, 5) },
    }], plotlyLayout('Average AQI by Pollutant', { yaxis: { title: 'Average AQI' } }), PLOTLY_CONFIG);
}

function renderGeographical(data) {
    Plotly.newPlot('chart-geo', [
        { x: data.countries, y: data.aqi_mean, type: 'bar', name: 'Mean AQI', marker: { color: '#C26A2E' } },
        { x: data.countries, y: data.aqi_max, type: 'scatter', mode: 'lines+markers', name: 'Max AQI', marker: { color: '#5E8A5C' } }
    ], plotlyLayout('Top 20 Countries by AQI', {
        xaxis: { title: 'Country', tickangle: -45 },
        yaxis: { title: 'AQI' },
        margin: { b: 120 }
    }), PLOTLY_CONFIG);

    const worst = data.countries[0];
    const best = data.countries[data.countries.length - 1];
    document.getElementById('geo-insights').innerHTML = `
        <h4><i class="fa-solid fa-map-location-dot"></i> Regional Signals</h4>
        <ul>
            <li><strong>${worst}</strong> currently holds the highest mean AQI (${data.aqi_mean[0]}) among the monitored countries.</li>
            <li><strong>${best}</strong> (tail of the ranking) offers a blueprint for low AQI management.</li>
            <li>Record coverage spans ${data.records.reduce((a, b) => a + b, 0).toLocaleString()} measurements across ${data.cities.reduce((a, b) => a + b, 0).toLocaleString()} cities.</li>
            <li>Use this leaderboard to prioritize on-the-ground sensing investments.</li>
        </ul>`;
}

function renderCorrelation(data) {
    Plotly.newPlot('chart-corr', [{
        z: data.matrix,
        x: data.labels,
        y: data.labels,
        type: 'heatmap',
        colorscale: [[0, '#6886A5'], [0.5, '#E9C46A'], [1, '#C26A2E']],
        zmin: -1,
        zmax: 1,
    }], plotlyLayout('Correlation Matrix', {
        height: 450,
        margin: { l: 90, r: 10, t: 40, b: 100 },
        xaxis: { tickangle: -45 },
    }), PLOTLY_CONFIG);

    let corrTable = '<thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr></thead><tbody>';
    data.top_pairs.slice(0, 8).forEach(pair => {
        const color = pair.correlation >= 0 ? '#5E8A5C' : '#C7727A';
        corrTable += `<tr><td>${pair.feature1}</td><td>${pair.feature2}</td><td class="num" style="color:${color}">${pair.correlation.toFixed(3)}</td></tr>`;
    });
    corrTable += '</tbody>';
    document.getElementById('corr-table').innerHTML = corrTable;
}

function renderPollutantCategories(data) {
    const pollutants = Object.keys(data);
    const categories = data[pollutants[0]].labels;
    const categoryColors = data[pollutants[0]].colors;
    const traces = categories.map((cat, idx) => ({
        x: pollutants,
        y: pollutants.map(p => data[p].counts[idx]),
        name: cat,
        type: 'bar',
        marker: { color: categoryColors[idx] || COLORS.qualitative[idx % COLORS.qualitative.length] },
    }));
    Plotly.newPlot('chart-pollutant-categories', traces, plotlyLayout('Category Mix per Pollutant', {
        barmode: 'stack',
        yaxis: { title: 'Records' },
    }), PLOTLY_CONFIG);
}

function switchScatter(button) {
    document.querySelectorAll('.scatter-tab-btn').forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
    const pollutant = button.dataset.pollutant;
    plotScatter(pollutant);
}

function plotScatter(pollutant) {
    ACTIVE_POLLUTANT = pollutant;
    const payload = SCATTER_DATA[pollutant];
    if (!payload) return;
    Plotly.newPlot('chart-scatter', [{
        x: payload.x,
        y: payload.y,
        mode: 'markers',
        type: 'scatter',
        marker: { size: 5, color: '#C26A2E', opacity: 0.6 },
    }], plotlyLayout(`${payload.labels[0]} vs ${payload.labels[1]}`, {
        xaxis: { title: payload.labels[0] },
        yaxis: { title: payload.labels[1] },
    }), PLOTLY_CONFIG);
}

function renderOutliers(data) {
    document.getElementById('outlier-stats').innerHTML = `
        <div class="stat-card stat-red"><div class="stat-label">Outlier Count</div><div class="stat-value">${data.outlier_count}</div></div>
        <div class="stat-card stat-orange"><div class="stat-label">Outlier %</div><div class="stat-value">${data.outlier_pct}%</div></div>
        <div class="stat-card stat-green"><div class="stat-label">IQR Bounds</div><div class="stat-value">${data.bounds.lower} – ${data.bounds.upper}</div></div>`;

    let table = '<thead><tr><th>Country</th><th>City</th><th>AQI</th><th>Category</th><th>PM2.5 AQI</th></tr></thead><tbody>';
    data.top_outliers.forEach(row => {
        table += `<tr><td>${row.country}</td><td>${row.city}</td><td class="num">${row.aqi_value}</td><td>${row.aqi_category}</td><td class="num">${row.pm25}</td></tr>`;
    });
    table += '</tbody>';
    document.getElementById('outlier-table').innerHTML = table;
}

function renderInsights(data) {
    document.getElementById('insight-list').innerHTML = data.key_findings.map(item => `<li>${item}</li>`).join('');
    document.getElementById('recommendation-list').innerHTML = data.recommendations.map(item => `<li>${item}</li>`).join('');
}
