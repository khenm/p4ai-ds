/* text.js — Text EDA Chart Rendering */

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [overview, categories, lengths, timeline, quality, terms, lengthByCategory, yearlyTrends, categoryTerms, tfidfTerms] = await Promise.all([
            loadJSON('text_overview.json'),
            loadJSON('text_category_dist.json'),
            loadJSON('text_lengths.json'),
            loadJSON('text_timeline.json'),
            loadJSON('text_quality.json'),
            loadJSON('text_terms.json'),
            loadJSON('text_length_by_category.json'),
            loadJSON('text_yearly_category_trends.json'),
            loadJSON('text_top_terms_by_category.json'),
            loadJSON('text_tfidf_by_category.json'),
        ]);

        renderOverview(overview);
        renderCategories(categories);
        renderLengths(lengths);
        renderLengthByCategory(lengthByCategory);
        renderTimeline(timeline);
        renderYearlyTrends(yearlyTrends);
        renderQuality(quality);
        renderTerms(terms);
        renderCategoryTerms(categoryTerms);
        renderTfidf(tfidfTerms);
        renderExamples(quality);
    } catch (e) {
        console.error('Failed to load text data:', e);
    }
});

function renderOverview(data) {
    document.getElementById('overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Articles</div><div class="stat-value">${data.total_articles.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Categories</div><div class="stat-value">${data.category_count}</div><div class="stat-sub">Editorial labels</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Date Range</div><div class="stat-value">${data.date_min.slice(0, 4)}-${data.date_max.slice(0, 4)}</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Avg Text Length</div><div class="stat-value">${data.combined_mean_words}</div><div class="stat-sub">Words per article text</div></div>`;

    document.getElementById('overview-desc').innerHTML = `The News Category dataset contains <strong>${data.total_articles.toLocaleString()}</strong> HuffPost articles with <strong>${data.category_count}</strong> category labels. Each record includes a headline, optional short description, author metadata, article link, and publication date from <strong>${data.date_min}</strong> to <strong>${data.date_max}</strong>.`;

    let sampleHtml = '<thead><tr>' + data.display_columns.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    data.sample_rows.forEach(row => {
        sampleHtml += '<tr>' + data.display_columns.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>';
    });
    sampleHtml += '</tbody>';
    document.getElementById('sample-table').innerHTML = sampleHtml;

    let dtypeHtml = '<thead><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>';
    data.columns.forEach(col => {
        dtypeHtml += `<tr><td>${col.name}</td><td>${col.dtype}</td><td class="num">${col.non_null.toLocaleString()}</td><td class="num">${col.missing.toLocaleString()}</td><td class="num">${col.missing_pct}%</td></tr>`;
    });
    dtypeHtml += '</tbody>';
    document.getElementById('dtypes-table').innerHTML = dtypeHtml;
}

function renderCategories(data) {
    Plotly.newPlot('chart-category', [{
        x: data.counts.slice().reverse(),
        y: data.labels.slice().reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.sequential[2] },
        text: data.shares.slice().reverse().map(v => `${v}%`),
        textposition: 'outside',
    }], plotlyLayout('Top 10 Categories by Article Count', {
        xaxis: { title: 'Articles' },
        yaxis: { title: '' },
        height: 520,
        margin: { l: 140, r: 40, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    document.getElementById('category-balance').innerHTML = `
        <div class="balance-card"><div class="label">Largest Class</div><div class="value">${data.max_count.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Smallest Class</div><div class="value">${data.min_count.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Imbalance Ratio</div><div class="value warning">${data.imbalance_ratio}x</div></div>
        <div class="balance-card"><div class="label">Small Classes</div><div class="value">${data.smallest_labels.length}</div></div>`;

    const list = data.smallest_labels.map((label, i) => `<li><strong>${label}</strong>: ${data.smallest_counts[i].toLocaleString()} articles</li>`).join('');
    document.getElementById('category-insights').innerHTML = `
        <li><strong>POLITICS</strong> dominates the dataset, which creates a clear class imbalance.</li>
        <li>Several categories have close semantics, such as arts and wellness variants, which may confuse classifiers.</li>
        ${list}`;
}

function renderLengths(data) {
    Plotly.newPlot('chart-headline-lengths', [{
        x: data.headline_bins,
        y: data.headline_counts,
        type: 'bar',
        marker: { color: COLORS.qualitative[0] },
    }], plotlyLayout('Headline Length Distribution', {
        xaxis: { title: 'Words per headline' },
        yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);

    Plotly.newPlot('chart-combined-lengths', [{
        x: data.combined_bins,
        y: data.combined_counts,
        type: 'bar',
        marker: { color: COLORS.qualitative[4] },
    }], plotlyLayout('Combined Text Length Distribution', {
        xaxis: { title: 'Words in headline + short description' },
        yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);

    document.getElementById('length-cards').innerHTML = `
        <div class="balance-card"><div class="label">Headline Mean</div><div class="value">${data.headline_mean}</div></div>
        <div class="balance-card"><div class="label">Headline Median</div><div class="value">${data.headline_median}</div></div>
        <div class="balance-card"><div class="label">Text Mean</div><div class="value">${data.combined_mean}</div></div>
        <div class="balance-card"><div class="label">Text Median</div><div class="value">${data.combined_median}</div></div>`;
}

function renderLengthByCategory(data) {
    Plotly.newPlot('chart-length-by-category', [{
        x: data.chart_categories.slice().reverse(),
        y: data.chart_combined_mean.slice().reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[3] },
        text: data.chart_combined_mean.slice().reverse().map(v => `${v}`),
        textposition: 'outside',
    }], plotlyLayout('Average Combined Text Length by Category', {
        xaxis: { title: 'Average words' },
        yaxis: { title: '' },
        height: 520,
        margin: { l: 150, r: 30, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    let html = '<thead><tr><th>Category</th><th>Articles</th><th>Headline Mean</th><th>Combined Mean</th><th>Combined Median</th></tr></thead><tbody>';
    data.categories.forEach((category, i) => {
        html += `<tr><td>${category}</td><td class="num">${data.article_count[i].toLocaleString()}</td><td class="num">${data.headline_mean[i]}</td><td class="num">${data.combined_mean[i]}</td><td class="num">${data.combined_median[i]}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('length-by-category-table').innerHTML = html;
}

function renderTimeline(data) {
    Plotly.newPlot('chart-timeline', [{
        x: data.years,
        y: data.counts,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.qualitative[5], width: 3 },
        marker: { size: 8, color: COLORS.qualitative[5] },
        fill: 'tozeroy',
        fillcolor: 'rgba(194,106,46,0.14)',
    }], plotlyLayout('Article Counts by Year', {
        xaxis: { title: 'Year', dtick: 1 },
        yaxis: { title: 'Articles' },
    }), PLOTLY_CONFIG);

    document.getElementById('timeline-insights').innerHTML = `
        <li>Peak coverage occurs in <strong>${data.peak_year}</strong> with <strong>${data.peak_count.toLocaleString()}</strong> articles.</li>
        <li>The latest year in the file is <strong>${data.latest_year}</strong>, but it only contains <strong>${data.latest_count.toLocaleString()}</strong> records.</li>
        <li>A random train/test split may hide time-based distribution shift, so date-aware evaluation is safer.</li>`;
}

function renderYearlyTrends(data) {
    const traces = data.categories.map((category, i) => ({
        x: data.years,
        y: data.series[category],
        type: 'scatter',
        mode: 'lines+markers',
        name: category,
        line: { width: 2 },
        marker: { size: 6 },
        color: COLORS.qualitative[i % COLORS.qualitative.length],
    }));

    Plotly.newPlot('chart-yearly-trends', traces, plotlyLayout('Top Category Trends by Year', {
        xaxis: { title: 'Year', dtick: 1 },
        yaxis: { title: 'Articles' },
        legend: { orientation: 'h', y: -0.25 },
        margin: { l: 55, r: 25, t: 55, b: 100 },
    }), PLOTLY_CONFIG);
}

function renderQuality(data) {
    Plotly.newPlot('chart-missing', [{
        x: data.missing.map(row => row.column),
        y: data.missing.map(row => row.missing_pct),
        type: 'bar',
        marker: { color: COLORS.qualitative[1] },
        text: data.missing.map(row => `${row.missing.toLocaleString()}`),
        textposition: 'outside',
    }], plotlyLayout('Missing Values by Column', {
        xaxis: { title: '' },
        yaxis: { title: 'Missing %' },
    }), PLOTLY_CONFIG);

    let qualityHtml = '<thead><tr><th>Column</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>';
    data.missing.forEach(row => {
        qualityHtml += `<tr><td>${row.column}</td><td class="num">${row.missing.toLocaleString()}</td><td class="num">${row.missing_pct}%</td></tr>`;
    });
    qualityHtml += '</tbody>';
    document.getElementById('quality-table').innerHTML = qualityHtml;

    document.getElementById('dup-summary').innerHTML = `
        <div class="balance-card"><div class="label">Duplicate Records</div><div class="value">${data.duplicate_records}</div></div>
        <div class="balance-card"><div class="label">Duplicate Links</div><div class="value">${data.duplicate_links}</div></div>`;
}

function renderTerms(data) {
    Plotly.newPlot('chart-keywords', [{
        x: data.keywords.slice(0, 12).map(row => row.word).reverse(),
        y: data.keywords.slice(0, 12).map(row => row.count).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[2] },
    }], plotlyLayout('Top Content Words', {
        xaxis: { title: 'Frequency' },
        yaxis: { title: '' },
        height: 480,
        margin: { l: 100, r: 30, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    let authorHtml = '<thead><tr><th>Author</th><th>Articles</th></tr></thead><tbody>';
    data.top_authors.forEach(row => {
        authorHtml += `<tr><td>${row.author}</td><td class="num">${row.count.toLocaleString()}</td></tr>`;
    });
    authorHtml += '</tbody>';
    document.getElementById('authors-table').innerHTML = authorHtml;
}

function renderCategoryTerms(data) {
    const first = data.groups[0] || { category: '', terms: [] };
    Plotly.newPlot('chart-category-terms', [{
        x: first.terms.map(row => row.word).reverse(),
        y: first.terms.map(row => row.count).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[4] },
    }], plotlyLayout(`Top Terms: ${first.category}`, {
        xaxis: { title: 'Frequency' },
        yaxis: { title: '' },
        height: 460,
        margin: { l: 110, r: 30, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    let html = '<thead><tr><th>Category</th><th>Top Terms</th></tr></thead><tbody>';
    data.groups.forEach(group => {
        html += `<tr><td>${group.category}</td><td>${group.terms.map(term => `${term.word} (${term.count})`).join(', ')}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('category-terms-table').innerHTML = html;
}

function renderTfidf(data) {
    const first = data.groups[0] || { category: '', terms: [] };
    Plotly.newPlot('chart-tfidf', [{
        x: first.terms.map(row => row.term).reverse(),
        y: first.terms.map(row => row.score).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[5] },
    }], plotlyLayout(`TF-IDF Keywords: ${first.category}`, {
        xaxis: { title: 'Average TF-IDF score' },
        yaxis: { title: '' },
        height: 460,
        margin: { l: 110, r: 30, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    let html = '<thead><tr><th>Category</th><th>TF-IDF Keywords</th></tr></thead><tbody>';
    data.groups.forEach(group => {
        html += `<tr><td>${group.category}</td><td>${group.terms.map(term => `${term.term} (${term.score})`).join(', ')}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('tfidf-table').innerHTML = html;
}

function renderExamples(data) {
    const renderExampleTable = (targetId, rows) => {
        let html = '<thead><tr><th>Date</th><th>Category</th><th>Headline</th><th>Link</th></tr></thead><tbody>';
        rows.forEach(row => {
            html += `<tr><td>${row.date}</td><td>${row.category}</td><td>${row.headline}</td><td><a href="${row.link}" target="_blank" rel="noopener noreferrer">Open</a></td></tr>`;
        });
        html += '</tbody>';
        document.getElementById(targetId).innerHTML = html;
    };

    renderExampleTable('duplicate-examples-table', data.duplicate_examples || []);
    renderExampleTable('missing-desc-table', (data.missing_examples && data.missing_examples.short_description) || []);
    renderExampleTable('missing-author-table', (data.missing_examples && data.missing_examples.authors) || []);
}
