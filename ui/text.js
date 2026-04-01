/* text.js — Text EDA Chart Rendering */

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [overview, categories, lengths, timeline, quality, terms, stopwords, vocabRichness, bigrams, lengthByCategory, yearlyTrends, tfidfTerms] = await Promise.all([
            loadJSON('text_overview.json'),
            loadJSON('text_category_dist.json'),
            loadJSON('text_lengths.json'),
            loadJSON('text_timeline.json'),
            loadJSON('text_quality.json'),
            loadJSON('text_terms.json'),
            loadJSON('text_stopwords.json'),
            loadJSON('text_vocabulary_richness.json'),
            loadJSON('text_bigrams.json'),
            loadJSON('text_length_by_category.json'),
            loadJSON('text_yearly_category_trends.json'),
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
        renderStopwords(stopwords);
        renderVocabulary(vocabRichness);
        renderFinalInsights(categories, lengths, timeline, quality, terms, vocabRichness);
        renderBigrams(bigrams);
        renderTfidf(tfidfTerms);
    } catch (e) {
        console.error('Failed to load text data:', e);
    }
});

function renderOverview(data) {
    document.getElementById('overview-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Articles</div><div class="stat-value">${data.total_articles.toLocaleString()}</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Categories</div><div class="stat-value">${data.category_count}</div><div class="stat-sub">Editorial labels</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Date Range</div><div class="stat-value">${data.date_min.slice(0, 4)}-${data.date_max.slice(0, 4)}</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Avg Text Length</div><div class="stat-value">${data.combined_mean_words}</div><div class="stat-sub">Words per article text</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Avg Char Length</div><div class="stat-value">${data.combined_mean_chars}</div><div class="stat-sub">Characters per article text</div></div>`;

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
        <div class="balance-card"><div class="label">Text Median</div><div class="value">${data.combined_median}</div></div>
        <div class="balance-card"><div class="label">Headline Char Mean</div><div class="value">${data.headline_char_mean}</div></div>
        <div class="balance-card"><div class="label">Text Char Mean</div><div class="value">${data.combined_char_mean}</div></div>`;

    Plotly.newPlot('chart-headline-char-lengths', [{
        x: data.headline_char_bins,
        y: data.headline_char_counts,
        type: 'bar',
        marker: { color: COLORS.qualitative[1] },
    }], plotlyLayout('Headline Character Length Distribution', {
        xaxis: { title: 'Characters per headline' },
        yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);

    Plotly.newPlot('chart-combined-char-lengths', [{
        x: data.combined_char_bins,
        y: data.combined_char_counts,
        type: 'bar',
        marker: { color: COLORS.qualitative[3] },
    }], plotlyLayout('Combined Text Character Distribution', {
        xaxis: { title: 'Characters in headline + short description' },
        yaxis: { title: 'Count' },
    }), PLOTLY_CONFIG);
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

function renderStopwords(data) {
    Plotly.newPlot('chart-stopwords', [{
        x: data.stopwords.slice(0, 12).map(row => row.word).reverse(),
        y: data.stopwords.slice(0, 12).map(row => row.count).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[0] },
    }], plotlyLayout('Most Frequent Stopwords', {
        xaxis: { title: 'Frequency' },
        yaxis: { title: '' },
        height: 480,
        margin: { l: 90, r: 30, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    document.getElementById('stopword-summary').innerHTML = `
        <div class="balance-card"><div class="label">All Stopword Tokens</div><div class="value">${data.total_stopword_tokens.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Top-20 Coverage</div><div class="value">${data.stopword_share_top20}%</div></div>`;
}

function renderVocabulary(data) {
    document.getElementById('vocab-cards').innerHTML = `
        <div class="balance-card"><div class="label">Total Tokens</div><div class="value">${data.total_tokens.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Unique Tokens</div><div class="value">${data.unique_tokens.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Type-Token Ratio</div><div class="value">${data.type_token_ratio}</div></div>
        <div class="balance-card"><div class="label">Hapax Ratio</div><div class="value">${data.hapax_ratio}</div></div>
        <div class="balance-card"><div class="label">Hapax Count</div><div class="value">${data.hapax_count.toLocaleString()}</div></div>
        <div class="balance-card"><div class="label">Avg Unique / Article</div><div class="value">${data.avg_unique_per_article}</div></div>`;

    let html = '<thead><tr><th>Word</th><th>Count</th></tr></thead><tbody>';
    data.top_unique_words.forEach(row => {
        html += `<tr><td>${row.word}</td><td class="num">${row.count.toLocaleString()}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('vocab-table').innerHTML = html;
}

function renderBigrams(data) {
    Plotly.newPlot('chart-bigrams', [{
        x: data.bigrams.slice(0, 12).map(row => row.bigram).reverse(),
        y: data.bigrams.slice(0, 12).map(row => row.count).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[5] },
    }], plotlyLayout('Top Bigrams', {
        xaxis: { title: 'Frequency' },
        yaxis: { title: '' },
        height: 500,
        margin: { l: 140, r: 30, t: 50, b: 50 },
    }), PLOTLY_CONFIG);

    let html = '<thead><tr><th>Bigram</th><th>Count</th></tr></thead><tbody>';
    data.bigrams.forEach(row => {
        html += `<tr><td>${row.bigram}</td><td class="num">${row.count.toLocaleString()}</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('bigrams-table').innerHTML = html;
}

function renderTfidf(data) {
    const first = data.groups[0] || { category: '', terms: [] };
    const chartEl = document.getElementById('chart-tfidf');
    const tableEl = document.getElementById('tfidf-table');
    if (!chartEl || !tableEl) return;

    try {
        Plotly.newPlot('chart-tfidf', [{
            x: first.terms.map(row => row.score).reverse(),
            y: first.terms.map(row => row.term).reverse(),
            type: 'bar',
            orientation: 'h',
            marker: { color: COLORS.qualitative[5] },
        }], plotlyLayout(`TF-IDF Keywords: ${first.category}`, {
            xaxis: { title: 'Average TF-IDF score' },
            yaxis: { title: '' },
            height: 460,
            margin: { l: 110, r: 30, t: 50, b: 50 },
        }), PLOTLY_CONFIG);
    } catch (error) {
        console.error('Failed to render TF-IDF chart:', error);
        chartEl.innerHTML = '<div class="desc-box">Unable to render the TF-IDF chart, but the keyword table below is still available.</div>';
    }

    let html = '<thead><tr><th>Category</th><th>TF-IDF Keywords</th></tr></thead><tbody>';
    data.groups.forEach(group => {
        html += `<tr><td>${group.category}</td><td>${group.terms.map(term => `${term.term} (${term.score})`).join(', ')}</td></tr>`;
    });
    html += '</tbody>';
    tableEl.innerHTML = html;
}

function renderFinalInsights(categories, lengths, timeline, quality, terms, vocab) {
    const insightsEl = document.getElementById('final-insights');
    if (!insightsEl) return;

    const topCategory = categories.labels[0];
    const topShare = categories.shares[0];
    const missingShortDesc = quality.missing.find(row => row.column === 'short_description');
    const missingAuthors = quality.missing.find(row => row.column === 'authors');
    const topKeyword = terms.keywords[0];

    insightsEl.innerHTML = `
        <li><strong>${topCategory}</strong> is the dominant class with <strong>${topShare}%</strong> of all articles, while the largest-to-smallest class ratio reaches <strong>${categories.imbalance_ratio}x</strong>, so label imbalance is a major modeling concern.</li>
        <li>The text is concise overall: headlines average <strong>${lengths.headline_mean}</strong> words, and the combined headline-plus-description field averages <strong>${lengths.combined_mean}</strong> words, which is suitable for lightweight text classification pipelines.</li>
        <li>The dataset is temporally skewed: article volume peaks in <strong>${timeline.peak_year}</strong> with <strong>${timeline.peak_count.toLocaleString()}</strong> records, then drops to <strong>${timeline.latest_count.toLocaleString()}</strong> in <strong>${timeline.latest_year}</strong>, so random splitting can hide time drift.</li>
        <li>Data completeness issues are concentrated in metadata rather than labels: <strong>${missingShortDesc.missing_pct}%</strong> of rows lack short descriptions and <strong>${missingAuthors.missing_pct}%</strong> lack author names, while category, link, and date remain complete.</li>
        <li>The corpus is broad but still repetitive in topic focus, with <strong>${vocab.unique_tokens.toLocaleString()}</strong> unique tokens overall and <strong>${topKeyword.word}</strong> as the most frequent content word at <strong>${topKeyword.count.toLocaleString()}</strong> occurrences.</li>`;
}
