/* Text Classification Results UI */

const TEXT_RESULT_BASE = 'text_classification';

const RESULT_FILES = {
    traditionalOverview: `${TEXT_RESULT_BASE}/traditional_ml/text_ml_overview.json`,
    traditionalRows: `${TEXT_RESULT_BASE}/traditional_ml/text_ml_model_comparison.json`,
    traditionalReport: `${TEXT_RESULT_BASE}/traditional_ml/text_ml_classification_report.json`,
    traditionalErrors: `${TEXT_RESULT_BASE}/traditional_ml/text_ml_error_samples.json`,
    gridOverview: `${TEXT_RESULT_BASE}/pipeline_grid/text_pipeline_grid_overview.json`,
    gridRows: `${TEXT_RESULT_BASE}/pipeline_grid/text_pipeline_grid_comparison.json`,
    bertOverview: `${TEXT_RESULT_BASE}/bert/text_transformer_grid_overview.json`,
    bertRows: `${TEXT_RESULT_BASE}/bert/text_transformer_grid_comparison.json`,
};

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [traditionalOverview, traditionalRows, traditionalReport, traditionalErrors, gridOverview, gridRows, bertOverview, bertRows] = await Promise.all([
            loadJSON(RESULT_FILES.traditionalOverview),
            loadJSON(RESULT_FILES.traditionalRows),
            loadJSON(RESULT_FILES.traditionalReport),
            loadJSON(RESULT_FILES.traditionalErrors),
            loadJSON(RESULT_FILES.gridOverview),
            loadJSON(RESULT_FILES.gridRows),
            loadJSON(RESULT_FILES.bertOverview),
            loadJSON(RESULT_FILES.bertRows),
        ]);

        const bestBertId = bertOverview.best_pipeline?.pipeline_id || bertRows.rows?.[0]?.pipeline_id;
        const bestBertReport = await loadJSON(`${TEXT_RESULT_BASE}/bert/text_transformer_${bestBertId}_classification_report.json`);
        const bestBertErrors = await loadJSON(`${TEXT_RESULT_BASE}/bert/text_transformer_${bestBertId}_error_samples.json`);

        const state = {
            traditionalOverview,
            traditionalRows: cleanRows(traditionalRows.rows),
            traditionalReport,
            traditionalErrors,
            gridOverview,
            gridRows: cleanRows(gridRows.rows),
            bertOverview,
            bertRows: cleanRows(bertRows.rows).filter(row => !row.freeze_encoder),
            bestBertReport,
            bestBertErrors,
        };

        renderResultSummary(state);
        renderTraditional(state);
        renderGrid(state);
        renderBert(state);
        renderPerClass(state);
        renderErrors(state);
    } catch (error) {
        console.error('Failed to load text classification results:', error);
        document.querySelector('.page-container').insertAdjacentHTML('beforeend', `<div class="insight-box"><h4><i class="fa-solid fa-triangle-exclamation"></i> Data Load Error</h4><ul><li>${escapeHTML(error.message)}</li></ul></div>`);
    }
});

function cleanRows(rows = []) {
    return rows.filter(row => row && row.status !== 'failed');
}

function fmtMetric(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '-';
}

function fmtSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return '-';
    if (seconds >= 3600) return `${(seconds / 3600).toFixed(2)}h`;
    if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
    return `${seconds.toFixed(1)}s`;
}

function escapeHTML(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function htmlCell(html) {
    return { html };
}

function chipClass(value, group = 'model') {
    const normalized = String(value || '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '');
    return `chip-${group}-${normalized || 'unknown'}`;
}

function tableChip(value, group = 'model') {
    return htmlCell(`<span class="table-chip ${chipClass(value, group)}">${escapeHTML(value)}</span>`);
}

function modelLabel(row) {
    return row.pipeline || row.pipeline_id || row.model || row.name || 'model';
}

function shortReducerLabel(value) {
    const labelMap = {
        chi2_k20000: 'chi2',
        svd_100: 'svd100',
        svd_300: 'svd300',
    };
    return labelMap[value] || value || '-';
}

function shortGridPipelineLabel(rowOrName) {
    const pipeline = typeof rowOrName === 'string' ? rowOrName : rowOrName?.pipeline;
    if (!pipeline) return '-';
    const [feature, reducer, ...modelParts] = pipeline.split('__');
    const model = modelParts.join('__');
    return [feature, shortReducerLabel(reducer), model].filter(Boolean).join(' + ');
}

function gridRankLabel(row, index) {
    return `${String(index + 1).padStart(2, '0')}. ${row.feature_step} + ${shortReducerLabel(row.reduction_step)}`;
}

function reducerColor(reducer) {
    const colorMap = {
        chi2_k20000: COLORS.qualitative[0],
        svd_100: COLORS.qualitative[2],
        svd_300: COLORS.qualitative[4],
    };
    return colorMap[reducer] || COLORS.qualitative[3];
}

function shortBertEncoder(value) {
    const label = String(value || '').toLowerCase();
    if (label.includes('distilbert')) return 'distilbert';
    if (label.includes('bert')) return 'bert';
    return value || '-';
}

function shortBertPipelineLabel(rowOrId) {
    const pipelineId = typeof rowOrId === 'string' ? rowOrId : rowOrId?.pipeline_id;
    const modelName = typeof rowOrId === 'string' ? pipelineId : rowOrId?.model_name;
    const pooling = typeof rowOrId === 'string'
        ? pipelineId?.match(/_(cls|mean|pooler)_/)?.[1]
        : rowOrId?.pooling;
    return [shortBertEncoder(modelName), pooling].filter(Boolean).join(' + ');
}

function bertEncoderColor(modelName) {
    return shortBertEncoder(modelName) === 'distilbert'
        ? COLORS.qualitative[0]
        : COLORS.qualitative[3];
}

function renderResultSummary(state) {
    const traditionalBest = state.traditionalOverview.best_model;
    const gridBest = state.gridOverview.best_pipeline;
    const bertBest = state.bertOverview.best_pipeline;
    const delta = bertBest.macro_f1 - traditionalBest.macro_f1;

    document.getElementById('result-stats').innerHTML = `
        <div class="stat-card stat-orange"><div class="stat-label">Train / Test</div><div class="stat-value">${state.traditionalOverview.train_size.toLocaleString()} / ${state.traditionalOverview.test_size.toLocaleString()}</div><div class="stat-sub">Full dataset split</div></div>
        <div class="stat-card stat-brown"><div class="stat-label">Classes</div><div class="stat-value">${state.traditionalOverview.class_count}</div><div class="stat-sub">News categories</div></div>
        <div class="stat-card stat-green"><div class="stat-label">Best Overall</div><div class="stat-value">${fmtMetric(bertBest.macro_f1)}</div><div class="stat-sub">${escapeHTML(shortBertPipelineLabel(bertBest))} macro-F1</div></div>
        <div class="stat-card stat-blue"><div class="stat-label">Gain vs ML</div><div class="stat-value">+${fmtMetric(delta)}</div><div class="stat-sub">Macro-F1 improvement</div></div>
        <div class="stat-card stat-amber"><div class="stat-label">Pipelines</div><div class="stat-value">${state.traditionalRows.length + state.gridRows.length + state.bertRows.length}</div><div class="stat-sub">ML + grid + BERT results</div></div>`;

    const familyRows = [
        { family: 'Traditional ML', model: traditionalBest.name, accuracy: traditionalBest.accuracy, macro_f1: traditionalBest.macro_f1, weighted_f1: traditionalBest.weighted_f1, train_seconds: traditionalBest.train_seconds },
        { family: 'Pipeline Grid', model: shortGridPipelineLabel(gridBest), accuracy: gridBest.accuracy, macro_f1: gridBest.macro_f1, weighted_f1: gridBest.weighted_f1, train_seconds: gridBest.train_seconds },
        { family: 'BERT Fine-Tune', model: shortBertPipelineLabel(bertBest), accuracy: bertBest.accuracy, macro_f1: bertBest.macro_f1, weighted_f1: bertBest.weighted_f1, train_seconds: bertBest.train_seconds },
    ];

    Plotly.newPlot('chart-family-comparison', [
        { x: familyRows.map(row => row.family), y: familyRows.map(row => row.macro_f1), name: 'Macro-F1', type: 'bar', marker: { color: COLORS.qualitative[0] }, text: familyRows.map(row => fmtMetric(row.macro_f1)), textposition: 'outside' },
        { x: familyRows.map(row => row.family), y: familyRows.map(row => row.accuracy), name: 'Accuracy', type: 'bar', marker: { color: COLORS.qualitative[2] }, text: familyRows.map(row => fmtMetric(row.accuracy)), textposition: 'outside' },
    ], plotlyLayout('Best Model by Experiment Family', {
        yaxis: { title: 'Score', range: [0, 0.75] },
        barmode: 'group',
        height: 430,
    }), PLOTLY_CONFIG);

    renderTable('table-family-comparison', familyRows, [
        ['Family', row => row.family],
        ['Best Model', row => row.model],
        ['Accuracy', row => fmtMetric(row.accuracy), 'num'],
        ['Macro-F1', row => fmtMetric(row.macro_f1), 'num'],
        ['Weighted-F1', row => fmtMetric(row.weighted_f1), 'num'],
        ['Train Time', row => fmtSeconds(row.train_seconds), 'num'],
    ]);
}

function renderTraditional(state) {
    const overview = state.traditionalOverview;
    document.getElementById('traditional-desc').innerHTML = `This experiment establishes a scalable classical baseline for short-news classification. Sparse lexical representations are paired with fast linear classifiers and ensemble variants, providing a strong reference point before applying dimensionality reduction or transformer fine-tuning. The best baseline is <strong>${escapeHTML(overview.best_model.name)}</strong>, reaching macro-F1 <strong>${fmtMetric(overview.best_model.macro_f1)}</strong>.`;

    const rows = state.traditionalRows.slice(0, 12);
    Plotly.newPlot('chart-traditional', [{
        x: rows.map(row => row.macro_f1).reverse(),
        y: rows.map(row => row.model).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[5] },
        text: rows.map(row => fmtMetric(row.macro_f1)).reverse(),
        textposition: 'outside',
    }], plotlyLayout('Traditional ML Models Ranked by Macro-F1', {
        xaxis: { title: 'Macro-F1' },
        yaxis: { title: '' },
        height: 520,
        margin: { l: 155, r: 40, t: 55, b: 45 },
    }), PLOTLY_CONFIG);

    renderTable('table-traditional', state.traditionalRows, metricColumns('model', { chipModel: false }));
}

function renderGrid(state) {
    const overview = state.gridOverview;
    const reducerNames = overview.reduction_steps.map(shortReducerLabel).join(', ');
    document.getElementById('grid-desc').innerHTML = `This grid studies whether reducing high-dimensional sparse text features improves downstream classification. It compares <strong>${overview.feature_steps.join(', ')}</strong> feature extraction with <strong>${reducerNames}</strong> reduction strategies, then evaluates the retained classifier set after removing consistently weak dense models. The best reduced pipeline is <strong>${escapeHTML(shortGridPipelineLabel(overview.best_pipeline))}</strong>, with macro-F1 <strong>${fmtMetric(overview.best_pipeline.macro_f1)}</strong>.`;

    const topRows = [...state.gridRows]
        .sort((a, b) => b.macro_f1 - a.macro_f1)
        .slice(0, 10);
    const topRowsForPlot = [...topRows].reverse();
    const yLabels = topRowsForPlot.map((row) => gridRankLabel(row, topRows.indexOf(row)));
    const minScore = Math.min(...topRows.map(row => Number(row.macro_f1)));
    const maxScore = Math.max(...topRows.map(row => Number(row.macro_f1)));
    Plotly.newPlot('chart-grid-top', [{
        x: topRowsForPlot.map(row => row.macro_f1),
        y: yLabels,
        type: 'scatter',
        mode: 'markers+text',
        marker: {
            color: topRowsForPlot.map(row => reducerColor(row.reduction_step)),
            size: 13,
            line: { color: '#FFFFFF', width: 1.5 },
        },
        text: topRowsForPlot.map(row => `${row.model} · ${fmtMetric(row.macro_f1)}`),
        textposition: 'middle right',
        textfont: { size: 11, color: '#4B3A2A' },
        hovertemplate: [
            '<b>%{customdata[0]}</b>',
            'Feature: %{customdata[1]}',
            'Reducer: %{customdata[2]}',
            'Model: %{customdata[3]}',
            'Macro-F1: %{x:.4f}',
            '<extra></extra>',
        ].join('<br>'),
        customdata: topRowsForPlot.map(row => [
            shortGridPipelineLabel(row),
            row.feature_step,
            shortReducerLabel(row.reduction_step),
            row.model,
        ]),
    }], plotlyLayout('Top Pipeline Grid Results', {
        xaxis: {
            title: 'Macro-F1',
            range: [Math.max(0, minScore - 0.015), Math.min(1, maxScore + 0.07)],
        },
        yaxis: {
            title: '',
            categoryorder: 'array',
            categoryarray: yLabels,
            gridcolor: '#F1E7DA',
        },
        height: 470,
        margin: { l: 125, r: 95, t: 55, b: 45 },
        showlegend: false,
    }), PLOTLY_CONFIG);

    const reducerGroups = groupBest(state.gridRows, 'reduction_step');
    Plotly.newPlot('chart-grid-reducers', [{
        x: reducerGroups.map(row => shortReducerLabel(row.key)),
        y: reducerGroups.map(row => row.macro_f1),
        type: 'bar',
        marker: { color: COLORS.qualitative[1] },
        text: reducerGroups.map(row => fmtMetric(row.macro_f1)),
        textposition: 'outside',
    }], plotlyLayout('Best Macro-F1 by Reduction Step', {
        yaxis: { title: 'Best macro-F1', range: [0, 0.55] },
        height: 420,
    }), PLOTLY_CONFIG);

    renderTable('table-grid', state.gridRows, [
        ['Pipeline', row => shortGridPipelineLabel(row)],
        ['Feature', row => tableChip(row.feature_step, 'feature')],
        ['Reducer', row => tableChip(shortReducerLabel(row.reduction_step), 'reducer')],
        ['Model', row => tableChip(row.model, 'model')],
        ['Accuracy', row => fmtMetric(row.accuracy), 'num'],
        ['Macro-F1', row => fmtMetric(row.macro_f1), 'num'],
        ['Weighted-F1', row => fmtMetric(row.weighted_f1), 'num'],
        ['Train Time', row => fmtSeconds(row.train_seconds), 'num'],
    ]);
}

function renderBert(state) {
    const overview = state.bertOverview;
    const modelNames = [...new Set(overview.model_names.map(shortBertEncoder))].join(', ');
    document.getElementById('bert-desc').innerHTML = `The transformer experiment fine-tunes <strong>${escapeHTML(modelNames)}</strong> encoders directly on the classification task and compares pooling choices for converting token-level representations into document-level predictions. Frozen-encoder and TinyBERT variants are excluded from the selected workflow because they underperformed or were outside the final experiment scope. The best transformer pipeline is <strong>${escapeHTML(shortBertPipelineLabel(overview.best_pipeline))}</strong>, with macro-F1 <strong>${fmtMetric(overview.best_pipeline.macro_f1)}</strong>.`;

    const bertRows = [...state.bertRows].sort((a, b) => b.macro_f1 - a.macro_f1);
    const bertRowsForPlot = [...bertRows].reverse();
    const bertYLabels = bertRowsForPlot.map((row) => shortBertPipelineLabel(row));
    const bertMinScore = Math.min(...bertRows.map(row => Number(row.macro_f1)));
    const bertMaxScore = Math.max(...bertRows.map(row => Number(row.macro_f1)));
    Plotly.newPlot('chart-bert', [{
        x: bertRowsForPlot.map(row => row.macro_f1),
        y: bertYLabels,
        type: 'scatter',
        mode: 'markers+text',
        marker: {
            color: bertRowsForPlot.map(row => bertEncoderColor(row.model_name)),
            size: 14,
            symbol: bertRowsForPlot.map(row => row.pooling === 'mean' ? 'diamond' : row.pooling === 'pooler' ? 'square' : 'circle'),
            line: { color: '#FFFFFF', width: 1.5 },
        },
        text: bertRowsForPlot.map(row => fmtMetric(row.macro_f1)),
        textposition: 'middle right',
        textfont: { size: 11, color: '#4B3A2A' },
        hovertemplate: [
            '<b>%{customdata[0]}</b>',
            'Encoder: %{customdata[1]}',
            'Pooling: %{customdata[2]}',
            'Accuracy: %{customdata[3]:.4f}',
            'Macro-F1: %{x:.4f}',
            '<extra></extra>',
        ].join('<br>'),
        customdata: bertRowsForPlot.map(row => [
            shortBertPipelineLabel(row),
            shortBertEncoder(row.model_name),
            row.pooling,
            row.accuracy,
        ]),
    }], plotlyLayout('BERT Fine-Tuning Results', {
        xaxis: {
            title: 'Macro-F1',
            range: [Math.max(0, bertMinScore - 0.01), Math.min(1, bertMaxScore + 0.035)],
        },
        yaxis: {
            title: '',
            categoryorder: 'array',
            categoryarray: bertYLabels,
            gridcolor: '#F1E7DA',
        },
        height: 390,
        margin: { l: 115, r: 95, t: 55, b: 45 },
        showlegend: false,
    }), PLOTLY_CONFIG);

    Plotly.newPlot('chart-bert-time', [{
        x: state.bertRows.map(shortBertPipelineLabel),
        y: state.bertRows.map(row => row.train_seconds / 60),
        type: 'bar',
        marker: { color: COLORS.qualitative[4] },
        text: state.bertRows.map(row => fmtSeconds(row.train_seconds)),
        textposition: 'outside',
    }], plotlyLayout('BERT Training Time', {
        yaxis: { title: 'Minutes' },
        xaxis: { tickangle: -35 },
        height: 450,
        margin: { l: 60, r: 25, t: 55, b: 130 },
    }), PLOTLY_CONFIG);

    renderTable('table-bert', state.bertRows, [
        ['Pipeline', row => shortBertPipelineLabel(row)],
        ['Encoder', row => tableChip(shortBertEncoder(row.model_name), 'encoder')],
        ['Pooling', row => tableChip(row.pooling, 'pooling')],
        ['Accuracy', row => fmtMetric(row.accuracy), 'num'],
        ['Macro-F1', row => fmtMetric(row.macro_f1), 'num'],
        ['Weighted-F1', row => fmtMetric(row.weighted_f1), 'num'],
        ['Loss', row => fmtMetric(row.loss), 'num'],
        ['Train Time', row => fmtSeconds(row.train_seconds), 'num'],
    ]);
}

function renderPerClass(state) {
    const rows = state.bestBertReport.per_class || [];
    const topRows = rows.slice(0, 20);
    Plotly.newPlot('chart-per-class', [{
        x: topRows.map(row => row.f1_score).reverse(),
        y: topRows.map(row => row.category).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.qualitative[2] },
        text: topRows.map(row => fmtMetric(row.f1_score)).reverse(),
        textposition: 'outside',
    }], plotlyLayout('Best BERT Per-Class F1 on Largest Classes', {
        xaxis: { title: 'F1 score', range: [0, 1] },
        yaxis: { title: '' },
        height: 620,
        margin: { l: 155, r: 35, t: 55, b: 45 },
    }), PLOTLY_CONFIG);

    renderTable('table-per-class', rows, [
        ['Category', row => row.category],
        ['Support', row => row.support.toLocaleString(), 'num'],
        ['Precision', row => fmtMetric(row.precision), 'num'],
        ['Recall', row => fmtMetric(row.recall), 'num'],
        ['F1', row => fmtMetric(row.f1_score), 'num'],
    ]);
}

function renderErrors(state) {
    const rows = (state.bestBertErrors.rows || []).map(row => ({ model: state.bestBertErrors.pipeline_id, ...row }));
    renderTable('table-errors', rows, [
        ['Actual', row => row.actual],
        ['Predicted', row => row.predicted],
        ['Text Preview', row => row.text_preview],
    ]);
}

function metricColumns(labelKey, options = {}) {
    return [
        ['Model', row => options.chipModel === false ? row[labelKey] : tableChip(row[labelKey], 'model')],
        ['Accuracy', row => fmtMetric(row.accuracy), 'num'],
        ['Macro-F1', row => fmtMetric(row.macro_f1), 'num'],
        ['Weighted-F1', row => fmtMetric(row.weighted_f1), 'num'],
        ['Precision', row => fmtMetric(row.macro_precision), 'num'],
        ['Recall', row => fmtMetric(row.macro_recall), 'num'],
        ['Train Time', row => fmtSeconds(row.train_seconds), 'num'],
    ];
}

function groupBest(rows, key) {
    const best = new Map();
    rows.forEach(row => {
        const groupKey = row[key] || 'unknown';
        if (!best.has(groupKey) || row.macro_f1 > best.get(groupKey).macro_f1) {
            best.set(groupKey, { key: groupKey, ...row });
        }
    });
    return [...best.values()].sort((a, b) => b.macro_f1 - a.macro_f1);
}

function renderTable(elementId, rows, columns) {
    const head = `<thead><tr>${columns.map(([name]) => `<th>${escapeHTML(name)}</th>`).join('')}</tr></thead>`;
    const body = rows.map(row => `<tr>${columns.map(([, getter, className]) => {
        const value = getter(row);
        const content = value && typeof value === 'object' && 'html' in value ? value.html : escapeHTML(value);
        return `<td class="${className || ''}">${content}</td>`;
    }).join('')}</tr>`).join('');
    document.getElementById(elementId).innerHTML = `${head}<tbody>${body}</tbody>`;
}
