/* salary_ml_dashboard.js - Data Loading and Plotly Rendering for ML Metrics */

let featureDataGlobal = null; // Store globally to render dynamically

document.addEventListener("DOMContentLoaded", () => {
    // 1. Fetch Model Metrics
    fetch('assets/data/jobsalary_ml/model_metrics.json')
        .then(res => res.json())
        .then(data => renderModelMetrics(data))
        .catch(err => console.error("Error loading model metrics:", err));

    // 2. Fetch Feature Importances
    fetch('assets/data/jobsalary_ml/feature_importances.json')
        .then(res => res.json())
        .then(data => {
            featureDataGlobal = data;
            renderFeatureImportance(); // render default selected
        })
        .catch(err => console.error("Error loading feature importances:", err));

    // 3. Fetch Preprocessing Mappings
    fetch('assets/data/jobsalary_ml/preprocessing_mappings.json')
        .then(res => res.json())
        .then(data => renderPreprocessingCharts(data))
        .catch(err => console.error("Error loading preprocessing mappings:", err));
});

function renderModelMetrics(metrics) {
    if (!metrics || metrics.length === 0) return;

    // Pluck values
    const models = metrics.map(m => m.Model);
    const mses = metrics.map(m => m.RMSE);
    const maes = metrics.map(m => m.MAE);
    const r2s = metrics.map(m => (m.R2_Score * 100).toFixed(2)); // Display as %

    const traceRMSE = {
        x: models,
        y: mses,
        name: 'RMSE',
        type: 'bar',
        marker: { color: '#E57373' },
        hovertemplate: 'RMSE: %{y:.2f}<extra></extra>'
    };

    const traceMAE = {
        x: models,
        y: maes,
        name: 'MAE',
        type: 'bar',
        marker: { color: '#FFB74D' },
        hovertemplate: 'MAE: %{y:.2f}<extra></extra>'
    };

    const traceR2 = {
        x: models,
        y: metrics.map(m => m.R2_Score),
        name: 'R² Score (Axis 2)',
        type: 'scatter',
        mode: 'lines+markers+text',
        text: r2s.map(r => r + '%'),
        textposition: 'top center',
        yaxis: 'y2',
        marker: { size: 12, color: '#4CAF50' },
        line: { width: 3, dash: 'dot' },
        hovertemplate: 'R² Score: %{y:.4f}<extra></extra>'
    };

    const layout = {
        barmode: 'group',
        margin: { t: 40, r: 50, b: 60, l: 60 },
        height: 500,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { 
            title: 'Algorithm',
            tickfont: { size: 14, color: '#333' }
        },
        yaxis: { 
            title: 'Error Magnitude ($)',
            gridcolor: '#e0e0e0',
            zerolinecolor: '#ccc'
        },
        yaxis2: {
            title: 'R² Accuracy Score',
            overlaying: 'y',
            side: 'right',
            range: [0.90, 1.0], // Zoom in to show stark contrast between 95 and 98
            tickformat: '.2f',
            gridcolor: 'rgba(0,0,0,0)', // Hide grid for y2
            zeroline: false
        },
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center'
        }
    };

    Plotly.newPlot('chart-model-metrics', [traceRMSE, traceMAE, traceR2], layout, {responsive: true, displayModeBar: true});
}

function renderFeatureImportance() {
    if (!featureDataGlobal) return;

    const selectedModel = document.getElementById('model-selector').value;
    const data = featureDataGlobal[selectedModel];
    
    if (!data) {
        console.error("No importance data for model:", selectedModel);
        return;
    }

    // Sort ascending so highest is at top in horizontal bar
    const sortedData = [...data].sort((a, b) => a.Importance - b.Importance);

    const features = sortedData.map(d => d.Feature.replace('_', ' ').toUpperCase());
    const importances = sortedData.map(d => d.Importance);

    const trace = {
        type: 'bar',
        x: importances,
        y: features,
        orientation: 'h',
        marker: {
            color: importances,
            colorscale: 'Blues',
            reversescale: true
        },
        hovertemplate: '%{x:.4f}<extra></extra>'
    };

    const layout = {
        margin: { t: 20, r: 30, b: 50, l: 150 },
        height: 600,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Information Gain / Weight',
            gridcolor: '#e0e0e0',
            zerolinecolor: '#ccc'
        },
        yaxis: {
            tickfont: { size: 13, color: '#333' }
        }
    };

    Plotly.newPlot('chart-feature-importance', [trace], layout, {responsive: true, displayModeBar: true});
}

function renderPreprocessingCharts(data) {
    if (!data) return;

    // 1. Ordinal Encoding Chart (Education Level)
    const eduMap = data.Ordinal['Education Level'];
    // Sort logic by integer value
    const eduSorted = Object.entries(eduMap).sort((a,b) => a[1] - b[1]);
    const eduLabels = eduSorted.map(x => x[0]);
    const eduVals = eduSorted.map(x => x[1]);

    const ordTrace = {
        x: eduLabels,
        y: eduVals,
        type: 'scatter',
        mode: 'lines+markers+text',
        text: eduVals.map(String),
        textposition: 'top center',
        line: { shape: 'hv', width: 4, color: '#3f51b5' },
        marker: { size: 12, color: '#3f51b5' },
        fill: 'tozeroy',
        fillcolor: 'rgba(63, 81, 181, 0.1)',
        hovertemplate: 'Level %{y}<extra></extra>'
    };

    const ordLayout = {
        margin: { t: 30, r: 20, b: 80, l: 60 },
        height: 380,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { tickfont: { size: 11 }, automargin: true },
        yaxis: { 
            title: 'Assigned Integer Value',
            tickvals: [0,1,2,3,4],
            gridcolor: '#e0e0e0',
            zeroline: false,
            automargin: true
        }
    };
    Plotly.newPlot('chart-ordinal', [ordTrace], ordLayout, {responsive: true, displayModeBar: true});

    // 2. Target Encoding Chart (Job Title)
    // Extract top salaries visually
    const jobMap = data.Target['job_title'];
    let jobSorted = Object.entries(jobMap).sort((a,b) => a[1] - b[1]);
    // Take top 8 and bottom 2, or just top 10 for simplicity
    jobSorted = jobSorted.slice(-10); // Keep top 10 paying jobs

    const jobLabels = jobSorted.map(x => x[0]);
    const jobVals = jobSorted.map(x => x[1]);

    const tgtTrace = {
        type: 'bar',
        x: jobVals,
        y: jobLabels,
        orientation: 'h',
        marker: {
            color: jobVals,
            colorscale: 'Greens',
            line: { color: '#2E7D32', width: 1 }
        },
        hovertemplate: '$%{x:,.0f}<extra></extra>'
    };

    const tgtLayout = {
        margin: { t: 30, r: 40, b: 80, l: 200 },
        height: 380,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Mapped Target Mean (Salary $)',
            gridcolor: '#e0e0e0',
            automargin: true
        },
        yaxis: { tickfont: { size: 11, color: '#333' }, automargin: true }
    };
    Plotly.newPlot('chart-target', [tgtTrace], tgtLayout, {responsive: true, displayModeBar: true});
}
