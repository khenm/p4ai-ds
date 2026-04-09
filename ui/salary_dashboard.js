/**
 * Tải file JSON và render biểu đồ bằng Plotly
 * @param {string} jsonUrl - Đường dẫn tới file json chứa dữ liệu biểu đồ
 * @param {string} containerId - ID của thẻ div để chứa biểu đồ
 */
async function fetchAndRenderPlotly(jsonUrl, containerId) {
    try {
        const response = await fetch(jsonUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const figData = await response.json();
        
        const config = { 
            responsive: true,
            displayModeBar: true 
        };
        
        Plotly.newPlot(containerId, figData.data, figData.layout, config);
        
    } catch (error) {
        console.error(`Không thể hiển thị biểu đồ cho "${containerId}":`, error);
        
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="display: flex; height: 100%; justify-content: center; align-items: center; color: #ef4444; background: #fee2e2; border-radius: 8px;">
                    <p><strong>Lỗi dữ liệu:</strong> Chưa thể hiển thị biểu đồ. Vui lòng đảm bảo script Python (salary_eda.py) đã được chạy thành công và tạo ra file dữ liệu JSON tại "assets/data/jobsalary/".</p>
                </div>
            `;
        }
    }
}

async function loadAndRenderOverview(jsonUrl) {
    try {
        const response = await fetch(jsonUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();

        // Stats
        document.getElementById('overview-stats').innerHTML = `
            <div class="stat-card stat-blue"><div class="stat-label">Total Listings</div><div class="stat-value">${data.total_listings.toLocaleString()}</div></div>
            <div class="stat-card stat-brown"><div class="stat-label">Features</div><div class="stat-value">${data.feature_count}</div><div class="stat-sub">${data.feature_count} columns</div></div>
            <div class="stat-card stat-green"><div class="stat-label">Numerical Features</div><div class="stat-value">${data.num_features}</div></div>
            <div class="stat-card stat-amber"><div class="stat-label">Categorical Features</div><div class="stat-value">${data.cat_features}</div></div>
        `;

        // Description
        document.getElementById('overview-desc').innerHTML = `The Job Salary Prediction dataset contains <strong>${data.total_listings.toLocaleString()}</strong> records. Each record has <strong>${data.feature_count}</strong> features detailing the job properties (Job Title, Industry, Company Size, Location), candidate details (Experience Years, Education Level, Skills Count, Certifications), and the target variable <strong>Salary</strong>.`;

        // Sample Rows
        const cols = data.display_columns;
        let html = '<thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        data.sample_rows.forEach(row => {
            html += '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>';
        });
        html += '</tbody>';
        document.getElementById('sample-table').innerHTML = html;

        // Dtypes Table
        let dhtml = '<thead><tr><th>Column</th><th>Feature Type</th><th>Data Type</th><th>Non-Null</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>';
        data.columns.forEach(c => {
            let badgeClass = c.feature_type === 'Target' ? 'badge-image' : (c.feature_type === 'Numerical' ? 'badge-tabular' : '');
            let fTypeStr = badgeClass ? `<span class="badge ${badgeClass}" style="margin:0; padding:2px 8px; font-size:0.7rem;">${c.feature_type}</span>` : `<span class="badge" style="margin:0; padding:2px 8px; font-size:0.7rem; background:#E5E0F5; color:#4B3C8E;">${c.feature_type}</span>`;
            
            dhtml += `<tr><td><strong>${c.name}</strong></td><td>${fTypeStr}</td><td>${c.dtype}</td><td class="num">${c.non_null.toLocaleString()}</td><td class="num">${c.missing}</td><td class="num">${c.missing_pct}%</td></tr>`;
        });
        dhtml += '</tbody>';
        document.getElementById('dtypes-table').innerHTML = dhtml;

    } catch (error) {
        console.error('Failed to load overview data:', error);
        document.getElementById('overview-desc').innerHTML = '<span style="color:red">Error loading dataset overview. Please ensure Python script was run successfully.</span>';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const DATA_BASE_DIR = 'assets/data/jobsalary/';

    loadAndRenderOverview(DATA_BASE_DIR + 'dataset_overview.json');

    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'education_boxplot.json', 
        'chart-education-boxplot'
    );
    
    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'company_size_boxplot.json', 
        'chart-company_size-boxplot'
    );
    
    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'job_title_boxplot.json', 
        'chart-job_title-boxplot'
    );
    
    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'industry_boxplot.json', 
        'chart-industry-boxplot'
    );
    
    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'remote_work_boxplot.json', 
        'chart-remote_work-boxplot'
    );
    
    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'location_boxplot.json', 
        'chart-location-boxplot'
    );

    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'education_company_size_boxplot.json', 
        'chart-education-company-interaction'
    );

    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'job_title_freq_salary.json', 
        'chart-job-title-freq'
    );

    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'experience_salary_scatter.json', 
        'chart-experience-scatter'
    );
    
    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'numerical_correlation.json', 
        'chart-numerical-correlation'
    );

    fetchAndRenderPlotly(
        DATA_BASE_DIR + 'categorical_correlation.json', 
        'chart-categorical-correlation'
    );

});
