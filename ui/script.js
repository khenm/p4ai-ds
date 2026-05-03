/* ═══════════════════════════════════════════════════════ */
/*  P4AI-DS EDA — Shared JavaScript Utilities              */
/* ═══════════════════════════════════════════════════════ */

// ── Plotly Theme (warm pastel) ──
const COLORS = {
    sequential: ['#FAE8D4', '#E6C9A8', '#C26A2E', '#9E5420', '#6B3A15'],
    qualitative: ['#6886A5', '#C7727A', '#5E8A5C', '#9B7DB8', '#B8942F', '#C26A2E'],
    diverging: ['#6886A5', '#E9C46A', '#C26A2E'],
    speedColors: ['#6886A5', '#5E8A5C', '#B8942F', '#C26A2E', '#C7727A'],
    speedNames: ['Same Day', 'First Week', 'First Month', '2nd-3rd Month', 'No Adoption'],
};

function plotlyLayout(title, extra = {}) {
    const base = {
        title: { text: title, font: { size: 14, family: 'Inter, sans-serif', color: '#3D2C1E' } },
        paper_bgcolor: '#FFFFFF',
        plot_bgcolor: '#FDFAF6',
        font: { family: 'Inter, sans-serif', size: 11, color: '#6B5744' },
        xaxis: { gridcolor: '#EDE5D8', gridwidth: 0.5, showline: true, linecolor: '#DDD2C3', linewidth: 0.5, zeroline: false, ...(extra.xaxis || {}) },
        yaxis: { gridcolor: '#EDE5D8', gridwidth: 0.5, showline: true, linecolor: '#DDD2C3', linewidth: 0.5, zeroline: false, ...(extra.yaxis || {}) },
        margin: { l: 55, r: 40, t: 60, b: 45, ...(extra.margin || {}) },
        hoverlabel: { bgcolor: '#FDF5EC', font: { family: 'Inter, sans-serif', size: 12 } },
        colorway: COLORS.qualitative,
    };
    const { xaxis, yaxis, margin, ...rest } = extra;
    return { ...base, ...rest };
}

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
};

// ── Data Loading ─────────────────────────────────────
async function loadJSON(filename) {
    const resp = await fetch(`assets/data/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load ${filename}: ${resp.status}`);
    return resp.json();
}

// ── Section Collapse / Expand ────────────────────────
function toggleSection(headerEl) {
    headerEl.classList.toggle('collapsed');
    const body = headerEl.nextElementSibling;
    if (body) body.classList.toggle('collapsed');
}

function expandAll() {
    document.querySelectorAll('.section-header.collapsed').forEach(h => {
        h.classList.remove('collapsed');
        const body = h.nextElementSibling;
        if (body) body.classList.remove('collapsed');
    });
}

function collapseAll() {
    document.querySelectorAll('.section-header:not(.collapsed)').forEach(h => {
        h.classList.add('collapsed');
        const body = h.nextElementSibling;
        if (body) body.classList.add('collapsed');
    });
}

// ── Modal ────────────────────────────────────────────
function showModal(title, contentHTML) {
    let overlay = document.getElementById('modal-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'modal-overlay';
        overlay.className = 'modal-overlay';
        overlay.innerHTML = `<div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <h3 id="modal-title"></h3>
            <div id="modal-body"></div>
        </div>`;
        document.body.appendChild(overlay);
        overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });
    }
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = contentHTML;
    overlay.classList.add('active');
}

function closeModal() {
    const overlay = document.getElementById('modal-overlay');
    if (overlay) overlay.classList.remove('active');
}

// ── Toast ────────────────────────────────────────────
function showToast(msg) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.style.cssText = `
            position: fixed; bottom: 20px; right: 20px;
            background: #3D2C1E; color: #FDF5EC; padding: 12px 20px;
            border-radius: 8px; font-size: 0.9rem; z-index: 999;
            opacity: 0; transition: opacity 0.3s;
            font-family: Inter, sans-serif;
        `;
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 2500);
}

// ── Nav active state ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const page = globalThis.location.pathname.split('/').pop() || 'index.html';
    const activePage = {
        'text_results.html': 'text.html',
        'image_results.html': 'image.html',
    }[page] || page;
    document.querySelectorAll('.nav-links a').forEach(a => {
        const href = a.getAttribute('href');
        if (href === activePage || (activePage === '' && href === 'index.html')) {
            a.classList.add('active');
        }
    });
});
