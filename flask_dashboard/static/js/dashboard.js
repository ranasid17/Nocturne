// flask_dashboard/static/js/dashboard.js

document.addEventListener('DOMContentLoaded', function() {
    const tickerSelect = document.getElementById('tickerSelect');
    const tickerTitle = document.getElementById('tickerTitle');
    const refreshPipelineBtn = document.getElementById('refreshPipelineBtn');
    const runInferenceBtn = document.getElementById('runInferenceBtn');

    // Initial load
    loadTickers();
    loadDashboardData();

    // Event Listeners
    tickerSelect.addEventListener('change', function() {
        tickerTitle.textContent = `${this.value} Dashboard`;
        loadDashboardData();
    });

    refreshPipelineBtn.addEventListener('click', function() {
        refreshPipeline();
    });

    runInferenceBtn.addEventListener('click', function() {
        runInference();
    });

    // Helper functions
    function loadTickers() {
        fetch('/api/tickers')
            .then(response => response.json())
            .then(tickers => {
                tickerSelect.innerHTML = '';
                tickers.forEach(ticker => {
                    const option = document.createElement('option');
                    option.value = ticker;
                    option.textContent = ticker;
                    tickerSelect.appendChild(option);
                });
            });
    }

    function loadDashboardData() {
        const ticker = tickerSelect.value;
        if (!ticker) return;

        loadPredictionData(ticker);
        loadPerformanceData(ticker);
    }

    function loadPredictionData(ticker) {
        fetch(`/api/prediction/${ticker}`)
            .then(response => response.json())
            .then(data => {
                updatePredictionUI(data);
            });
    }

    function updatePredictionUI(data) {
        const latest = data.latest;
        const history = data.history;

        if (latest) {
            document.getElementById('metric-bias').textContent = latest.direction || '--';
            document.getElementById('metric-prob').textContent = latest.probability_up ? `${(latest.probability_up * 100).toFixed(1)}%` : '--';
            document.getElementById('metric-conviction').textContent = latest.confidence || '--';

            const statusBox = document.getElementById('signalStatusBox');
            statusBox.classList.remove('d-none', 'alert-success', 'alert-warning');
            if (latest.confidence === 'HIGH') {
                statusBox.classList.add('alert-success');
                statusBox.textContent = `STRATEGIC ${latest.direction} SIGNAL DETECTED`;
            } else {
                statusBox.classList.add('alert-warning');
                statusBox.textContent = 'NO HIGH-CONVICTION SIGNAL PRESENT';
            }
        }

        const tableBody = document.getElementById('predictionLogTableBody');
        tableBody.innerHTML = '';
        history.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.date}</td>
                <td><span class="badge ${row.direction === 'UP' ? 'bg-success' : 'bg-danger'}">${row.direction}</span></td>
                <td>${(row.probability_up * 100).toFixed(1)}%</td>
                <td><span class="badge ${row.confidence === 'HIGH' ? 'bg-primary' : 'bg-secondary'}">${row.confidence}</span></td>
                <td>${parseFloat(row.atr_pct).toFixed(3)}</td>
                <td>${row.volatility_filter_triggered ? 'TRIGGERED' : 'CLEAR'}</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    function loadPerformanceData(ticker) {
        fetch(`/api/performance/${ticker}`)
            .then(response => {
                if (!response.ok) throw new Error('Performance data not available');
                return response.json();
            })
            .then(data => {
                renderCharts(data);
                updatePerformanceMetrics(data.metrics);
            })
            .catch(error => {
                console.error(error);
                // Clear charts or show error
            });
    }

    function updatePerformanceMetrics(metrics) {
        if (!metrics) return;
        
        const updateMetric = (id, value, isPct = false) => {
            const el = document.getElementById(id);
            const valEl = el.querySelector('.h3');
            valEl.textContent = isPct ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
            
            // Update color based on alpha/returns
            if (id === 'perf-alpha' || id === 'perf-total-return') {
                el.style.backgroundColor = value < 0 ? '#9f1239' : '#065f46';
            }
        };

        updateMetric('perf-total-return', metrics.strategy_return, true);
        updateMetric('perf-sharpe', metrics.sharpe_ratio);
        updateMetric('perf-max-dd', Math.abs(metrics.max_draw_down), true);
        updateMetric('perf-alpha', metrics.alpha);
    }

    function renderCharts(data) {
        // Equity Chart
        const equityTrace = {
            x: data.dates,
            y: data.strategy_value,
            name: 'Strategy',
            type: 'scatter',
            line: { color: '#2563eb', width: 2.5 }
        };
        const bhTrace = {
            x: data.dates,
            y: data.buy_hold_value,
            name: 'B&H',
            type: 'scatter',
            line: { color: '#94a3b8', dash: 'dash' }
        };
        const equityLayout = {
            title: 'Cumulative Equity',
            template: 'plotly_white',
            margin: { t: 40, b: 40, l: 40, r: 20 },
            hovermode: 'x unified'
        };
        Plotly.newPlot('equityChart', [equityTrace, bhTrace], equityLayout);

        // Drawdown Chart
        const ddTrace = {
            x: data.dates,
            y: data.drawdown,
            fill: 'tozeroy',
            name: 'Drawdown',
            type: 'scatter',
            line: { color: '#ef4444' }
        };
        const ddLayout = {
            title: 'Strategy Drawdown (%)',
            template: 'plotly_white',
            margin: { t: 40, b: 40, l: 40, r: 20 },
            yaxis: { title: 'Drawdown %' }
        };
        Plotly.newPlot('drawdownChart', [ddTrace], ddLayout);
    }

    function refreshPipeline() {
        const ticker = tickerSelect.value;
        refreshPipelineBtn.disabled = true;
        refreshPipelineBtn.textContent = 'Refreshing...';

        fetch('/api/refresh-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        })
        .then(response => response.json())
        .then(result => {
            pollTaskStatus(result.task_id, () => {
                refreshPipelineBtn.disabled = false;
                refreshPipelineBtn.textContent = 'Refresh Pipeline';
                loadDashboardData();
            });
        });
    }

    function runInference() {
        const ticker = tickerSelect.value;
        const recipients = document.getElementById('recipientInput').value.split(',').map(r => r.trim()).filter(r => r);
        
        runInferenceBtn.disabled = true;
        runInferenceBtn.textContent = 'Running Model...';

        fetch('/api/run-inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker, recipients: recipients })
        })
        .then(response => response.json())
        .then(result => {
            pollTaskStatus(result.task_id, (finalData) => {
                runInferenceBtn.disabled = false;
                runInferenceBtn.textContent = 'Generate New Inference';
                loadDashboardData();
                if (finalData.email) {
                    alert(finalData.email.sent ? 'Prediction email sent!' : `Email error: ${finalData.email.error}`);
                }
            });
        });
    }

    function pollTaskStatus(taskId, callback) {
        const interval = setInterval(() => {
            fetch(`/api/task-status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        clearInterval(interval);
                        callback(data);
                    } else if (data.status === 'failed') {
                        clearInterval(interval);
                        alert(`Task failed: ${data.error}`);
                        callback(data);
                    }
                });
        }, 2000);
    }
});
