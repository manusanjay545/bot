/**
 * RL Trading Bot Dashboard - JavaScript
 */

// Global state
let currentInstrument = 'NIFTY';
let currentDays = 7;
let priceChart = null;
let rsiChart = null;
let macdChart = null;
let autoRefreshInterval = null;

// API base URL
const API_BASE = '';

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    setupEventListeners();
    loadAllData();

    // Auto-refresh every 1 minute for live market data
    autoRefreshInterval = setInterval(() => {
        loadAllData();
        updateLiveIndicator();
    }, 60 * 1000);

    // Update live indicator
    updateLiveIndicator();
});

function updateLiveIndicator() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const day = now.getDay();

    // Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    const isMarketHours = day >= 1 && day <= 5 &&
        ((hours === 9 && minutes >= 15) || (hours > 9 && hours < 15) || (hours === 15 && minutes <= 30));

    const statusEl = document.getElementById('api-status');
    const liveBadge = document.getElementById('live-badge');
    const lastUpdate = document.getElementById('last-update');

    if (statusEl) {
        if (isMarketHours) {
            statusEl.innerHTML = '🟢 LIVE';
            statusEl.className = 'online';
        } else {
            statusEl.innerHTML = '🔴 Market Closed';
            statusEl.className = 'offline';
        }
    }

    if (liveBadge) {
        if (isMarketHours) {
            liveBadge.textContent = '● LIVE';
            liveBadge.classList.remove('closed');
        } else {
            liveBadge.textContent = '● CLOSED';
            liveBadge.classList.add('closed');
        }
    }

    if (lastUpdate) {
        lastUpdate.textContent = now.toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

/**
 * Initialize Lightweight Charts
 */
function initializeCharts() {
    const chartOptions = {
        layout: {
            background: { color: '#1a1a24' },
            textColor: '#a0a0b0',
        },
        grid: {
            vertLines: { color: '#2a2a3a' },
            horzLines: { color: '#2a2a3a' },
        },
        crosshair: {
            mode: 0,
        },
        rightPriceScale: {
            borderColor: '#2a2a3a',
        },
        timeScale: {
            borderColor: '#2a2a3a',
            timeVisible: true,
        },
    };

    // Price chart
    const priceContainer = document.getElementById('price-chart');
    priceChart = LightweightCharts.createChart(priceContainer, {
        ...chartOptions,
        height: 400,
    });

    // RSI chart
    const rsiContainer = document.getElementById('rsi-chart');
    if (rsiContainer) {
        rsiChart = LightweightCharts.createChart(rsiContainer, {
            ...chartOptions,
            height: 200,
        });
    }

    // MACD chart
    const macdContainer = document.getElementById('macd-chart');
    if (macdContainer) {
        macdChart = LightweightCharts.createChart(macdContainer, {
            ...chartOptions,
            height: 200,
        });
    }

    // Handle resize
    window.addEventListener('resize', () => {
        priceChart.applyOptions({ width: priceContainer.clientWidth });
        if (rsiChart) rsiChart.applyOptions({ width: rsiContainer.clientWidth });
        if (macdChart) macdChart.applyOptions({ width: macdContainer.clientWidth });
    });
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Instrument selector
    document.querySelectorAll('.instrument-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.instrument-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentInstrument = e.target.dataset.instrument;
            loadAllData();
        });
    });

    // Search functionality
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('symbol-search');

    if (searchBtn && searchInput) {
        const handleSearch = () => {
            const symbol = searchInput.value.trim().toUpperCase();
            if (symbol) {
                // Remove active class from buttons
                document.querySelectorAll('.instrument-btn').forEach(b => b.classList.remove('active'));
                currentInstrument = symbol;
                loadAllData();
            }
        };

        searchBtn.addEventListener('click', handleSearch);
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSearch();
        });
    }

    // Chart timeframe buttons
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentDays = parseInt(e.target.dataset.days);
            loadChartData();
        });
    });
}

/**
 * Refresh all data
 */
function refreshData() {
    const refreshIcon = document.querySelector('.refresh-icon');
    refreshIcon.style.transform = 'rotate(360deg)';
    setTimeout(() => refreshIcon.style.transform = '', 500);
    loadAllData();
}

/**
 * Load all dashboard data
 */
async function loadAllData() {
    await Promise.all([
        loadIndicators(),
        loadSignal(),
        loadChartData(),
        loadPatterns(),
        checkApiStatus()
    ]);
}

/**
 * Load indicator values
 */
async function loadIndicators() {
    try {
        const response = await fetch(`${API_BASE}/api/indicators/${currentInstrument}`);
        const data = await response.json();

        if (data.error) {
            console.error('Indicators error:', data.error);
            return;
        }

        // Update price
        document.getElementById('current-price').textContent = `₹${data.price.toLocaleString()}`;

        const changeEl = document.getElementById('price-change');
        const changeSign = data.change >= 0 ? '+' : '';
        changeEl.textContent = `${changeSign}${data.change} (${changeSign}${data.changePct}%)`;
        changeEl.className = `stat-change ${data.change >= 0 ? 'positive' : 'negative'}`;

        // Update RSI
        document.getElementById('rsi-value').textContent = data.rsi || '--';
        const rsiStatus = data.rsi > 70 ? 'Overbought' : data.rsi < 30 ? 'Oversold' : 'Neutral';
        document.getElementById('rsi-status').textContent = rsiStatus;

        // Update Trend
        document.getElementById('trend-value').textContent = data.trend;
        document.getElementById('trend-status').textContent = data.trend === 'Bullish' ? '📈' : '📉';

        // Update Volatility
        document.getElementById('volatility-value').textContent = data.atr ? `₹${data.atr}` : '--';
        document.getElementById('volatility-status').textContent = data.volatility;

    } catch (error) {
        console.error('Failed to load indicators:', error);
    }
}

/**
 * Load trading signal
 */
async function loadSignal() {
    const signalCard = document.getElementById('signal-card');
    const signalAction = document.getElementById('signal-action');
    const signalIcon = document.getElementById('signal-icon');
    const signalTime = document.getElementById('signal-time');
    const confidenceValue = document.getElementById('confidence-value');

    try {
        const response = await fetch(`${API_BASE}/api/signal/${currentInstrument}`);
        const data = await response.json();

        if (data.error) {
            signalAction.textContent = 'ERROR';
            signalTime.textContent = data.error;
            return;
        }

        const signal = data.signal;
        signalCard.className = 'signal-card ' + (signal.action === 'BUY' ? 'buy' : signal.action === 'SELL' ? 'sell' : '');

        signalIcon.textContent = signal.action === 'BUY' ? '🟢' : signal.action === 'SELL' ? '🔴' : '⏸️';
        signalAction.textContent = signal.action + ' ' + currentInstrument;
        signalTime.textContent = signal.timestamp;
        confidenceValue.textContent = signal.confidence ? `${signal.confidence}%` : '--';

        // Update levels
        if (signal.entry) {
            document.getElementById('entry-price').textContent = `₹${signal.entry.toLocaleString()}`;
            document.getElementById('stop-loss').textContent = `₹${signal.stopLoss.toLocaleString()}`;
            document.getElementById('target-1').textContent = `₹${signal.target1.toLocaleString()}`;
            document.getElementById('target-2').textContent = `₹${signal.target2.toLocaleString()}`;
            document.getElementById('risk-reward').textContent = `1:${signal.riskReward}`;
        } else {
            document.getElementById('entry-price').textContent = '--';
            document.getElementById('stop-loss').textContent = '--';
            document.getElementById('target-1').textContent = '--';
            document.getElementById('target-2').textContent = '--';
            document.getElementById('risk-reward').textContent = '--';
        }

        // Update reasoning
        const reasoningList = document.getElementById('reasoning-list');
        reasoningList.innerHTML = '';
        (signal.reasoning || []).forEach(reason => {
            const li = document.createElement('li');
            li.textContent = reason;
            reasoningList.appendChild(li);
        });

    } catch (error) {
        console.error('Failed to load signal:', error);
        signalAction.textContent = 'CONNECTION ERROR';
        signalTime.textContent = 'Unable to connect to server';
    }
}

/**
 * Load chart data
 */
async function loadChartData() {
    try {
        const response = await fetch(`${API_BASE}/api/market-data/${currentInstrument}?days=${currentDays}`);
        const data = await response.json();

        if (data.error) {
            console.error('Chart data error:', data.error);
            return;
        }

        // Prepare candlestick data
        const candleData = data.timestamps.map((time, i) => ({
            time: new Date(time).getTime() / 1000,
            open: data.open[i],
            high: data.high[i],
            low: data.low[i],
            close: data.close[i],
        }));

        // Clear existing series and add new
        if (priceChart.series) {
            priceChart.series.forEach(s => priceChart.removeSeries(s));
        }

        // Add candlestick series
        const candleSeries = priceChart.addCandlestickSeries({
            upColor: '#00d4aa',
            downColor: '#ff4757',
            borderUpColor: '#00d4aa',
            borderDownColor: '#ff4757',
            wickUpColor: '#00d4aa',
            wickDownColor: '#ff4757',
        });
        candleSeries.setData(candleData);

        // Add Bollinger Bands if available
        if (data.bb_upper && data.bb_upper.length > 0) {
            const bbUpperSeries = priceChart.addLineSeries({
                color: 'rgba(124, 58, 237, 0.5)',
                lineWidth: 1,
            });
            bbUpperSeries.setData(data.timestamps.map((time, i) => ({
                time: new Date(time).getTime() / 1000,
                value: data.bb_upper[i],
            })));

            const bbLowerSeries = priceChart.addLineSeries({
                color: 'rgba(124, 58, 237, 0.5)',
                lineWidth: 1,
            });
            bbLowerSeries.setData(data.timestamps.map((time, i) => ({
                time: new Date(time).getTime() / 1000,
                value: data.bb_lower[i],
            })));
        }

        // Add EMAs
        if (data.ema_9 && data.ema_9.length > 0) {
            const ema9Series = priceChart.addLineSeries({
                color: '#00d4aa',
                lineWidth: 1,
            });
            ema9Series.setData(data.timestamps.map((time, i) => ({
                time: new Date(time).getTime() / 1000,
                value: data.ema_9[i],
            })));
        }

        if (data.ema_21 && data.ema_21.length > 0) {
            const ema21Series = priceChart.addLineSeries({
                color: '#7c3aed',
                lineWidth: 1,
            });
            ema21Series.setData(data.timestamps.map((time, i) => ({
                time: new Date(time).getTime() / 1000,
                value: data.ema_21[i],
            })));
        }

        // Add signal markers on the chart
        try {
            const signalResponse = await fetch(`${API_BASE}/api/signal/${currentInstrument}`);
            const signalData = await signalResponse.json();

            if (signalData.signal) {
                const signal = signalData.signal;
                const lastTime = new Date(data.timestamps[data.timestamps.length - 1]).getTime() / 1000;
                const isSignal = signal.action !== 'HOLD';

                // Entry line
                candleSeries.createPriceLine({
                    price: signal.entry,
                    color: isSignal ? (signal.action === 'BUY' ? '#00d4aa' : '#ff4757') : '#606070',
                    lineWidth: isSignal ? 2 : 1,
                    lineStyle: isSignal ? 0 : 2, // Solid if signal, dashed if potential
                    axisLabelVisible: true,
                    title: isSignal ? `${signal.action} Entry` : 'Potential Entry',
                });

                // Stop Loss line
                candleSeries.createPriceLine({
                    price: signal.stopLoss,
                    color: '#ff4757',
                    lineWidth: 1,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'Stop Loss',
                });

                // Target 1 line
                candleSeries.createPriceLine({
                    price: signal.target1,
                    color: '#00d4aa',
                    lineWidth: 1,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'Target 1',
                });

                // Target 2 line
                candleSeries.createPriceLine({
                    price: signal.target2,
                    color: '#00d4aa',
                    lineWidth: 1,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'Target 2',
                });

                // Add marker only if it's a real signal
                if (isSignal) {
                    candleSeries.setMarkers([{
                        time: lastTime,
                        position: signal.action === 'BUY' ? 'belowBar' : 'aboveBar',
                        color: signal.action === 'BUY' ? '#00d4aa' : '#ff4757',
                        shape: signal.action === 'BUY' ? 'arrowUp' : 'arrowDown',
                        text: `${signal.action} ${signal.confidence}%`,
                    }]);
                } else {
                    candleSeries.setMarkers([]);
                }
            }
        } catch (signalError) {
            console.log('Signal overlay skipped:', signalError);
        }

        priceChart.timeScale().fitContent();

        // RSI Chart
        if (rsiChart && data.rsi && data.rsi.length > 0) {
            if (rsiChart.series) {
                rsiChart.series.forEach(s => rsiChart.removeSeries(s));
            }

            const rsiSeries = rsiChart.addLineSeries({
                color: '#ffa502',
                lineWidth: 2,
            });
            rsiSeries.setData(data.timestamps.map((time, i) => ({
                time: new Date(time).getTime() / 1000,
                value: data.rsi[i],
            })));

            rsiChart.timeScale().fitContent();
        }

        // MACD Chart
        if (macdChart && data.macd && data.macd.length > 0) {
            if (macdChart.series) {
                macdChart.series.forEach(s => macdChart.removeSeries(s));
            }

            const macdLine = macdChart.addLineSeries({
                color: '#00d4aa',
                lineWidth: 2,
            });
            macdLine.setData(data.timestamps.map((time, i) => ({
                time: new Date(time).getTime() / 1000,
                value: data.macd[i],
            })));

            if (data.macd_signal && data.macd_signal.length > 0) {
                const signalLine = macdChart.addLineSeries({
                    color: '#ff4757',
                    lineWidth: 2,
                });
                signalLine.setData(data.timestamps.map((time, i) => ({
                    time: new Date(time).getTime() / 1000,
                    value: data.macd_signal[i],
                })));
            }

            macdChart.timeScale().fitContent();
        }

    } catch (error) {
        console.error('Failed to load chart data:', error);
    }
}

/**
 * Load detected patterns
 */
async function loadPatterns() {
    const grid = document.getElementById('patterns-grid');

    try {
        const response = await fetch(`${API_BASE}/api/patterns/${currentInstrument}`);
        const data = await response.json();

        if (!data.patterns || data.patterns.length === 0) {
            grid.innerHTML = '<div class="pattern-card">No patterns detected at this time</div>';
            return;
        }

        grid.innerHTML = data.patterns.map(pattern => `
            <div class="pattern-card ${pattern.type.toLowerCase()} fade-in">
                <span class="pattern-icon">${pattern.type === 'BULLISH' ? '🟢' : pattern.type === 'BEARISH' ? '🔴' : '⚪'}</span>
                <div class="pattern-info">
                    <h4>${pattern.name}</h4>
                    <p>${pattern.description}</p>
                </div>
                <span class="pattern-strength">${pattern.strength}%</span>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load patterns:', error);
        grid.innerHTML = '<div class="pattern-card">Failed to load patterns</div>';
    }
}

/**
 * Check API status
 */
async function checkApiStatus() {
    const statusEl = document.getElementById('api-status');

    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        if (data.status === 'online') {
            statusEl.textContent = `Online (${data.models.length} models)`;
            statusEl.className = 'online';
        } else {
            statusEl.textContent = 'Offline';
            statusEl.className = 'offline';
        }
    } catch (error) {
        statusEl.textContent = 'Offline';
        statusEl.className = 'offline';
    }
}
