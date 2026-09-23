// Real Flask/SQLite integration; only response timing is intercepted for races.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const base = process.argv[2] || 'http://127.0.0.1:5053';
  const output = process.env.SCREENSHOT_DIR || '/tmp/nocturne-93-screenshots';
  fs.mkdirSync(output, { recursive: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.locator('#ticker').fill('UPRO');
    await page.waitForFunction(() => !document.querySelector('#prediction-result').hidden);
    assert.equal(await page.locator('#result-source').textContent(), 'Saved latest result');
    assert.equal(await page.locator('#result-freshness').textContent(), 'stale');
    assert.notEqual(await page.locator('#result-date').textContent(), '--');
    await page.waitForFunction(() => document.querySelector('#history-body').textContent.includes('2025-'));
    assert(await page.locator('button img').evaluateAll(images => images.every(img => img.complete && img.naturalWidth > 0)));
    await page.locator('#fetch-latest').uncheck();
    await page.locator('#prediction-button').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent.includes('Prediction complete'));
    await page.waitForFunction(() => !document.querySelector('#run-controls').disabled);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({ path: path.join(output, `dashboard-${width}.png`), fullPage: true });
    }
    await page.reload();
    await page.locator('#ticker').fill('UPRO');
    await page.waitForFunction(() => document.querySelector('#result-source').textContent === 'Saved latest result');

    // Hold an actual saved response while a new prediction for the same ticker completes.
    let release, started;
    const gate = new Promise(resolve => { release = resolve; });
    const intercepted = new Promise(resolve => { started = resolve; });
    await page.route('**/api/predictions/latest*', async route => {
      const response = await route.fetch();
      started();
      await gate;
      await route.fulfill({ response });
    });
    await page.locator('#ticker').fill('upro');
    await intercepted;
    await page.locator('#prediction-button').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent.includes('Prediction complete'));
    const finished = page.waitForResponse('**/api/predictions/latest*');
    release();
    await finished;
    await page.waitForTimeout(150);
    assert.equal(await page.locator('#result-source').textContent(), 'New result');
    await page.unroute('**/api/predictions/latest*');

    // A delayed UPRO response must not repopulate the result after switching to AAPL.
    let releaseTicker, startedTicker;
    const tickerGate = new Promise(resolve => { releaseTicker = resolve; });
    const tickerIntercepted = new Promise(resolve => { startedTicker = resolve; });
    await page.route('**/api/predictions/latest*', async route => {
      const response = await route.fetch();
      if (new URL(route.request().url()).searchParams.get('ticker') === 'UPRO') {
        startedTicker();
        await tickerGate;
      }
      await route.fulfill({ response });
    });
    await page.waitForFunction(() => !document.querySelector('#run-controls').disabled);
    await page.locator('#ticker').fill('UPRO');
    await tickerIntercepted;
    await page.locator('#ticker').fill('AAPL');
    await page.waitForResponse('**/api/predictions/latest?ticker=AAPL');
    const tickerFinished = page.waitForResponse('**/api/predictions/latest?ticker=UPRO');
    releaseTicker();
    await tickerFinished;
    await page.waitForTimeout(150);
    assert(await page.locator('#prediction-result').isHidden());
    await page.locator('#prediction-button').click();
    await page.waitForFunction(() => document.querySelector('#run-status').className === 'error');
    assert(await page.locator('#prediction-result').isHidden());
    await page.unroute('**/api/predictions/latest*');
    await page.route('**/api/predictions/latest*', route => route.fulfill({ json: {
      success: true, prediction: { ticker: 'LEGACY', date: null, direction: 'UP',
        volatility_filter_triggered: null, volatility_state: 'unknown',
        readiness_status: null, freshness: { status: 'unavailable' } },
    } }));
    await page.locator('#ticker').fill('LEGACY');
    await page.waitForFunction(() => document.querySelector('#result-filter').textContent === 'Unknown');
    assert.equal(await page.locator('#result-freshness').textContent(), 'unavailable');
    await page.unroute('**/api/predictions/latest*');
    assert.deepEqual(errors, []);
    console.log('Installed browser checks passed: real inference/history/reload/errors/assets, same-ticker and ticker-switch races, desktop/mobile.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
