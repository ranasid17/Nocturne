// With Playwright installed: node tests/web_dashboard.cjs [base URL]
// Set CHROME_PATH to use an existing Chrome installation.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.argv[2] || 'http://127.0.0.1:5052';
    await page.goto(base);
    await page.waitForFunction(() => !document.querySelector('#history-status').textContent.includes('Loading'));
    assert.equal(await page.title(), 'Nocturne | Predictions');
    assert(await page.locator('#tickers option').count() > 0);
    assert(await page.locator('button img').evaluateAll(images => images.every(img => img.complete && img.naturalWidth > 0)));
    await page.screenshot({ path: '/tmp/qusa-sprint3-desktop.png', fullPage: true });

    let historyCount = 0;
    await page.route('**/api/predictions/history*', async route => {
      historyCount++;
      const symbol = new URL(route.request().url()).searchParams.get('ticker') || 'UPRO';
      await route.fulfill({ json: { success: true, history: [{
        ticker: symbol, timestamp: '2026-05-22 10:00:00', date: '2026-05-22',
        direction: 'UP', probability_up: 0.72, confidence: 'HIGH',
      }] } });
    });
    await page.locator('#ticker').fill('UPRO');
    await page.waitForFunction(() => document.querySelector('#history-body').textContent.includes('UPRO'));
    await page.locator('#fetch-latest').check();
    await page.locator('#volatility').fill('2.5');
    let releasePrediction;
    let predictionCalls = 0;
    const predictionGate = new Promise(resolve => { releasePrediction = resolve; });
    await page.route('**/api/predictions/run', async route => {
      predictionCalls++;
      assert.deepEqual(route.request().postDataJSON(), { ticker: 'UPRO', fetch_latest: true, volatility: 2.5 });
      await predictionGate;
      await route.fulfill({ json: { success: true, ticker: 'UPRO', volatility_filter: { enabled: true },
        prediction: { date: '2026-05-22T00:00:00', direction: 'UP', confidence: 'HIGH', probability_up: 0.72, volatility_filter_triggered: false, volatility_state: 'pass' } } });
    });
    await page.locator('#volatility').press('Enter');
    assert(await page.locator('#pipeline-button').isDisabled());
    assert(await page.locator('#prediction-button').isDisabled());
    assert(await page.locator('#ticker').isDisabled());
    const beforePrediction = historyCount;
    releasePrediction();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent.includes('Prediction complete'));
    await page.waitForFunction(() => !document.querySelector('#run-controls').disabled);
    assert.equal(predictionCalls, 1);
    assert(historyCount > beforePrediction);
    assert.equal(await page.locator('#result-probability').textContent(), '72.0%');
    assert.equal(await page.locator('#result-filter').textContent(), 'Within limit');
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({ path: `/tmp/qusa-sprint3-result-${width}.png`, fullPage: true });
    }
    await page.setViewportSize({ width: 1440, height: 1000 });

    await page.route('**/api/pipeline/run', async route => {
      assert.deepEqual(route.request().postDataJSON(), { ticker: 'UPRO', fetch_latest: true });
      await route.fulfill({ json: { success: true, rows: 513, output_path: '/data/UPRO_processed.csv' } });
    });
    await page.locator('#pipeline-button').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent.includes('513 rows saved'));
    await page.unroute('**/api/predictions/run');
    await page.route('**/api/predictions/run', route => route.fulfill({ status: 404, json: { success: false, error: 'Model not found' } }));
    await page.locator('#prediction-button').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent.includes('Model not found'));
    assert(await page.locator('#prediction-result').isHidden());
    assert(await page.locator('#prediction-button').isEnabled());

    await page.unroute('**/api/predictions/history*');
    await page.route('**/api/predictions/history*', route => route.fulfill({ json: { success: true, history: [] } }));
    await page.locator('#refresh-history').click();
    await page.waitForFunction(() => document.querySelector('#history-status').textContent.includes('No predictions'));
    await page.unroute('**/api/predictions/history*');
    await page.route('**/api/predictions/history*', route => route.fulfill({ status: 500, json: { success: false, error: 'Log unavailable' } }));
    await page.locator('#refresh-history').click();
    await page.waitForFunction(() => document.querySelector('#history-status').textContent.includes('Log unavailable'));

    await page.unroute('**/api/predictions/history*');
    let releaseOld;
    const oldGate = new Promise(resolve => { releaseOld = resolve; });
    await page.route('**/api/predictions/history*', async route => {
      const symbol = new URL(route.request().url()).searchParams.get('ticker');
      if (symbol === 'AAPL') await oldGate;
      await route.fulfill({ json: { success: true, history: [{ ticker: symbol, direction: '<img src=x onerror=alert(1)>', probability_up: null }] } });
    });
    const oldStarted = page.waitForRequest('**/api/predictions/history?ticker=AAPL');
    await page.locator('#ticker').fill('AAPL');
    await oldStarted;
    await page.locator('#ticker').fill('UPRO');
    await page.waitForFunction(() => document.querySelector('#history-body').textContent.includes('UPRO'));
    const oldFinished = page.waitForResponse('**/api/predictions/history?ticker=AAPL');
    releaseOld();
    await oldFinished;
    await page.waitForTimeout(100);
    assert((await page.locator('#history-body').textContent()).includes('UPRO'));
    assert.equal(await page.locator('#history-body img').count(), 0);

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({ path: `/tmp/qusa-sprint3-mobile-${width}.png`, fullPage: true });
    }
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: live page, assets, payloads, busy states, results, history, errors, stale responses, escaping, mobile overflow.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
