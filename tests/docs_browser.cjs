// Browser acceptance for the built documentation preview.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const base = process.argv[2] || 'http://127.0.0.1:4001/Nocturne/';
  const output = process.env.SCREENSHOT_DIR || '/tmp/nocturne-docs-screenshots';
  fs.mkdirSync(output, { recursive: true });
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await Promise.all([
        page.waitForResponse('**/assets/js/search-data.json'),
        page.goto(base),
      ]);
      await page.waitForTimeout(200);
      await assert.equal(await page.title(), 'Overview | Nocturne');
      await assert.equal((await page.locator('main h1').first().textContent()).trim(), 'Nocturne');
      await assert.equal(await page.locator('main img').first().evaluate(img => img.complete && img.naturalWidth > 0), true);
      if (width <= 390) {
        await page.locator('#menu-button').click();
        await assert.equal(await page.locator('#menu-button').getAttribute('aria-expanded'), 'true');
      }
      await page.locator('#search-input').pressSequentially('volatility');
      await page.waitForFunction(() => document.querySelector('.search-results-list')?.textContent.includes('Prediction'));
      await page.keyboard.press('Escape');
      if (width <= 390) await page.locator('#menu-button').click();
      await assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      await page.screenshot({ path: path.join(output, `docs-${width}.png`), fullPage: true });
      await page.goto(new URL('api/', base).toString());
      await assert.equal((await page.locator('main h1').first().textContent()).trim(), 'Local API');
      await assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Docs browser passed: theme, image, nav, search, keyboard, API, desktop/mobile overflow.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
