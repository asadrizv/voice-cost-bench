import { expect, test } from "@playwright/test";

const API = process.env.E2E_API_URL ?? "http://localhost:8080";

test("a scripted call renders non-zero live costs and lands in history", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("pipeline-selfhosted").click();
  await page.getByTestId("start-call").click();

  // The greeting turn arrives first, then at least one caller turn from the fake mic.
  await expect(page.getByTestId("transcript").locator(".bubble").first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("transcript").locator(".bubble.caller").first()).toBeVisible({ timeout: 60_000 });

  const total = page.getByTestId("running-total");
  await expect(total).not.toHaveText("$0.0000", { timeout: 15_000 });
  await expect(page.getByTestId("cost-per-minute")).not.toHaveText("$0.0000");
  await expect(page.getByTestId("latency").locator("table tbody tr").first()).toBeVisible();

  await page.getByTestId("end-call").click();

  const calls = await (await page.request.get(`${API}/calls?pipeline=selfhosted`)).json();
  expect(calls.length).toBeGreaterThan(0);
  const cost = await (await page.request.get(`${API}/calls/${calls[0].id}/cost`)).json();
  expect(cost.stages.gpu).toBeGreaterThan(0);

  await page.goto("/calls");
  await expect(page.getByTestId("calls-table")).toBeVisible();
  await expect(page.getByTestId("compare-selfhosted")).toContainText("Cost per minute");
});
