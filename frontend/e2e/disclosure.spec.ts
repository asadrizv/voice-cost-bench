import { expect, test } from "@playwright/test";

// EU AI Act Art. 50(1): the caller hears that Clara is an AI before anything else.
for (const [persona, disclosure] of [
  ["law_firm", "an AI assistant"],
  ["law_firm_de", "eine KI-Assistentin"],
] as const) {
  test(`the ${persona} greeting discloses that Clara is an AI`, async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("pipeline-selfhosted").click();
    await page.getByLabel("Language").selectOption(persona);
    await page.getByTestId("start-call").click();
    const greeting = page.getByTestId("transcript").locator(".bubble").first();
    await expect(greeting).toContainText(disclosure, { timeout: 30_000 });
    await page.getByTestId("end-call").click();
  });
}
