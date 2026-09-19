import { expect, test } from "@playwright/test";

test("the navigation links to the public transparency endpoint", async ({ page }) => {
  await page.goto("/");
  const link = page.getByRole("navigation").getByRole("link", { name: "Transparency" });
  await expect(link).toHaveAttribute("href", /\/transparency$/);
});
