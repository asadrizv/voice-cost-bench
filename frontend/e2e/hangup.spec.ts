import path from "node:path";
import { expect, test } from "@playwright/test";

// The caller says goodbye once and then stays silent; only the agent can end this call.
test.use({
  launchOptions: {
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${path.resolve(__dirname, "wrong_number.wav")}`,
    ],
  },
});

test("the agent hangs up after the caller says goodbye", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("pipeline-selfhosted").click();
  await page.getByTestId("start-call").click();
  await expect(page.getByTestId("transcript").locator(".bubble.caller").first()).toContainText("wrong number", {
    timeout: 40_000,
  });
  await expect(page.getByTestId("call-notice")).toHaveText("Clara ended the call.", { timeout: 20_000 });
  await expect(page.getByTestId("start-call")).toBeVisible();
});
