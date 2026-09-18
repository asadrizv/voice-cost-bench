import path from "node:path";
import { defineConfig } from "@playwright/test";

// Chromium's fake capture device plays a fixture WAV as the microphone, so the scripted
// call speaks real audio through WebRTC, LiveKit and the agent.
const fixture = path.resolve(__dirname, "e2e/caller.wav");

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    permissions: ["microphone"],
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-audio-capture=${fixture}`,
        "--autoplay-policy=no-user-gesture-required",
      ],
    },
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
