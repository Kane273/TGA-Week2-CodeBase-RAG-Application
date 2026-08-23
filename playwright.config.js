import {defineConfig, devices} from "@playwright/test";

export default defineConfig({
  testDir: "tests/e2e",
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? [["html", {open: "never"}], ["line"]] : "line",
  use: {
    baseURL: process.env.BASE_URL || "http://127.0.0.1:8000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {name: "chromium", use: {...devices["Desktop Chrome"]}},
    {name: "mobile", use: {...devices["Pixel 7"]}},
  ],
});
