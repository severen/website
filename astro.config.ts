// SPDX-FileCopyrightText: 2025 Severen Redwood <sev@severen.dev>
// SPDX-License-Identifier: CC0-1.0

import { defineConfig, fontProviders } from "astro/config";

export default defineConfig({
  site: "https://severen.dev",
  fonts: [
    {
      provider: fontProviders.fontsource(),
      name: "Inter",
      cssVariable: "--font-inter",
      // Inter is a variable font, so request the full weight range.
      weights: ["100 900"],
      styles: ["normal"],
      subsets: ["latin"],
    },
  ],
  vite: {
    css: {
      transformer: "lightningcss",
    },
    build: {
      cssMinify: "lightningcss",
    },
  },
});
