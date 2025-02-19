// SPDX-FileCopyrightText: 2025 Severen Redwood <sev@severen.dev>
// SPDX-License-Identifier: CC0-1.0

import { defineConfig } from "astro/config";

export default defineConfig({
  vite: {
    css: {
      transformer: "lightningcss",
    },
    build: {
      cssMinify: "lightningcss",
    },
  },
});
