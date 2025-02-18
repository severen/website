// SPDX-FileCopyrightText: 2025 Severen Redwood <sev@severen.dev>
// SPDX-License-Identifier: CC0-1.0

// @ts-check

/** @type {import("prettier").Config} */
export default {
  bracketSameLine: true,

  plugins: ["prettier-plugin-astro"],
  overrides: [
    {
      files: "*.astro",
      options: { parser: "astro" },
    },
  ],
};
