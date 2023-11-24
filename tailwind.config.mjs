/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <sev@severen.dev>
 * SPDX-License-Identifier: CC0-1.0
 */

import typography from "@tailwindcss/typography";

/** @type {import("tailwindcss").Config} */
export default {
  content: ["./src/**/*.{html,js,svelte,ts}"],

  theme: {
    extend: {},
  },

  plugins: [typography],
};
