/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <sev@severen.dev>
 * SPDX-License-Identifier: CC0-1.0
 */

import netlifyAdapter from "@sveltejs/adapter-netlify";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import("@sveltejs/kit").Config} */
export default {
  preprocess: [vitePreprocess()],
  kit: {
    adapter: netlifyAdapter({ edge: true }),
  },
};
