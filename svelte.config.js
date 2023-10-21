/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <me@severen.dev>
 * SPDX-License-Identifier: CC0-1.0
 */

import netlifyAdapter from "@sveltejs/adapter-netlify";
import { vitePreprocess } from "@sveltejs/kit/vite";

/** @type {import("@sveltejs/kit").Config} */
export default {
  preprocess: [vitePreprocess()],
  kit: {
    adapter: netlifyAdapter(),
  },
};
