/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <severen@shrike.me>
 * SPDX-License-Identifier: CC0-1.0
 */

import preprocess from "svelte-preprocess";
import adapter from "@sveltejs/adapter-netlify";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: [preprocess({ postcss: true })],

  kit: {
    adapter: adapter(),
  },
};

export default config;
