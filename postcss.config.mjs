/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <sev@severen.dev>
 * SPDX-License-Identifier: CC0-1.0
 */

import tailwindcss from "tailwindcss";
import autoprefixer from "autoprefixer";

/** @type {import("postcss-load-config").Config} */
export default {
  plugins: [tailwindcss(), autoprefixer],
};
