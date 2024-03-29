/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <sev@severen.dev>
 * SPDX-License-Identifier: CC0-1.0
 */

import { sveltekit } from "@sveltejs/kit/vite";
import type { UserConfig } from "vite";

const config: UserConfig = {
  plugins: [sveltekit()],
};

export default config;
