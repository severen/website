/** @license
 * SPDX-FileCopyrightText: 2022 Severen Redwood <severen@shrike.me>
 * SPDX-License-Identifier: CC0-1.0
 */

const tailwindcss = require("tailwindcss");
const autoprefixer = require("autoprefixer");

module.exports = {
  plugins: [
    tailwindcss(),
    autoprefixer,
  ],
};
