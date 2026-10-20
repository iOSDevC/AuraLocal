---
layout: docs
title: License
nav_order: 4
description: "Tonamox is dual-licensed: AGPL-3.0-or-later, or a commercial license from iOSDevC. Earlier commits stay under MIT; dependencies and models keep their own licenses."
---

# License
{: .no_toc }

Tonamox is dual-licensed: under the GNU Affero General Public License, or under a commercial license from the
copyright holder. This page follows the [License](https://github.com/iOSDevC/AuraLocal#license) section of the
README; the AGPL text is in [LICENSE](https://github.com/iOSDevC/AuraLocal/blob/main/LICENSE).

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Two options

**AGPL-3.0-or-later.** Tonamox is free software: you can redistribute it and/or modify it under the terms of the
[GNU Affero General Public License](https://github.com/iOSDevC/AuraLocal/blob/main/LICENSE) as published by the
Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Copyright (C) 2026 iOSDevC

Tonamox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

**Commercial license.** To use Tonamox where the AGPL's terms don't fit, such as a closed-source app, an App
Store release, or a network service whose source you don't publish, get a commercial license from the copyright
holder. See [Commercial license](#commercial-license).

## What the AGPL means for your app

- **Linking and distributing.** An app or package that links Tonamox and is distributed must be released under
  AGPL-3.0-compatible terms, with its source available.
- **Network use.** If you modify Tonamox and let people use it over a network, you must offer them the source of
  your version too.
- **App Store.** The AGPL's conditions conflict with the App Store's terms. An App Store release needs the
  commercial license.

## Commercial license

Contact [iOSDevC](https://github.com/iOSDevC), the copyright holder, on GitHub.

## Earlier MIT versions

Commits published before the license change (up to `1647874`, including the
[`v0.1.0`](https://github.com/iOSDevC/AuraLocal/releases/tag/v0.1.0) tag) remain available under the MIT License
they shipped with.

## Contributions

Contributions are accepted under the terms in
[CONTRIBUTING.md](https://github.com/iOSDevC/AuraLocal/blob/main/CONTRIBUTING.md), which keep the commercial
license possible. By submitting a contribution (a pull request, a patch, or any code, documentation or asset), you
confirm that:

1. You wrote it, or otherwise have the right to submit it under these terms.
2. You license it to everyone under AGPL-3.0-or-later.
3. You grant iOSDevC, the copyright holder of Tonamox, a perpetual, worldwide, non-exclusive, royalty-free,
   irrevocable license to use, reproduce, modify, sublicense and distribute your contribution under any terms,
   including commercial and proprietary licenses. This grant includes a license to any patent claims you can
   license that your contribution necessarily infringes.
4. You keep the copyright in your contribution.

Sign off each commit (`git commit -s`) to confirm these terms. A pull request with unsigned commits is not merged.
Code under a license that can't be relicensed this way, such as GPL code written by someone else, can't be
contributed. A dependency on such code needs discussion in an issue first.

## Dependencies

Dependencies keep their own licenses (MIT, Apache-2.0, BSD); all are compatible with AGPL-3.0 and with the
commercial license. [THIRD_PARTY_NOTICES.md](https://github.com/iOSDevC/AuraLocal/blob/main/THIRD_PARTY_NOTICES.md)
lists each Swift package resolved from `Package.resolved`, with its pinned version and license. Binaries built from
Tonamox (the `tonamox` CLI, the example apps) include code from these packages, so their license texts and NOTICE
files must ship with those binaries.

## Models

Models downloaded at run time are not part of Tonamox and carry their own licenses; check each model card before
you use or redistribute one.

For language and vision models, the Hugging Face compatibility check reads the license from the model card and
never blocks on it: a gated repo, a missing license, a non-commercial or custom (`other`) license, or one with use
restrictions (such as OpenRAIL, Llama or Gemma terms) is a caveat. See
[What it checks]({{ '/guide/models' | relative_url }}#what-it-checks). For Core ML Stable Diffusion repos, a missing
or `other` license, or one marked non-commercial or `community`, is a caveat; see
[Other models from Hugging Face]({{ '/guide/imagegen' | relative_url }}#other-models-from-hugging-face).
