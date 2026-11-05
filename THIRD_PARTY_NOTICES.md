# Third-party and project asset notices

The project source and SVG assets authored in this project are licensed under
[MIT](LICENSE), copyright (c) 2026 ltczdw. Third-party fonts retain their own
licenses. Installed Python/npm dependencies retain the licenses shipped in
their packages; this notice identifies the font binaries bundled in our source
and frontend distribution.

| Distributed asset | Origin | License and notice |
|---|---|---|
| `frontend/public/fitzen/fonts/InterVariable.woff2` | [Inter upstream, fixed commit](https://github.com/rsms/inter/blob/353b61b9f4430d5f420d56605a6e7993e0941470/docs/font-files/InterVariable.woff2); original bytes unchanged | Copyright (c) 2016 The Inter Project Authors. [SIL OFL 1.1](frontend/public/licenses/Inter-OFL.txt) |
| `frontend/public/fitzen/fonts/MaterialSymbolsOutlined.woff2` | [Google Material Symbols upstream, fixed commit](https://github.com/google/material-design-icons/blob/49d4db35df873165d6bd6ba09b063c7dafbac2f4/variablefont/MaterialSymbolsOutlined%5BFILL%2CGRAD%2Copsz%2Cwght%5D.ttf) | Copyright 2026 Google LLC. All Rights Reserved. [Apache 2.0](frontend/public/licenses/Material-Symbols-APACHE-2.0.txt) |
| `frontend/public/fitzen/logo.svg` and `frontend/public/fitzen/assets/*.svg` | Authored in this project for G010 on 2026-10-14; inspectable SVG source, no external images embedded | Copyright (c) 2026 ltczdw. [MIT](frontend/public/licenses/MIT.txt) |

Material Symbols was converted from its upstream TTF container to WOFF2 using
fontTools 4.66.1 and Brotli 1.2.0. Glyph outlines/order, character mapping,
ligature substitution tables, and variable axes are unchanged. The conversion
is a project modification; the font remains licensed under Apache 2.0.

The [asset provenance manifest](frontend/public/licenses/asset-provenance.json)
records every distributed image/font, its SHA-256, the upstream commit and
download URL, the original upstream SHA-256, and the applicable license.
[NOTICE.txt](frontend/public/licenses/NOTICE.txt) and the full license texts
are also copied into the built frontend at `/licenses/`, so these notices
accompany standalone frontend distributions.

The runtime/generated-style notices for [Vue](frontend/public/licenses/Vue-MIT.txt),
[Pinia](frontend/public/licenses/Pinia-MIT.txt),
[Vue Devtools API](frontend/public/licenses/Vue-Devtools-API-MIT.txt), and
[Tailwind CSS preflight](frontend/public/licenses/Tailwind-CSS-MIT.txt) are
included at `/licenses/` as well. Their installed package versions and license
hashes are recorded in the provenance manifest; `frontend/package-lock.json`
pins the registry package integrity. These packages retain their own copyright
notices.

The SVG training and meal images are decorative illustrations, not exercise
instructions, meal photographs, or records of a real user's activity. Default
avatars do not depict a real person.

Inherited raster images without verifiable provenance, unused font binaries,
and superseded asset files were archived locally for recovery. They are absent
from the public assets directory, frontend build, and release manifest. This
notice does not assign a license to those excluded files or imply endorsement
by the font authors.


## Official public health text snapshots (2026-10-20)

`backend/seed/authority_coverage_snapshot.json` contains text-only article excerpts from the NHS website and US government health agencies (NIAMS, NHLBI, NIDDK, NCCIH and CDC). Every article retains its source/final URL, publisher, retrieval date, text SHA-256 and license metadata. Failed/blocked URLs remain listed as failures and are not represented as imported evidence. No photos, logos, videos or third-party figures are copied.

Information from the NHS website is licensed under the Open Government Licence v3.0.
See the [NHS terms](https://www.nhs.uk/our-policies/terms-and-conditions/) and [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). HTML navigation is removed and headings retained; the clinical article body remains English source text. Chinese discovery titles and clearly labelled excerpt translations are project editorial adaptations, not NHS-approved translations. Derived Chinese documents retain the original article SHA-256 and attribution; English source bodies are preserved. The NHS does not endorse this project.

US government materials retain agency/source attribution; third-party content is excluded. Refer to the [NIH disclaimer](https://www.nih.gov/about-nih/what-we-do/nih-almanac/disclaimer) and [CDC reuse guidance](https://www.cdc.gov/other/agencymaterials.html). Jurisdiction is recorded (UK/US); regional services and emergency numbers must not be presented as universally applicable. These source texts retain their original terms rather than inheriting this project's MIT license.


## DeepSeek V4.1 offline token estimation

The pinned tokenizer and Python prompt-encoding reference in
`backend/app/vendor/deepseek_v41/` are from the official
[DeepSeek-V4.1-Flash repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(revision `314113f8a7dadb388215396ba5fbee2c610a04d6`). They are distributed under
the MIT License, Copyright (c) 2023 DeepSeek. The complete notice is retained
in that directory's `LICENSE`; `manifest.json` records source URLs and SHA-256.
Only offline text/token encoding is used; no model weights or remote code loading.
