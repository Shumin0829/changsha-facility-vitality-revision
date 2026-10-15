# Public Code Release 2026-10-16

## Source And Scope

This release derives from the audited `park-sports-vitality-code_1009.zip` package.
The frozen source archive SHA256 is:

```text
eccb636562bd562c3cab02d4645e053d922a7dff76d5a3bba8e533459ed4103b
```

It contains the revision's six-model reanalysis, nested random/spatial validation,
SHAP attribution, validation helpers, and Figures 5-11 renderer. Real inputs are
not distributed. The 400-row examples are synthetic and cannot reproduce the
paper's numerical findings.

It is not the untouched historical exploratory notebook, the upstream GIS and
heat-map extraction pipeline, or the later GWR/MGWR and alternative benchmark
experiments. It must not be cited as covering those additional analyses.

## Publication Changes

- English/Chinese README status and release scope were updated.
- SciPy 1.14.1, already recorded in the historical rendering environment, was
  added to the plotting requirements because the renderer reads its package
  version when writing QA metadata.
- This notice, licensing notice, packaging metadata, and checksums were updated.
- The archive builder's private-path guard now distinguishes ordinary HTTPS URLs
  from Windows drive paths; it still rejects personal-path and credential patterns.
- The checksum file is written with fixed UTF-8/LF bytes so its working-copy and
  archive checksums agree on Windows as well as other platforms.
- The analysis, schema, result-verification, synthetic-generator, smoke-test, and
  figure-rendering Python source files remain byte-identical to the frozen ZIP.

No real-data model fitting or new scientific validation is implied by publication.
Historical test JSON files retain their original scope. A local archive build or
synthetic smoke test is not proof of a new real-data analysis or a repository DOI.
