# Release checklist — overnight supplements v2

Verified at 2026-09-19T21:31:17.225136+00:00. Final scientific release: **PASS**. Git commit/push identity is reported separately after this immutable release is committed.

training_invocations = 0
inference_invocations = 0

- [x] Source paths, sizes, SHA-256 and source schema: `freeze/source_manifest.json`, SHA-256 `24d674f895dfc8bf23e95dbc3c07427ae4e541862e612918afa13b9ae8dbb27d`; 5179 local consumed historical files rehashed, no changes.
- [x] Code and configuration versions: `freeze/code_manifest.json`, SHA-256 `8aaf2c1857b08ffab5a84bbe643d3ed2fdbf75a7877083160575fe150503621a`; frozen baseline `79c9c3dc838ecaed7fe247f2f1f61c08cbab519c`.
- [x] P0 sole primary scope: Gaussian, heldout combinations,15/5dB,ResNet/TCN,five seeds. No train-combination calculation or selection by result.
- [x] P0 four-arm common identities; same-test-structure input equality; within-training-strategy checkpoint equality; separate clean/noisy/checkpoint/prediction hashes. E inputs never required to equal I inputs.
- [x] P0 complete4000noisy probability sources,4020point checks including20clean references;431600unique case-record input identities;1000paired case rows;20seed values;4summary rows;48000draw audit rows.
- [x] Lowest-case pairing before equal condition/noise aggregation; within-seed DiD before five-seed mean;2000original multiplicity draws;NaN preservation and ddof1seed SD separate from conditional patient CI.
- [x] P0 independent raw-probability/weight recomputation: QA units A/B, point and draws0/1999, all4groups/all20seeds; 399518 comparisons; max error `1.7542391150815462e-15` versus1e-12. Remaining1998draws audited for consistency, not claimed independently recomputed from raw probabilities.
- [x] P1 GEO-DIRECT: all126final input hashes,271908record rows; old545974q rows and271908input diagnostics crosschecked. P symmetry/idempotence gates passed; residual defined from finalfloat32 minus cleanfloat32 in float64.
- [x] P1 independent five fixedIDs×126cases=630SVD geometry checks; nine boundary fixtures; noepsilon/clipping. Real geometry anomalies0; overlay anomalies0.
- [x] P1 overlay:756noisy×2158=1631448exact record joins;762source metrics including6cleanreferences; independent42noisy+6clean metric checks; E/Iweights not replicated. Five-class positive/negative exposure distributions retained.
- [x] Publication: 249 displayed numeric cells/counts checked against accepted CSVs; 26 local links;3visually inspectedPNGfigures with9hashedPDF/PNG/SVGfiles. Full-resolution text/legends/axes/captions checked. Ground-truth labels not described as prediction true-positives.
- [x] Initial54overlay rejections retained in qa/attempts with root-cause/correction record; repairedone-to-manyworkerprovenance, then fullproductionandindependentQApassed. No historical scientific artifact correction was needed.
- [x] Caption and completed-status-note corrections changed no accepted numeric artifacts;10artifact identities retained in qa/presentation_corrections.json.
- [x] Lossless alignment packaging: 143811089→64249453 bytes; 1631448 rows×22 columns exactly equal, including schema/nulls/order. Actual production writer rerun; independent P1 QA repeated on final bytes. No sample or column omission.
- [x] P1 independent maximum errors: absolute projected SNR `4.263256414560601e-14` dB (absolute tolerance1e-10); noise q `5.6066262743570405e-15` (absolute tolerance1e-12); largest of four energy absolute errors `3.865352482534945e-12` (each comparison passed relative tolerance1e-12, not an absolute-energy1e-12 gate).
- [x] P3 DEFERRED: publiccommit/license/topology/path/deviceinventoryonly; zero modelimports/forward/backward/epoch timing.
- [x] Unexecuted by design: training,inference,checkpoint/thresholdselection,pvalues/newtestingfamilies,binningstandardization,reweightedAUROC,propensity,equalexposurecomparisons,mediation,thirdarchitectureexecution.
- [x] No raw waveforms,input caches,original probability arrays or checkpoints copied into the release namespace. Isolated packagevendor excluded from Git. Audited scalar/linkParquet files may be committed.
- [x] Producer status files are timestamped production snapshots; final independent acceptance is the separate qa evidence and this checklist, not retroactively inferred from a producer's pending-QA label.

## Exact evidence paths

- `qa/did_recompute.json` and `did/independent_recompute.json`
- `qa/snrp_recompute.json` and `snrp/qa.json`
- `qa/source_protection.json`, `qa/visual_acceptance.json`, `qa/publication_verification.json`
- `reports/overnight_supplement_final_report.md`, `reports/manifest.json`
