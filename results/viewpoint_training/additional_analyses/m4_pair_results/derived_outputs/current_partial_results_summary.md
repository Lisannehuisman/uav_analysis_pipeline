# Current Partial M4 Pair-Sweep Summary

- Generated at: 2026-04-07T10:03:38
- Source snapshot: `results/viewpoint_training/additional_analyses/m4_pair_results/snapshot`

## Sweep Status

- Total defined pairs: 2556
- Completed Option A evaluations: 2535
- Failed pairs so far: 3
- Remaining pending / incomplete pairs: 18
- Completion rate: 99.2%
- Sweep status: still in progress.

## Best Current Duo Viewpoints

- Best by mAP50-95: `p0569` = `ellow-radmid-az000` + `elmid-radmid-az225`
- Scores: mAP50-95 `0.4958`, mAP50 `0.7252`, F1 `0.7478`, precision `0.7252`, recall `0.7719`
- Delta vs full M4 baseline (mAP50-95): `nan`
- Best by mAP50: `p0864` (0.7274)
- Best by F1: `p1325` (0.7538)

## Top 10 Completed Pairs So Far

1. `p0569`: `ellow-radmid-az000` + `elmid-radmid-az225` | mAP50-95 `0.4958` | mAP50 `0.7252` | F1 `0.7478`
2. `p0692`: `ellow-radmid-az090` + `elmid-radmid-az225` | mAP50-95 `0.4951` | mAP50 `0.7219` | F1 `0.7443`
3. `p1196`: `ellow-radfar-az135` + `elmid-radmid-az225` | mAP50-95 `0.4948` | mAP50 `0.7258` | F1 `0.7481`
4. `p1092`: `ellow-radfar-az045` + `elmid-radmid-az270` | mAP50-95 `0.4941` | mAP50 `0.7235` | F1 `0.7462`
5. `p0107`: `ellow-radnear-az045` + `elmid-radmid-az225` | mAP50-95 `0.4935` | mAP50 `0.7200` | F1 `0.7442`
6. `p0752`: `ellow-radmid-az135` + `elmid-radmid-az225` | mAP50-95 `0.4935` | mAP50 `0.7208` | F1 `0.7419`
7. `p0502`: `ellow-radnear-az315` + `elmid-radmid-az045` | mAP50-95 `0.4934` | mAP50 `0.7193` | F1 `0.7416`
8. `p1145`: `ellow-radfar-az090` + `elmid-radmid-az270` | mAP50-95 `0.4924` | mAP50 `0.7199` | F1 `0.7419`
9. `p0839`: `ellow-radmid-az180` + `elhigh-radfar-az045` | mAP50-95 `0.4921` | mAP50 `0.7219` | F1 `0.7448`
10. `p1091`: `ellow-radfar-az045` + `elmid-radmid-az225` | mAP50-95 `0.4917` | mAP50 `0.7245` | F1 `0.7483`

## Current Strongest Individual Viewpoints

1. `elmid-radmid-az225` | avg mAP50-95 across completed pairs `0.4432` | completed pair count `70` | best pair `p0569`
2. `elmid-radmid-az000` | avg mAP50-95 across completed pairs `0.4398` | completed pair count `71` | best pair `p0864`
3. `elmid-radfar-az180` | avg mAP50-95 across completed pairs `0.4397` | completed pair count `71` | best pair `p0876`
4. `elmid-radfar-az090` | avg mAP50-95 across completed pairs `0.4392` | completed pair count `71` | best pair `p0042`
5. `elmid-radmid-az045` | avg mAP50-95 across completed pairs `0.4388` | completed pair count `71` | best pair `p0502`
6. `elmid-radfar-az225` | avg mAP50-95 across completed pairs `0.4385` | completed pair count `71` | best pair `p0819`
7. `elmid-radfar-az045` | avg mAP50-95 across completed pairs `0.4382` | completed pair count `71` | best pair `p0381`
8. `elmid-radmid-az180` | avg mAP50-95 across completed pairs `0.4367` | completed pair count `71` | best pair `p0810`
9. `elmid-radmid-az270` | avg mAP50-95 across completed pairs `0.4364` | completed pair count `70` | best pair `p1092`
10. `elmid-radfar-az000` | avg mAP50-95 across completed pairs `0.4348` | completed pair count `71` | best pair `p0814`

## What The Current Duo Viewpoints Seem To Suggest

- Based on the current top 25 completed pairs.
- Most common elevation labels in those top pairs: [('ellow', 25), ('elmid', 22), ('elhigh', 3)]
- Most common radius labels in those top pairs: [('radmid', 33), ('radfar', 10), ('radnear', 7)]
- Most frequently recurring viewpoints in those top pairs: [('elmid-radmid-az225', 10), ('elmid-radmid-az045', 5), ('ellow-radmid-az180', 5), ('ellow-radnear-az000', 4), ('ellow-radmid-az090', 3), ('ellow-radfar-az045', 3), ('elmid-radmid-az270', 3), ('ellow-radfar-az090', 2), ('elmid-radmid-az090', 2), ('elmid-radmid-az180', 2)]
- These patterns are still provisional because the full 2556-pair sweep is not finished yet.

