# Current Partial Trend Analysis

## Main Caution

- The sweep is not yet mathematically complete, but the successful pairs already cover the design very evenly.
- The main remaining caveat is the small set of pending or failed pairs, not a strong coverage distortion.

## Coverage Bias So Far

- Completed pairs analyzed: 2535
- Pair-id range represented in the synced results starts at `p0001` and currently reaches at least `p2556`.
- Failed pairs currently excluded from these trend summaries: 3.
- Coverage is already close to the full design: minimum coverage ratio versus design is 0.964 across the tracked grouping schemes.

## What Does Look Real So Far

- Pure low+low (`ellow-ellow`) pairs are clearly weaker on average than mixed-elevation pairs in the completed subset.
- Far+far is not emerging as a strong pattern; the best-performing completed radius groups are `radmid-radmid`, `radmid-radnear`, and `radnear-radnear`.
- The strongest current pairs are not 'close+far' in a consistent way; they are mostly `radmid-radmid` or `radmid-radnear`.
- Among the current top completed pairs, azimuth separations of `90`, `135`, and `45` degrees appear most often, with exact same-azimuth pairs almost absent from the top 25.
- Certain mid-radius viewpoints, especially `elmid-radmid-az225`, keep recurring as strong partners.

## Top-25 Pattern Snapshot

- Elevation combos in top 25: [('ellow-elmid', 22), ('elhigh-ellow', 3)]
- Radius combos in top 25: [('radmid-radmid', 9), ('radfar-radmid', 9), ('radmid-radnear', 6), ('radfar-radnear', 1)]
- Azimuth separations in top 25: [(135, 10), (90, 7), (180, 4), (45, 3), (0, 1)]

## Interpretation

- The signal is not simply 'always high' and not simply 'always low'.
- The stronger trend is that mixed-elevation pairs outperform low+low pairs, and mid-radius viewpoints appear repeatedly in the strongest duos.
- There is not clear evidence that 'close+far' is the dominant recipe.
- Because design coverage is already very even, these patterns are close to the final picture, with only a small number of unresolved pairs still absent.

## Output Files

- `elevation_combo_trends.csv`
- `radius_combo_trends.csv`
- `azimuth_separation_trends.csv`
- `elevation_combo_coverage_bias.png`
- `radius_combo_coverage_bias.png`
- `elevation_combo_mean_map50_95.png`
- `radius_combo_mean_map50_95.png`
- `azimuth_separation_mean_map50_95.png`
