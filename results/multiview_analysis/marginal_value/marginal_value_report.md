# Marginal Viewpoint Value Report

This report reframes the current M4 swarm outputs around a thesis question:

> how much new information do extra UAV views still add, and when do they mostly become redundant?

## 1. Diminishing Returns

- `1 -> 2` views: target AP50-95 changes by `0.0761` and target strict quality changes by `0.0357`.
- `2 -> 3` views: target AP50-95 changes by `0.0210` and target strict quality changes by `0.0101`.
- `2` views already capture `78.4%` of the total `1 -> 3` AP50-95 gain.
- `2` views already capture `78.0%` of the total `1 -> 3` target strict quality gain.

Gain targets:
- `3` view(s) are enough to capture `90%` of the available `AP50-95` gain (achieved `100.0%`).
- `3` view(s) are enough to capture `90%` of the available `Target Strict Quality` gain (achieved `100.0%`).
- `3` view(s) are enough to capture `95%` of the available `AP50-95` gain (achieved `100.0%`).
- `3` view(s) are enough to capture `95%` of the available `Target Strict Quality` gain (achieved `100.0%`).

## 2. Strongest Single Views

- `ellow-radnear-az225`: strict quality `0.9423`, AP50-95 `0.8585`, scenes `36`.
- `elmid-radnear-az315`: strict quality `0.9383`, AP50-95 `0.9613`, scenes `33`.
- `elmid-radnear-az000`: strict quality `0.9311`, AP50-95 `0.9263`, scenes `21`.
- `elmid-radnear-az045`: strict quality `0.9301`, AP50-95 `0.9156`, scenes `32`.
- `elhigh-radnear-az000`: strict quality `0.9291`, AP50-95 `0.9378`, scenes `34`.
- `ellow-radmid-az315`: strict quality `0.9265`, AP50-95 `0.8816`, scenes `33`.
- `ellow-radnear-az135`: strict quality `0.9251`, AP50-95 `0.8765`, scenes `32`.
- `elmid-radnear-az135`: strict quality `0.9234`, AP50-95 `0.9076`, scenes `43`.
- `elmid-radnear-az270`: strict quality `0.9142`, AP50-95 `0.9137`, scenes `43`.
- `ellow-radnear-az090`: strict quality `0.9094`, AP50-95 `0.8485`, scenes `32`.
- `elmid-radfar-az000`: strict quality `0.9086`, AP50-95 `0.8897`, scenes `27`.
- `elhigh-radfar-az315`: strict quality `0.9065`, AP50-95 `0.9139`, scenes `30`.
- `elmid-radmid-az270`: strict quality `0.9054`, AP50-95 `0.9074`, scenes `40`.
- `elmid-radmid-az090`: strict quality `0.9044`, AP50-95 `0.8960`, scenes `29`.
- `elhigh-radnear-az225`: strict quality `0.9027`, AP50-95 `0.9315`, scenes `36`.

## 3. Most Complementary Pairs

Only pairs with at least `8` matched scenes are used in this headline section.

Pair complementarity is defined here as:

`E[max(view_i, view_j)] - max(E[view_i], E[view_j])`

evaluated on the matched scene subset where both viewpoints are available.

- `ellow-radfar-az000 + ellow-radmid-az225`: complementarity gain `0.1209`, pair strict quality `0.8860`, matched scenes `8`.
- `elhigh-radmid-az090 + ellow-radmid-az000`: complementarity gain `0.0709`, pair strict quality `0.8233`, matched scenes `8`.
- `elhigh-radfar-az000 + ellow-radfar-az135`: complementarity gain `0.0705`, pair strict quality `0.8815`, matched scenes `8`.
- `elhigh-radfar-az090 + ellow-radnear-az000`: complementarity gain `0.0683`, pair strict quality `0.9211`, matched scenes `8`.
- `ellow-radnear-az000 + elmid-radfar-az090`: complementarity gain `0.0677`, pair strict quality `0.8537`, matched scenes `9`.
- `ellow-radmid-az225 + elmid-radfar-az135`: complementarity gain `0.0661`, pair strict quality `0.8563`, matched scenes `8`.
- `elhigh-radfar-az045 + ellow-radmid-az045`: complementarity gain `0.0590`, pair strict quality `0.9044`, matched scenes `8`.
- `ellow-radmid-az045 + ellow-radnear-az000`: complementarity gain `0.0564`, pair strict quality `0.8907`, matched scenes `10`.
- `ellow-radmid-az045 + ellow-radnear-az180`: complementarity gain `0.0528`, pair strict quality `0.9007`, matched scenes `8`.
- `ellow-radmid-az045 + elmid-radfar-az135`: complementarity gain `0.0466`, pair strict quality `0.8738`, matched scenes `10`.
- `ellow-radnear-az180 + elmid-radmid-az315`: complementarity gain `0.0434`, pair strict quality `0.9334`, matched scenes `8`.
- `ellow-radnear-az000 + elmid-radnear-az090`: complementarity gain `0.0433`, pair strict quality `0.9284`, matched scenes `8`.
- `elhigh-radnear-az135 + ellow-radnear-az270`: complementarity gain `0.0432`, pair strict quality `0.9457`, matched scenes `8`.
- `ellow-radfar-az000 + ellow-radmid-az045`: complementarity gain `0.0415`, pair strict quality `0.8763`, matched scenes `9`.
- `elhigh-radfar-az225 + ellow-radfar-az000`: complementarity gain `0.0380`, pair strict quality `0.9356`, matched scenes `8`.

## 4. Triples With Useful Third Views

Only triples with at least `3` matched scenes are used in this headline section.
The maximum exact-triple overlap in the current dataset is `6` scenes.

Third-view gain is defined here as:

`E[max(view_i, view_j, view_k)] - max(E[max(view_i, view_j)], E[max(view_i, view_k)], E[max(view_j, view_k)])`

again on the matched scene subset where all three viewpoints are available.

- `elhigh-radfar-az090 + ellow-radmid-az225 + elmid-radfar-az135`: third-view strict-quality gain `0.0198`, best pair `elhigh-radfar-az090 + elmid-radfar-az135`, matched scenes `3`.
- `elhigh-radfar-az090 + ellow-radmid-az045 + elmid-radfar-az135`: third-view strict-quality gain `0.0156`, best pair `elhigh-radfar-az090 + elmid-radfar-az135`, matched scenes `4`.
- `ellow-radmid-az045 + elmid-radmid-az135 + elmid-radnear-az090`: third-view strict-quality gain `0.0096`, best pair `elmid-radmid-az135 + elmid-radnear-az090`, matched scenes `3`.
- `elhigh-radmid-az090 + ellow-radfar-az135 + elmid-radnear-az225`: third-view strict-quality gain `0.0094`, best pair `elhigh-radmid-az090 + ellow-radfar-az135`, matched scenes `3`.
- `ellow-radfar-az000 + ellow-radmid-az045 + elmid-radnear-az090`: third-view strict-quality gain `0.0094`, best pair `ellow-radfar-az000 + ellow-radmid-az045`, matched scenes `4`.
- `ellow-radfar-az135 + ellow-radnear-az000 + ellow-radnear-az315`: third-view strict-quality gain `0.0092`, best pair `ellow-radfar-az135 + ellow-radnear-az315`, matched scenes `3`.
- `elhigh-radfar-az090 + ellow-radfar-az000 + ellow-radmid-az225`: third-view strict-quality gain `0.0087`, best pair `elhigh-radfar-az090 + ellow-radfar-az000`, matched scenes `3`.
- `elhigh-radnear-az180 + ellow-radnear-az090 + ellow-radnear-az135`: third-view strict-quality gain `0.0071`, best pair `elhigh-radnear-az180 + ellow-radnear-az090`, matched scenes `3`.
- `ellow-radnear-az270 + elmid-radfar-az225 + elmid-radmid-az090`: third-view strict-quality gain `0.0068`, best pair `elmid-radfar-az225 + elmid-radmid-az090`, matched scenes `3`.
- `elhigh-radfar-az000 + ellow-radmid-az045 + ellow-radnear-az315`: third-view strict-quality gain `0.0066`, best pair `elhigh-radfar-az000 + ellow-radmid-az045`, matched scenes `3`.
- `elhigh-radmid-az135 + elmid-radmid-az225 + elmid-radnear-az225`: third-view strict-quality gain `0.0065`, best pair `elhigh-radmid-az135 + elmid-radnear-az225`, matched scenes `3`.
- `ellow-radmid-az045 + elmid-radfar-az135 + elmid-radnear-az090`: third-view strict-quality gain `0.0064`, best pair `elmid-radfar-az135 + elmid-radnear-az090`, matched scenes `4`.
- `elhigh-radfar-az000 + elhigh-radmid-az225 + ellow-radnear-az045`: third-view strict-quality gain `0.0063`, best pair `elhigh-radfar-az000 + ellow-radnear-az045`, matched scenes `3`.
- `elhigh-radmid-az000 + elmid-radmid-az180 + elmid-radnear-az270`: third-view strict-quality gain `0.0062`, best pair `elhigh-radmid-az000 + elmid-radmid-az180`, matched scenes `4`.
- `ellow-radmid-az045 + ellow-radmid-az225 + elmid-radfar-az135`: third-view strict-quality gain `0.0061`, best pair `ellow-radmid-az045 + elmid-radfar-az135`, matched scenes `5`.

## 5. Observed Marginal View Scores

The score below averages a viewpoint's marginal contribution when added to:

- one existing view
- an existing pair

This is an observed-coalition marginal summary, not a Shapley analysis.

- `elmid-radnear-az090`: combined marginal strict quality `0.0491`, pair-stage marginal `0.0733`, third-stage marginal `0.0249`.
- `ellow-radnear-az090`: combined marginal strict quality `0.0445`, pair-stage marginal `0.0588`, third-stage marginal `0.0302`.
- `elmid-radnear-az045`: combined marginal strict quality `0.0401`, pair-stage marginal `0.0593`, third-stage marginal `0.0209`.
- `elmid-radnear-az180`: combined marginal strict quality `0.0394`, pair-stage marginal `0.0607`, third-stage marginal `0.0181`.
- `ellow-radnear-az225`: combined marginal strict quality `0.0363`, pair-stage marginal `0.0483`, third-stage marginal `0.0243`.
- `ellow-radnear-az135`: combined marginal strict quality `0.0342`, pair-stage marginal `0.0471`, third-stage marginal `0.0213`.
- `ellow-radmid-az045`: combined marginal strict quality `0.0340`, pair-stage marginal `0.0485`, third-stage marginal `0.0196`.
- `ellow-radnear-az270`: combined marginal strict quality `0.0335`, pair-stage marginal `0.0439`, third-stage marginal `0.0231`.
- `ellow-radnear-az000`: combined marginal strict quality `0.0332`, pair-stage marginal `0.0454`, third-stage marginal `0.0209`.
- `elmid-radmid-az045`: combined marginal strict quality `0.0331`, pair-stage marginal `0.0518`, third-stage marginal `0.0144`.
- `ellow-radmid-az270`: combined marginal strict quality `0.0325`, pair-stage marginal `0.0527`, third-stage marginal `0.0122`.
- `elmid-radmid-az000`: combined marginal strict quality `0.0324`, pair-stage marginal `0.0534`, third-stage marginal `0.0114`.
- `elmid-radnear-az225`: combined marginal strict quality `0.0323`, pair-stage marginal `0.0491`, third-stage marginal `0.0155`.
- `ellow-radnear-az045`: combined marginal strict quality `0.0318`, pair-stage marginal `0.0448`, third-stage marginal `0.0188`.
- `elmid-radmid-az225`: combined marginal strict quality `0.0292`, pair-stage marginal `0.0445`, third-stage marginal `0.0139`.

## 6. Object Classes With The Largest Multi-View Gain

- `barrel`: strict-quality gain `1 -> 2` = `0.0806`, `1 -> 3` = `0.1080`.
- `male`: strict-quality gain `1 -> 2` = `0.0670`, `1 -> 3` = `0.0899`.
- `suv`: strict-quality gain `1 -> 2` = `0.0528`, `1 -> 3` = `0.0640`.
- `tank`: strict-quality gain `1 -> 2` = `0.0521`, `1 -> 3` = `0.0667`.
- `whitevan`: strict-quality gain `1 -> 2` = `0.0363`, `1 -> 3` = `0.0428`.
- `rock`: strict-quality gain `1 -> 2` = `0.0180`, `1 -> 3` = `0.0217`.
- `tree`: strict-quality gain `1 -> 2` = `0.0150`, `1 -> 3` = `0.0196`.
- `tower`: strict-quality gain `1 -> 2` = `0.0139`, `1 -> 3` = `0.0169`.
- `container`: strict-quality gain `1 -> 2` = `0.0135`, `1 -> 3` = `0.0173`.
- `tent`: strict-quality gain `1 -> 2` = `0.0090`, `1 -> 3` = `0.0117`.

## 7. Thesis Interpretation

These outputs support a thesis narrative in which:

- single-view performance maps the baseline information landscape;
- `k-view` curves quantify diminishing returns instead of only asking whether multiview beats single-view;
- complementarity is about marginal added coverage, not just individually strong viewpoints;
- the best swarm size is therefore the smallest `k` that captures most of the available gain.

