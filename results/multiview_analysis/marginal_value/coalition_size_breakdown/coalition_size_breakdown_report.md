# Exact Added-Drone Gain Analysis

This analysis keeps the same coalition value as the fusion-based ring Shapley analysis:

- Per scene: `Noisy-OR(confidences in coalition) x best IoU in coalition`.
- Per ring: average that coalition value over all scenes that contain any observation in the ring.

The difference is the output:

- Shapley asks: `which viewpoint adds the most value on average across all coalition contexts?`
- This file asks: `how much extra value does the 2nd, 3rd, 4th, ... drone add?`

## Overall pattern across the 9 controlled rings

| Drones in coalition | Mean of ring mean values | Mean of ring best values | Mean extra value of kth drone | Mean best-case extra value of kth drone |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0000 | 0.0000 | n/a | n/a |
| 1 | 0.1783 | 0.2289 | 0.1783 | 0.2289 |
| 2 | 0.3329 | 0.4096 | 0.1546 | 0.1807 |
| 3 | 0.4668 | 0.5508 | 0.1339 | 0.1412 |
| 4 | 0.5826 | 0.6629 | 0.1158 | 0.1121 |
| 5 | 0.6829 | 0.7518 | 0.1002 | 0.0888 |
| 6 | 0.7696 | 0.8194 | 0.0868 | 0.0677 |
| 7 | 0.8448 | 0.8706 | 0.0752 | 0.0511 |
| 8 | 0.9099 | 0.9099 | 0.0651 | 0.0393 |

## Focus ring: elmid-radnear

This is the strongest ring by full 8-drone coalition value: `0.9434`.

| Drones | Mean coalition value | Best coalition value | Average extra value of this drone | Best-case extra value of this drone | Best coalition members |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.0000 | 0.0000 | n/a | n/a | (empty) |
| 1 | 0.1966 | 0.2629 | 0.1966 | 0.2629 | az135 |
| 2 | 0.3646 | 0.4789 | 0.1679 | 0.2160 | az135 az270 |
| 3 | 0.5071 | 0.6166 | 0.1425 | 0.1376 | az135 az180 az270 |
| 4 | 0.6274 | 0.7309 | 0.1203 | 0.1144 | az090 az135 az180 az270 |
| 5 | 0.7285 | 0.8277 | 0.1011 | 0.0967 | az090 az135 az180 az270 az315 |
| 6 | 0.8131 | 0.8777 | 0.0846 | 0.0500 | az045 az090 az135 az180 az270 az315 |
| 7 | 0.8840 | 0.9171 | 0.0708 | 0.0394 | az045 az090 az135 az180 az225 az270 az315 |
| 8 | 0.9434 | 0.9434 | 0.0595 | 0.0264 | az000 az045 az090 az135 az180 az225 az270 az315 |

## How to read this for the thesis question

- In `elmid-radnear`, the **2nd drone** adds `0.1679` on average.
- The **3rd drone** adds `0.1425` on average.
- The **4th drone** adds `0.1203` on average.
- After that, gains continue to be positive but keep shrinking, which is the diminishing-returns story the swarm-size question needs.

## Methodological takeaway

- Use this coalition-size breakdown to answer `how many drones are worth adding?`.
- Use Shapley on top of that to answer `which viewpoints are the best teammates within a fixed swarm size setting?`.