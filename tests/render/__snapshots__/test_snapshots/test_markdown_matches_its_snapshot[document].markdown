# Unemployment model

## Data

The series are quarterly.

Table T3.1. Series summary \<all>

| Series | Mean (pp) | Share | Observations | Last |
| :--- | ---: | ---: | ---: | :--- |
| Unemployment \| total | 1,234.57 | 12.50% | 1,500 | 2024-03-31 |
| GDP growth & more | -0.25 | – | 96 | – |

Means over the whole sample.

![Figure F3.1. Unemployment rate](F3.1.svg)

## Diagnostics

1 of 3 checks pass; the p-value is 0.04 \<strong>.

Table T8.1. Diagnostic checks

| Check | Observed | Rule | Threshold | Unit | Result | Reason |
| :--- | ---: | :--- | ---: | :--- | :--- | :--- |
| adf | 0.20 | \< | 0.05 |  | **Overridden** | Short sample; see section 4. |
| r2 | 0.91 | >= | 0.50 | share | Pass |  |
| dw | 0.40 | > | 1.50 |  | **Fail** |  |

Source: FRED; run run-7, data 0123456789ab, 2026-11-25.

### Detail

Nothing else.
