# Part A summary

Annotators: André. Inventory source: vlm.
Rows (annotator x item): 20; step 1 done 20, step 2 20, step 3 20.
Majority baseline: 0.30 (always 'chart').

## Type accuracy (pooled over annotators; Wilson 95% intervals)

| gold type | step 1 | step 2 |
|---|---|---|
| chart | 0.67 [0.30, 0.90] (4/6) | 0.67 [0.30, 0.90] (4/6) |
| table | 0.17 [0.03, 0.56] (1/6) | 0.33 [0.10, 0.70] (2/6) |
| figure | 0.50 [0.15, 0.85] (2/4) | 0.50 [0.15, 0.85] (2/4) |
| text | 0.50 [0.15, 0.85] (2/4) | 0.75 [0.30, 0.95] (3/4) |
| all | 0.45 [0.26, 0.66] (9/20) | 0.55 [0.34, 0.74] (11/20) |

Pooled intervals treat annotator-item pairs as independent. Per annotator:

- André: step 1 0.45 [0.26, 0.66] (9/20), step 2 0.55 [0.34, 0.74] (11/20), McNemar exact p = 0.500

## Changes after the inventory

Any change: 3. Type changed: 2. Render changed: 0.
Fixed an error: 2. Broke a correct answer: 0. Pooled McNemar exact p = 0.500.

## Confusion matrix, step1 (rows gold, columns predicted)

| gold | chart | table | figure | text | layout |
|---|---|---|---|---|---|
| chart | 4 | 1 | 1 | 0 | 0 |
| table | 2 | 1 | 2 | 1 | 0 |
| figure | 1 | 0 | 2 | 1 | 0 |
| text | 0 | 1 | 1 | 2 | 0 |

## Confusion matrix, step2 (rows gold, columns predicted)

| gold | chart | table | figure | text | layout |
|---|---|---|---|---|---|
| chart | 4 | 1 | 1 | 0 | 0 |
| table | 1 | 2 | 2 | 1 | 0 |
| figure | 1 | 0 | 2 | 1 | 0 |
| text | 0 | 1 | 0 | 3 | 0 |

## Render rate per gold type

| gold | step 1 | step 2 |
|---|---|---|
| chart | 1.00 | 1.00 |
| table | 1.00 | 1.00 |
| figure | 1.00 | 1.00 |
| text | 0.50 | 0.50 |

## Subtype match (chart items typed chart at step 2, n = 4)

Counts: {'yes': 3, 'partly': 1}. Yes: 0.75. Yes or partly: 1.00.

## Labels

label_ok = no: 1. Unsure: 2. No clear region: 1.

- André A09 (figure): 

## Agreement (step 1)

One annotator only.

## Accuracy by confidence (step 1)

| confidence | n | accuracy |
|---|---|---|
| 3 | 8 | 0.38 |
| 4 | 8 | 0.50 |
| 5 | 4 | 0.50 |

Median seconds per step: step 1 73, step 2 9, step 3 23.

## Word overlap, step1 specification

| words | n items | gold hit | other pages hit | Wilcoxon p | gold strictly top | gold top (ties) | median words |
|---|---|---|---|---|---|---|---|
| all | 20 | 0.35 | 0.12 | 0.0217 | 0.25 | 0.50 | 4 |
| question | 15 | 0.63 | 0.18 | 0.00373 | 0.27 | 0.67 | 3 |
| added | 18 | 0.14 | 0.08 | 0.834 | 0.17 | 0.22 | 3 |

## Word overlap, step2 specification

| words | n items | gold hit | other pages hit | Wilcoxon p | gold strictly top | gold top (ties) | median words |
|---|---|---|---|---|---|---|---|
| all | 20 | 0.33 | 0.12 | 0.0386 | 0.20 | 0.45 | 4 |
| question | 14 | 0.63 | 0.18 | 0.00604 | 0.29 | 0.64 | 3 |
| added | 18 | 0.13 | 0.08 | 0.583 | 0.11 | 0.17 | 3 |

OCR coverage: 203 of 204 thin pages (fewer than 20 text-layer words) in these documents have OCR text. Thin gold pages: 8.
