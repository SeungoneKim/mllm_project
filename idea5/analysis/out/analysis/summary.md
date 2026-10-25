# Part B: what does the page image add beyond the text layer?

72 questions (gold page only; 20 pages have no text layer). Reader: Qwen2.5-VL-7B, greedy short answers. Accuracy in % (MMLongBench-Doc rules; correct = score > 0).

| condition | text (n=16) | table (n=16) | chart (n=16) | figure (n=16) | layout (n=8) | all (n=72) | 95% CI (all) | input tokens |
|---|---|---|---|---|---|---|---|---|
| question only | 6 | 6 | 6 | 6 | 12 | 7 | [1, 14] | 79 |
| text layer | 44 | 31 | 31 | 44 | 50 | 39 | [28, 50] | 539 |
| VLM transcription | 44 | 19 | 62 | 62 | 75 | 50 | [38, 61] | 644 |
| image (1280) | 50 | 6 | 56 | 81 | 100 | 54 | [43, 65] | 1301 |
| image (256) | 38 | 12 | 56 | 31 | 100 | 42 | [31, 53] | 320 |
| text + image (1280) | 56 | 31 | 56 | 75 | 100 | 60 | [49, 71] | 1775 |
| text + image (256) | 50 | 31 | 56 | 50 | 100 | 53 | [40, 64] | 794 |
| masked image (256) | 6 | 6 | 31 | 44 | 75 | 28 | [18, 38] | 326 |
| text + masked image (256) | 50 | 25 | 56 | 62 | 100 | 54 | [43, 65] | 801 |

## H1: text alone against text + image (1280), paired, percentage points

| type | text layer - text+image [CI] | b/s/w | transcription - text+image [CI] | b/s/w |
|---|---|---|---|---|
| text | -12 [-31, +0] | 0/14/2 | -12 [-31, +0] | 0/14/2 |
| table | +0 [-19, +19] | 1/14/1 | -12 [-31, +0] | 0/14/2 |
| chart | -25 [-50, -6] | 0/12/4 | +6 [-12, +25] | 2/13/1 |
| figure | -31 [-56, -12] | 0/11/5 | -12 [-38, +19] | 2/10/4 |
| layout | -50 [-88, -12] | 0/4/4 | -25 [-62, +0] | 0/6/2 |
| all | -21 [-31, -11] | 1/55/16 | -10 [-21, +0] | 4/57/11 |

## H2: interaction responses (from T, I_hi, TI_hi)

| type | n | equivalence | text-dominant | image-dominant | emergence | interference | none |
|---|---|---|---|---|---|---|---|
| text | 16 | 7 | 0 | 1 | 1 | 0 | 7 |
| table | 16 | 1 | 3 | 0 | 1 | 1 | 10 |
| chart | 16 | 5 | 0 | 4 | 0 | 0 | 7 |
| figure | 16 | 7 | 0 | 5 | 0 | 1 | 3 |
| layout | 8 | 4 | 0 | 4 | 0 | 0 | 0 |
| all | 72 | 24 | 3 | 14 | 2 | 2 | 27 |

Regression y = w0 + w1 I + w2 T + w3 I*T (accuracy, points), estimate [95% CI]:

| type | w0 | w1 image | w2 text | w3 interaction |
|---|---|---|---|---|
| text | +6 [+0, +19] | +44 [+19, +69] | +38 [+19, +62] | -31 [-62, +0] |
| table | +6 [+0, +19] | +0 [-19, +19] | +25 [+6, +50] | +0 [-25, +25] |
| chart | +6 [+0, +19] | +50 [+25, +75] | +25 [+6, +50] | -25 [-50, -6] |
| figure | +6 [+0, +19] | +75 [+50, +94] | +38 [+6, +69] | -44 [-69, -12] |
| layout | +12 [+0, +38] | +88 [+62, +100] | +38 [+12, +75] | -38 [-75, -12] |
| all | +7 [+1, +14] | +47 [+35, +58] | +32 [+21, +43] | -26 [-39, -14] |

Regression y = w0 + w1 I + w2 T + w3 I*T (mean log-prob of the reference answer), estimate [95% CI]:

| type | w0 | w1 image | w2 text | w3 interaction |
|---|---|---|---|---|
| text | -2.76 [-3.47, -2.11] | +1.56 [+0.96, +2.16] | +1.11 [+0.53, +1.74] | -1.03 [-1.69, -0.40] |
| table | -3.15 [-4.10, -2.27] | +1.30 [+0.41, +2.25] | +1.16 [+0.04, +2.20] | -0.97 [-1.87, -0.10] |
| chart | -3.90 [-5.19, -2.69] | +2.10 [+1.26, +3.00] | +1.02 [+0.11, +1.99] | -0.97 [-1.93, -0.06] |
| figure | -3.31 [-4.48, -2.29] | +2.63 [+1.67, +3.76] | +1.42 [+0.36, +2.59] | -1.32 [-2.49, -0.28] |
| layout | -4.25 [-6.74, -2.10] | +4.20 [+2.09, +6.69] | +2.37 [+0.53, +5.08] | -2.37 [-5.08, -0.53] |
| all | -3.39 [-3.90, -2.85] | +2.15 [+1.66, +2.67] | +1.31 [+0.81, +1.85] | -1.22 [-1.74, -0.74] |

## H3 (pages with a text layer, n = 52), percentage points

- Accuracy lost going from 1280 to 256 image tokens, without text: +21 [+12, +33]; with the text layer: +10 [+0, +19]; difference (modulation): +12 [+0, +23].
- text + masked image - text + image (256): +2 [-6, +10], better/same/worse 3/47/2.
- text + masked image - text layer alone: +4 [-4, +12], better/same/worse 3/48/1.
- masked image - image (256): -17 [-33, -2], better/same/worse 4/35/13.

Pages without a text layer (n = 20): question only 15, text layer 20, VLM transcription 55, image (1280) 55, image (256) 65, text + image (1280) 65, text + image (256) 65, masked image (256) 60, text + masked image (256) 65.

Reference-answer tokens aligned with the tokenized prompt: 100% of rows (log-prob rows that are not aligned are still scored).
