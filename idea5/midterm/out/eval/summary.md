# Idea 5 midterm: text layer + K selected visual tokens per page

829 questions (475 single-page; 219 with a gold page without a text layer). Reader: Qwen2.5-VL-7B on the gold pages, greedy. Accuracy in % (rule-based scorer, correct = score > 0). K is per page; the selectors choose K x N tokens across a question's N pages.

Group sizes: all 829, single-page 475, multi-page 354, text 128, table 140, chart 107, figure 173, layout 32, visual 509, text layer 610, no text layer 219, image-needed 203, text-sufficient 232

| condition | all | single-page | multi-page | visual | text layer | no text layer | image-needed | text-sufficient | text | table | chart | figure | layout | 95% CI (all) | visual tokens / layer | sequence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text | 28 | 37 | 16 | 22 | 35 | 7 | 0 | 100 | 44 | 31 | 28 | 18 | 34 | [25, 31] | 0 | 977 |
| full | 49 | 63 | 32 | 49 | 47 | 56 | 100 | 89 | 56 | 44 | 50 | 54 | 69 | [46, 53] | 2318 | 3311 |
| full_imgfirst | 50 | 63 | 32 | 50 | 48 | 55 | 89 | 91 | 59 | 42 | 46 | 54 | 78 | [47, 54] | 2318 | 3307 |
| down_64 | 36 | 44 | 25 | 33 | 38 | 31 | 36 | 91 | 49 | 34 | 34 | 39 | 56 | [33, 40] | 109 | 1102 |
| down_128 | 41 | 49 | 29 | 38 | 39 | 45 | 53 | 90 | 50 | 41 | 39 | 39 | 62 | [37, 44] | 213 | 1206 |
| down_256 | 42 | 53 | 29 | 40 | 40 | 49 | 62 | 88 | 50 | 42 | 43 | 46 | 69 | [39, 46] | 455 | 1449 |
| rand_64 | 31 | 40 | 20 | 26 | 36 | 18 | 20 | 91 | 47 | 34 | 28 | 28 | 47 | [28, 35] | 120 | 1114 |
| rand_128 | 33 | 42 | 20 | 29 | 36 | 23 | 28 | 89 | 45 | 34 | 30 | 32 | 41 | [30, 36] | 241 | 1234 |
| rand_256 | 38 | 47 | 25 | 34 | 40 | 32 | 47 | 90 | 51 | 39 | 32 | 38 | 53 | [35, 41] | 481 | 1475 |
| ink_64 | 33 | 44 | 19 | 29 | 38 | 20 | 31 | 91 | 46 | 35 | 31 | 33 | 50 | [30, 37] | 120 | 1114 |
| ink_128 | 35 | 45 | 21 | 32 | 39 | 24 | 38 | 90 | 47 | 36 | 31 | 36 | 56 | [32, 39] | 241 | 1234 |
| ink_256 | 39 | 50 | 24 | 36 | 40 | 34 | 48 | 90 | 51 | 37 | 34 | 41 | 56 | [35, 42] | 481 | 1475 |
| comp_64 | 35 | 45 | 21 | 31 | 40 | 20 | 34 | 91 | 45 | 35 | 31 | 35 | 56 | [31, 38] | 120 | 1114 |
| comp_128 | 36 | 48 | 21 | 34 | 41 | 24 | 42 | 89 | 48 | 34 | 34 | 38 | 62 | [33, 40] | 241 | 1234 |
| comp_256 | 40 | 51 | 26 | 38 | 42 | 34 | 55 | 89 | 52 | 35 | 36 | 44 | 62 | [37, 44] | 481 | 1475 |
| fastv_64 | 30 | 39 | 19 | 25 | 36 | 16 | 19 | 88 | 45 | 36 | 29 | 23 | 50 | [27, 34] | 277 | 1114 |
| fastv_128 | 34 | 43 | 23 | 31 | 37 | 29 | 34 | 90 | 48 | 35 | 34 | 31 | 56 | [31, 38] | 389 | 1234 |
| fastv_256 | 40 | 48 | 29 | 37 | 38 | 44 | 57 | 88 | 52 | 39 | 34 | 41 | 56 | [37, 43] | 612 | 1475 |
| svlm_64 | 31 | 41 | 19 | 27 | 35 | 21 | 22 | 88 | 45 | 34 | 32 | 26 | 50 | [28, 35] | 356 | 1114 |
| svlm_128 | 36 | 45 | 24 | 33 | 37 | 32 | 36 | 90 | 48 | 36 | 34 | 35 | 56 | [33, 39] | 463 | 1234 |
| svlm_256 | 40 | 50 | 27 | 38 | 40 | 41 | 52 | 92 | 54 | 36 | 34 | 43 | 56 | [37, 44] | 678 | 1475 |
| svlmnr_64 | 33 | 42 | 20 | 28 | 36 | 23 | 24 | 91 | 47 | 34 | 30 | 27 | 56 | [29, 36] | 356 | 1114 |
| svlmnr_128 | 35 | 45 | 22 | 31 | 36 | 32 | 35 | 91 | 49 | 36 | 30 | 34 | 56 | [32, 39] | 463 | 1234 |
| svlmnr_256 | 40 | 48 | 28 | 37 | 39 | 42 | 52 | 90 | 52 | 36 | 33 | 40 | 56 | [36, 43] | 678 | 1475 |

## Paired differences (percentage points)

| comparison | n | diff [95% CI] | better / same / worse |
|---|---|---|---|
| svlm_64 - text (all) | 829 | +3.4 [+1.2, +5.5] | 55 / 747 / 27 |
| svlm_64 - text (multi-page) | 354 | +3.1 [+0.0, +6.2] | 21 / 323 / 10 |
| svlm_64 - text (visual) | 509 | +5.3 [+2.4, +8.4] | 44 / 448 / 17 |
| svlm_64 - text (image-needed) | 203 | +21.7 [+16.3, +27.1] | 44 / 159 / 0 |
| svlm_64 - down_64 (all) | 829 | -4.7 [-7.0, -2.4] | 25 / 740 / 64 |
| svlm_64 - down_64 (multi-page) | 354 | -6.2 [-9.6, -2.8] | 9 / 314 / 31 |
| svlm_64 - down_64 (visual) | 509 | -6.3 [-9.4, -3.1] | 20 / 437 / 52 |
| svlm_64 - down_64 (image-needed) | 203 | -14.8 [-22.2, -7.9] | 14 / 145 / 44 |
| svlm_64 - rand_64 (all) | 829 | +0.0 [-2.1, +2.2] | 42 / 745 / 42 |
| svlm_64 - rand_64 (multi-page) | 354 | -1.4 [-4.5, +1.7] | 13 / 323 / 18 |
| svlm_64 - rand_64 (visual) | 509 | +0.8 [-2.2, +3.7] | 34 / 445 / 30 |
| svlm_64 - rand_64 (image-needed) | 203 | +1.5 [-4.9, +8.4] | 26 / 154 / 23 |
| svlm_64 - fastv_64 (all) | 829 | +1.0 [-0.7, +2.8] | 31 / 775 / 23 |
| svlm_64 - fastv_64 (multi-page) | 354 | -0.6 [-3.1, +2.0] | 10 / 332 / 12 |
| svlm_64 - fastv_64 (visual) | 509 | +2.2 [-0.4, +4.5] | 26 / 468 / 15 |
| svlm_64 - fastv_64 (image-needed) | 203 | +3.0 [-2.5, +8.4] | 19 / 171 / 13 |
| svlm_64 - svlmnr_64 (all) | 829 | -1.2 [-3.0, +0.6] | 22 / 775 / 32 |
| svlm_64 - svlmnr_64 (multi-page) | 354 | -1.4 [-4.0, +0.8] | 8 / 333 / 13 |
| svlm_64 - svlmnr_64 (visual) | 509 | -1.4 [-3.7, +1.0] | 17 / 468 / 24 |
| svlm_64 - svlmnr_64 (image-needed) | 203 | -2.5 [-7.9, +2.5] | 11 / 176 / 16 |
| comp_64 - svlm_64 (all) | 829 | +3.3 [+1.0, +5.7] | 60 / 736 / 33 |
| comp_64 - svlm_64 (multi-page) | 354 | +2.0 [-0.8, +4.8] | 18 / 325 / 11 |
| comp_64 - svlm_64 (visual) | 509 | +4.5 [+1.2, +7.9] | 50 / 432 / 27 |
| comp_64 - svlm_64 (image-needed) | 203 | +12.3 [+4.9, +19.2] | 42 / 144 / 17 |
| comp_64 - ink_64 (all) | 829 | +1.2 [+0.1, +2.4] | 17 / 805 / 7 |
| comp_64 - ink_64 (multi-page) | 354 | +2.0 [+0.6, +3.7] | 8 / 345 / 1 |
| comp_64 - ink_64 (visual) | 509 | +2.2 [+0.6, +3.7] | 14 / 492 / 3 |
| comp_64 - ink_64 (image-needed) | 203 | +3.4 [+0.0, +6.9] | 10 / 190 / 3 |
| comp_64 - down_64 (all) | 829 | -1.4 [-4.0, +1.0] | 49 / 719 / 61 |
| comp_64 - down_64 (multi-page) | 354 | -4.2 [-7.6, -0.8] | 12 / 315 / 27 |
| comp_64 - down_64 (visual) | 509 | -1.8 [-5.3, +1.8] | 39 / 422 / 48 |
| comp_64 - down_64 (image-needed) | 203 | -2.5 [-10.8, +5.9] | 34 / 130 / 39 |
| full - svlm_64 (all) | 829 | +18.0 [+14.8, +21.1] | 175 / 628 / 26 |
| full - svlm_64 (multi-page) | 354 | +13.0 [+8.5, +16.9] | 55 / 290 / 9 |
| full - svlm_64 (visual) | 509 | +22.0 [+17.9, +26.3] | 130 / 361 / 18 |
| full - svlm_64 (image-needed) | 203 | +78.3 [+72.9, +83.7] | 159 / 44 / 0 |
| svlm_128 - text (all) | 829 | +7.8 [+5.4, +10.3] | 88 / 718 / 23 |
| svlm_128 - text (multi-page) | 354 | +8.5 [+5.1, +11.9] | 36 / 312 / 6 |
| svlm_128 - text (visual) | 509 | +11.0 [+7.5, +14.3] | 69 / 427 / 13 |
| svlm_128 - text (image-needed) | 203 | +36.0 [+29.6, +42.9] | 73 / 130 / 0 |
| svlm_128 - down_128 (all) | 829 | -4.7 [-7.1, -2.4] | 30 / 730 / 69 |
| svlm_128 - down_128 (multi-page) | 354 | -4.8 [-7.6, -1.7] | 7 / 323 / 24 |
| svlm_128 - down_128 (visual) | 509 | -5.3 [-8.4, -2.2] | 20 / 442 / 47 |
| svlm_128 - down_128 (image-needed) | 203 | -17.2 [-24.2, -10.3] | 13 / 142 / 48 |
| svlm_128 - rand_128 (all) | 829 | +3.1 [+0.7, +5.4] | 63 / 729 / 37 |
| svlm_128 - rand_128 (multi-page) | 354 | +3.7 [+0.8, +6.8] | 20 / 327 / 7 |
| svlm_128 - rand_128 (visual) | 509 | +3.5 [+0.2, +7.1] | 48 / 431 / 30 |
| svlm_128 - rand_128 (image-needed) | 203 | +8.4 [+1.0, +15.8] | 38 / 144 / 21 |
| svlm_128 - fastv_128 (all) | 829 | +1.3 [-0.6, +3.1] | 35 / 770 / 24 |
| svlm_128 - fastv_128 (multi-page) | 354 | +0.6 [-1.7, +2.8] | 10 / 336 / 8 |
| svlm_128 - fastv_128 (visual) | 509 | +2.0 [-0.8, +4.7] | 29 / 461 / 19 |
| svlm_128 - fastv_128 (image-needed) | 203 | +1.5 [-3.9, +7.4] | 19 / 168 / 16 |
| svlm_128 - svlmnr_128 (all) | 829 | +0.6 [-1.0, +2.1] | 21 / 792 / 16 |
| svlm_128 - svlmnr_128 (multi-page) | 354 | +1.7 [+0.3, +3.4] | 7 / 346 / 1 |
| svlm_128 - svlmnr_128 (visual) | 509 | +1.4 [-0.8, +3.3] | 19 / 478 / 12 |
| svlm_128 - svlmnr_128 (image-needed) | 203 | +1.0 [-3.4, +5.4] | 11 / 183 / 9 |
| comp_128 - svlm_128 (all) | 829 | +0.5 [-2.1, +3.0] | 60 / 713 / 56 |
| comp_128 - svlm_128 (multi-page) | 354 | -3.4 [-6.5, -0.3] | 10 / 322 / 22 |
| comp_128 - svlm_128 (visual) | 509 | +1.0 [-2.8, +4.9] | 51 / 412 / 46 |
| comp_128 - svlm_128 (image-needed) | 203 | +5.9 [-2.5, +13.8] | 42 / 131 / 30 |
| comp_128 - ink_128 (all) | 829 | +1.1 [-0.2, +2.4] | 20 / 798 / 11 |
| comp_128 - ink_128 (multi-page) | 354 | -0.8 [-2.5, +0.8] | 4 / 343 / 7 |
| comp_128 - ink_128 (visual) | 509 | +2.0 [+0.4, +3.5] | 15 / 489 / 5 |
| comp_128 - ink_128 (image-needed) | 203 | +3.9 [+1.5, +6.9] | 8 / 195 / 0 |
| comp_128 - down_128 (all) | 829 | -4.2 [-6.9, -1.6] | 47 / 700 / 82 |
| comp_128 - down_128 (multi-page) | 354 | -8.2 [-11.6, -4.8] | 6 / 313 / 35 |
| comp_128 - down_128 (visual) | 509 | -4.3 [-8.3, -0.4] | 42 / 403 / 64 |
| comp_128 - down_128 (image-needed) | 203 | -11.3 [-20.2, -3.0] | 29 / 122 / 52 |
| full - svlm_128 (all) | 829 | +13.5 [+10.6, +16.5] | 143 / 655 / 31 |
| full - svlm_128 (multi-page) | 354 | +7.6 [+3.7, +11.3] | 37 / 307 / 10 |
| full - svlm_128 (visual) | 509 | +16.3 [+12.2, +20.4] | 105 / 382 / 22 |
| full - svlm_128 (image-needed) | 203 | +64.0 [+57.1, +70.4] | 130 / 73 / 0 |
| svlm_256 - text (all) | 829 | +12.2 [+9.5, +14.8] | 119 / 692 / 18 |
| svlm_256 - text (multi-page) | 354 | +11.9 [+7.9, +15.5] | 48 / 300 / 6 |
| svlm_256 - text (visual) | 509 | +16.1 [+12.4, +19.8] | 94 / 403 / 12 |
| svlm_256 - text (image-needed) | 203 | +51.7 [+44.8, +58.6] | 105 / 98 / 0 |
| svlm_256 - down_256 (all) | 829 | -2.3 [-4.7, +0.1] | 42 / 726 / 61 |
| svlm_256 - down_256 (multi-page) | 354 | -1.1 [-4.2, +2.0] | 15 / 320 / 19 |
| svlm_256 - down_256 (visual) | 509 | -2.8 [-5.9, +0.4] | 28 / 439 / 42 |
| svlm_256 - down_256 (image-needed) | 203 | -10.3 [-17.2, -3.4] | 17 / 148 / 38 |
| svlm_256 - rand_256 (all) | 829 | +2.3 [+0.1, +4.5] | 56 / 736 / 37 |
| svlm_256 - rand_256 (multi-page) | 354 | +2.3 [-0.3, +5.1] | 16 / 330 / 8 |
| svlm_256 - rand_256 (visual) | 509 | +3.5 [+0.4, +6.7] | 41 / 445 / 23 |
| svlm_256 - rand_256 (image-needed) | 203 | +4.9 [-2.0, +11.8] | 31 / 151 / 21 |
| svlm_256 - fastv_256 (all) | 829 | +0.4 [-1.6, +2.3] | 35 / 762 / 32 |
| svlm_256 - fastv_256 (multi-page) | 354 | -1.1 [-3.1, +0.8] | 5 / 340 / 9 |
| svlm_256 - fastv_256 (visual) | 509 | +0.6 [-2.2, +3.3] | 26 / 460 / 23 |
| svlm_256 - fastv_256 (image-needed) | 203 | -5.4 [-11.3, +0.0] | 12 / 168 / 23 |
| svlm_256 - svlmnr_256 (all) | 829 | +0.6 [-0.7, +1.9] | 18 / 798 / 13 |
| svlm_256 - svlmnr_256 (multi-page) | 354 | -0.6 [-2.5, +1.1] | 4 / 344 / 6 |
| svlm_256 - svlmnr_256 (visual) | 509 | +1.0 [-0.8, +2.8] | 13 / 488 / 8 |
| svlm_256 - svlmnr_256 (image-needed) | 203 | +0.0 [-3.5, +3.4] | 7 / 189 / 7 |
| comp_256 - svlm_256 (all) | 829 | +0.0 [-2.7, +2.8] | 64 / 701 / 64 |
| comp_256 - svlm_256 (multi-page) | 354 | -1.7 [-4.8, +1.7] | 13 / 322 / 19 |
| comp_256 - svlm_256 (visual) | 509 | +0.6 [-3.1, +4.3] | 49 / 414 / 46 |
| comp_256 - svlm_256 (image-needed) | 203 | +3.4 [-5.4, +11.8] | 44 / 122 / 37 |
| comp_256 - ink_256 (all) | 829 | +1.4 [-0.1, +3.0] | 28 / 785 / 16 |
| comp_256 - ink_256 (multi-page) | 354 | +1.7 [-0.3, +4.0] | 11 / 338 / 5 |
| comp_256 - ink_256 (visual) | 509 | +2.8 [+0.8, +4.9] | 21 / 481 / 7 |
| comp_256 - ink_256 (image-needed) | 203 | +7.4 [+3.4, +11.8] | 18 / 182 / 3 |
| comp_256 - down_256 (all) | 829 | -2.3 [-5.1, +0.6] | 62 / 686 / 81 |
| comp_256 - down_256 (multi-page) | 354 | -2.8 [-6.5, +0.8] | 17 / 310 / 27 |
| comp_256 - down_256 (visual) | 509 | -2.2 [-6.1, +1.8] | 45 / 408 / 56 |
| comp_256 - down_256 (image-needed) | 203 | -6.9 [-15.8, +1.5] | 33 / 123 / 47 |
| full - svlm_256 (all) | 829 | +9.2 [+6.6, +11.7] | 106 / 693 / 30 |
| full - svlm_256 (multi-page) | 354 | +4.2 [+0.6, +7.9] | 28 / 313 / 13 |
| full - svlm_256 (visual) | 509 | +11.2 [+7.5, +14.9] | 77 / 412 / 20 |
| full - svlm_256 (image-needed) | 203 | +48.3 [+41.4, +55.2] | 98 / 105 / 0 |

## Response to the image, against text only

| condition | recovered | interference | n |
|---|---|---|---|
| comp_128 | 94 | 25 | 829 |
| comp_256 | 127 | 26 | 829 |
| comp_64 | 77 | 22 | 829 |
| down_128 | 127 | 23 | 829 |
| down_256 | 148 | 28 | 829 |
| down_64 | 87 | 20 | 829 |
| fastv_128 | 78 | 24 | 829 |
| fastv_256 | 127 | 29 | 829 |
| fastv_64 | 47 | 27 | 829 |
| full | 203 | 26 | 829 |
| full_imgfirst | 202 | 21 | 829 |
| ink_128 | 84 | 24 | 829 |
| ink_256 | 113 | 24 | 829 |
| ink_64 | 67 | 22 | 829 |
| rand_128 | 65 | 26 | 829 |
| rand_256 | 106 | 24 | 829 |
| rand_64 | 49 | 21 | 829 |
| svlm_128 | 88 | 23 | 829 |
| svlm_256 | 119 | 18 | 829 |
| svlm_64 | 55 | 27 | 829 |
| svlmnr_128 | 82 | 22 | 829 |
| svlmnr_256 | 120 | 24 | 829 |
| svlmnr_64 | 60 | 22 | 829 |

## Where the kept tokens fall (pages with a text layer)

All tokens of these pages: 23% on text-layer words, 26% other ink, 51% blank.

| condition | on words | other ink | blank | residual kept | residual kept, right / wrong (image-needed) |
|---|---|---|---|---|---|
| rand_64 | 22 | 26 | 52 | 5 | 5 / 5 |
| rand_128 | 22 | 26 | 52 | 10 | 10 / 11 |
| rand_256 | 22 | 26 | 52 | 21 | 21 / 21 |
| ink_64 | 54 | 46 | 0 | 49 | 72 / 58 |
| ink_128 | 59 | 41 | 0 | 48 | 65 / 52 |
| ink_256 | 60 | 39 | 1 | 57 | 73 / 57 |
| comp_64 | 10 | 90 | 0 | 100 | 100 / 100 |
| comp_128 | 17 | 83 | 0 | 100 | 100 / 100 |
| comp_256 | 27 | 71 | 1 | 100 | 100 / 100 |
| fastv_64 | 9 | 20 | 70 | 5 | 6 / 4 |
| fastv_128 | 17 | 27 | 55 | 12 | 13 / 11 |
| fastv_256 | 23 | 30 | 46 | 26 | 28 / 24 |
| svlm_64 | 21 | 27 | 52 | 3 | 3 / 3 |
| svlm_128 | 28 | 29 | 43 | 10 | 12 / 9 |
| svlm_256 | 30 | 29 | 41 | 23 | 23 / 22 |
| svlmnr_64 | 26 | 29 | 45 | 6 | 5 / 7 |
| svlmnr_128 | 29 | 29 | 42 | 13 | 15 / 12 |
| svlmnr_256 | 30 | 29 | 41 | 26 | 25 / 25 |

## Examples in fig_examples.pdf

- qid 21 (Research report / Introduction, layout): What year is the report for? Answer: 2015-2016. text: 2015; full: 2015-2016; comp_64: 2015-2016; svlm_64: 2015
- qid 30 (Tutorial/Workshop, figure): What is the color of the zone Mali in the map used to demonstrate GeoShapes using SPARQL and OSM? Answer: purple. text: Not specified; full: Purple; comp_64: Purple; svlm_64: Green
- qid 20 (Research report / Introduction, figure): What does the map in the report shows? Answer: The centres of Indian Space Programme. text: Population density.; full: The Centres of Indian Space Programme; comp_64: Indian states; svlm_64: Centres of Indian Space Programme
