# Idea 2+3 baseline results

## multi

|                          |   n |   answered |   complete |   recall |   acc |   score |
|:-------------------------|----:|-----------:|-----------:|---------:|------:|--------:|
| ('topk', 3)              | 310 |        310 |       40   |     59   |  24.2 |    23.7 |
| ('topk', 5)              | 310 |        310 |       51.3 |     69.1 |  24.2 |    23.8 |
| ('coverage', 3)          | 310 |        310 |       42.6 |     61   |  25.8 |    25.4 |
| ('coverage', 5)          | 310 |        310 |       52.9 |     69.9 |  24.5 |    24.1 |
| ('coverage_mod', 3)      | 310 |        310 |       41.9 |     61   |  24.8 |    24.4 |
| ('coverage_mod', 5)      | 310 |        310 |       52.9 |     70.2 |  24.2 |    23.8 |
| ('elem_topk', 3)         | 310 |        310 |       37.7 |     58.6 |  22.3 |    22.1 |
| ('elem_topk', 5)         | 310 |        310 |       49.7 |     68.8 |  25.2 |    24.9 |
| ('elem_coverage', 3)     | 310 |        310 |       40.6 |     60.2 |  23.2 |    23   |
| ('elem_coverage', 5)     | 310 |        310 |       52.3 |     70   |  25.5 |    25.1 |
| ('elem_coverage_mod', 3) | 310 |        310 |       38.4 |     59.2 |  22.9 |    22.6 |
| ('elem_coverage_mod', 5) | 310 |        310 |       51.9 |     69.9 |  25.5 |    25.1 |
| ('evidence', 3)          | 310 |        310 |       42.3 |     62.5 |  25.2 |    24.9 |
| ('evidence', 5)          | 310 |        310 |       53.9 |     71.4 |  26.8 |    26.3 |
| ('gold', 0)              | 310 |        310 |      100   |    100   |  32.9 |    32.3 |

## same_mod

|                          |   n |   answered |   complete |   recall |   acc |   score |
|:-------------------------|----:|-----------:|-----------:|---------:|------:|--------:|
| ('topk', 3)              | 201 |        201 |       36.3 |     54.7 |  22.4 |    21.8 |
| ('topk', 5)              | 201 |        201 |       45.8 |     64.3 |  22.9 |    22.3 |
| ('coverage', 3)          | 201 |        201 |       38.8 |     56.6 |  23.9 |    23.3 |
| ('coverage', 5)          | 201 |        201 |       47.8 |     65.5 |  22.9 |    22.3 |
| ('coverage_mod', 3)      | 201 |        201 |       38.8 |     56.6 |  22.4 |    21.8 |
| ('coverage_mod', 5)      | 201 |        201 |       48.3 |     65.8 |  22.4 |    21.8 |
| ('elem_topk', 3)         | 201 |        201 |       35.3 |     54.5 |  21.4 |    21.1 |
| ('elem_topk', 5)         | 201 |        201 |       45.3 |     64.6 |  23.9 |    23.5 |
| ('elem_coverage', 3)     | 201 |        201 |       37.8 |     55.9 |  22.4 |    22.1 |
| ('elem_coverage', 5)     | 201 |        201 |       48.3 |     66.2 |  24.4 |    23.8 |
| ('elem_coverage_mod', 3) | 201 |        201 |       35.8 |     54.5 |  21.9 |    21.4 |
| ('elem_coverage_mod', 5) | 201 |        201 |       47.8 |     65.7 |  24.4 |    23.8 |
| ('evidence', 3)          | 201 |        201 |       37.3 |     57.4 |  23.9 |    23.5 |
| ('evidence', 5)          | 201 |        201 |       47.8 |     66.4 |  26.4 |    25.7 |
| ('gold', 0)              | 201 |        201 |      100   |    100   |  30.8 |    30   |

## cross_multi

|                          |   n |   answered |   complete |   recall |   acc |   score |
|:-------------------------|----:|-----------:|-----------:|---------:|------:|--------:|
| ('topk', 3)              | 109 |        109 |       46.8 |     67.2 |  27.5 |    27.4 |
| ('topk', 5)              | 109 |        109 |       61.5 |     77.9 |  26.6 |    26.6 |
| ('coverage', 3)          | 109 |        109 |       49.5 |     69.2 |  29.4 |    29.2 |
| ('coverage', 5)          | 109 |        109 |       62.4 |     77.9 |  27.5 |    27.5 |
| ('coverage_mod', 3)      | 109 |        109 |       47.7 |     69   |  29.4 |    29.2 |
| ('coverage_mod', 5)      | 109 |        109 |       61.5 |     78.2 |  27.5 |    27.5 |
| ('elem_topk', 3)         | 109 |        109 |       42.2 |     66.1 |  23.9 |    23.9 |
| ('elem_topk', 5)         | 109 |        109 |       57.8 |     76.4 |  27.5 |    27.5 |
| ('elem_coverage', 3)     | 109 |        109 |       45.9 |     68.1 |  24.8 |    24.8 |
| ('elem_coverage', 5)     | 109 |        109 |       59.6 |     77.1 |  27.5 |    27.5 |
| ('elem_coverage_mod', 3) | 109 |        109 |       43.1 |     68   |  24.8 |    24.7 |
| ('elem_coverage_mod', 5) | 109 |        109 |       59.6 |     77.6 |  27.5 |    27.5 |
| ('evidence', 3)          | 109 |        109 |       51.4 |     71.9 |  27.5 |    27.5 |
| ('evidence', 5)          | 109 |        109 |       65.1 |     80.6 |  27.5 |    27.5 |
| ('gold', 0)              | 109 |        109 |      100   |    100   |  36.7 |    36.6 |

## control

|                          |   n |   answered |   complete |   recall |   acc |   score |
|:-------------------------|----:|-----------:|-----------:|---------:|------:|--------:|
| ('topk', 3)              | 453 |        453 |       82.3 |     82.3 |  52.3 |    51.2 |
| ('topk', 5)              | 453 |        453 |       86.3 |     86.3 |  53.4 |    52.7 |
| ('coverage', 3)          | 453 |        453 |       83.4 |     83.4 |  52.3 |    51.2 |
| ('coverage', 5)          | 453 |        453 |       86.5 |     86.5 |  53.9 |    53   |
| ('coverage_mod', 3)      | 453 |        453 |       83.7 |     83.7 |  52.5 |    51.4 |
| ('coverage_mod', 5)      | 453 |        453 |       86.8 |     86.8 |  53.9 |    53   |
| ('elem_topk', 3)         | 453 |        453 |       83.9 |     83.9 |  54.3 |    53.1 |
| ('elem_topk', 5)         | 453 |        453 |       86.3 |     86.3 |  51.9 |    51.2 |
| ('elem_coverage', 3)     | 453 |        453 |       84.1 |     84.1 |  53.9 |    52.7 |
| ('elem_coverage', 5)     | 453 |        453 |       87   |     87   |  52.5 |    51.7 |
| ('elem_coverage_mod', 3) | 453 |        453 |       84.8 |     84.8 |  54.7 |    53.6 |
| ('elem_coverage_mod', 5) | 453 |        453 |       87.2 |     87.2 |  52.8 |    51.9 |
| ('evidence', 3)          | 453 |        453 |       85.9 |     85.9 |  56.1 |    54.9 |
| ('evidence', 5)          | 453 |        453 |       88.3 |     88.3 |  54.3 |    53.5 |
| ('gold', 0)              | 453 |        453 |      100   |    100   |  59.8 |    59   |

## Paired against top-k (k = 5): wins / losses, exact McNemar p

|                                      |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:-------------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'coverage')                |   5 |             2 |               1 |     1      |               6 |                 1 |       0.125  |
| ('multi', 'coverage_mod')            |   5 |             2 |               2 |     1      |               6 |                 1 |       0.125  |
| ('multi', 'elem_topk')               |   5 |            10 |               7 |     0.6291 |              10 |                15 |       0.4244 |
| ('multi', 'elem_coverage')           |   5 |            10 |               6 |     0.4545 |              16 |                13 |       0.7111 |
| ('multi', 'elem_coverage_mod')       |   5 |            10 |               6 |     0.4545 |              15 |                13 |       0.8506 |
| ('multi', 'evidence')                |   5 |            15 |               7 |     0.1338 |              22 |                14 |       0.243  |
| ('multi', 'gold')                    |   5 |            36 |               9 |     0.0001 |             151 |                 0 |       0      |
| ('same_mod', 'coverage')             |   5 |             1 |               1 |     1      |               4 |                 0 |       0.125  |
| ('same_mod', 'coverage_mod')         |   5 |             1 |               2 |     1      |               5 |                 0 |       0.0625 |
| ('same_mod', 'elem_topk')            |   5 |             9 |               7 |     0.8036 |               7 |                 8 |       1      |
| ('same_mod', 'elem_coverage')        |   5 |             9 |               6 |     0.6072 |              11 |                 6 |       0.3323 |
| ('same_mod', 'elem_coverage_mod')    |   5 |             9 |               6 |     0.6072 |              10 |                 6 |       0.4545 |
| ('same_mod', 'evidence')             |   5 |            13 |               6 |     0.1671 |              12 |                 8 |       0.5034 |
| ('same_mod', 'gold')                 |   5 |            24 |               8 |     0.007  |             109 |                 0 |       0      |
| ('cross_multi', 'coverage')          |   5 |             1 |               0 |     1      |               2 |                 1 |       1      |
| ('cross_multi', 'coverage_mod')      |   5 |             1 |               0 |     1      |               1 |                 1 |       1      |
| ('cross_multi', 'elem_topk')         |   5 |             1 |               0 |     1      |               3 |                 7 |       0.3438 |
| ('cross_multi', 'elem_coverage')     |   5 |             1 |               0 |     1      |               5 |                 7 |       0.7744 |
| ('cross_multi', 'elem_coverage_mod') |   5 |             1 |               0 |     1      |               5 |                 7 |       0.7744 |
| ('cross_multi', 'evidence')          |   5 |             2 |               1 |     1      |              10 |                 6 |       0.4545 |
| ('cross_multi', 'gold')              |   5 |            12 |               1 |     0.0034 |              42 |                 0 |       0      |
| ('control', 'coverage')              |   5 |             2 |               0 |     0.5    |               2 |                 1 |       1      |
| ('control', 'coverage_mod')          |   5 |             3 |               1 |     0.625  |               3 |                 1 |       0.625  |
| ('control', 'elem_topk')             |   5 |            10 |              17 |     0.2478 |              10 |                10 |       1      |
| ('control', 'elem_coverage')         |   5 |            13 |              17 |     0.5847 |              12 |                 9 |       0.6636 |
| ('control', 'elem_coverage_mod')     |   5 |            14 |              17 |     0.7201 |              13 |                 9 |       0.5235 |
| ('control', 'evidence')              |   5 |            19 |              15 |     0.6076 |              18 |                 9 |       0.1221 |
| ('control', 'gold')                  |   5 |            50 |              21 |     0.0008 |              62 |                 0 |       0      |

## Paired against coverage (k = 5): wins / losses, exact McNemar p

|                                      |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:-------------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'topk')                    |   5 |             1 |               2 |     1      |               1 |                 6 |       0.125  |
| ('multi', 'coverage_mod')            |   5 |             0 |               1 |     1      |               1 |                 1 |       1      |
| ('multi', 'elem_topk')               |   5 |            10 |               8 |     0.8145 |               9 |                19 |       0.0872 |
| ('multi', 'elem_coverage')           |   5 |             9 |               6 |     0.6072 |              10 |                12 |       0.8318 |
| ('multi', 'elem_coverage_mod')       |   5 |             9 |               6 |     0.6072 |              10 |                13 |       0.6776 |
| ('multi', 'evidence')                |   5 |            14 |               7 |     0.1892 |              16 |                13 |       0.7111 |
| ('multi', 'gold')                    |   5 |            34 |               8 |     0.0001 |             146 |                 0 |       0      |
| ('same_mod', 'topk')                 |   5 |             1 |               1 |     1      |               0 |                 4 |       0.125  |
| ('same_mod', 'coverage_mod')         |   5 |             0 |               1 |     1      |               1 |                 0 |       1      |
| ('same_mod', 'elem_topk')            |   5 |             9 |               7 |     0.8036 |               6 |                11 |       0.3323 |
| ('same_mod', 'elem_coverage')        |   5 |             8 |               5 |     0.5811 |               7 |                 6 |       1      |
| ('same_mod', 'elem_coverage_mod')    |   5 |             8 |               5 |     0.5811 |               6 |                 6 |       1      |
| ('same_mod', 'evidence')             |   5 |            12 |               5 |     0.1435 |               8 |                 8 |       1      |
| ('same_mod', 'gold')                 |   5 |            23 |               7 |     0.0052 |             105 |                 0 |       0      |
| ('cross_multi', 'topk')              |   5 |             0 |               1 |     1      |               1 |                 2 |       1      |
| ('cross_multi', 'coverage_mod')      |   5 |             0 |               0 |     1      |               0 |                 1 |       1      |
| ('cross_multi', 'elem_topk')         |   5 |             1 |               1 |     1      |               3 |                 8 |       0.2266 |
| ('cross_multi', 'elem_coverage')     |   5 |             1 |               1 |     1      |               3 |                 6 |       0.5078 |
| ('cross_multi', 'elem_coverage_mod') |   5 |             1 |               1 |     1      |               4 |                 7 |       0.5488 |
| ('cross_multi', 'evidence')          |   5 |             2 |               2 |     1      |               8 |                 5 |       0.5811 |
| ('cross_multi', 'gold')              |   5 |            11 |               1 |     0.0063 |              41 |                 0 |       0      |
| ('control', 'topk')                  |   5 |             0 |               2 |     0.5    |               1 |                 2 |       1      |
| ('control', 'coverage_mod')          |   5 |             1 |               1 |     1      |               1 |                 0 |       1      |
| ('control', 'elem_topk')             |   5 |            10 |              19 |     0.136  |              11 |                12 |       1      |
| ('control', 'elem_coverage')         |   5 |            11 |              17 |     0.3449 |              12 |                10 |       0.8318 |
| ('control', 'elem_coverage_mod')     |   5 |            12 |              17 |     0.4583 |              13 |                10 |       0.6776 |
| ('control', 'evidence')              |   5 |            17 |              15 |     0.8601 |              17 |                 9 |       0.1686 |
| ('control', 'gold')                  |   5 |            49 |              22 |     0.0018 |              61 |                 0 |       0      |

## Paired against elem_coverage_mod (k = 5): wins / losses, exact McNemar p

|                                  |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:---------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'topk')                |   5 |             6 |              10 |     0.4545 |              13 |                15 |       0.8506 |
| ('multi', 'coverage')            |   5 |             6 |               9 |     0.6072 |              13 |                10 |       0.6776 |
| ('multi', 'coverage_mod')        |   5 |             6 |              10 |     0.4545 |              13 |                10 |       0.6776 |
| ('multi', 'elem_topk')           |   5 |             1 |               2 |     1      |               0 |                 7 |       0.0156 |
| ('multi', 'elem_coverage')       |   5 |             0 |               0 |     1      |               3 |                 2 |       1      |
| ('multi', 'evidence')            |   5 |             6 |               2 |     0.2891 |              10 |                 4 |       0.1796 |
| ('multi', 'gold')                |   5 |            35 |              12 |     0.0011 |             149 |                 0 |       0      |
| ('same_mod', 'topk')             |   5 |             6 |               9 |     0.6072 |               6 |                10 |       0.4545 |
| ('same_mod', 'coverage')         |   5 |             5 |               8 |     0.5811 |               6 |                 6 |       1      |
| ('same_mod', 'coverage_mod')     |   5 |             5 |               9 |     0.424  |               7 |                 6 |       1      |
| ('same_mod', 'elem_topk')        |   5 |             1 |               2 |     1      |               0 |                 5 |       0.0625 |
| ('same_mod', 'elem_coverage')    |   5 |             0 |               0 |     1      |               2 |                 1 |       1      |
| ('same_mod', 'evidence')         |   5 |             5 |               1 |     0.2188 |               3 |                 3 |       1      |
| ('same_mod', 'gold')             |   5 |            24 |              11 |     0.041  |             105 |                 0 |       0      |
| ('cross_multi', 'topk')          |   5 |             0 |               1 |     1      |               7 |                 5 |       0.7744 |
| ('cross_multi', 'coverage')      |   5 |             1 |               1 |     1      |               7 |                 4 |       0.5488 |
| ('cross_multi', 'coverage_mod')  |   5 |             1 |               1 |     1      |               6 |                 4 |       0.7539 |
| ('cross_multi', 'elem_topk')     |   5 |             0 |               0 |     1      |               0 |                 2 |       0.5    |
| ('cross_multi', 'elem_coverage') |   5 |             0 |               0 |     1      |               1 |                 1 |       1      |
| ('cross_multi', 'evidence')      |   5 |             1 |               1 |     1      |               7 |                 1 |       0.0703 |
| ('cross_multi', 'gold')          |   5 |            11 |               1 |     0.0063 |              44 |                 0 |       0      |
| ('control', 'topk')              |   5 |            17 |              14 |     0.7201 |               9 |                13 |       0.5235 |
| ('control', 'coverage')          |   5 |            17 |              12 |     0.4583 |              10 |                13 |       0.6776 |
| ('control', 'coverage_mod')      |   5 |            16 |              11 |     0.4421 |              10 |                12 |       0.8318 |
| ('control', 'elem_topk')         |   5 |             0 |               4 |     0.125  |               1 |                 5 |       0.2188 |
| ('control', 'elem_coverage')     |   5 |             0 |               1 |     1      |               1 |                 2 |       1      |
| ('control', 'evidence')          |   5 |             7 |               0 |     0.0156 |               7 |                 2 |       0.1797 |
| ('control', 'gold')              |   5 |            51 |              19 |     0.0002 |              58 |                 0 |       0      |

## Paired against top-k (k = 3): wins / losses, exact McNemar p

|                                      |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:-------------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'coverage')                |   3 |             5 |               0 |     0.0625 |               9 |                 1 |       0.0215 |
| ('multi', 'coverage_mod')            |   3 |             4 |               2 |     0.6875 |              10 |                 4 |       0.1796 |
| ('multi', 'elem_topk')               |   3 |             4 |              10 |     0.1796 |               9 |                16 |       0.2295 |
| ('multi', 'elem_coverage')           |   3 |             7 |              10 |     0.6291 |              15 |                13 |       0.8506 |
| ('multi', 'elem_coverage_mod')       |   3 |             6 |              10 |     0.4545 |              12 |                17 |       0.4583 |
| ('multi', 'evidence')                |   3 |            13 |              10 |     0.6776 |              25 |                18 |       0.3604 |
| ('multi', 'gold')                    |   3 |            39 |              12 |     0.0002 |             186 |                 0 |       0      |
| ('same_mod', 'coverage')             |   3 |             3 |               0 |     0.25   |               5 |                 0 |       0.0625 |
| ('same_mod', 'coverage_mod')         |   3 |             2 |               2 |     1      |               7 |                 2 |       0.1797 |
| ('same_mod', 'elem_topk')            |   3 |             3 |               5 |     0.7266 |               7 |                 9 |       0.8036 |
| ('same_mod', 'elem_coverage')        |   3 |             5 |               5 |     1      |              10 |                 7 |       0.6291 |
| ('same_mod', 'elem_coverage_mod')    |   3 |             5 |               6 |     1      |               8 |                 9 |       1      |
| ('same_mod', 'evidence')             |   3 |             9 |               6 |     0.6072 |              12 |                10 |       0.8318 |
| ('same_mod', 'gold')                 |   3 |            25 |               8 |     0.0046 |             128 |                 0 |       0      |
| ('cross_multi', 'coverage')          |   3 |             2 |               0 |     0.5    |               4 |                 1 |       0.375  |
| ('cross_multi', 'coverage_mod')      |   3 |             2 |               0 |     0.5    |               3 |                 2 |       1      |
| ('cross_multi', 'elem_topk')         |   3 |             1 |               5 |     0.2188 |               2 |                 7 |       0.1797 |
| ('cross_multi', 'elem_coverage')     |   3 |             2 |               5 |     0.4531 |               5 |                 6 |       1      |
| ('cross_multi', 'elem_coverage_mod') |   3 |             1 |               4 |     0.375  |               4 |                 8 |       0.3877 |
| ('cross_multi', 'evidence')          |   3 |             4 |               4 |     1      |              13 |                 8 |       0.3833 |
| ('cross_multi', 'gold')              |   3 |            14 |               4 |     0.0309 |              58 |                 0 |       0      |
| ('control', 'coverage')              |   3 |             2 |               2 |     1      |               5 |                 0 |       0.0625 |
| ('control', 'coverage_mod')          |   3 |             3 |               2 |     1      |               6 |                 0 |       0.0312 |
| ('control', 'elem_topk')             |   3 |            16 |               7 |     0.0931 |              15 |                 8 |       0.21   |
| ('control', 'elem_coverage')         |   3 |            14 |               7 |     0.1892 |              16 |                 8 |       0.1516 |
| ('control', 'elem_coverage_mod')     |   3 |            17 |               6 |     0.0347 |              19 |                 8 |       0.0522 |
| ('control', 'evidence')              |   3 |            23 |               6 |     0.0023 |              27 |                11 |       0.0139 |
| ('control', 'gold')                  |   3 |            54 |              20 |     0.0001 |              80 |                 0 |       0      |

## Failure decomposition, topk, multi, k = 5

| outcome   |   n |   correct |   wrong |   wrong_fixed_by_gold |
|:----------|----:|----------:|--------:|----------------------:|
| complete  | 159 |        56 |     103 |                    11 |
| partial   | 108 |        15 |      93 |                    18 |
| none      |  43 |         4 |      39 |                     7 |

## Failure decomposition, evidence, multi, k = 5

| outcome   |   n |   correct |   wrong |   wrong_fixed_by_gold |
|:----------|----:|----------:|--------:|----------------------:|
| complete  | 167 |        61 |     106 |                    16 |
| partial   | 105 |        19 |      86 |                    10 |
| none      |  38 |         3 |      35 |                     7 |

## By stratum (multi, k = 5; complete-evidence recall and accuracy, %)

| stratum     |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:------------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| same_mod    | 201 |            45.8 |       22.9 |                47.8 |           22.9 |                47.8 |           26.4 |       30.8 |
| cross_multi | 109 |            61.5 |       26.6 |                62.4 |           27.5 |                65.1 |           27.5 |       36.7 |

## By needs (multi, k = 5; complete-evidence recall and accuracy, %)

|   needs |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|--------:|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
|       1 | 173 |            39.9 |       22.5 |                39.9 |           22.5 |                39.9 |           27.7 |       34.7 |
|       2 | 117 |            66.7 |       23.9 |                70.9 |           24.8 |                73.5 |           23.1 |       29.9 |
|       3 |  20 |            60   |       40   |                60   |           40   |                60   |           40   |       35   |

## By qtype (multi, k = 5; complete-evidence recall and accuracy, %)

| qtype           |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:----------------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| count           | 125 |            48.8 |       24   |                50.4 |           23.2 |                48.8 |           24   |       28.8 |
| lookup          |  86 |            52.3 |       32.6 |                52.3 |           33.7 |                54.7 |           34.9 |       39.5 |
| list            |  53 |            56.6 |       17   |                60.4 |           18.9 |                64.2 |           26.4 |       41.5 |
| compare/compute |  46 |            50   |       17.4 |                52.2 |           17.4 |                54.3 |           19.6 |       21.7 |

## By gap (multi, k = 5; complete-evidence recall and accuracy, %)

| gap      |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:---------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| adjacent | 143 |            62.2 |       26.6 |                64.3 |           27.3 |                62.2 |           30.1 |       35.7 |
| gap >=5  |  88 |            37.5 |       17   |                38.6 |           15.9 |                45.5 |           21.6 |       23.9 |
| gap 2-4  |  79 |            46.8 |       27.8 |                48.1 |           29.1 |                48.1 |           26.6 |       38   |

## By n_gold (multi, k = 5; complete-evidence recall and accuracy, %)

|   n_gold |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|---------:|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
|        2 | 235 |            60   |       25.1 |                62.1 |           25.5 |                63   |           26.4 |       31.1 |
|        3 |  40 |            35   |       32.5 |                35   |           32.5 |                40   |           37.5 |       47.5 |
|        4 |  21 |            14.3 |        4.8 |                14.3 |            4.8 |                14.3 |           19   |       28.6 |
|        5 |  14 |             7.1 |       14.3 |                 7.1 |           14.3 |                 0   |           14.3 |       28.6 |

## By combo (multi, k = 5; complete-evidence recall and accuracy, %)

| combo                |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:---------------------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| Table                |  64 |            28.1 |        9.4 |                29.7 |            9.4 |                31.2 |            9.4 |       17.2 |
| Figure               |  59 |            47.5 |       28.8 |                49.2 |           27.1 |                50.8 |           30.5 |       33.9 |
| Text                 |  41 |            56.1 |       29.3 |                58.5 |           31.7 |                56.1 |           41.5 |       41.5 |
| Chart                |  27 |            63   |       25.9 |                66.7 |           25.9 |                66.7 |           29.6 |       25.9 |
| other                |  25 |            44   |       16   |                44   |           16   |                52   |           16   |       28   |
| Chart + Text         |  21 |            52.4 |       14.3 |                52.4 |           14.3 |                47.6 |           14.3 |       14.3 |
| Table + Text         |  17 |            64.7 |       23.5 |                64.7 |           23.5 |                76.5 |           29.4 |       35.3 |
| Figure + Layout      |  15 |            80   |       40   |                80   |           40   |                73.3 |           33.3 |       53.3 |
| Figure + Text        |  11 |            81.8 |       72.7 |                81.8 |           72.7 |                90.9 |           72.7 |       81.8 |
| Layout               |  10 |            60   |       40   |                60   |           40   |                50   |           40   |       70   |
| Layout + Text        |   9 |            66.7 |       22.2 |                66.7 |           22.2 |                66.7 |           22.2 |       44.4 |
| Chart + Table + Text |   6 |            33.3 |       16.7 |                50   |           33.3 |                50   |           16.7 |       16.7 |
| Figure + Table       |   5 |           100   |       20   |               100   |           20   |               100   |           40   |       40   |

## By answer_format (multi, k = 5; complete-evidence recall and accuracy, %)

| answer_format   |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:----------------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| Int             | 128 |            52.3 |       26.6 |                53.1 |           25.8 |                50.8 |           25.8 |       29.7 |
| Float           |  63 |            31.7 |        6.3 |                36.5 |            7.9 |                38.1 |           11.1 |       12.7 |
| List            |  62 |            56.5 |       17.7 |                59.7 |           19.4 |                66.1 |           25.8 |       41.9 |
| Str             |  57 |            64.9 |       45.6 |                63.2 |           45.6 |                64.9 |           47.4 |       52.6 |

## By doc_type (multi, k = 5; complete-evidence recall and accuracy, %)

| doc_type                       |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:-------------------------------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| Research report / Introduction |  85 |            55.3 |       17.6 |                58.8 |           20   |                61.2 |           23.5 |       27.1 |
| Academic paper                 |  62 |            54.8 |       27.4 |                58.1 |           25.8 |                64.5 |           27.4 |       29   |
| Financial report               |  43 |            20.9 |        0   |                20.9 |            0   |                20.9 |            4.7 |        7   |
| Tutorial/Workshop              |  40 |            60   |       52.5 |                60   |           52.5 |                57.5 |           55   |       62.5 |
| Guidebook                      |  38 |            63.2 |       39.5 |                63.2 |           39.5 |                60.5 |           39.5 |       47.4 |
| Brochure                       |  26 |            50   |       11.5 |                50   |           11.5 |                42.3 |           15.4 |       34.6 |
| Administration/Industry file   |  16 |            50   |       25   |                50   |           25   |                56.2 |           18.8 |       37.5 |

## By Analysis 2 category (annotated questions in multi only)

| category    |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   evidence complete |   evidence acc |   gold acc |
|:------------|----:|----------------:|-----------:|--------------------:|---------------:|--------------------:|---------------:|-----------:|
| composition |   8 |            62.5 |       12.5 |                62.5 |           12.5 |                62.5 |           12.5 |       25   |
| partition   |   7 |            85.7 |       28.6 |                85.7 |           28.6 |                85.7 |           28.6 |       28.6 |
| redundant   |   2 |           100   |       50   |               100   |           50   |               100   |           50   |      100   |

## Gold pages by evidence form (multi, k = 5): how often each method misses them

|                             |   pages |   median ColQwen rank |   topk missed % |   coverage missed % |   evidence missed % |
|:----------------------------|--------:|----------------------:|----------------:|--------------------:|--------------------:|
| ('cross_multi', 'Chart')    |      37 |                     2 |            18.9 |                21.6 |                18.9 |
| ('cross_multi', 'Figure')   |      47 |                     2 |            17   |                17   |                21.3 |
| ('cross_multi', 'Layout')   |      20 |                     2 |            20   |                20   |                15   |
| ('cross_multi', 'Table')    |      59 |                     3 |            30.5 |                27.1 |                18.6 |
| ('cross_multi', 'Text')     |      82 |                     3 |            30.5 |                31.7 |                29.3 |
| ('cross_multi', 'untagged') |       3 |                     1 |             0   |                 0   |                 0   |
| ('same_mod', 'Chart')       |      66 |                     3 |            34.8 |                33.3 |                28.8 |
| ('same_mod', 'Figure')      |     161 |                     3 |            36   |                35.4 |                35.4 |
| ('same_mod', 'Layout')      |      29 |                     3 |            27.6 |                27.6 |                34.5 |
| ('same_mod', 'Table')       |     136 |                     6 |            52.9 |                51.5 |                49.3 |
| ('same_mod', 'Text')        |     104 |                     3 |            29.8 |                28.8 |                27.9 |

## Cross-modal questions retrieved in part by top-k: form of found -> missed page

| found -> missed      |   pairs |
|:---------------------|--------:|
| ('Text', 'Text')     |      19 |
| ('Table', 'Table')   |      11 |
| ('Chart', 'Chart')   |       6 |
| ('Figure', 'Layout') |       5 |
| ('Layout', 'Layout') |       4 |
| ('Chart', 'Text')    |       4 |
| ('Chart', 'Table')   |       2 |
| ('Text', 'Table')    |       2 |
| ('Table', 'Text')    |       2 |
| ('Text', 'Figure')   |       2 |
| ('Figure', 'Text')   |       1 |
| ('Text', 'Chart')    |       1 |
| ('Layout', 'Figure') |       1 |
| ('Layout', 'Table')  |       1 |

## Decomposition diagnostics (multi)

```
{
 "n": 310,
 "unparsed": 0,
 "needs_hist": {
  "1": 173,
  "2": 117,
  "3": 17,
  "4": 3
 },
 "gold_pages_ranked_first_by_some_need_pct": 39.0,
 "all_gold_ranked_first_by_needs_pct": 14.5,
 "question_top1_in_gold_pct": 63.5
}
```

## Need forms and page forms (multi)

```
{
 "n": 310,
 "unparsed": 4,
 "all_needs_any_pct": 1.9,
 "predicted_forms_include_a_labelled_source_pct": 72.3,
 "predicted_forms_include_every_labelled_source_pct": 48.1,
 "gold_pages_holding_an_expected_form_pct": 79.1
}
```

## VLM aggregation (aggregate.py) and the evidence selector (multi)

```
{
 "n": 310,
 "model": "Qwen3-VL-8B-Instruct",
 "unparsed": 18,
 "facts_per_question": 1.97,
 "no_fact_pct": 25.2,
 "with_joint_fact_pct": 9.0,
 "with_cross_page_joint_fact_pct": 6.5,
 "with_redundant_copy_pct": 1.0,
 "with_missing_need_pct": 8.1,
 "gold_recall_of_candidate_pages": 82.6,
 "gold_recall_of_fact_pages": 44.1,
 "all_gold_in_candidate_pages_pct": 70.0,
 "all_gold_in_fact_pages_pct": 26.1
}
```

## Questions with VLM facts and those that fall back to elem_coverage_mod (k = 5, %)

|                                   |   ('n', 'coverage') |   ('n', 'elem_coverage_mod') |   ('n', 'evidence') |   ('complete', 'coverage') |   ('complete', 'elem_coverage_mod') |   ('complete', 'evidence') |   ('acc', 'coverage') |   ('acc', 'elem_coverage_mod') |   ('acc', 'evidence') |
|:----------------------------------|--------------------:|-----------------------------:|--------------------:|---------------------------:|------------------------------------:|---------------------------:|----------------------:|-------------------------------:|----------------------:|
| ('control', 'fallback (no fact)') |                  56 |                           56 |                  56 |                       66.1 |                                64.3 |                       64.3 |                  37.5 |                           39.3 |                  39.3 |
| ('control', 'with facts')         |                 397 |                          397 |                 397 |                       89.4 |                                90.4 |                       91.7 |                  56.2 |                           54.7 |                  56.4 |
| ('multi', 'fallback (no fact)')   |                  78 |                           78 |                  78 |                       33.3 |                                32.1 |                       32.1 |                  12.8 |                           10.3 |                  10.3 |
| ('multi', 'with facts')           |                 232 |                          232 |                 232 |                       59.5 |                                58.6 |                       61.2 |                  28.4 |                           30.6 |                  32.3 |

## Multi-page questions without a VLM fact, by doc_type

| doc_type                       |   n |   no_fact_pct |
|:-------------------------------|----:|--------------:|
| Research report / Introduction |  85 |          16.5 |
| Academic paper                 |  62 |          25.8 |
| Financial report               |  43 |          48.8 |
| Tutorial/Workshop              |  40 |          17.5 |
| Guidebook                      |  38 |          31.6 |
| Brochure                       |  26 |          26.9 |
| Administration/Industry file   |  16 |           6.2 |

## Multi-page questions without a VLM fact, by qtype

| qtype           |   n |   no_fact_pct |
|:----------------|----:|--------------:|
| count           | 125 |          21.6 |
| lookup          |  86 |          30.2 |
| list            |  53 |          18.9 |
| compare/compute |  46 |          32.6 |

## Multi-page questions without a VLM fact, by needs

|   needs |   n |   no_fact_pct |
|--------:|----:|--------------:|
|       1 | 173 |          26.6 |
|       2 | 117 |          22.2 |
|       3 |  20 |          30   |
