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
| ('coverage_mod_beam', 3) | 310 |        310 |       41.6 |     60.8 |  24.8 |    24.4 |
| ('coverage_mod_beam', 5) | 310 |        310 |       52.9 |     70.4 |  24.2 |    23.8 |
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
| ('coverage_mod_beam', 3) | 201 |        201 |       38.3 |     56.4 |  22.4 |    21.8 |
| ('coverage_mod_beam', 5) | 201 |        201 |       48.3 |     65.8 |  22.4 |    21.8 |
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
| ('coverage_mod_beam', 3) | 109 |        109 |       47.7 |     69   |  29.4 |    29.2 |
| ('coverage_mod_beam', 5) | 109 |        109 |       61.5 |     78.7 |  27.5 |    27.5 |
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
| ('coverage_mod_beam', 3) | 453 |        453 |       83.7 |     83.7 |  52.5 |    51.4 |
| ('coverage_mod_beam', 5) | 453 |        453 |       86.8 |     86.8 |  53.9 |    53   |
| ('gold', 0)              | 453 |        453 |      100   |    100   |  59.8 |    59   |

## Paired against top-k (k = 5): wins / losses, exact McNemar p

|                                      |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:-------------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'coverage')                |   5 |             2 |               1 |     1      |               6 |                 1 |       0.125  |
| ('multi', 'coverage_mod')            |   5 |             2 |               2 |     1      |               6 |                 1 |       0.125  |
| ('multi', 'coverage_mod_beam')       |   5 |             2 |               2 |     1      |               6 |                 1 |       0.125  |
| ('multi', 'gold')                    |   5 |            36 |               9 |     0.0001 |             151 |                 0 |       0      |
| ('same_mod', 'coverage')             |   5 |             1 |               1 |     1      |               4 |                 0 |       0.125  |
| ('same_mod', 'coverage_mod')         |   5 |             1 |               2 |     1      |               5 |                 0 |       0.0625 |
| ('same_mod', 'coverage_mod_beam')    |   5 |             1 |               2 |     1      |               5 |                 0 |       0.0625 |
| ('same_mod', 'gold')                 |   5 |            24 |               8 |     0.007  |             109 |                 0 |       0      |
| ('cross_multi', 'coverage')          |   5 |             1 |               0 |     1      |               2 |                 1 |       1      |
| ('cross_multi', 'coverage_mod')      |   5 |             1 |               0 |     1      |               1 |                 1 |       1      |
| ('cross_multi', 'coverage_mod_beam') |   5 |             1 |               0 |     1      |               1 |                 1 |       1      |
| ('cross_multi', 'gold')              |   5 |            12 |               1 |     0.0034 |              42 |                 0 |       0      |
| ('control', 'coverage')              |   5 |             2 |               0 |     0.5    |               2 |                 1 |       1      |
| ('control', 'coverage_mod')          |   5 |             3 |               1 |     0.625  |               3 |                 1 |       0.625  |
| ('control', 'coverage_mod_beam')     |   5 |             3 |               1 |     0.625  |               3 |                 1 |       0.625  |
| ('control', 'gold')                  |   5 |            50 |              21 |     0.0008 |              62 |                 0 |       0      |

## Paired against coverage (k = 5): wins / losses, exact McNemar p

|                                      |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:-------------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'topk')                    |   5 |             1 |               2 |     1      |               1 |                 6 |        0.125 |
| ('multi', 'coverage_mod')            |   5 |             0 |               1 |     1      |               1 |                 1 |        1     |
| ('multi', 'coverage_mod_beam')       |   5 |             0 |               1 |     1      |               1 |                 1 |        1     |
| ('multi', 'gold')                    |   5 |            34 |               8 |     0.0001 |             146 |                 0 |        0     |
| ('same_mod', 'topk')                 |   5 |             1 |               1 |     1      |               0 |                 4 |        0.125 |
| ('same_mod', 'coverage_mod')         |   5 |             0 |               1 |     1      |               1 |                 0 |        1     |
| ('same_mod', 'coverage_mod_beam')    |   5 |             0 |               1 |     1      |               1 |                 0 |        1     |
| ('same_mod', 'gold')                 |   5 |            23 |               7 |     0.0052 |             105 |                 0 |        0     |
| ('cross_multi', 'topk')              |   5 |             0 |               1 |     1      |               1 |                 2 |        1     |
| ('cross_multi', 'coverage_mod')      |   5 |             0 |               0 |     1      |               0 |                 1 |        1     |
| ('cross_multi', 'coverage_mod_beam') |   5 |             0 |               0 |     1      |               0 |                 1 |        1     |
| ('cross_multi', 'gold')              |   5 |            11 |               1 |     0.0063 |              41 |                 0 |        0     |
| ('control', 'topk')                  |   5 |             0 |               2 |     0.5    |               1 |                 2 |        1     |
| ('control', 'coverage_mod')          |   5 |             1 |               1 |     1      |               1 |                 0 |        1     |
| ('control', 'coverage_mod_beam')     |   5 |             1 |               1 |     1      |               1 |                 0 |        1     |
| ('control', 'gold')                  |   5 |            49 |              22 |     0.0018 |              61 |                 0 |        0     |

## Paired against top-k (k = 3): wins / losses, exact McNemar p

|                                      |   k |   answer_wins |   answer_losses |   answer_p |   complete_wins |   complete_losses |   complete_p |
|:-------------------------------------|----:|--------------:|----------------:|-----------:|----------------:|------------------:|-------------:|
| ('multi', 'coverage')                |   3 |             5 |               0 |     0.0625 |               9 |                 1 |       0.0215 |
| ('multi', 'coverage_mod')            |   3 |             4 |               2 |     0.6875 |              10 |                 4 |       0.1796 |
| ('multi', 'coverage_mod_beam')       |   3 |             4 |               2 |     0.6875 |              10 |                 5 |       0.3018 |
| ('multi', 'gold')                    |   3 |            39 |              12 |     0.0002 |             186 |                 0 |       0      |
| ('same_mod', 'coverage')             |   3 |             3 |               0 |     0.25   |               5 |                 0 |       0.0625 |
| ('same_mod', 'coverage_mod')         |   3 |             2 |               2 |     1      |               7 |                 2 |       0.1797 |
| ('same_mod', 'coverage_mod_beam')    |   3 |             2 |               2 |     1      |               7 |                 3 |       0.3438 |
| ('same_mod', 'gold')                 |   3 |            25 |               8 |     0.0046 |             128 |                 0 |       0      |
| ('cross_multi', 'coverage')          |   3 |             2 |               0 |     0.5    |               4 |                 1 |       0.375  |
| ('cross_multi', 'coverage_mod')      |   3 |             2 |               0 |     0.5    |               3 |                 2 |       1      |
| ('cross_multi', 'coverage_mod_beam') |   3 |             2 |               0 |     0.5    |               3 |                 2 |       1      |
| ('cross_multi', 'gold')              |   3 |            14 |               4 |     0.0309 |              58 |                 0 |       0      |
| ('control', 'coverage')              |   3 |             2 |               2 |     1      |               5 |                 0 |       0.0625 |
| ('control', 'coverage_mod')          |   3 |             3 |               2 |     1      |               6 |                 0 |       0.0312 |
| ('control', 'coverage_mod_beam')     |   3 |             3 |               2 |     1      |               6 |                 0 |       0.0312 |
| ('control', 'gold')                  |   3 |            54 |              20 |     0.0001 |              80 |                 0 |       0      |

## Failure decomposition, topk, multi, k = 5

| outcome   |   n |   correct |   wrong |   wrong_fixed_by_gold |
|:----------|----:|----------:|--------:|----------------------:|
| complete  | 159 |        56 |     103 |                    11 |
| partial   | 108 |        15 |      93 |                    18 |
| none      |  43 |         4 |      39 |                     7 |

## Failure decomposition, coverage_mod, multi, k = 5

| outcome   |   n |   correct |   wrong |   wrong_fixed_by_gold |
|:----------|----:|----------:|--------:|----------------------:|
| complete  | 164 |        58 |     106 |                    11 |
| partial   | 104 |        13 |      91 |                    18 |
| none      |  42 |         4 |      38 |                     6 |

## By stratum (multi, k = 5; complete-evidence recall and accuracy, %)

| stratum     |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:------------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| same_mod    | 201 |            45.8 |       22.9 |                47.8 |           22.9 |                    48.3 |               22.4 |       30.8 |
| cross_multi | 109 |            61.5 |       26.6 |                62.4 |           27.5 |                    61.5 |               27.5 |       36.7 |

## By needs (multi, k = 5; complete-evidence recall and accuracy, %)

|   needs |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|--------:|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
|       1 | 173 |            39.9 |       22.5 |                39.9 |           22.5 |                    39.9 |               22.5 |       34.7 |
|       2 | 117 |            66.7 |       23.9 |                70.9 |           24.8 |                    70.1 |               24.8 |       29.9 |
|       3 |  20 |            60   |       40   |                60   |           40   |                    65   |               35   |       35   |

## By qtype (multi, k = 5; complete-evidence recall and accuracy, %)

| qtype           |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:----------------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| count           | 125 |            48.8 |       24   |                50.4 |           23.2 |                    51.2 |               22.4 |       28.8 |
| lookup          |  86 |            52.3 |       32.6 |                52.3 |           33.7 |                    52.3 |               33.7 |       39.5 |
| list            |  53 |            56.6 |       17   |                60.4 |           18.9 |                    58.5 |               18.9 |       41.5 |
| compare/compute |  46 |            50   |       17.4 |                52.2 |           17.4 |                    52.2 |               17.4 |       21.7 |

## By gap (multi, k = 5; complete-evidence recall and accuracy, %)

| gap      |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:---------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| adjacent | 143 |            62.2 |       26.6 |                64.3 |           27.3 |                    64.3 |               26.6 |       35.7 |
| gap >=5  |  88 |            37.5 |       17   |                38.6 |           15.9 |                    37.5 |               15.9 |       23.9 |
| gap 2-4  |  79 |            46.8 |       27.8 |                48.1 |           29.1 |                    49.4 |               29.1 |       38   |

## By n_gold (multi, k = 5; complete-evidence recall and accuracy, %)

|   n_gold |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|---------:|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
|        2 | 235 |            60   |       25.1 |                62.1 |           25.5 |                    62.1 |               25.5 |       31.1 |
|        3 |  40 |            35   |       32.5 |                35   |           32.5 |                    35   |               30   |       47.5 |
|        4 |  21 |            14.3 |        4.8 |                14.3 |            4.8 |                    14.3 |                4.8 |       28.6 |
|        5 |  14 |             7.1 |       14.3 |                 7.1 |           14.3 |                     7.1 |               14.3 |       28.6 |

## By combo (multi, k = 5; complete-evidence recall and accuracy, %)

| combo                |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:---------------------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| Table                |  64 |            28.1 |        9.4 |                29.7 |            9.4 |                    29.7 |                9.4 |       17.2 |
| Figure               |  59 |            47.5 |       28.8 |                49.2 |           27.1 |                    49.2 |               27.1 |       33.9 |
| Text                 |  41 |            56.1 |       29.3 |                58.5 |           31.7 |                    58.5 |               29.3 |       41.5 |
| Chart                |  27 |            63   |       25.9 |                66.7 |           25.9 |                    70.4 |               25.9 |       25.9 |
| other                |  25 |            44   |       16   |                44   |           16   |                    40   |               16   |       28   |
| Chart + Text         |  21 |            52.4 |       14.3 |                52.4 |           14.3 |                    52.4 |               14.3 |       14.3 |
| Table + Text         |  17 |            64.7 |       23.5 |                64.7 |           23.5 |                    64.7 |               23.5 |       35.3 |
| Figure + Layout      |  15 |            80   |       40   |                80   |           40   |                    80   |               40   |       53.3 |
| Figure + Text        |  11 |            81.8 |       72.7 |                81.8 |           72.7 |                    81.8 |               72.7 |       81.8 |
| Layout               |  10 |            60   |       40   |                60   |           40   |                    60   |               40   |       70   |
| Layout + Text        |   9 |            66.7 |       22.2 |                66.7 |           22.2 |                    66.7 |               22.2 |       44.4 |
| Chart + Table + Text |   6 |            33.3 |       16.7 |                50   |           33.3 |                    50   |               33.3 |       16.7 |
| Figure + Table       |   5 |           100   |       20   |               100   |           20   |                   100   |               20   |       40   |

## By answer_format (multi, k = 5; complete-evidence recall and accuracy, %)

| answer_format   |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:----------------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| Int             | 128 |            52.3 |       26.6 |                53.1 |           25.8 |                    53.1 |               25   |       29.7 |
| Float           |  63 |            31.7 |        6.3 |                36.5 |            7.9 |                    38.1 |                7.9 |       12.7 |
| List            |  62 |            56.5 |       17.7 |                59.7 |           19.4 |                    58.1 |               19.4 |       41.9 |
| Str             |  57 |            64.9 |       45.6 |                63.2 |           45.6 |                    63.2 |               45.6 |       52.6 |

## By doc_type (multi, k = 5; complete-evidence recall and accuracy, %)

| doc_type                       |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:-------------------------------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| Research report / Introduction |  85 |            55.3 |       17.6 |                58.8 |           20   |                    58.8 |               20   |       27.1 |
| Academic paper                 |  62 |            54.8 |       27.4 |                58.1 |           25.8 |                    58.1 |               25.8 |       29   |
| Financial report               |  43 |            20.9 |        0   |                20.9 |            0   |                    20.9 |                0   |        7   |
| Tutorial/Workshop              |  40 |            60   |       52.5 |                60   |           52.5 |                    60   |               52.5 |       62.5 |
| Guidebook                      |  38 |            63.2 |       39.5 |                63.2 |           39.5 |                    63.2 |               39.5 |       47.4 |
| Brochure                       |  26 |            50   |       11.5 |                50   |           11.5 |                    50   |               11.5 |       34.6 |
| Administration/Industry file   |  16 |            50   |       25   |                50   |           25   |                    50   |               18.8 |       37.5 |

## By Analysis 2 category (annotated questions in multi only)

| category    |   n |   topk complete |   topk acc |   coverage complete |   coverage acc |   coverage_mod complete |   coverage_mod acc |   gold acc |
|:------------|----:|----------------:|-----------:|--------------------:|---------------:|------------------------:|-------------------:|-----------:|
| composition |   8 |            62.5 |       12.5 |                62.5 |           12.5 |                    62.5 |               12.5 |       25   |
| partition   |   7 |            85.7 |       28.6 |                85.7 |           28.6 |                    85.7 |               28.6 |       28.6 |
| redundant   |   2 |           100   |       50   |               100   |           50   |                   100   |               50   |      100   |

## Gold pages by evidence form (multi, k = 5): how often each method misses them

|                             |   pages |   median ColQwen rank |   topk missed % |   coverage missed % |   coverage_mod missed % |
|:----------------------------|--------:|----------------------:|----------------:|--------------------:|------------------------:|
| ('cross_multi', 'Chart')    |      37 |                     2 |            18.9 |                21.6 |                    21.6 |
| ('cross_multi', 'Figure')   |      47 |                     2 |            17   |                17   |                    17   |
| ('cross_multi', 'Layout')   |      20 |                     2 |            20   |                20   |                    20   |
| ('cross_multi', 'Table')    |      59 |                     3 |            30.5 |                27.1 |                    25.4 |
| ('cross_multi', 'Text')     |      82 |                     3 |            30.5 |                31.7 |                    31.7 |
| ('cross_multi', 'untagged') |       3 |                     1 |             0   |                 0   |                     0   |
| ('same_mod', 'Chart')       |      66 |                     3 |            34.8 |                33.3 |                    31.8 |
| ('same_mod', 'Figure')      |     161 |                     3 |            36   |                35.4 |                    35.4 |
| ('same_mod', 'Layout')      |      29 |                     3 |            27.6 |                27.6 |                    27.6 |
| ('same_mod', 'Table')       |     136 |                     6 |            52.9 |                51.5 |                    50.7 |
| ('same_mod', 'Text')        |     104 |                     3 |            29.8 |                28.8 |                    29.8 |

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
