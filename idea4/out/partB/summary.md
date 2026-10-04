# Part B summary

144 questions, retrieval within the document (median 23 pages). Random-ranking MRR: 0.146.
Retriever: ColQwen2.5 (vidore/colqwen2.5-v0.2). Proposer: Claude Opus 5.5 (fresh context, question + inventory only). Generator: Qwen-Image-2.1.

## all pool questions

| variant | type | n | R@1 | R@5 | MRR | median rank |
|---|---|---|---|---|---|---|
| Question (text) | chart | 34 | 0.85 | 0.91 | 0.885 | 1 |
| Question (text) | table | 37 | 0.59 | 0.73 | 0.676 | 1 |
| Question (text) | text | 37 | 0.84 | 0.92 | 0.885 | 1 |
| Question (text) | figure | 31 | 0.77 | 0.90 | 0.832 | 1 |
| Question (text) | all | 144 | 0.75 | 0.85 | 0.806 | 1 |
| Spec as text | chart | 34 | 0.85 | 0.91 | 0.891 | 1 |
| Spec as text | table | 37 | 0.73 | 0.95 | 0.810 | 1 |
| Spec as text | text | 37 | 0.78 | 0.92 | 0.860 | 1 |
| Spec as text | figure | 31 | 0.74 | 0.84 | 0.796 | 1 |
| Spec as text | all | 144 | 0.76 | 0.90 | 0.829 | 1 |
| Spec drawn (code) | chart | 34 | 0.85 | 0.94 | 0.886 | 1 |
| Spec drawn (code) | table | 37 | 0.68 | 0.92 | 0.787 | 1 |
| Spec drawn (code) | text | 37 | 0.78 | 0.97 | 0.874 | 1 |
| Spec drawn (code) | figure | 31 | 0.74 | 0.84 | 0.797 | 1 |
| Spec drawn (code) | all | 144 | 0.76 | 0.91 | 0.829 | 1 |
| Spec generated (Qwen-Image) | chart | 34 | 0.88 | 0.97 | 0.913 | 1 |
| Spec generated (Qwen-Image) | table | 37 | 0.70 | 0.89 | 0.789 | 1 |
| Spec generated (Qwen-Image) | text | 37 | 0.62 | 0.89 | 0.753 | 1 |
| Spec generated (Qwen-Image) | figure | 31 | 0.55 | 0.74 | 0.654 | 1 |
| Spec generated (Qwen-Image) | all | 144 | 0.67 | 0.86 | 0.765 | 1 |
| Question + spec as text | chart | 34 | 0.85 | 0.94 | 0.904 | 1 |
| Question + spec as text | table | 37 | 0.68 | 0.86 | 0.777 | 1 |
| Question + spec as text | text | 37 | 0.86 | 0.97 | 0.906 | 1 |
| Question + spec as text | figure | 31 | 0.74 | 0.87 | 0.803 | 1 |
| Question + spec as text | all | 144 | 0.77 | 0.90 | 0.836 | 1 |
| Question + code drawing | chart | 34 | 0.85 | 0.94 | 0.906 | 1 |
| Question + code drawing | table | 37 | 0.65 | 0.97 | 0.786 | 1 |
| Question + code drawing | text | 37 | 0.89 | 0.97 | 0.920 | 1 |
| Question + code drawing | figure | 31 | 0.74 | 0.87 | 0.808 | 1 |
| Question + code drawing | all | 144 | 0.78 | 0.93 | 0.849 | 1 |
| Question + generated image | chart | 34 | 0.85 | 0.94 | 0.901 | 1 |
| Question + generated image | table | 37 | 0.70 | 0.92 | 0.798 | 1 |
| Question + generated image | text | 37 | 0.70 | 0.95 | 0.817 | 1 |
| Question + generated image | figure | 31 | 0.74 | 0.87 | 0.804 | 1 |
| Question + generated image | all | 144 | 0.74 | 0.90 | 0.817 | 1 |
| Question as image | chart | 34 | 0.82 | 0.94 | 0.870 | 1 |
| Question as image | table | 37 | 0.57 | 0.81 | 0.662 | 1 |
| Question as image | text | 37 | 0.81 | 0.86 | 0.842 | 1 |
| Question as image | figure | 31 | 0.65 | 0.84 | 0.756 | 1 |
| Question as image | all | 144 | 0.70 | 0.86 | 0.774 | 1 |

Paired against the question-as-text baseline (all types):

| variant | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |
|---|---|---|---|---|
| Spec as text | 144 | +0.023 [-0.029, +0.070] | 25 / 100 / 19 | 0.138 |
| Spec drawn (code) | 144 | +0.024 [-0.031, +0.076] | 28 / 93 / 23 | 0.138 |
| Spec generated (Qwen-Image) | 144 | -0.041 [-0.107, +0.018] | 25 / 86 / 33 | 0.957 |
| Question + spec as text | 144 | +0.030 [-0.007, +0.066] | 24 / 106 / 14 | 0.00999 |
| Question + code drawing | 144 | +0.043 [+0.002, +0.084] | 27 / 102 / 15 | 0.00715 |
| Question + generated image | 144 | +0.011 [-0.031, +0.052] | 25 / 100 / 19 | 0.0249 |
| Question as image | 144 | -0.032 [-0.061, -0.006] | 16 / 103 / 25 | 0.292 |

## proposer drew (render = true)

| variant | type | n | R@1 | R@5 | MRR | median rank |
|---|---|---|---|---|---|---|
| Question (text) | chart | 34 | 0.85 | 0.91 | 0.885 | 1 |
| Question (text) | table | 36 | 0.58 | 0.72 | 0.667 | 1 |
| Question (text) | text | 17 | 0.82 | 0.88 | 0.864 | 1 |
| Question (text) | figure | 31 | 0.77 | 0.90 | 0.832 | 1 |
| Question (text) | all | 122 | 0.73 | 0.83 | 0.787 | 1 |
| Spec as text | chart | 34 | 0.85 | 0.91 | 0.891 | 1 |
| Spec as text | table | 36 | 0.72 | 0.94 | 0.805 | 1 |
| Spec as text | text | 17 | 0.71 | 0.88 | 0.808 | 1 |
| Spec as text | figure | 31 | 0.74 | 0.84 | 0.796 | 1 |
| Spec as text | all | 122 | 0.75 | 0.89 | 0.813 | 1 |
| Spec drawn (code) | chart | 34 | 0.85 | 0.94 | 0.886 | 1 |
| Spec drawn (code) | table | 36 | 0.67 | 0.92 | 0.781 | 1 |
| Spec drawn (code) | text | 17 | 0.71 | 0.94 | 0.814 | 1 |
| Spec drawn (code) | figure | 31 | 0.74 | 0.84 | 0.797 | 1 |
| Spec drawn (code) | all | 122 | 0.74 | 0.89 | 0.811 | 1 |
| Spec generated (Qwen-Image) | chart | 34 | 0.88 | 0.97 | 0.913 | 1 |
| Spec generated (Qwen-Image) | table | 36 | 0.69 | 0.89 | 0.783 | 1 |
| Spec generated (Qwen-Image) | text | 17 | 0.47 | 0.82 | 0.652 | 2 |
| Spec generated (Qwen-Image) | figure | 31 | 0.55 | 0.74 | 0.654 | 1 |
| Spec generated (Qwen-Image) | all | 122 | 0.66 | 0.84 | 0.753 | 1 |
| Question + spec as text | chart | 34 | 0.85 | 0.94 | 0.904 | 1 |
| Question + spec as text | table | 36 | 0.67 | 0.86 | 0.771 | 1 |
| Question + spec as text | text | 17 | 0.88 | 0.94 | 0.908 | 1 |
| Question + spec as text | figure | 31 | 0.74 | 0.87 | 0.803 | 1 |
| Question + spec as text | all | 122 | 0.75 | 0.89 | 0.822 | 1 |
| Question + code drawing | chart | 34 | 0.85 | 0.94 | 0.906 | 1 |
| Question + code drawing | table | 36 | 0.64 | 0.97 | 0.781 | 1 |
| Question + code drawing | text | 17 | 0.82 | 0.94 | 0.873 | 1 |
| Question + code drawing | figure | 31 | 0.74 | 0.87 | 0.808 | 1 |
| Question + code drawing | all | 122 | 0.75 | 0.92 | 0.828 | 1 |
| Question + generated image | chart | 34 | 0.85 | 0.94 | 0.901 | 1 |
| Question + generated image | table | 36 | 0.69 | 0.92 | 0.792 | 1 |
| Question + generated image | text | 17 | 0.59 | 0.88 | 0.737 | 1 |
| Question + generated image | figure | 31 | 0.74 | 0.87 | 0.804 | 1 |
| Question + generated image | all | 122 | 0.72 | 0.89 | 0.803 | 1 |
| Question as image | chart | 34 | 0.82 | 0.94 | 0.870 | 1 |
| Question as image | table | 36 | 0.56 | 0.81 | 0.653 | 1 |
| Question as image | text | 17 | 0.76 | 0.88 | 0.816 | 1 |
| Question as image | figure | 31 | 0.65 | 0.84 | 0.756 | 1 |
| Question as image | all | 122 | 0.67 | 0.86 | 0.756 | 1 |

Paired against the question-as-text baseline (all types):

| variant | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |
|---|---|---|---|---|
| Spec as text | 122 | +0.027 [-0.030, +0.083] | 23 / 81 / 18 | 0.14 |
| Spec drawn (code) | 122 | +0.024 [-0.040, +0.086] | 25 / 76 / 21 | 0.193 |
| Spec generated (Qwen-Image) | 122 | -0.034 [-0.105, +0.031] | 23 / 71 / 28 | 0.821 |
| Question + spec as text | 122 | +0.035 [-0.007, +0.079] | 22 / 87 / 13 | 0.0122 |
| Question + code drawing | 122 | +0.041 [-0.007, +0.089] | 24 / 83 / 15 | 0.0152 |
| Question + generated image | 122 | +0.016 [-0.031, +0.061] | 23 / 82 / 17 | 0.0294 |
| Question as image | 122 | -0.031 [-0.060, -0.005] | 15 / 85 / 22 | 0.652 |

## proposer type correct

| variant | type | n | R@1 | R@5 | MRR | median rank |
|---|---|---|---|---|---|---|
| Question (text) | chart | 30 | 0.90 | 0.97 | 0.927 | 1 |
| Question (text) | table | 33 | 0.58 | 0.70 | 0.651 | 1 |
| Question (text) | text | 21 | 0.86 | 0.95 | 0.907 | 1 |
| Question (text) | figure | 21 | 0.81 | 0.90 | 0.852 | 1 |
| Question (text) | all | 108 | 0.75 | 0.84 | 0.803 | 1 |
| Spec as text | chart | 30 | 0.87 | 0.93 | 0.906 | 1 |
| Spec as text | table | 33 | 0.73 | 0.94 | 0.808 | 1 |
| Spec as text | text | 21 | 0.86 | 0.95 | 0.908 | 1 |
| Spec as text | figure | 21 | 0.81 | 0.86 | 0.846 | 1 |
| Spec as text | all | 108 | 0.79 | 0.91 | 0.845 | 1 |
| Spec drawn (code) | chart | 30 | 0.87 | 0.93 | 0.898 | 1 |
| Spec drawn (code) | table | 33 | 0.70 | 0.91 | 0.792 | 1 |
| Spec drawn (code) | text | 21 | 0.86 | 1.00 | 0.929 | 1 |
| Spec drawn (code) | figure | 21 | 0.76 | 0.86 | 0.819 | 1 |
| Spec drawn (code) | all | 108 | 0.78 | 0.91 | 0.842 | 1 |
| Spec generated (Qwen-Image) | chart | 30 | 0.90 | 0.97 | 0.928 | 1 |
| Spec generated (Qwen-Image) | table | 33 | 0.73 | 0.91 | 0.805 | 1 |
| Spec generated (Qwen-Image) | text | 21 | 0.76 | 0.95 | 0.846 | 1 |
| Spec generated (Qwen-Image) | figure | 21 | 0.57 | 0.71 | 0.666 | 1 |
| Spec generated (Qwen-Image) | all | 108 | 0.73 | 0.87 | 0.800 | 1 |
| Question + spec as text | chart | 30 | 0.87 | 0.97 | 0.920 | 1 |
| Question + spec as text | table | 33 | 0.67 | 0.85 | 0.765 | 1 |
| Question + spec as text | text | 21 | 0.86 | 1.00 | 0.909 | 1 |
| Question + spec as text | figure | 21 | 0.81 | 0.86 | 0.846 | 1 |
| Question + spec as text | all | 108 | 0.77 | 0.90 | 0.835 | 1 |
| Question + code drawing | chart | 30 | 0.87 | 0.97 | 0.921 | 1 |
| Question + code drawing | table | 33 | 0.67 | 0.97 | 0.791 | 1 |
| Question + code drawing | text | 21 | 0.95 | 1.00 | 0.962 | 1 |
| Question + code drawing | figure | 21 | 0.81 | 0.86 | 0.844 | 1 |
| Question + code drawing | all | 108 | 0.80 | 0.94 | 0.860 | 1 |
| Question + generated image | chart | 30 | 0.90 | 0.97 | 0.938 | 1 |
| Question + generated image | table | 33 | 0.70 | 0.91 | 0.794 | 1 |
| Question + generated image | text | 21 | 0.81 | 1.00 | 0.890 | 1 |
| Question + generated image | figure | 21 | 0.81 | 0.86 | 0.839 | 1 |
| Question + generated image | all | 108 | 0.78 | 0.91 | 0.842 | 1 |
| Question as image | chart | 30 | 0.87 | 0.93 | 0.902 | 1 |
| Question as image | table | 33 | 0.55 | 0.79 | 0.642 | 1 |
| Question as image | text | 21 | 0.86 | 0.86 | 0.870 | 1 |
| Question as image | figure | 21 | 0.67 | 0.86 | 0.772 | 1 |
| Question as image | all | 108 | 0.70 | 0.85 | 0.773 | 1 |

Paired against the question-as-text baseline (all types):

| variant | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |
|---|---|---|---|---|
| Spec as text | 108 | +0.042 [-0.018, +0.097] | 19 / 77 / 12 | 0.0768 |
| Spec drawn (code) | 108 | +0.039 [-0.028, +0.105] | 22 / 70 / 16 | 0.132 |
| Spec generated (Qwen-Image) | 108 | -0.003 [-0.077, +0.069] | 21 / 67 / 20 | 0.507 |
| Question + spec as text | 108 | +0.033 [-0.010, +0.078] | 18 / 79 / 11 | 0.0154 |
| Question + code drawing | 108 | +0.057 [+0.009, +0.107] | 21 / 76 / 11 | 0.0114 |
| Question + generated image | 108 | +0.040 [-0.006, +0.090] | 21 / 76 / 11 | 0.0133 |
| Question as image | 108 | -0.029 [-0.066, +0.002] | 12 / 78 / 18 | 0.405 |

## Part A items

| variant | type | n | R@1 | R@5 | MRR | median rank |
|---|---|---|---|---|---|---|
| Question (text) | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Question (text) | table | 6 | 0.83 | 1.00 | 0.889 | 1 |
| Question (text) | text | 4 | 1.00 | 1.00 | 1.000 | 1 |
| Question (text) | figure | 4 | 1.00 | 1.00 | 1.000 | 1 |
| Question (text) | all | 20 | 0.95 | 1.00 | 0.967 | 1 |
| Spec as text | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Spec as text | table | 6 | 0.83 | 1.00 | 0.917 | 1 |
| Spec as text | text | 4 | 0.50 | 0.75 | 0.661 | 2 |
| Spec as text | figure | 4 | 0.75 | 0.75 | 0.786 | 1 |
| Spec as text | all | 20 | 0.80 | 0.90 | 0.864 | 1 |
| Spec drawn (code) | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Spec drawn (code) | table | 6 | 0.83 | 1.00 | 0.917 | 1 |
| Spec drawn (code) | text | 4 | 0.50 | 1.00 | 0.750 | 2 |
| Spec drawn (code) | figure | 4 | 0.50 | 0.75 | 0.653 | 2 |
| Spec drawn (code) | all | 20 | 0.75 | 0.95 | 0.856 | 1 |
| Spec generated (Qwen-Image) | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Spec generated (Qwen-Image) | table | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Spec generated (Qwen-Image) | text | 4 | 0.50 | 1.00 | 0.750 | 2 |
| Spec generated (Qwen-Image) | figure | 4 | 0.25 | 0.50 | 0.448 | 4 |
| Spec generated (Qwen-Image) | all | 20 | 0.75 | 0.90 | 0.840 | 1 |
| Question + code drawing | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Question + code drawing | table | 6 | 0.83 | 1.00 | 0.917 | 1 |
| Question + code drawing | text | 4 | 0.75 | 1.00 | 0.875 | 1 |
| Question + code drawing | figure | 4 | 0.75 | 1.00 | 0.812 | 1 |
| Question + code drawing | all | 20 | 0.85 | 1.00 | 0.912 | 1 |
| Question + generated image | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Question + generated image | table | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Question + generated image | text | 4 | 0.50 | 1.00 | 0.750 | 2 |
| Question + generated image | figure | 4 | 0.75 | 1.00 | 0.833 | 1 |
| Question + generated image | all | 20 | 0.85 | 1.00 | 0.917 | 1 |
| Human spec as text | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Human spec as text | table | 6 | 0.50 | 0.50 | 0.534 | 4 |
| Human spec as text | text | 4 | 0.50 | 1.00 | 0.750 | 2 |
| Human spec as text | figure | 4 | 0.50 | 0.75 | 0.578 | 3 |
| Human spec as text | all | 20 | 0.65 | 0.80 | 0.726 | 1 |
| Human spec drawn (code) | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Human spec drawn (code) | table | 6 | 0.50 | 0.67 | 0.550 | 3 |
| Human spec drawn (code) | text | 4 | 0.25 | 1.00 | 0.542 | 2 |
| Human spec drawn (code) | figure | 4 | 0.25 | 0.75 | 0.528 | 2 |
| Human spec drawn (code) | all | 20 | 0.55 | 0.85 | 0.679 | 1 |
| Human spec generated | chart | 6 | 0.67 | 0.83 | 0.769 | 1 |
| Human spec generated | table | 6 | 0.33 | 0.50 | 0.473 | 4 |
| Human spec generated | text | 4 | 0.25 | 1.00 | 0.508 | 2 |
| Human spec generated | figure | 4 | 0.50 | 0.50 | 0.567 | 4 |
| Human spec generated | all | 20 | 0.45 | 0.70 | 0.588 | 2 |
| Question + human code drawing | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Question + human code drawing | table | 6 | 0.50 | 1.00 | 0.672 | 2 |
| Question + human code drawing | text | 4 | 0.50 | 1.00 | 0.750 | 2 |
| Question + human code drawing | figure | 4 | 0.75 | 1.00 | 0.875 | 1 |
| Question + human code drawing | all | 20 | 0.70 | 1.00 | 0.827 | 1 |
| Question + human generated image | chart | 6 | 0.83 | 1.00 | 0.917 | 1 |
| Question + human generated image | table | 6 | 0.50 | 1.00 | 0.750 | 2 |
| Question + human generated image | text | 4 | 0.75 | 1.00 | 0.875 | 1 |
| Question + human generated image | figure | 4 | 0.75 | 1.00 | 0.875 | 1 |
| Question + human generated image | all | 20 | 0.70 | 1.00 | 0.850 | 1 |
| Oracle crop (real region) | chart | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Oracle crop (real region) | table | 6 | 1.00 | 1.00 | 1.000 | 1 |
| Oracle crop (real region) | text | 4 | 1.00 | 1.00 | 1.000 | 1 |
| Oracle crop (real region) | figure | 3 | 1.00 | 1.00 | 1.000 | 1 |
| Oracle crop (real region) | all | 19 | 1.00 | 1.00 | 1.000 | 1 |

Paired against the question-as-text baseline (all types):

| variant | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |
|---|---|---|---|---|
| Spec as text | 20 | -0.102 [-0.254, +0.042] | 1 / 15 / 4 | 0.221 |
| Spec drawn (code) | 20 | -0.111 [-0.253, +0.025] | 1 / 14 / 5 | 0.236 |
| Spec generated (Qwen-Image) | 20 | -0.127 [-0.269, +0.017] | 1 / 14 / 5 | 0.168 |
| Question + code drawing | 20 | -0.054 [-0.175, +0.067] | 1 / 16 / 3 | 0.461 |
| Question + generated image | 20 | -0.050 [-0.167, +0.067] | 1 / 16 / 3 | 0.577 |
| Human spec as text | 20 | -0.241 [-0.409, -0.098] | 0 / 13 / 7 | 0.0178 |
| Human spec drawn (code) | 20 | -0.288 [-0.455, -0.145] | 0 / 11 / 9 | 0.00726 |
| Human spec generated | 20 | -0.379 [-0.572, -0.199] | 1 / 9 / 10 | 0.00575 |
| Question + human code drawing | 20 | -0.140 [-0.260, -0.050] | 0 / 15 / 5 | 0.0339 |
| Question + human generated image | 20 | -0.117 [-0.225, -0.025] | 1 / 14 / 5 | 0.102 |
| Oracle crop (real region) | 19 | +0.035 [+0.000, +0.105] | 1 / 18 / 0 | 0.317 |

## Head to head (all questions; change in MRR of the first minus the second)

| comparison | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |
|---|---|---|---|---|
| Question + code drawing vs Question + spec as text | 144 | +0.012 [-0.011, +0.037] | 14 / 119 / 11 | 0.133 |
| Question + generated image vs Question + spec as text | 144 | -0.019 [-0.048, +0.011] | 18 / 106 / 20 | 0.578 |
| Spec generated (Qwen-Image) vs Spec drawn (code) | 144 | -0.065 [-0.109, -0.022] | 14 / 104 / 26 | 0.0737 |
| Spec drawn (code) vs Spec as text | 144 | +0.001 [-0.032, +0.035] | 14 / 112 / 18 | 0.909 |
| Question + generated image vs Question + code drawing | 144 | -0.032 [-0.059, -0.006] | 10 / 116 / 18 | 0.254 |

## Fusion weight curve (all questions, MRR)

- Spec drawn (code): w=0: 0.806, w=0.25: 0.834, w=0.5: 0.837, w=0.75: 0.845, w=1: 0.849, w=1.25: 0.862, w=1.5: 0.857, w=1.75: 0.857, w=2: 0.854
- Spec generated (Qwen-Image): w=0: 0.806, w=0.25: 0.828, w=0.5: 0.839, w=0.75: 0.829, w=1: 0.817, w=1.25: 0.816, w=1.5: 0.812, w=1.75: 0.806, w=2: 0.806
