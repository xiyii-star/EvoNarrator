# Deep Paper Evaluation Report

**Generated:** 2025-12-17 23:30:45

**Golden Set:** data/golden_set_79papers.xlsx

**Annotated Papers:** 79

**Metrics Used:**
- ROUGE-1/2/L
- BLEU

## Overall Comparison

| Method                         |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |   BLEU |   Num Papers |   problem_rouge1 |   method_rouge1 |   limitation_rouge1 |   future_work_rouge1 |
|:-------------------------------|----------:|----------:|----------:|-------:|-------------:|-----------------:|----------------:|--------------------:|---------------------:|
| newmethod_deeppaper2           |    0.4126 |    0.1608 |    0.2549 | 0.0673 |           79 |           0.2802 |          0.4008 |              0.4483 |               0.521  |
| naive_llm_gpt-4o-2024-11-20    |    0.3676 |    0.1084 |    0.2201 | 0.0344 |           79 |           0.4619 |          0.423  |              0.2772 |               0.3083 |
| mymethod_deeppaper             |    0.2921 |    0.126  |    0.1966 | 0.0684 |           79 |           0.0726 |          0.1233 |              0.4938 |               0.4788 |
| ablation_no_critic_dual_stream |    0.254  |    0.0907 |    0.161  | 0.0454 |           79 |           0.0625 |          0.1299 |              0.4047 |               0.419  |
| ablation_no_navigator          |    0.2484 |    0.0753 |    0.1485 | 0.0331 |           79 |           0.1256 |          0.1433 |              0.3326 |               0.392  |
| pure_rag                       |    0.2504 |    0.041  |    0.1448 | 0.0112 |           79 |           0.2601 |          0.2489 |              0.24   |               0.2527 |
| llm_rag                        |    0.0205 |    0.0031 |    0.019  | 0      |           79 |           0.0183 |          0.0374 |              0.0049 |               0.0215 |

## naive_llm_gpt-4o-2024-11-20

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.3676
- ROUGE-2 F1: 0.1084
- ROUGE-L F1: 0.2201
- BLEU: 0.0344

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.4619 ± 0.1174
- ROUGE-2: 0.1962 ± 0.0842
- ROUGE-L: 0.3111 ± 0.1004
- BLEU: 0.0743 ± 0.0696

#### METHOD

- ROUGE-1: 0.4230 ± 0.1116
- ROUGE-2: 0.1533 ± 0.0801
- ROUGE-L: 0.2505 ± 0.0819
- BLEU: 0.0415 ± 0.0447

#### LIMITATION

- ROUGE-1: 0.2772 ± 0.0785
- ROUGE-2: 0.0368 ± 0.0346
- ROUGE-L: 0.1525 ± 0.0442
- BLEU: 0.0105 ± 0.0092

#### FUTURE_WORK

- ROUGE-1: 0.3083 ± 0.0786
- ROUGE-2: 0.0474 ± 0.0353
- ROUGE-L: 0.1661 ± 0.0463
- BLEU: 0.0112 ± 0.0130

## newmethod_deeppaper2

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.4126
- ROUGE-2 F1: 0.1608
- ROUGE-L F1: 0.2549
- BLEU: 0.0673

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.2802 ± 0.0956
- ROUGE-2: 0.1087 ± 0.0817
- ROUGE-L: 0.1978 ± 0.0799
- BLEU: 0.0095 ± 0.0144

#### METHOD

- ROUGE-1: 0.4008 ± 0.1020
- ROUGE-2: 0.1172 ± 0.0610
- ROUGE-L: 0.2190 ± 0.0650
- BLEU: 0.0372 ± 0.0376

#### LIMITATION

- ROUGE-1: 0.4483 ± 0.1559
- ROUGE-2: 0.1856 ± 0.1173
- ROUGE-L: 0.2819 ± 0.1235
- BLEU: 0.0965 ± 0.0825

#### FUTURE_WORK

- ROUGE-1: 0.5210 ± 0.1201
- ROUGE-2: 0.2319 ± 0.1294
- ROUGE-L: 0.3209 ± 0.1352
- BLEU: 0.1258 ± 0.1027

## llm_rag

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.0205
- ROUGE-2 F1: 0.0031
- ROUGE-L F1: 0.0190
- BLEU: 0.0000

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0183 ± 0.0210
- ROUGE-2: 0.0020 ± 0.0070
- ROUGE-L: 0.0180 ± 0.0209
- BLEU: 0.0000 ± 0.0000

#### METHOD

- ROUGE-1: 0.0374 ± 0.0398
- ROUGE-2: 0.0073 ± 0.0154
- ROUGE-L: 0.0334 ± 0.0341
- BLEU: 0.0000 ± 0.0000

#### LIMITATION

- ROUGE-1: 0.0049 ± 0.0144
- ROUGE-2: 0.0003 ± 0.0027
- ROUGE-L: 0.0049 ± 0.0144
- BLEU: 0.0000 ± 0.0000

#### FUTURE_WORK

- ROUGE-1: 0.0215 ± 0.0317
- ROUGE-2: 0.0030 ± 0.0094
- ROUGE-L: 0.0197 ± 0.0293
- BLEU: 0.0000 ± 0.0000

## ablation_no_critic_dual_stream

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2540
- ROUGE-2 F1: 0.0907
- ROUGE-L F1: 0.1610
- BLEU: 0.0454

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0625 ± 0.1361
- ROUGE-2: 0.0220 ± 0.0691
- ROUGE-L: 0.0506 ± 0.1049
- BLEU: 0.0099 ± 0.0371

#### METHOD

- ROUGE-1: 0.1299 ± 0.1635
- ROUGE-2: 0.0390 ± 0.0674
- ROUGE-L: 0.0856 ± 0.0895
- BLEU: 0.0090 ± 0.0259

#### LIMITATION

- ROUGE-1: 0.4047 ± 0.1600
- ROUGE-2: 0.1447 ± 0.1222
- ROUGE-L: 0.2509 ± 0.1289
- BLEU: 0.0751 ± 0.0804

#### FUTURE_WORK

- ROUGE-1: 0.4190 ± 0.1705
- ROUGE-2: 0.1572 ± 0.1212
- ROUGE-L: 0.2569 ± 0.1325
- BLEU: 0.0874 ± 0.0835

## mymethod_deeppaper

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2921
- ROUGE-2 F1: 0.1260
- ROUGE-L F1: 0.1966
- BLEU: 0.0684

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0726 ± 0.1375
- ROUGE-2: 0.0226 ± 0.0616
- ROUGE-L: 0.0547 ± 0.0929
- BLEU: 0.0071 ± 0.0242

#### METHOD

- ROUGE-1: 0.1233 ± 0.1429
- ROUGE-2: 0.0323 ± 0.0548
- ROUGE-L: 0.0850 ± 0.0814
- BLEU: 0.0088 ± 0.0261

#### LIMITATION

- ROUGE-1: 0.4938 ± 0.1590
- ROUGE-2: 0.2254 ± 0.1589
- ROUGE-L: 0.3211 ± 0.1491
- BLEU: 0.1335 ± 0.1428

#### FUTURE_WORK

- ROUGE-1: 0.4788 ± 0.2007
- ROUGE-2: 0.2237 ± 0.1458
- ROUGE-L: 0.3254 ± 0.1735
- BLEU: 0.1239 ± 0.1090

## pure_rag

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2504
- ROUGE-2 F1: 0.0410
- ROUGE-L F1: 0.1448
- BLEU: 0.0112

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.2601 ± 0.0895
- ROUGE-2: 0.0428 ± 0.0423
- ROUGE-L: 0.1517 ± 0.0457
- BLEU: 0.0129 ± 0.0215

#### METHOD

- ROUGE-1: 0.2489 ± 0.0750
- ROUGE-2: 0.0382 ± 0.0392
- ROUGE-L: 0.1370 ± 0.0415
- BLEU: 0.0076 ± 0.0107

#### LIMITATION

- ROUGE-1: 0.2400 ± 0.0795
- ROUGE-2: 0.0361 ± 0.0523
- ROUGE-L: 0.1424 ± 0.0580
- BLEU: 0.0110 ± 0.0209

#### FUTURE_WORK

- ROUGE-1: 0.2527 ± 0.0890
- ROUGE-2: 0.0469 ± 0.0545
- ROUGE-L: 0.1481 ± 0.0611
- BLEU: 0.0131 ± 0.0194

## ablation_no_navigator

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2484
- ROUGE-2 F1: 0.0753
- ROUGE-L F1: 0.1485
- BLEU: 0.0331

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.1256 ± 0.1864
- ROUGE-2: 0.0485 ± 0.0902
- ROUGE-L: 0.0889 ± 0.1255
- BLEU: 0.0179 ± 0.0474

#### METHOD

- ROUGE-1: 0.1433 ± 0.1574
- ROUGE-2: 0.0412 ± 0.0636
- ROUGE-L: 0.0950 ± 0.0891
- BLEU: 0.0100 ± 0.0287

#### LIMITATION

- ROUGE-1: 0.3326 ± 0.1173
- ROUGE-2: 0.0824 ± 0.0852
- ROUGE-L: 0.1868 ± 0.0827
- BLEU: 0.0362 ± 0.0541

#### FUTURE_WORK

- ROUGE-1: 0.3920 ± 0.1470
- ROUGE-2: 0.1290 ± 0.1228
- ROUGE-L: 0.2233 ± 0.1162
- BLEU: 0.0684 ± 0.0841

