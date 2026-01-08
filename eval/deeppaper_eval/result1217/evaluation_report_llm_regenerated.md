# Deep Paper Evaluation Report

**Generated:** 2025-12-18 16:22:33

**Golden Set:** data/golden_set_79papers.xlsx

**Annotated Papers:** 79

**Metrics Used:**
- ROUGE-1/2/L
- BLEU
- LLM Evaluation (gpt-4o-2024-11-20)

## Overall Comparison

| Method                         |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |   BLEU |   Num Papers |   LLM_Similarity |   LLM_Coverage |   LLM_Accuracy |
|:-------------------------------|----------:|----------:|----------:|-------:|-------------:|-----------------:|---------------:|---------------:|
| newmethod_deeppaper2           |    0.4126 |    0.1608 |    0.2549 | 0.0673 |           79 |           0.7325 |         0.6576 |         0.8981 |
| naive_llm_gpt-4o-2024-11-20    |    0.3676 |    0.1084 |    0.2201 | 0.0344 |           79 |           0.6717 |         0.598  |         0.8595 |
| mymethod_deeppaper             |    0.2921 |    0.126  |    0.1966 | 0.0684 |           79 |           0.7813 |         0.7272 |         0.8966 |
| ablation_no_critic_dual_stream |    0.254  |    0.0907 |    0.161  | 0.0454 |           79 |           0.7174 |         0.6619 |         0.8816 |
| ablation_no_navigator          |    0.2484 |    0.0753 |    0.1485 | 0.0331 |           79 |           0.6933 |         0.6325 |         0.8709 |
| pure_rag                       |    0.2504 |    0.041  |    0.1448 | 0.0112 |           79 |           0.3647 |         0.2862 |         0.7856 |
| llm_rag                        |    0.0205 |    0.0031 |    0.019  | 0      |           79 |           0.5375 |         0.4756 |         0.7729 |

## naive_llm_gpt-4o-2024-11-20

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.3676
- ROUGE-2 F1: 0.1084
- ROUGE-L F1: 0.2201
- BLEU: 0.0344
- LLM Similarity: 0.6717
- LLM Coverage: 0.5980
- LLM Accuracy: 0.8595

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.4619 ± 0.1174
- ROUGE-2: 0.1962 ± 0.0842
- ROUGE-L: 0.3111 ± 0.1004
- BLEU: 0.0743 ± 0.0696
- LLM Similarity: 0.8165 ± 0.1597
- LLM Coverage: 0.7373 ± 0.1675
- LLM Accuracy: 0.9051 ± 0.1532

#### METHOD

- ROUGE-1: 0.4230 ± 0.1116
- ROUGE-2: 0.1533 ± 0.0801
- ROUGE-L: 0.2505 ± 0.0819
- BLEU: 0.0415 ± 0.0447
- LLM Similarity: 0.7844 ± 0.1586
- LLM Coverage: 0.7143 ± 0.1511
- LLM Accuracy: 0.8816 ± 0.1563

#### LIMITATION

- ROUGE-1: 0.2772 ± 0.0785
- ROUGE-2: 0.0368 ± 0.0346
- ROUGE-L: 0.1525 ± 0.0442
- BLEU: 0.0105 ± 0.0092
- LLM Similarity: 0.4759 ± 0.2267
- LLM Coverage: 0.4108 ± 0.2233
- LLM Accuracy: 0.8095 ± 0.1613

#### FUTURE_WORK

- ROUGE-1: 0.3083 ± 0.0786
- ROUGE-2: 0.0474 ± 0.0353
- ROUGE-L: 0.1661 ± 0.0463
- BLEU: 0.0112 ± 0.0130
- LLM Similarity: 0.6101 ± 0.1834
- LLM Coverage: 0.5297 ± 0.1822
- LLM Accuracy: 0.8418 ± 0.1479

## newmethod_deeppaper2

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.4126
- ROUGE-2 F1: 0.1608
- ROUGE-L F1: 0.2549
- BLEU: 0.0673
- LLM Similarity: 0.7325
- LLM Coverage: 0.6576
- LLM Accuracy: 0.8981

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.2802 ± 0.0956
- ROUGE-2: 0.1087 ± 0.0817
- ROUGE-L: 0.1978 ± 0.0799
- BLEU: 0.0095 ± 0.0144
- LLM Similarity: 0.6487 ± 0.2024
- LLM Coverage: 0.5139 ± 0.1904
- LLM Accuracy: 0.8741 ± 0.1006

#### METHOD

- ROUGE-1: 0.4008 ± 0.1020
- ROUGE-2: 0.1172 ± 0.0610
- ROUGE-L: 0.2190 ± 0.0650
- BLEU: 0.0372 ± 0.0376
- LLM Similarity: 0.7462 ± 0.1425
- LLM Coverage: 0.6367 ± 0.1500
- LLM Accuracy: 0.8892 ± 0.1124

#### LIMITATION

- ROUGE-1: 0.4483 ± 0.1559
- ROUGE-2: 0.1856 ± 0.1173
- ROUGE-L: 0.2819 ± 0.1235
- BLEU: 0.0965 ± 0.0825
- LLM Similarity: 0.7405 ± 0.2240
- LLM Coverage: 0.7139 ± 0.2502
- LLM Accuracy: 0.9089 ± 0.1078

#### FUTURE_WORK

- ROUGE-1: 0.5210 ± 0.1201
- ROUGE-2: 0.2319 ± 0.1294
- ROUGE-L: 0.3209 ± 0.1352
- BLEU: 0.1258 ± 0.1027
- LLM Similarity: 0.7947 ± 0.1464
- LLM Coverage: 0.7658 ± 0.1674
- LLM Accuracy: 0.9203 ± 0.0736

## llm_rag

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.0205
- ROUGE-2 F1: 0.0031
- ROUGE-L F1: 0.0190
- BLEU: 0.0000
- LLM Similarity: 0.5375
- LLM Coverage: 0.4756
- LLM Accuracy: 0.7729

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0183 ± 0.0210
- ROUGE-2: 0.0020 ± 0.0070
- ROUGE-L: 0.0180 ± 0.0209
- BLEU: 0.0000 ± 0.0000
- LLM Similarity: 0.7063 ± 0.2417
- LLM Coverage: 0.6513 ± 0.2336
- LLM Accuracy: 0.8424 ± 0.1967

#### METHOD

- ROUGE-1: 0.0374 ± 0.0398
- ROUGE-2: 0.0073 ± 0.0154
- ROUGE-L: 0.0334 ± 0.0341
- BLEU: 0.0000 ± 0.0000
- LLM Similarity: 0.6595 ± 0.2678
- LLM Coverage: 0.5962 ± 0.2618
- LLM Accuracy: 0.7930 ± 0.2633

#### LIMITATION

- ROUGE-1: 0.0049 ± 0.0144
- ROUGE-2: 0.0003 ± 0.0027
- ROUGE-L: 0.0049 ± 0.0144
- BLEU: 0.0000 ± 0.0000
- LLM Similarity: 0.2443 ± 0.2934
- LLM Coverage: 0.1810 ± 0.2852
- LLM Accuracy: 0.6152 ± 0.3706

#### FUTURE_WORK

- ROUGE-1: 0.0215 ± 0.0317
- ROUGE-2: 0.0030 ± 0.0094
- ROUGE-L: 0.0197 ± 0.0293
- BLEU: 0.0000 ± 0.0000
- LLM Similarity: 0.5399 ± 0.3114
- LLM Coverage: 0.4741 ± 0.2982
- LLM Accuracy: 0.8411 ± 0.1864

## ablation_no_critic_dual_stream

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2540
- ROUGE-2 F1: 0.0907
- ROUGE-L F1: 0.1610
- BLEU: 0.0454
- LLM Similarity: 0.7174
- LLM Coverage: 0.6619
- LLM Accuracy: 0.8816

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0625 ± 0.1361
- ROUGE-2: 0.0220 ± 0.0691
- ROUGE-L: 0.0506 ± 0.1049
- BLEU: 0.0099 ± 0.0371
- LLM Similarity: 0.7709 ± 0.1203
- LLM Coverage: 0.7013 ± 0.1465
- LLM Accuracy: 0.8867 ± 0.0664

#### METHOD

- ROUGE-1: 0.1299 ± 0.1635
- ROUGE-2: 0.0390 ± 0.0674
- ROUGE-L: 0.0856 ± 0.0895
- BLEU: 0.0090 ± 0.0259
- LLM Similarity: 0.7639 ± 0.1222
- LLM Coverage: 0.7019 ± 0.1417
- LLM Accuracy: 0.8690 ± 0.0730

#### LIMITATION

- ROUGE-1: 0.4047 ± 0.1600
- ROUGE-2: 0.1447 ± 0.1222
- ROUGE-L: 0.2509 ± 0.1289
- BLEU: 0.0751 ± 0.0804
- LLM Similarity: 0.6215 ± 0.2668
- LLM Coverage: 0.5905 ± 0.2700
- LLM Accuracy: 0.8772 ± 0.0896

#### FUTURE_WORK

- ROUGE-1: 0.4190 ± 0.1705
- ROUGE-2: 0.1572 ± 0.1212
- ROUGE-L: 0.2569 ± 0.1325
- BLEU: 0.0874 ± 0.0835
- LLM Similarity: 0.7133 ± 0.1755
- LLM Coverage: 0.6538 ± 0.1952
- LLM Accuracy: 0.8937 ± 0.0681

## mymethod_deeppaper

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2921
- ROUGE-2 F1: 0.1260
- ROUGE-L F1: 0.1966
- BLEU: 0.0684
- LLM Similarity: 0.7813
- LLM Coverage: 0.7272
- LLM Accuracy: 0.8966

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0726 ± 0.1375
- ROUGE-2: 0.0226 ± 0.0616
- ROUGE-L: 0.0547 ± 0.0929
- BLEU: 0.0071 ± 0.0242
- LLM Similarity: 0.7829 ± 0.1201
- LLM Coverage: 0.7114 ± 0.1260
- LLM Accuracy: 0.8924 ± 0.0675

#### METHOD

- ROUGE-1: 0.1233 ± 0.1429
- ROUGE-2: 0.0323 ± 0.0548
- ROUGE-L: 0.0850 ± 0.0814
- BLEU: 0.0088 ± 0.0261
- LLM Similarity: 0.7494 ± 0.1157
- LLM Coverage: 0.6829 ± 0.1321
- LLM Accuracy: 0.8658 ± 0.0736

#### LIMITATION

- ROUGE-1: 0.4938 ± 0.1590
- ROUGE-2: 0.2254 ± 0.1589
- ROUGE-L: 0.3211 ± 0.1491
- BLEU: 0.1335 ± 0.1428
- LLM Similarity: 0.7728 ± 0.1833
- LLM Coverage: 0.7380 ± 0.1920
- LLM Accuracy: 0.8968 ± 0.0929

#### FUTURE_WORK

- ROUGE-1: 0.4788 ± 0.2007
- ROUGE-2: 0.2237 ± 0.1458
- ROUGE-L: 0.3254 ± 0.1735
- BLEU: 0.1239 ± 0.1090
- LLM Similarity: 0.8203 ± 0.1014
- LLM Coverage: 0.7766 ± 0.1394
- LLM Accuracy: 0.9314 ± 0.0501

## pure_rag

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2504
- ROUGE-2 F1: 0.0410
- ROUGE-L F1: 0.1448
- BLEU: 0.0112
- LLM Similarity: 0.3647
- LLM Coverage: 0.2862
- LLM Accuracy: 0.7856

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.2601 ± 0.0895
- ROUGE-2: 0.0428 ± 0.0423
- ROUGE-L: 0.1517 ± 0.0457
- BLEU: 0.0129 ± 0.0215
- LLM Similarity: 0.4013 ± 0.2471
- LLM Coverage: 0.3190 ± 0.2302
- LLM Accuracy: 0.7690 ± 0.1572

#### METHOD

- ROUGE-1: 0.2489 ± 0.0750
- ROUGE-2: 0.0382 ± 0.0392
- ROUGE-L: 0.1370 ± 0.0415
- BLEU: 0.0076 ± 0.0107
- LLM Similarity: 0.4095 ± 0.2401
- LLM Coverage: 0.3203 ± 0.2166
- LLM Accuracy: 0.7703 ± 0.1966

#### LIMITATION

- ROUGE-1: 0.2400 ± 0.0795
- ROUGE-2: 0.0361 ± 0.0523
- ROUGE-L: 0.1424 ± 0.0580
- BLEU: 0.0110 ± 0.0209
- LLM Similarity: 0.2924 ± 0.1997
- LLM Coverage: 0.2247 ± 0.1947
- LLM Accuracy: 0.7968 ± 0.1688

#### FUTURE_WORK

- ROUGE-1: 0.2527 ± 0.0890
- ROUGE-2: 0.0469 ± 0.0545
- ROUGE-L: 0.1481 ± 0.0611
- BLEU: 0.0131 ± 0.0194
- LLM Similarity: 0.3557 ± 0.2268
- LLM Coverage: 0.2810 ± 0.2178
- LLM Accuracy: 0.8063 ± 0.1165

## ablation_no_navigator

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2484
- ROUGE-2 F1: 0.0753
- ROUGE-L F1: 0.1485
- BLEU: 0.0331
- LLM Similarity: 0.6933
- LLM Coverage: 0.6325
- LLM Accuracy: 0.8709

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.1256 ± 0.1864
- ROUGE-2: 0.0485 ± 0.0902
- ROUGE-L: 0.0889 ± 0.1255
- BLEU: 0.0179 ± 0.0474
- LLM Similarity: 0.7823 ± 0.1153
- LLM Coverage: 0.7076 ± 0.1290
- LLM Accuracy: 0.9051 ± 0.0560

#### METHOD

- ROUGE-1: 0.1433 ± 0.1574
- ROUGE-2: 0.0412 ± 0.0636
- ROUGE-L: 0.0950 ± 0.0891
- BLEU: 0.0100 ± 0.0287
- LLM Similarity: 0.7778 ± 0.1024
- LLM Coverage: 0.7222 ± 0.1166
- LLM Accuracy: 0.8627 ± 0.0950

#### LIMITATION

- ROUGE-1: 0.3326 ± 0.1173
- ROUGE-2: 0.0824 ± 0.0852
- ROUGE-L: 0.1868 ± 0.0827
- BLEU: 0.0362 ± 0.0541
- LLM Similarity: 0.5395 ± 0.1862
- LLM Coverage: 0.4882 ± 0.1988
- LLM Accuracy: 0.8297 ± 0.0761

#### FUTURE_WORK

- ROUGE-1: 0.3920 ± 0.1470
- ROUGE-2: 0.1290 ± 0.1228
- ROUGE-L: 0.2233 ± 0.1162
- BLEU: 0.0684 ± 0.0841
- LLM Similarity: 0.6734 ± 0.1712
- LLM Coverage: 0.6120 ± 0.1921
- LLM Accuracy: 0.8861 ± 0.0573

