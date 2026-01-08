# Deep Paper Evaluation Report

**Generated:** 2025-12-18 15:57:56

**Golden Set:** data/golden_set_79papers.xlsx

**Annotated Papers:** 79

**Metrics Used:**
- ROUGE-1/2/L
- BLEU
- BERTScore

## Overall Comparison

| Method                      |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |   BLEU |   Num Papers |   BERTScore |   problem_rouge1 |   method_rouge1 |   limitation_rouge1 |   future_work_rouge1 |
|:----------------------------|----------:|----------:|----------:|-------:|-------------:|------------:|-----------------:|----------------:|--------------------:|---------------------:|
| newmethod_deeppaper2        |    0.4126 |    0.1608 |    0.2549 | 0.0673 |           79 |      0.8463 |           0.2802 |          0.4008 |              0.4483 |               0.521  |
| naive_llm_gpt-4o-2024-11-20 |    0.3676 |    0.1084 |    0.2201 | 0.0344 |           79 |      0.8182 |           0.4619 |          0.423  |              0.2772 |               0.3083 |
| pure_rag                    |    0.2504 |    0.041  |    0.1448 | 0.0112 |           79 |      0.8075 |           0.2601 |          0.2489 |              0.24   |               0.2527 |
| llm_rag                     |    0.0205 |    0.0031 |    0.019  | 0      |           79 |      0.5769 |           0.0183 |          0.0374 |              0.0049 |               0.0215 |

## naive_llm_gpt-4o-2024-11-20

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.3676
- ROUGE-2 F1: 0.1084
- ROUGE-L F1: 0.2201
- BLEU: 0.0344
- BERTScore F1: 0.8182

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.4619 ± 0.1174
- ROUGE-2: 0.1962 ± 0.0842
- ROUGE-L: 0.3111 ± 0.1004
- BLEU: 0.0743 ± 0.0696
- BERTScore F1: 0.8361 ± 0.2530

#### METHOD

- ROUGE-1: 0.4230 ± 0.1116
- ROUGE-2: 0.1533 ± 0.0801
- ROUGE-L: 0.2505 ± 0.0819
- BLEU: 0.0415 ± 0.0447
- BERTScore F1: 0.8231 ± 0.2099

#### LIMITATION

- ROUGE-1: 0.2772 ± 0.0785
- ROUGE-2: 0.0368 ± 0.0346
- ROUGE-L: 0.1525 ± 0.0442
- BLEU: 0.0105 ± 0.0092
- BERTScore F1: 0.8028 ± 0.1998

#### FUTURE_WORK

- ROUGE-1: 0.3083 ± 0.0786
- ROUGE-2: 0.0474 ± 0.0353
- ROUGE-L: 0.1661 ± 0.0463
- BLEU: 0.0112 ± 0.0130
- BERTScore F1: 0.8107 ± 0.1846

## newmethod_deeppaper2

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.4126
- ROUGE-2 F1: 0.1608
- ROUGE-L F1: 0.2549
- BLEU: 0.0673
- BERTScore F1: 0.8463

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.2802 ± 0.0956
- ROUGE-2: 0.1087 ± 0.0817
- ROUGE-L: 0.1978 ± 0.0799
- BLEU: 0.0095 ± 0.0144
- BERTScore F1: 0.8448 ± 0.0528

#### METHOD

- ROUGE-1: 0.4008 ± 0.1020
- ROUGE-2: 0.1172 ± 0.0610
- ROUGE-L: 0.2190 ± 0.0650
- BLEU: 0.0372 ± 0.0376
- BERTScore F1: 0.8416 ± 0.0488

#### LIMITATION

- ROUGE-1: 0.4483 ± 0.1559
- ROUGE-2: 0.1856 ± 0.1173
- ROUGE-L: 0.2819 ± 0.1235
- BLEU: 0.0965 ± 0.0825
- BERTScore F1: 0.8293 ± 0.0715

#### FUTURE_WORK

- ROUGE-1: 0.5210 ± 0.1201
- ROUGE-2: 0.2319 ± 0.1294
- ROUGE-L: 0.3209 ± 0.1352
- BLEU: 0.1258 ± 0.1027
- BERTScore F1: 0.8696 ± 0.0292

## llm_rag

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.0205
- ROUGE-2 F1: 0.0031
- ROUGE-L F1: 0.0190
- BLEU: 0.0000
- BERTScore F1: 0.5769

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.0183 ± 0.0210
- ROUGE-2: 0.0020 ± 0.0070
- ROUGE-L: 0.0180 ± 0.0209
- BLEU: 0.0000 ± 0.0000
- BERTScore F1: 0.6177 ± 0.0553

#### METHOD

- ROUGE-1: 0.0374 ± 0.0398
- ROUGE-2: 0.0073 ± 0.0154
- ROUGE-L: 0.0334 ± 0.0341
- BLEU: 0.0000 ± 0.0000
- BERTScore F1: 0.6160 ± 0.0683

#### LIMITATION

- ROUGE-1: 0.0049 ± 0.0144
- ROUGE-2: 0.0003 ± 0.0027
- ROUGE-L: 0.0049 ± 0.0144
- BLEU: 0.0000 ± 0.0000
- BERTScore F1: 0.4803 ± 0.1056

#### FUTURE_WORK

- ROUGE-1: 0.0215 ± 0.0317
- ROUGE-2: 0.0030 ± 0.0094
- ROUGE-L: 0.0197 ± 0.0293
- BLEU: 0.0000 ± 0.0000
- BERTScore F1: 0.5935 ± 0.0775

## pure_rag

**Papers Evaluated:** 79

### Overall Metrics

- ROUGE-1 F1: 0.2504
- ROUGE-2 F1: 0.0410
- ROUGE-L F1: 0.1448
- BLEU: 0.0112
- BERTScore F1: 0.8075

### Per-Field Metrics

#### PROBLEM

- ROUGE-1: 0.2601 ± 0.0895
- ROUGE-2: 0.0428 ± 0.0423
- ROUGE-L: 0.1517 ± 0.0457
- BLEU: 0.0129 ± 0.0215
- BERTScore F1: 0.8072 ± 0.0461

#### METHOD

- ROUGE-1: 0.2489 ± 0.0750
- ROUGE-2: 0.0382 ± 0.0392
- ROUGE-L: 0.1370 ± 0.0415
- BLEU: 0.0076 ± 0.0107
- BERTScore F1: 0.8082 ± 0.0352

#### LIMITATION

- ROUGE-1: 0.2400 ± 0.0795
- ROUGE-2: 0.0361 ± 0.0523
- ROUGE-L: 0.1424 ± 0.0580
- BLEU: 0.0110 ± 0.0209
- BERTScore F1: 0.8055 ± 0.0349

#### FUTURE_WORK

- ROUGE-1: 0.2527 ± 0.0890
- ROUGE-2: 0.0469 ± 0.0545
- ROUGE-L: 0.1481 ± 0.0611
- BLEU: 0.0131 ± 0.0194
- BERTScore F1: 0.8090 ± 0.0373

