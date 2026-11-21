# Model size table inputs (old vs new n / s / m)

Latency and parameter counts are the old benchmark's (`model_size_benchmark/results.csv`): the architecture and input size are unchanged, so they were not re-measured (desktop RTX 3080, PyTorch FP16, end-to-end).

| model | params (fused) | PyTorch FP16 ms | DUT drone mAP50 | all4 mAP50 | Det-Fly 020 R@0.25 [CI] | Det-Fly 020 mAP50 | Det-Fly 010-test R@0.25 [CI] | false-drone images /1,534 |
|---|---|---|---|---|---|---|---|---|
| old n | 2.39 M | 4.5 | n/a | n/a | n/a | n/a | n/a | n/a |
| new n | 2.39 M | 4.5 | n/a | n/a | n/a | n/a | n/a | n/a |
| old s | 9.47 M | 5.3 | n/a | 0.861 | 0.444 [0.375, 0.506] | 0.497 | 0.667 [0.524, 0.796] | n/a |
| new s | 9.47 M | 5.3 | n/a | n/a | n/a | n/a | n/a | n/a |
| old m | 20.35 M | 8.8 | n/a | n/a | n/a | n/a | n/a | n/a |
| new m | 20.35 M | 8.8 | n/a | n/a | n/a | n/a | n/a | n/a |
