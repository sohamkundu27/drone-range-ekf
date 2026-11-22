# Model size table inputs (old vs new n / s / m)

Latency and parameter counts are the old benchmark's (`model_size_benchmark/results.csv`): the architecture and input size are unchanged, so they were not re-measured (desktop RTX 3080, PyTorch FP16, end-to-end).

| model | params (fused) | PyTorch FP16 ms | DUT drone mAP50 | all4 mAP50 | Det-Fly 020 R@0.25 [CI] | Det-Fly 020 mAP50 | Det-Fly 010-test R@0.25 [CI] | false-drone images /1,534 |
|---|---|---|---|---|---|---|---|---|
| old n | 2.39 M | 4.5 | 0.976 | 0.860 | 0.460 [0.392, 0.521] | 0.498 | 0.650 [0.514, 0.782] | 9 |
| new n | 2.39 M | 4.5 | 0.976 | 0.856 | 0.684 [0.612, 0.746] | 0.730 | 0.767 [0.642, 0.890] | 7 |
| old s | 9.47 M | 5.3 | 0.979 | 0.861 | 0.444 [0.375, 0.506] | 0.497 | 0.667 [0.524, 0.796] | 5 |
| new s | 9.47 M | 5.3 | 0.980 | 0.861 | 0.711 [0.649, 0.768] | 0.752 | 0.762 [0.633, 0.889] | 8 |
| old m | 20.35 M | 8.8 | 0.982 | 0.860 | 0.485 [0.409, 0.557] | 0.541 | 0.657 [0.520, 0.784] | 3 |
| new m | 20.35 M | 8.8 | 0.988 | 0.865 | 0.729 [0.661, 0.785] | 0.775 | 0.762 [0.637, 0.885] | 2 |
