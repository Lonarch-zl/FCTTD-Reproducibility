# Released checkpoint record

| checkpoint | FC1 method | seed | FC1 weight params | FC1 CR | Top-1 |
|---|---:|---:|---:|---:|---:|
| `vgg16_cifar10_dense_best.pt` | Dense | 42 | 102,760,448 | 1x | 94.22% (clean reference) |
| `vgg16_cifar10_fcttd_r400_seed43.pt` | FCTTD, `R_t=(8,8,8,8,8,8)`, `R_k=400` | 43 | 20,360 | 5,047.17x | 90.68% (matched-control FC1-only fine-tuning) |

The dense and compressed accuracies come from different explicitly documented protocols (clean reference versus the reviewer-requested matched-control fine-tuning run). They are therefore checkpoint provenance values, not a claim of a single paired leaderboard.
