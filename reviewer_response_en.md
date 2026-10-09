We agree that the end-to-end latency gain is modest and
have revised the positioning accordingly. To isolate the efficiency of the
compressed fully connected (FC) layers, we added a training-free structural
profile on the standard 1000-class AlexNet and GoogLeNet at batch size 1. We
anchored the fixed configuration to the existing AlexNet fc1/fc2 setting (six
Tucker ranks of 8 and Kruskal rank 128) and applied the same rule to AlexNet
fc3 and GoogLeNet fc. We counted all supported operations executed by the
implementation, including outer/core factor products and Khatri--Rao
contractions, rather than only the two idealized low-rank data GEMMs.

For AlexNet's three FC layers, FCTTD reduces the aggregate FC parameters from
58,631,144 to 29,912
(1960.12x) and the FC operator count from
58,621,952 to 7,103,164
MAC-equivalents (8.25x). For GoogLeNet's final FC
layer, the corresponding reductions are 134.16x
in parameters and 1.76x in operator count.
These layer-local structural ratios should not be compared one-to-one with the
reported 13.5x whole-model, accuracy-constrained parameter ratio: this
training-free diagnostic uses fixed ranks and a different denominator solely
to isolate FC arithmetic.
However, FC layers account for only 8.21%
and 0.0680% of the supported
whole-model operator counts in AlexNet and GoogLeNet, respectively. The implied
whole-model operator-count gains are therefore only
1.0777x and
1.0003x.

These results reconcile the large parameter reduction with the modest
end-to-end speed-up: FCTTD primarily improves memory efficiency and also
reduces computation inside the replaced FC layers, while convolutional
backbones dominate total computation. We now explicitly distinguish
FLOP/operator-count reductions from measured latency and acknowledge that
fragmented tensor contractions and kernel-launch overhead can further reduce
the realized GPU/NPU speed-up. This experiment does not evaluate accuracy and
is presented only as a structural efficiency analysis.
