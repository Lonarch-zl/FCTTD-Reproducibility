# ImageNet AlexNet / GoogLeNet 全连接层 FCTTD 效率实验

## 实验边界

- 使用 torchvision ImageNet 1000 类 AlexNet 与 GoogLeNet，输入几何
  `1×3×224×224`。
- 不读取 ImageNet 样本，不训练，不报告准确率；权重数值不影响参数量和
  operator count。
- AlexNet 压缩 3 个 classifier Linear；GoogLeNet 压缩最终 `fc`。
- 固定配置以现有 AlexNet 原型的 fc1/fc2 设置为锚点：平衡三阶输入/输出
  tensorization，六个 Tucker ranks 均为 8，Kruskal rank `Rc=128`；同一
  规则外推到 AlexNet fc3 和 GoogLeNet fc。
- 参数量包含 bias。计算量使用 fvcore batch=1 operator-count 口径；Dense
  Linear 与 FCTTD bias add 均不计。
- FCTTD 数值包含当前 forward 实际执行的 6 个 outer/core factor matmul、
  两个 Khatri–Rao einsum 和两个数据 matmul。因此不是只计算理想化低秩 GEMM。

## 逐层结果

| Model | Layer | Shape | Dense Params | FCTTD Params | Param CR | Dense MAC-eq | FCTTD MAC-eq | MAC reduction ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AlexNet | classifier.1 | 9216→4096 | 37,752,832 | 11,136 | 3390.16× | 37,748,736 | 3,522,624 | 10.72× |
| AlexNet | classifier.4 | 4096→4096 | 16,781,312 | 11,008 | 1524.47× | 16,777,216 | 2,195,880 | 7.64× |
| AlexNet | classifier.6 | 4096→1000 | 4,097,000 | 7,768 | 527.42× | 4,096,000 | 1,384,660 | 2.96× |
| GoogLeNet | fc | 1024→1000 | 1,025,000 | 7,640 | 134.16× | 1,024,000 | 581,610 | 1.76× |

## 压缩层聚合结果（主表）

| Model | Compressed FC layers | Dense FC Params | FCTTD FC Params | FC Param CR | Dense FC MAC-eq | FCTTD FC MAC-eq | FC MAC reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AlexNet | classifier.1;classifier.4;classifier.6 | 58,631,144 | 29,912 | 1960.12× | 58,621,952 | 7,103,164 | 8.25× |
| GoogLeNet | fc | 1,025,000 | 7,640 | 134.16× | 1,024,000 | 581,610 | 1.76× |

这里的压缩率是被替换 FC 层的局部结构指标，不应与论文中 13.5× 的整网、
带准确率约束参数压缩率作一一等预算比较。本实验使用固定 ranks 且不评估
准确率，目的仅是隔离 FC 层的参数和算术效率。

## 为什么总体加速有限

| Model | Dense FC compute share | Non-FC compute share | Implied full-model operator speed-up | Absolute ceiling if FC cost=0 |
| --- | ---: | ---: | ---: | ---: |
| AlexNet | 8.2081% | 91.7919% | 1.0777× | 1.0894× |
| GoogLeNet | 0.0680% | 99.9320% | 1.0003× | 1.0007× |

AlexNet 和 GoogLeNet 的卷积/非 FC 主干分别占统一统计口径下约
91.79% 和
99.93% 的计算。即使被替换的
FC 层本身的 operator count 分别缩减
8.25× 和
1.76×，按
Amdahl 定律摊到整网后也只有 1.0777× 和
1.0003×。

因此该小实验支持的准确表述是：FCTTD 首先带来显著的 FC 参数/存储压缩，同时
也降低了被替换 FC 层的理论计算量；但卷积主干占据绝大多数计算，使端到端
加速受到严格上限约束。这里没有测量 GPU/NPU wall-clock latency，不能用
operator-count speed-up 否定审稿人关于 kernel fragmentation 和硬件利用率的
意见。

## 限制

由于按要求不训练、不评估准确率，本实验只证明固定 rank 配置的结构效率，
不证明该极高压缩率是 accuracy-preserving operating point。
