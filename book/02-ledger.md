# GPU-02｜硬件｜性能账本：Roofline、算术强度与三类瓶颈

> 🎯 **这篇文档解决什么问题**：优化前先要知道还能快多少。一个 kernel 跑了 200 微秒，是已经贴着硬件上限，还是只用了一成算力？要靠一本账来回答：运算量有多少，搬运字节数多少，这张卡每秒能算多少、能搬多少。本篇从 Roofline 模型讲起，教你给逐元素、归一化、矩阵乘、注意力这些常见算子算强度，测出本机实际的两根线，用开销、带宽、算力三类瓶颈给实测时间归类。读完应当能在写代码前估出算子的时间下限，拿到 profile 后说清它卡在哪一类、该往哪个方向改。
>
> **使用方式**：第 1、2 章讲记账方法；第 3 章把纸面的线换成实测的线；第 4、5 章是全篇重点，讲如何判瓶颈、融合为什么有效；第 6 章用 ncu 验证；第 7、8 章分别算 LLM 推理和训练的账；第 9 章讲这本账在哪些地方会骗人。文中版本、规格和基准数字核对于 2026 年 11 月 4 日。
>
> **代码**：`repo/zero-to-gpu/02-ledger/`。以 `sim_` 开头的两个计算器只需 Python，文中所有表格都能在没有 GPU 的机器上复现；以 `gpu_` 开头的两个脚本和 `ncu_roofline.sh` 需要一张 Ampere 及以上的卡和 PyTorch 2.x。

<b>🧭 主线位置</b>：GPU-01 讲 SM 上驻留的 warp 不够多时访存延迟藏不住；本篇处理的等待是建坐标，给每个 kernel 和每个训练步算出理论时间，判断等待落在算力、带宽还是固定开销。同时首次给出全系列的时间账记号（T_compute、W_mem、W_host、W_comm、W_idle）和四把尺子（GPU-Util、SM Activity、Tensor Core Active、MFU/HFU）；读完后瓶颈会转到 [GPU-03｜硬件｜CUDA 编程模型：线程层级、stream 与编译链](./03-cuda.md)，由它讲主机与设备如何互相等待。

本篇的核心观点是：<b>不测不知。</b>一个看似“跑了 200 微秒”的 kernel，到底是达到了硬件的 80% 利用率（已经很优秀），还是只用了 10% 算力（还有 9 倍的优化空间），两种情况下的改进策略完全不同。[[1]](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) 定义的 Roofline 模型提供了一套硬定律：给定运算量和搬运字节数，任何 GPU 上的执行时间都不可能比“运算量÷峰值算力”和“字节数÷峰值带宽”的最大值更短。这两个数决定了上下限，中间所有的变化都来自编译、调度、访问 pattern 的优化。建立这套账本是后续所有优化的前提；判断错了瓶颈，优化方向就反效果。[[2]](https://horace.io/brrr_intro.html) 的案例表明，一个看似简单的融合（把 4 个逐元素算子合并），不仅要计算能省多少字节，还要理解融合后是否还卡在带宽或转到了算力，才能决定下一步的方向。

## 导读 · 先给结论

1. <b>时间下限只由两个数决定。</b>运算量除以算力峰值，字节数除以显存带宽，取较大的那个就是下限，两项完全不重叠时是二者之和，所以下限和上限最多差 2 倍。[[1]](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)[[4]](https://jax-ml.github.io/scaling-book/)
2. <b>拐点是一张卡最重要的比值。</b>H100 SXM 的 BF16 稠密峰值 989 TFLOPS、显存带宽 3.35 TB/s，拐点约 295 FLOP/字节。算子的算术强度低于它就受带宽限制，高于它才有机会吃满 Tensor Core。[[5]](https://www.nvidia.com/en-us/data-center/h100/)[[4]](https://jax-ml.github.io/scaling-book/)
3. <b>深度学习里大部分算子都在拐点左边。</b>ReLU 的强度是 0.25，LayerNorm 不到 10，batch 为 1 的线性层约为 1。BERT 训练里矩阵乘占 99.8% 的运算量，只占 61% 的时间，剩下近四成花在几乎不算东西的归一化和逐元素算子上。[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)[[6]](https://arxiv.org/abs/2007.00072)
4. <b>厂商峰值要打折。</b>H100 SXM 上实测矩阵乘最好跑到 853 TFLOPS，持续负载下 786，分别是峰值的 86% 和 79%；B200 是 77% 和 64%。拐点要用实测值重算。[[7]](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)
5. <b>先判瓶颈再动手。</b>batch 翻倍时间只涨一成左右是开销受限，药是 CUDA Graph 和编译；带宽受限的药是融合和降精度；算力受限才轮到 Tensor Core 和低精度格式。[[2]](https://horace.io/brrr_intro.html)
6. <b>LLM 里算术强度差异最大。</b>prefill（序列很长）Roofline 左边高，decode（每次只生成 1 token）左边低；训练需要同时考虑参数梯度、激活重算和通信的额外字节。[[8]](https://arxiv.org/abs/2211.05102)[[9]](https://arxiv.org/abs/2204.02311)
7. <b>Roofline 的三根线分别出自谁。</b>算力峰值线来自 kernel 计算密度上限，HBM 带宽线来自硬件设计；实测时开销线取决于编译器与运行时，是三根里最飘的。[[10]](https://arxiv.org/abs/2009.05257)

### 总览图

![图](assets/02-ledger/fig-01.jpg)

### 📎 与系列其他文档的分工

GPU-00 索引给出全系列的路线图；GPU-01 讲 SM 和缓存层级；本篇给出算账的工具和方法，作为后续所有章节的统一尺度。GPU-03～08 深挖 L1 内的各项等待，每篇找到一个具体的消除手段；GPU-09～12 处理主机与驱动的延迟；GPU-14 讲怎么用 profiling 工具读这本账；GPU-16 定下单机验收的基线，后续集群与 LLM 层的所有等待都以此为参照。

## 第 1 章 · 在等什么：时间账的坐标系

### 1.1 一条折线隐藏的两个维度

一个 kernel 的运行时间 T 来自两项竞争：计算需要多久、搬运需要多久，取二者之和（如果完全不能重叠）或较大者。对同一个 kernel，如果搬运的字节数 B 按一倍、二倍、十倍增长，运行时间怎么走？如果运算量 F 增长呢？这条折线就是 Roofline 模型，它把高维的性能投影到二维上：横轴是算术强度 I = F/B（每字节做多少次运算），纵轴是吞吐 T = F/t（单位时间的运算数），折点的位置由 P（峰值算力）和 W（峰值带宽）决定。[[1]](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)

Roofline 画在对数坐标系上，左边（强度低于 P/W）是一条斜线，高度等于带宽 W 乘以强度，在对数坐标上斜率为 1；右边（强度高于 P/W）是一条水平线，高度等于峰值算力 P。这两条线把所有计算都分为三类：计算受 HBM 带宽制约、受 Tensor Core 算力制约、或者受固定开销制约。每一类的优化手段不同，在错误的那一侧优化会事倍功半。

从数据流的角度，一个 kernel 是三元组 (F, B, W_latency)：做 F 次浮点运算，搬运 B 字节数据，暴露出 W_latency 的延迟。硬件有两个硬天花板：每秒能做 P 次运算（TFLOPS），每秒能搬 W 字节（TB/s）。实际时间下限取决于谁更紧：

**实际时间 T = T_compute + W_mem + W_host + W_comm + W_idle**

这是全系列统一的时间账。“暴露的”是关键词：被重叠藏到计算后面的等待不计入关键路径。T_compute 是有效计算的时间，等于 F / P（理想情况下完全用满 Tensor Core）；W_mem 是 SM 等 HBM 的访存延迟，来自 B / W 或更复杂的缓存 miss 模式；W_host 是主机与驱动侧的等待（PCIe、显存分配、kernel 发射），这章不展开；W_comm 是卡等卡或网络的时间，本篇范围外；W_idle 是硬件空转的时间，比如 GPU 等 CPU 任务分配。本章重点是 T_compute 和 W_mem 的比较。

<b>四把尺子（GPU-02 首次定义，后续章节不重复）</b>：GPU-Util 回答采样窗口内是否有任意 kernel 在跑，SM Activity 回答有多少 SM 和 warp 在干活，Tensor Core Active 和 DRAM Active 回答矩阵单元和 HBM 是否被喂饱，MFU/HFU 和 tokens/s 回答端到端有效产出。后续章节引用时只用名字，不再解释。

**这一层在等什么**：单个 kernel 内部，计算单元在等数据。搬运与计算的时间关系决定了峰值打多满。

### 1.2 Roofline 为什么有两条线

任何计算都需要运算和搬运两个动作。算力峰值 P（单位时间能做多少运算）和带宽 W（单位时间能搬多少字节）是卡的两条硬限。对于算术强度 I 的计算，理论时间下限是 max(F/P, B/W)，即 Roofline 模型的两条线。折点在 I = P/W，小于折点时带宽限制，大于折点才能用满算力。[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

H100 SXM 的 BF16 稠密峰值是 989 TFLOPS，HBM3 带宽 3.35 TB/s，拐点 = 989 / 3.35 ≈ 295 FLOP/字节。同一张卡，BF16 Tensor Core 峰值 989 TFLOPS，但 FP32 非 Tensor Core 只有 67 TFLOPS，拐点会不同。[[5]](https://www.nvidia.com/en-us/data-center/h100/)

厂商给的都是理论峰值，实测会更低：H100 上 BF16 矩阵乘最好跑到 853 TFLOPS（86% 峰值），持续负载 786 TFLOPS（79%）。拐点要改用实测值，称为”实测拐点”。所以有理论拐点 295 和实测拐点 261，两者差不多一成。这就是为什么优化前必须在本机实测一遍 Roofline。用 sim_roofline.py 或在 GPU 上跑一个简单的矩阵乘测试，记下完全矩阵乘的吞吐（比如 4096³），就能得到本机的 MSMF 值；再量一个 HBM 读取的吞吐（比如用 reduce 或 memcpy），得到实测带宽。两个数字一组合，实测拐点就有了。

### 1.3 算术强度与三类等待

运算量 F 和搬运字节数 B 来自哪里？对一个循环或 kernel，通过代数化简能得到显式的 F 和 B，算出强度 I = F/B。然后在 Roofline 上找到对应点，判断它落在哪一侧或折点附近。若点在折点左边（I &lt; P/W），瓶颈是带宽；在右边（I > P/W），瓶颈是算力；在折点处，两项各占一半。

但 Roofline 没画出的第三类等待是固定开销，比如 kernel launch 延迟、显存分配、PCIe 握手，它与 F 和 B 无关，只跟”做多少件事”有关。当处理的总数据量很小时，一个 5 微秒的 kernel launch 开销可能比计算本身还久。三类瓶颈的治疗方向完全不同：开销是减少 kernel 个数（融合或编译）；带宽是减少搬运字节（融合、降精度、重新分块）；算力是改进计算密度（更好的 Tensor Core 用法、更小的数据类型）。[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

衡量开销的方法很直接：用 nsys 看”两个 kernel 发射之间的间隙”。每个间隙上百纳秒到几微秒，积累下来就是 W_host。如果一个 step 有 50 个 kernel，每个间隙 1 微秒，累计 50 微秒的开销，而整个 step 只有 100 微秒，那 50% 的时间都花在了开销上，远超 Roofline 预测的值。计算开销的相对占比：总开销 W_host / (T_compute + W_mem)，如果超过 10%，就值得投入精力优化。

### 1.4 时间账的使用场景

给定一个模型、一张卡和一个 batch，怎么估出这个 training step 最快能跑多少？第一步算出全步的 F（大多数来自矩阵乘）和 B（L1～L3 缓存 miss 的字节加上主机侧的通信字节）；第二步分别除以峰值算力和峰值带宽，得到两个时间，取较大者就是最乐观的下限。[[4]](https://jax-ml.github.io/scaling-book/) 如果实测比这个下限快不了，瓶颈要么是某项没算清，要么是该优化的方向用错了。

反之是算力瓶颈；这三类瓶颈的处理方向完全不同，诊断错就会白费优化工作。

时间账本身没有 profile 数据，所以会骗人：比如忽略了某个中间结果的回写、或者通信字节算少了。第 9 章会讲这些陷阱。Roofline 只预测运行时间，不直接预测能耗或通信成本，后续章节会引入”通信账”和”显存账”作为时间账的输入。

> 💡 **常见误区**：
>
> - <b>“GPU-Util 100% 就是没有等待”</b>：错。GPU-Util 采样 100 毫秒，其中哪怕只有 1 微秒在跑 kernel，也显示 100%。一个只占 1 个 SM 的 kernel 也能让整卡 GPU-Util 100%。看饱和度要看 SM Activity 或 MFU。
> - <b>“带宽受限的 kernel 没法加速”</b>：错。可以降低精度（4 bit 权重让每个字节搬的数据少一倍，强度翻倍），可以融合多个 kernel 减少搬运次数，可以用 cache-friendly 的分块减少 HBM 访问。
> - <b>“优化就是逼近峰值”</b>：错。LLM 推理 decode 阶段的线性层几乎不可能逼近峰值，因为强度太低。优化的目标是消掉不必要的等待（等主机、等内存、等通信），而不是盲目追峰值。

> 🔑 **本章要点**：时间账 T = T_compute + W_mem + W_host + W_comm + W_idle 给出了性能建模的坐标系；Roofline 的两条线和四把尺子是诊断任何 kernel 性能的起点；判断瓶颈是带宽、算力还是开销，决定了优化方向，本章通过建立这个坐标系，隐藏了 kernel 内部等数据的访存延迟。

## 第 2 章 · 常见算子的强度与分块

时间账的第一步是认清每个算子的强度，也就是算术密度。这决定了它是否靠近 Roofline 的左边（带宽限制）还是右边（算力限制）。不同的算子强度差异很大，从 0.25 FLOP/字节的逐元素操作到 1365 FLOP/字节的大矩阵乘。理解这些差异是优化的起点。

### 2.1 逐元素与归一化：内存密集的极端

ReLU、GELU、LayerNorm、Dropout，这些算子虽然计算逻辑简单，但数据搬运成本相对于运算来说太高。一个 64M 元素的 ReLU，做 64M 次乘法或比较，搬运 256 MiB（BF16 每元素读 2 字节、写 2 字节）；算术强度 64M ÷ 256 MiB ≈ 0.25 FLOP/字节。根据 [[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)，H100 的拐点是 295 FLOP/字节，所以 ReLU 远在左边，受 HBM 带宽限制。LayerNorm 给 8192×8192 矩阵，每行计算 8192 元素的均值和方差，每个元素同样读一次、写一次，搬运 256 MiB，每个元素只做求和、平方、减均值、乘缩放这几次运算，强度在 1 FLOP/字节的量级，仍然不到拐点的 1%。

这类算子的实测时间由 HBM 带宽决定，Roofline 的算力线对它们完全没用。[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) 给出的带宽实测约 3.02 TB/s（乘以 0.9 的有效系数），所以 ReLU 的理论下限是 256 MiB / 3.02 TB/s ≈ 85 µs。code_outputs.md 的模拟结果将这一类算子全部放在 0.1% 峰值的位置，证实了它们极其受带宽制约。

<b>融合的价值。</b>单个逐元素算子虽小，但往往串联使用。如 LayerNorm → 线性层 → GELU 这样的组合，分开跑时中间输出都要落回显存，总搬运是 4 倍。融合后中间结果留在寄存器或共享内存，减少 HBM 往返。这是优化的重点，但需权衡：共享内存有限，融合太多会寄存器溢出，反而降速。

### 2.2 矩阵乘：从左边到右边的跨越

矩阵乘 C = A × B，其中 A 为 m×k，B 为 k×n，运算量 m×n×2k（乘加算作 2 个操作），搬运 m×k + k×n + m×n 字节（假设 C 写回）。算术强度 I = m×n×2k / (m×k + k×n + m×n)。当 k 很大时，约等于 2mn/(m+n)。

方阵（m=n=k=4096）的情况：运算 137.4 GFLOP，搬运 96 MiB，强度 1365 FLOP/字节，远大于拐点 295，所以是算力受限。[[11]](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) 给出的 GEMM 性能指标表明，这样的矩阵乘能达到接近峰值的吞吐。但情况完全不同于单 token 的线性层：batch=1 时线性层是 1×4096 的输入乘以 4096×14336 的权重，运算 2×14336M = 115 GFLOP，搬运约 115 MiB，强度约 1 FLOP/字节，反而在拐点左边，是带宽受限。

<b>强度随 batch 增长。</b>这是 Transformer 推理和训练的核心：batch 翻倍时，矩阵乘的强度快速上升。根据 code_outputs.md 的 sim_roofline.py decode 输出，线性层 d=4096、f=14336 的强度随 token 数增长为 1.0、8.0、31.7、62.7、123.1、237.0 FLOP/字节，越过拐点 295 约需 295 个 token。如果权重用 4 bit 精度存储，搬运字节减半，拐点也减半，只需约 74 个 token。这说明量化不仅减少显存占用，还能显著提升推理时的算力利用。

### 2.3 分块强度与片上数据复用

在 Tensor Core 上跑满计算单元，必须保证持续有数据供应。一个 SM 的片上内存（共享内存）只有几百 KB，无法容纳完整的大矩阵，所以必须分块。一个 4096³ 的矩阵乘，分块成 64×64 的小块，每块的工作量是 64×64×2×64 = 512 KiB（double 精度计数），再加上两个输入块，总共 1.5 MiB 左右，超过 H100 的 192 KB 共享内存。

<b>多层分块与分层强度。</b>分块大小决定了片上数据复用和每一层的算术强度。据 code_outputs.md，4096³ 矩阵乘全局强度 1365 FLOP/字节，但分块 64×64 时片上强度仅 32 FLOP/字节，256×256 时 128 FLOP/字节。分块越大重复使用次数越多，片上强度越高，但管理成本也增加。[[12]](https://siboehm.com/articles/22/CUDA-MMM) 详细记录了多层分块优化接近 cuBLAS 性能的过程。

多层 Roofline 的用处在于此：全局 Roofline 告诉你数据来自 HBM，但片上 Roofline 告诉你数据来自 L2 或共享内存，优化方向完全不同。如果片上带宽是瓶颈，你要增加分块大小、提高数据复用；如果 HBM 带宽是瓶颈，你要减少数据往返、用融合或低精度。

### 2.4 手写 kernel 的目标设定

给定一个 kernel 的逻辑，怎么确定优化目标？第一步是测出当前性能，第二步是算出理论下限（根据 Roofline 和 code_outputs.md 里的数据），第三步是看差距从何而来，第四步才是优化。

如果差距来自未充分利用 Tensor Core，优化方向是减少搬入 HBM 的字节数（让两个输入完全走缓存）。如果来自带宽饱和，优化方向是增加强度（更大的分块以降低 L2 访问次数）或降低精度（用 FP8 或 FP4 减少搬运数据）。[[13]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) 的 CUDA Best Practices 提醒要根据 profiler 的数据来指导优化，而不是凭直觉。

<b>收益递减律。</b>多数情况下，前 20% 的优化工作能获得 80% 的收益。手工优化 CUDA kernel 的后续工作往往收益递减，甚至因为代码复杂度增加、寄存器压力、缓存一致性问题而反向优化。所以实践中要权衡投入和产出。矩阵乘接近实测上限（H100 BF16 的 MAMF 约为峰值的 86%）就到头了，继续抠 kernel 不如换算法，或交给编译器（如 Triton）。

![图](assets/02-ledger/fig-02.jpg)

> 💡 <b>常见误区。</b>强度是看错账的常见原因。一个算子的全局强度高，不代表它不受带宽制约，片上强度低也会构成瓶颈。另一个误区是混淆 FLOPS（浮点数操作数）和有效 FLOPS。一个 kernel 的算术吞吐可能很高，但如果在重复计算或搬运多余数据，有效 FLOPS（结果产出速率）会很低。Roofline 的两条线都是硬约束，超越不了；选错了方向，优化效果反而恶化。

> 🔑 <b>本章要点。</b>算术强度决定了算子是受带宽限制还是算力限制；矩阵乘的强度随 batch 快速增长，从带宽限制跨越到算力限制；分块的多层设计能优化片上的数据复用强度；本章通过强度这个指标，消除或隐藏了对每个算子性能上限的猜测。

## 第 3 章 · 实测的两根线：从理论到真卡

Roofline 的算力线和带宽线上写的是理论值，但实际 GPU 打不到。厂商给的 BF16 Tensor Core 峰值是 989 TFLOPS，但 H100 上最好的矩阵乘只跑到 853 TFLOPS（86% 峰值）；持续负载下更低，约 79%。B200 的情况更糟，实测比例只有 77% 和 64%。为什么？因为 Tensor Core 之间要互相等待数据，搬运单元和计算单元的吞吐比例受 GPU 芯片内部通路宽度的限制，所以永远有某个单元在空转。理论拐点必须用这个实测的峰值来算，不能用厂商数字。

### 3.1 峰值打不满：实测最好成绩与理论的鸿沟

[[7]](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md) 和 [[14]](https://arxiv.org/abs/2402.13499) 都详细分析了这个现象。NVIDIA 规格表里的 989 TFLOPS 是在理想的 access pattern 下的理论峰值；实际的 kernel 要处理数据对齐、缓存失效、通讯延迟，所以永远达不到。H100 上最好的矩阵乘实测吞吐约 786 TFLOPS（79% 峰值），这是卡在功耗上限、频率降下来之后还能稳定保持的成绩，叫 MSMF（Maximum Sustainable Matmul FLOPS）；短时跑出的最好成绩 853 TFLOPS 叫 MAMF（Maximum Achievable Matmul FLOPS）[[7]](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)。

code_outputs.md 的 sim_roofline.py ridge 输出给出了标准硬件的 MSMF。A100 只有 MAMF 271 TFLOPS（87% 峰值），表里用它代替；B200 的 MSMF 是 1439 TFLOPS（64% 峰值）[[7]](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)。这个差异来自芯片架构的演进：A100 的计算单元和缓存通路设计比较均衡，所以能达到较高的百分比；B200 增加了 Tensor Core 数量，但通路带宽没有同比增长，所以百分比反而下降。这说明新卡不一定比旧卡“更好”，而是要看应用的特征是否匹配。

<b>两套 Roofline 线。</b>本篇引入“实测拐点”的概念：用 MSMF 和实测 HBM 带宽来计算。理论拐点是 989 / 3.35 = 295 FLOP/字节，但实测拐点是 786 / 3.02 = 261 FLOP/字节（3.02 TB/s 是按规格带宽的 90% 假设的可达带宽，见 3.2 节）。这两个拐点差不多，但意义完全不同：理论拐点告诉你数据流动的瓶颈在哪，实测拐点告诉你在真实硬件上会遇到的折点。绘制 Roofline 时必须画两套线，这样才能看清实际的优化空间。

### 3.2 带宽也要实测

理论 HBM 带宽往往以厂商规格为准：H100 是 3.35 TB/s，B200 是 8 TB/s（来自 [[15]](https://www.nvidia.com/en-us/data-center/dgx-b200/)）。实际 kernel 跑到的带宽低于规格值，低多少取决于访问模式和芯片，要在真卡上用带宽测试程序测。本篇没有可引用的带宽实测表，估算时按规格值的 90% 取可达带宽，这是假设：H100 取 3.35 × 0.9 ≈ 3.02 TB/s，B200 取 8 × 0.9 = 7.2 TB/s。换成自己的实测值后，下文的“实测拐点”都要重算。

为了排除缓存热（某些数据可能被 L2 命中），多跑几次同一个 kernel，取中间几次的平均值，不要用单次结果。

这个可达带宽可以作为 Roofline 上的“现实线”。拐点的计算改用这个有效带宽，结果更接近实际优化的难度。一个算子的强度略高于理论拐点 295，看起来应该是算力受限；但用实测拐点 261，就只是勉强到达，带宽仍然是制约因素。这个差异对于 batch=1 推理很关键。

### 3.3 非 Tensor Core 算子的屋顶

GPU 上不止有 Tensor Core。FP32 的标量单元或 SIMD 单元吞吐远低于 Tensor Core：H100 SXM 上 FP32 的理论峰值是 67 TFLOPS[[5]](https://www.nvidia.com/en-us/data-center/h100/)，约为 BF16 Tensor Core 的 1/15（989 ÷ 67 ≈ 14.8）。如果用 FP32 做 ReLU 或其他逐元素算子，Roofline 的算力线应该在 67，不是 989，拐点也相应下移到 67 ÷ 3.35 = 20 FLOP/字节。

[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) 明确指出了不同数据类型的峰值。在实际应用中，ReLU、GELU、LayerNorm 这类逐元素操作用 BF16 或 FP8 实现；矩阵乘用 Tensor Core 的 BF16；这样才能获得最好的性能。误用 FP32 做这些操作，Roofline 的屋顶会大幅下降，优化空间也随之缩小。

具体对标时，查一下硬件规格说明书（如 H100 的 datasheet）找出不同单元的峰值。小批量或序列短的情况下，逐元素算子的开销相对较大（kernel launch 延迟占比高），所以优先考虑融合，不要为每个算子独立发起 kernel。如果逐元素算子无法融合（比如在复杂的控制流中），至少要用 BF16 或 FP8，不要用 FP32 白白浪费吞吐。

![图](assets/02-ledger/fig-03.jpg)

> 💡 <b>常见误区。</b>很多人看到测出的吞吐与厂商数字差距很大，就觉得硬件有问题。这是正常现象。厂商数字通常是在理想的 access pattern、完美的缓存命中率、理想的功耗限制下的峰值，实际应用难以达到。另一个误区是只关注算力线、忽视带宽线。Roofline 两条线都是硬约束，超越不了；理解它们的交点（拐点）位置，才能正确判断优化方向。

> 🔑 <b>本章要点。</b>理论峰值与实测最好成绩之间的打折（H100 约 80%）来自硬件内部的通路限制，不是 bug；实测拐点（261 FLOP/B）比理论拐点（295）更接近实际优化的难度；不同的计算单元（Tensor Core vs FP32）有不同的屋顶；本章通过实测线替代理论线，隐藏了厂商规格与实际性能之间的鸿沟。

## 第 4 章 · 三类瓶颈与判断方法

### 4.1 开销、带宽、算力：治疗方向完全不同

一个 kernel 的执行时间受三类因素制约，每类都有不同的根源和优化手段。第一类是**开销**：kernel launch、显存分配、PCIe 握手、Python 调度，这些固定成本不随计算量增长。第二类是**带宽**：从 HBM、L2、共享内存搬运数据需要时间，当数据流量成为瓶颈时，浮点单元空转。第三类是**算力**：Tensor Core、标量单元的吞吐有上限，当这个上限成为瓶颈时，存储系统有富余。三者的治疗方向完全不同。[[2]](https://horace.io/brrr_intro.html)

开销的消除靠**减少 kernel 个数**，手段包括算子融合（把多个小 kernel 合成一个）、编译优化、CUDA Graph 把图形绘制开销摊销。[[16]](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) 明确指出 CUDA Graph 可以把 Python 层面的调度开销从毫秒级降到微秒级。带宽的优化靠**减少搬运字节**，手段是融合中间结果使其不落显存、降精度（FP8、FP4）减少字节数、改进分块模式降低重复读取。[[6]](https://arxiv.org/abs/2007.00072) 的量化表明，在 Transformer 中，50% 的时间花在 9% 的数据搬运上。算力的优化靠**改进计算密度**，手段是更好地利用 Tensor Core（批处理更大的矩阵、cache-friendly 访问 pattern）、调整数据类型（BF16、FP8 有不同的吞吐上限）。三条路径互相不兼容：为了消除开销而增加 kernel 复杂度可能增加带宽需求；为了减少带宽而过度融合会降低 Tensor Core 利用率。所以在做优化前必须判断当前的瓶颈类别，否则方向错了，优化反而把性能拉低。

### 4.2 batch 翻倍判瓶颈

一个实用的实验方法是 batch 翻倍观察时间变化率。测量一个模型 forward pass 的端到端时间，首先用 batch=1，记下时间 t₁；然后改成 batch=2，记下时间 t₂。计算比率 t₂ / t₁，从这个比率可以推断当前的瓶颈。

当 t₂ / t₁ ≈ 1.1 时，时间只增加一成，说明**开销受限**。因为 batch 翻倍时，kernel 内的计算和数据搬运都翻倍，但 launch 开销、显存分配、同步等固定成本只多一点点，所以整体时间增长不明显。这种情况下，优化的重点是减少 kernel 个数，用 CUDA Graph 或 torch.compile。

当 t₂ / t₁ 在 1.5～2.0 之间时，说明**带宽受限**。batch 翻倍时搬运的字节翻倍，带宽饱和，算力单元无法充分利用。这种情况下，优化的重点是减少搬运，比如融合相邻算子，或者降精度。

当 t₂ / t₁ 接近 2.0 甚至更大时，说明**算力受限**。batch 翻倍时计算量翻倍，需要更多时间才能用光 Tensor Core 的吞吐。这种情况下，优化的方向是改进矩阵乘法的密度、增大批处理块的尺寸。

这个判断方法的前提是消除缓存效应，所以不要在相邻的两个 batch 大小（比如 8～9）之间测，而要选择变化明显的区间，比如 1～8。同时要注意，同一个模型在不同卡上的瓶颈可能完全不同。H100（高算力、高带宽）可能在某些 batch 大小下是算力受限，而 V100（算力和带宽都较低）同样的 batch 大小可能是带宽受限。所以这个方法需要每个硬件和工作负载上独立验证。[[2]](https://horace.io/brrr_intro.html)

### 4.3 带宽到算力的拐弯：增加强度的手段

本节通过一个完整的案例展示五步闭环如何应用到真实的优化问题。选择的案例是四个逐元素算子的融合，这是最常见的优化模式。

**定天花板**。四个逐元素算子（ReLU、gelu、dropout、layer norm），处理 64M 个 BF16 元素，约 128 MB 数据。每个算子单独跑时，都是搬运 256 MiB（读进来、改变值、写回去），四个共 1024 MiB。[[2]](https://horace.io/brrr_intro.html) H100 的 HBM 带宽是 3.35 TB/s，所以理论下限是 1024 MiB ÷ 3.35 TB/s ≈ 320 µs。

**测实际**。在真卡上用 nsys 分别量四个算子的时长，和每个 256 MiB ÷ 3.35 TB/s ≈ 80 µs 的下限比；本篇没有真卡数据，用 sim_regime.py 的模型值代替。实测如果远超下限，先查访问是否合并。此时每个算子的强度都很低（ReLU 约 0.25 FLOP/字节），远低于拐点 295，确实是带宽受限。

**定位等待**。问题在于，四个输入数据的每一份都被读了四次，但数据的四次流经可以合并成一次。理论上，整个管道的有效搬运只需要 256 MiB（第一个读、最后一个写），但硬生生搬了 1024 MiB。这是融合的机会。

**消除或隐藏**。把四个算子融合成一个 CUDA kernel，在每个线程上流水执行：读数据一次，执行四个操作，写回去。这样总搬运变成 256 MiB，下限 80 µs。本篇 `python3 sim_regime.py fuse --n 4` 的结果，融合后搬运 256 MiB，下限 80.1 µs。

**复测**。融合后的下限约 80 µs，是分开跑的四分之一，真卡上要用 nsys 复测确认。但新的瓶颈是什么？融合后每个元素做几次运算、搬 4 字节，强度仍在 1 FLOP/字节的量级，远低于拐点 295，所以**还是带宽受限**。继续优化的手段是降精度：如果用 FP8 或 FP4 存中间结果，搬运字节进一步减少，强度继续上升。或者，如果四个操作里有可以跳过的（比如某些层的 dropout），代码级别的消除也能减少搬运。

这个案例的关键点是，融合不是万能的。融合成功的前提是片上内存（共享内存、寄存器）足够；如果融合后 register spilling（寄存器溢出）或 occupancy 大幅下降，加载到片上的开销可能反而增加。此外，融合后的强度虽然提升了，但不一定越过拐点进入算力受限；如果没有越过，继续优化的方向应该是降精度或减少搬运，而不是盲目增加 kernel 复杂度。

> 💡 **常见误区：FLOPS 越高越好**
>
> 关键是有效的 FLOPS，即吐出结果的速度。重复计算（比如为了融合而重新读取已经在 L1 的数据）或搬运多余数据会拉低有效 FLOPS。有些高性能 kernel 故意降低 arithmetic intensity，换取更好的缓存 locality 或 Tensor Core 利用，最终吞吐反而更高。

> 💡 **常见误区：Tensor Core 越多越好**
>
> 如果数据搬运成为瓶颈，Tensor Core 再多也无法提升性能，甚至因为更多 Tensor Core 空转而增加热量和功耗。高性能 kernel 的设计首先要确保瓶颈不是带宽；如果是，增加 Tensor Core 个数反而会浪费功耗预算。

![图](assets/02-ledger/fig-04.jpg)

> 🔑 **本章要点**：三类瓶颈（开销、带宽、算力）的治疗路径截然不同，判断错了方向就反效果。batch 翻倍法是快速定位瓶颈的实用工具，通过观察时间增长率推断当前的制约条件。融合四个逐元素算子的完整案例展示了如何从带宽受限优化到接近拐点。本章消掉的是对“优化”的盲目假设，建立了“定位才能优化”的因果链条。

## 第 5 章 · 融合为什么有效

### 5.1 中间结果不落显存

融合的核心价值是**消除中间结果的显存往返**。考虑一个常见的融合场景：矩阵乘（GEMM）后接 ReLU 和 layer norm。如果三个操作分开跑，GEMM 的输出（几百 MB）必须写回 HBM；然后 ReLU 从 HBM 读这份数据，计算后再写回 HBM；最后 layer norm 又从 HBM 读。整个管道的总搬运量是原始 GEMM 搬运的 2～3 倍。而且，三个 kernel 的执行是串行的，GPU 的跨 SM 并行度被严重浪费。

融合后的执行过程是：GEMM 计算一个数据块，结果直接在寄存器或共享内存中被 ReLU 消费，不写回 HBM；ReLU 的输出再被 layer norm 消费；最后只有 layer norm 的结果落地。这样，整个管道的搬运量从 3 倍降到 1 倍，应用的 Roofline 模型中，横轴（强度）上升。当强度足够大时，应用从左边（带宽限制）走向右边（算力限制），甚至可能越过拐点，从内存密集变成计算密集。[[17]](https://pytorch.org/assets/pytorch2-2.pdf)

融合的限制来自**片上内存**。寄存器和共享内存是稀缺资源，中间结果太大时无法全部驻留。比如，64×64 的 BF16 输出块是 64 × 64 × 2 = 8 KB，三个操作各留一份中间结果是 3 × 8 = 24 KB；256×256 的输出块按 FP32 累加是 256 × 256 × 4 = 256 KB，已经超过 H100 每个 SM 最多 228 KB 的共享内存，只能溢出回 HBM。此时融合反而会增加带宽。另一个限制是**Tensor Core 的 spilling**：为了融合 Tensor Core GEMM 和标量 ReLU，需要把 Tensor Core 的输出转存到寄存器或共享内存，再让标量单元读取，这个中间转换的开销有时大于省下来的 HBM 往返。所以实际优化中，融合多了也会卡在片上带宽，需要多层分块来平衡。

### 5.2 数据搬运才是瓶颈

另一个角度来看融合的价值是**摊销固定开销**。kernel launch 有固定的开销，量级是几微秒；下面的模拟取 5 µs，这是假设值，实际要用 nsys 在自己的机器上量。对于大算子（比如大矩阵乘），这个开销的相对影响很小；但对于小算子，开销可能压倒有效工作。

本篇的 `python3 sim_regime.py launch --us 5` 给出了具体的数字：一个 5 µs 固定开销的 kernel，计算 a+b（逐元素加法）。对 4096 个 BF16 元素，有效计算时间约 0.01 µs，固定开销占 99.9%。对 65536 个元素，有效时间 0.12 µs，开销占 97.7%。对 1M 元素，开销占 72.7%。对 16M 元素，开销占 14.3%。所以，只有当元素数达到百万级甚至以上，固定开销才能被足够摊销。

这说明小算子的优化空间其实很小。再怎么优化逐元素加法的计算部分，开销不消，最多也就能得到 14.3% 的时间节省（对 16M 元素的情况）。相反，融合多个小算子成一个 kernel，可以摊销多次的固定开销，把总时间从 5N µs（N 个 kernel）降到接近 5 µs（1 个 kernel）。所以高性能框架（PyTorch、JAX）和编译器（torch.compile、TVM）都把算子融合作为优化的首要步骤。[[13]](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 5.3 谁来做融合

融合有三种方式，各有权衡。**手写 CUDA kernel 融合**是最底层的方式，给予最大控制力但工作量最大。需要手工管理共享内存（多少块驻留、如何同步）、warp-level 的同步点、register spilling 时的数据流。对于复杂的融合（比如融合一个 Tensor Core GEMM、一个 softmax 和一个 dropout），需要深入理解硬件的细节，开发周期长，容易出错。好处是能准确控制每一条指令，性能天花板最高。

**Triton 是一个中间层的方案**，提供了更接近算法、更少硬件细节的编程抽象。在 Triton 中，开发者声明数据的“瓦片”（tile）大小和访问模式，Triton 编译器自动生成 CUDA kernel，处理共享内存分配、warp-level 同步、register 溢出的策略。Triton 的 schedule 往往比手写更激进（更大的分块、更深的流水），所以性能有时能超过谨慎的手写 kernel。[[18]](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

**torch.compile + inductor 是自动融合**，开发者只需要写 PyTorch 代码，编译器自动分析计算图，识别可融合的算子对，生成融合后的 kernel。inductor 内部是 Triton，所以融合的结果质量接近手写。但不是所有融合模式都支持：比如分支逻辑复杂、数据依赖不规则的融合，编译器可能放弃。[[17]](https://pytorch.org/assets/pytorch2-2.pdf)

三种方式的选择取决于性能要求和开发成本。对于通用的融合（比如 GEMM + ReLU + LayerNorm），torch.compile 往往足够，甚至更优；对于特殊的业务融合（比如自定义的 attention 变种）可能需要 Triton；对于性能卡在极限的组件（比如 base attention kernel），可能才需要手写 CUDA。完整的生产系统往往三种都用：框架和编译器做通用优化，开源库（CUTLASS、cuBLAS）提供高度优化的手写 kernel，业务代码用 Triton 做定制化融合。

> 💡 **常见误区：融合了就一定快**
>
> 融合后如果片上内存不足导致 register spilling、或共享内存竞争增加导致 occupancy 大幅下降、或融合后的强度反而降低（因为增加了标量操作），性能反而可能更慢。最典型的例子是，某些 Tensor Core GEMM 后融合一个 softmax：直接融合会打破 Tensor Core 的流水，导致吞吐下降；但如果拆开来、用 softmax 的特殊 kernel（比如 FlashAttention），反而更快。

> 💡 **常见误区：能写出高性能 kernel 就不需要编译器**
>
> 编译器的融合和 schedule 往往比人的直觉更好，特别是对新的硬件特性（比如 Hopper 的 TMA、Tensor Memory Accelerator）。编译器能尝试更多的分块组合、自动调优超参数、考虑全局的数据流和缓存。而手写 kernel 往往是一个特定问题的最优解，泛化性差。

![图](assets/02-ledger/fig-05.jpg)

> 🔑 **本章要点**：融合通过消除中间结果的显存往返，减少搬运字节，提升应用的算术强度。同时融合摊销固定的 kernel launch 开销，对小算子特别有效。融合有手写、Triton、自动编译三种方式，各有权衡。本章消掉或隐藏的是无效的显存往返和 launch 开销。

## 第 6 章 · 用 ncu 验证理论与实测

前几章的 Roofline 和时间账都是理论计算，假设算术强度是理想的、缓存访问完美、没有协议开销。实际 GPU 芯片远复杂得多：缓存行对齐、预取策略、调度延迟、协议开销，每一项都会改变真实的性能。只有在真卡上跑 profiler 才能验证理论，也才能找到真正的瓶颈。nvidia-nsight-compute（ncu）是 NVIDIA 官方的 kernel profiler，能采样 Tensor Core 吞吐、HBM 字节、L2 缓存命中率、Tensor Core 流水线停顿等指标。[[19]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 它输出的数据与理论计算往往有 20～50% 的偏差，需要理解这些偏差的来源（硬件的真实样子与模型的假设差别）才能准确定位瓶颈并指导优化。

### 6.1 Speed Of Light 与分层 Roofline

ncu 的 Speed Of Light（SOL）分析会给每个 kernel 指定一个百分比，比如 32%，意思是这个 kernel 只用到 NVIDIA 厂商声称的理论峰值的 32%。但这里有个陷阱：SOL 的分母是厂商峰值（H100 BF16 是 989 TFLOPS，见 code_outputs.md），不是实测峰值。H100 持续满载时能保持的 BF16 矩阵乘约 786 TFLOPS，是厂商峰值的 79%，差额主要来自功耗上限迫使频率低于标称 boost 频率[[7]](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)。如果一个 kernel 的 SOL 是 32%，用厂商数字看是不达标的，但用实测峰值重算，32% × 989 / 786 ≈ 40%，反而相对不错。所以不要直接用 SOL 来评判优化结果，必须带上实测数据和问题上下文来重算。[[19]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

ncu 最有用的指标是 stall reason，直接显示 kernel 在等什么：HBM 延迟（mem_throttle）、L2 缓存冲突、Tensor Core 流水线停顿（pipe_busy）还是 SM 调度延迟。这种分层的停顿信息比笼统的峰值百分比有用得多，能直接指向优化方向。但也要注意，stall reason 是 profiler 采样窗口内看到的 top-1 现象，不一定代表全局的关键路径。长 kernel 内可能存在多个停顿阶段，采样频率不足时会遗漏某个阶段，导致判断失准。

### 6.2 分层 Roofline 与片上强度

除了全局的 Roofline（以 HBM 为参照），还可以根据数据来自 L2 或共享内存来分别画片上的 Roofline。一个 GEMM kernel，全局强度 1365 FLOP/字节（相对 HBM），但片上强度（相对 L2 和共享内存）只有 32～128 FLOP/字节，取决于分块大小（见 code_outputs.md 的 sim_roofline.py gemm 输出：分块 64×64 是 32，分块 256×256 是 128）。两个强度差 10 倍以上，说明虽然全局看起来是算力受限，但实际的瓶颈可能在片上的 L2 往返，而不是 HBM。[[10]](https://arxiv.org/abs/2009.05257) 这对优化指导意义很大：如果只看全局 Roofline，你会尝试增加 Tensor Core 并行度，但真实的限制来自 L2 缓存，应该增大分块以降低 L2 访问次数。

分层 Roofline 需要手工计算或用 ncu 的衍生指标（derived metrics）。ncu 提供的 "Achieved Bandwidth" 包括 HBM、L2、共享内存分层的读写，从这些数据可以反推每层的强度。比如，如果测得 L2 Achieved Bandwidth 为 200 TB/s（理论 L2 带宽的 50%），FLOPS 为 500 GFLOPS，那么片上强度是 500 GFLOPS / 200 GB/s ≈ 2.5 FLOP/字节，代表 L2 是瓶颈。但理解 ncu 计数定义很关键：L2 的字节数是指经过 L2 的数据，包括了从 HBM 重新读取的数据，不等于片上循环的数据量。

### 6.3 profile 时的数字为什么不一样

同一个 kernel，在 profiler 监控下跑出来的时间往往比不 profile 时慢 10～40%。为什么？profiler 的性能计数器工作原理是定期硬中断（采样），每次中断会从寄存器和计数器读数据，这过程会打破指令流水线，驱逐 L1 缓存，改变内存访问 pattern。一个 kernel 如果依赖 cache 的空间或时间局部性，硬中断每隔几千个周期就会插入采样，导致 cache 行被驱逐，后续访问变成 miss。所以 ncu 或 nsys 采到的数字是下界，实际裸跑时性能会更好。[[20]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

另一个常见的不一致是计数单位的定义差异。ncu 会输出 “DRAM Bytes Requested“，但这个”字节”的计数口径与理论计算不同。预取（prefetch）会主动多读，cache line 对齐会在一个请求里装入多个 8 字节单位，一个不对齐的 load 可能触发两条 cache line 的读取。所以按元素大小严格算出的理论字节数是下限，profiler 报的字节数通常会多一些，多出的部分来自预取和对齐的浪费。

### 6.4 把 ncu 的字节读回账本

ncu 输出的字节数需要对标理论计算，才能确认是否存在优化空间。一个解码线性层（batch=1，d=4096，f=14336）的理论搬运是权重（约 117 MB，即 112 MiB） + 输入向量（8 KiB） + 输出向量（28 KiB），总约 117 MB。实际运行时，ncu 报的 DRAM 字节数通常比 117 MB 略多，原因包括访问粒度的对齐浪费（显存按 32 字节的 sector 取数，一个 warp 合并访问 128 字节时正好取 4 个 sector[[21]](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)）和硬件预取。读数比理论下限多出一点，是硬件实现的开销，不是优化空间；如果多出好几倍，多半是访问没有合并，那才是优化空间。把这个 gap 搞清楚，才能在优化时知道该往哪个方向用力。[[19]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

这部分验证需要在真卡上跑 GPU 代码并对比 ncu 输出，超出本篇的纯理论范围。所以后续的 [GPU-14｜单机｜性能分析与 Profiling](./14-profiling.md) 会专门讲如何用 ncu 的工作流反推实际的数据搬运模式、缓存行为和停顿原因，并用实例对比理论与实测的鸿沟。现阶段重点是认清楚 Roofline 模型和理论时间账本身的局限，以及 profiler 数字背后的物理含义。

> 💡 **常见误区**：把 ncu 的 SOL 百分比直接当作优化目标。不同 kernel 的 peak utilization 因为数据局部性、调度开销等原因差异很大；80% SOL 的融合 kernel 可能效率已经不错，50% SOL 的大 GEMM 反而还有空间。另一个误区是信任 profiler 采样窗口的 stall reason：长 kernel 的分层 stall 现象会随时间变化，采样不足时可能漏掉真正的瓶颈。

> 🔑 **本章要点**：ncu 采到的 SOL、stall reason 和字节数都需要用硬件实现的物理含义来解释，而不是机械地对标理论。分层 Roofline 能区分瓶颈在 HBM、L2 还是 Tensor Core，直接指向优化方向。理论下限和 profiler 数据之间常见 20～50% 的差距，先逐项解释差距来自哪里（实测带宽低于标称、隐藏搬运、尾波），再决定它是可消的等待还是硬件本身的开销。本章消掉的等待项是“测实际”这个五步闭环里的关键一步，把测到的数据连接回 Roofline 模型中的“定位等待”。

## 第 7 章 · LLM 推理的账：prefill 与 decode

Transformer 推理分两个截然不同的阶段：prefill（一次性读入输入 context，可能有数千个 token）和 decode（逐个生成 token，每轮只新增一个）。这两个阶段的计算强度相差 2000 倍以上，数据搬运的成本结构完全不同，优化策略必须截然相反。理解它们各自的时间账和 Roofline 位置，是设计高效 LLM 推理系统的前提，也是理解为什么单卡吞吐和 batch 大小对推理性能影响这么大的根本原因。

### 7.1 decode 线性层的强度曲线

decode 阶段的线性层计算 y = W^T x，其中 W 是权重矩阵（通常 d_model × ffn_hidden_dim，比如 4096 × 14336），x 是当前 token 的向量（仅 1 个 token，1 × 4096）。运算量 2 × 4096 × 14336 ≈ 117 MFLOP（乘加算 2 个 op），权重搬运 4096 × 14336 × 2 字节 ≈ 117 MB（112 MiB），输入 8 KiB，输出 28 KiB，总搬运约 117 MB。算术强度 117 MFLOP / 117 MB ≈ 1 FLOP/字节，远低于 H100 的拐点 295 FLOP/字节。按 117 MB ÷ 3.35 TB/s 算，单个 token 的理论时间约 35 µs，本篇 sim_roofline.py decode 的输出是 35.1 µs。[[8]](https://arxiv.org/abs/2211.05102)

对标时，用实测的权重搬运量乘以 token/s，除以 HBM 带宽就能算出实际吞吐是否接近理论上限。如果实测吞吐明显低于理论值，问题往往在调度或融合不彻底。

但实际推理往往采用 batch 处理，同时生成 256 个 token 的情形是常见的。此时运算量翻 256 倍变成约 30 GFLOP，但权重 W 只需读一次，搬运只多了约 8%：权重 117 MB + 256 份输入 2 MiB + 256 份输出 7 MiB ≈ 127 MB。强度变成约 237 FLOP/字节，接近但仍低于拐点 295，时间 37.9 µs，几乎和 1 个 token 一样。batch 到 512，强度 441.1，越过拐点，变成算力受限；batch 到 1024，强度 774.9。sim_roofline.py decode 的输出显示：用 2 字节权重格式越过拐点需约 512 个 token，用 4 bit（0.5 字节）权重格式只需 128 个 token。这就解释了为什么 decode 的 batch size 对吞吐影响这么大：单个 token 的请求内存延迟很长，但 batch 聚合后能快速转换为算力受限，显著提升 GPU 利用率。

![图](assets/02-ledger/fig-06.jpg)

### 7.2 prefill 注意力的强度与成本

prefill 阶段计算自注意力 Q K^T V，其中 Q、K、V 都是 n×d 的矩阵（n 是 input context token 数，d 是每个 attention head 的维度，通常 128）。prefill 时 n 很大（比如 8192 token 的长 context），中间的 Q K^T 步骤是 n×d 和 d×n 的矩阵乘，结果是 n×n，运算量 2 × n^2 × d（乘加算 2 op）。搬运成本只是 Q、K、V 三个输入矩阵各 n×d 的数据，加上输出，约 4 × n × d 的字节（假设激活是同时存在的）。算术强度为 (2 × n^2 × d) / (4 × n × d) = n/2。当 n = 8192 时，强度达 4096 FLOP/字节，见 code_outputs.md 的 sim_roofline.py attn 输出，远超拐点 295，完全算力受限。这样的高强度意味着每个数据元素要被反复计算使用（在 n×n 的矩阵乘中），充分利用 Tensor Core 的吞吐，也是为什么 prefill 能接近峰值的原因。[[22]](https://arxiv.org/abs/2307.08691)

这与 decode 的注意力形成强烈的结构对比。decode 时每步只产生一个新 token，query 向量是 1×d，要与所有历史的 KV cache（n_old × d × 2）做注意力。运算量约 4 × n_old × d（QKᵀ 和乘 V 各 2 × n_old × d）。假设 context 长 32K token，用 FP8 精度存储 KV，每个 head 的 KV cache 约 32K × 128 × 2 × 1 byte = 8 MiB；一个 Transformer block 有 8 个 KV 头，总 64 MiB；32 层模型共 2 GB。不计 GQA 里多个 query 头共享一份 KV，每个头的强度为 (4 × 32K × 128) / (2 × 32K × 128 × 1 byte) = 2 FLOP/字节，又变回带宽受限。prefill 和 decode 的注意力落在 Roofline 的两个极端。prefill 该优化的是增加单个数据元素的重复计算（数据复用），decode 该优化的是减少搬运的字节数（降精度、减维度）。这决定了两个阶段的硬件需求、调度策略完全不同。

### 7.3 decode 注意力与 KV cache

decode 时每生成一个新 token，都要从 KV cache（过去所有 token 的 K 和 V）计算注意力权重。假设上下文有 32K token，KV cache 的大小是 32K × 128（head_dim） × 2（K 和 V） × 字节数。若用 FP8 压缩，每个注意力头约 8 MiB；一个 Transformer block 有 8 个 KV 头，总 64 MiB；一个 32 层模型就是 2 GB 左右。每生成一个 token，要把这 2 GB 都读一遍，用来计算注意力权重。运算量很小（矩阵-向量乘和 softmax），所以强度约 1 FLOP/字节（见 code_outputs.md sim_roofline.py attn），完全是内存延迟主导。[[23]](https://arxiv.org/abs/2309.06180)

![图](assets/02-ledger/fig-07.jpg)

新的优化技术是 PagedAttention，把 KV cache 分页存储到物理页而不是连续的虚拟显存，减少内存碎片和拷贝开销（这属于”隐藏等待”）。[[23]](https://arxiv.org/abs/2309.06180) GQA（grouped query attention）让多个 query 头共享少数 KV 头，比如 32 个 query 头共用 8 个 KV 头，相当于把 KV 搬运量减到 1/4，但这改变了模型结构，需要单独训练。这两个手段都不改变强度（仍是 1 FLOP/字节），但降低了显存占用或搬运字节，属于”消除等待”的一部分。更根本的优化是减少 KV 维度本身（蒸馏层数、量化到 1 bit）或改变存储排列（提高缓存友好度）。KV 从 FP8 进一步降到 4 bit，能把 KV cache 显存占用和搬运字节再减一半，但精度损失需要在离线评估（BLEU、ROUGE）上验证是否可接受。同一张卡，KV cache 越小，能容纳的并发请求越多，吞吐越高，这是提升 decode batch size 的关键。

> 💡 **常见误区**：认为 decode 阶段没有优化空间。虽然单 token 的计算强度低、完全被 HBM 带宽限制，但通过降精度（权重和 KV 用 4 bit）、减少搬运（GQA）、融合（在线 softmax、合并多头）等手段仍能提速 50% 以上。另一个误区是混淆 prefill 和 decode 的成本结构：prefill 的成本随序列长度平方增长（因为注意力矩阵 n×n），是算力受限；decode 成本随 batch 和上下文长度线性增长（每 token 要读一遍 KV），是带宽受限。两个阶段处于 Roofline 的两端，训推合一时必须分别优化，不能用同一套策略。

> 🔑 **本章要点**：prefill 时注意力的强度随序列长度平方增长，可达数千 FLOP/字节，是算力受限；decode 时强度固定在 1 FLOP/字节左右，是带宽受限。两个阶段的时间账结构、瓶颈位置和优化方向完全相反。batch size 对 decode 的影响极大，因为它改变了总的算术强度，从“单 token 带宽受限”变为“多 token 可能算力受限”。本章第一次把具体的 LLM 操作拉进 Roofline 和时间账模型中。

## 第 8 章 · LLM 训练的账：MFU 与每卡吞吐

### 8.1 每 token 运算量与 MFU 定义

训练一个稠密 transformer，每个迭代的运算来自两部分。矩阵乘贡献 6N（6 倍参数数做 forward 和 backward），注意力贡献 12LHQT（L 层数、H 头数、Q head_dim、T 序列长度）。对于 8.03B 稠密模型（L=32、H=32、head_dim=128），这两部分分别是 48.2 GFLOP 和 12.9 GFLOP，注意力只占 21%，说明绝大部分算力还在矩阵乘上。[[9]](https://arxiv.org/abs/2204.02311)

这个 6 倍系数怎么来的？如果用梯度检查点（gradient checkpointing）重算激活，前向会重跑一次，总成本变成 8N，代价是减少显存占用（见第 8.3 节）。用 FP8 训练的情况下，8.03B 稠密模型变成 FP8 权重 + 低精度梯度累积，实际搬运的字节数减半，所以显存账和时间账都会改变，对标论文时一定要明确是否混合精度、是否重算、是否量化。

MFU（Model FLOPs Utilization）= 模型每秒实际完成的浮点运算 ÷ 硬件峰值算力，衡量 GPU 上有效产出占峰值多少。[[24]](https://arxiv.org/abs/2407.21783) 在 H100 SXM 上，MFU 30% 需要每卡每秒 4859 个 token，MFU 50% 需要 8098 个 token。本篇 sim_roofline.py mfu 计算这些吞吐数字。换句话说，如果你的实际吞吐低于这个数字，要么是瓶颈没消除，要么是并行度不够。

MFU 是虚拟指标，分子必须明确定义为”有效”运算（不含重算、不含通信开销在关键路径上），分母也要明确定义（是否含所有层的注意力、是否含梯度求和）。[[25]](https://arxiv.org/abs/2410.06511) 不同论文的 MFU 口径可能不同，所以对标论文成绩时要细读其定义，否则会比出奇怪的结论。

一个容易犯的错误是用峰值算力（989 TFLOPS）直接除，但实际应该用”能达到的最好成绩”（约 786 TFLOPS，即实测 MSMF），拐点才对得上，MFU 才不会虚高。量实际 MFU 时，用 wall-clock time 计算：收集一整个训练周期（包括数据加载、forward、backward、通信、优化器更新）的端到端时间 T，以及这个周期里有多少个 token 被处理（N token），计算 MFU = (N × flops_per_token) / (card_peak_flops × T)。如果发现 MFU 低于预期（比如应该 40% 但实测 30%），用 ncu 和 nsys 对标：nsys 的时间线看是否有 kernel 间的空隙（W_host），ncu 的 warp 占用率看是否 Tensor Core 经常空闲（W_mem），这样才能定位具体的等待项。

![图](assets/02-ledger/fig-08.jpg)

### 8.2 注意力成本与通信开销

注意力虽然只占 21% 的运算量，但这部分计算与通信混杂，不能简单忽视。在分布式训练里，all-reduce（梯度同步）的字节数与矩阵乘的运算量无关，与参数总量有关，约 16N（FP8 时只需 2N）。[[26]](https://arxiv.org/abs/2104.04473) 如果通信没有被计算 overlap 掉，通信时间会直接加到训练周期上，把有效吞吐从时间账的 T_compute + W_mem 变成 T_compute + W_mem + W_comm，拉低 MFU。

所以高效训练需要两项并行：计算用 Tensor Core（消耗大部分时间），通信用 NVLink（机内）或网卡（跨机），彼此不互相阻塞。这等价于把 W_comm 从关键路径上挪下来，让 GPU 一边算梯度一边发送，而不是先算完再等通信。[[27]](https://arxiv.org/abs/2402.15627) 时间账变成只有 T_compute 和 W_mem 在关键路径，W_comm 被隐藏。这个目标需要 gradient accumulation、通信融合和 ring allreduce 等技术，第 21 章《训练并行》会详细讲。

还有一个细节：注意力计算里的 softmax 归一化算子，强度只有 1～2，如果逐个 attention head 计算，会产生大量的小 kernel，开销受限。好的实现会把注意力融合成一个大 kernel，或者用 FlashAttention 这样的优化算法，把搬运次数从 4 次降到 2 次，有效地提高了强度。

### 8.3 重算与显存账的权衡

Forward pass 的激活（intermediate tensor）需要保存用于 backward，但激活的显存占用往往很大。比如一个 8B 模型，128 token 的序列，每一层的激活有几 GB。一个优化手段是不保存激活，backward 时重算，代价是额外的计算开销（重算约占原计算量的 50%）。[[28]](https://arxiv.org/abs/2205.05198) 显存账会告诉最坏情况需要多少显存，但重算值不值得要用时间账来衡量：如果重算成本低于显存外溢导致的数据搬运成本（以及可能的 swap 到 CPU），就该重算。

这是显存账和时间账联动优化的典型例子。一个激活张量如果无法装进显存，GPU 必须把中间结果换页到 CPU 显存或磁盘，搬运的成本远高于重算。换页一次可能成本十几 GB，重算只是再跑一遍 forward（算力受限的矩阵乘，不怕）。反之，如果显存充足，不重算能省 50% 的计算。所以最优的选择取决于显存容量和显存带宽相对于计算的比例。

实践中，小模型或 GPU 显存充足时，不重算。大模型或使用了 LoRA 微调时，显存紧张，重算更划算。PyTorch 的 gradient checkpointing 提供了重算的 API，TorchTitan 等分布式框架也内置了这个选择。

> 💡 **常见误区**：
>
> - MFU 越高越好。不一定。为了提高 MFU 而用极大的 batch size，可能导致内存压力恶化、通信成本增加、延迟变长，综合体验反而下降。
> - 注意力的 MFU 一定很低。错。注意力虽然强度不如矩阵乘，但可以用好的分块和融合（FlashAttention）来弥补，[[22]](https://arxiv.org/abs/2307.08691) 并且是计算密集型（prefill）还是内存密集型（decode）取决于具体场景。

> 🔑 **本章要点**：训练阶段，MFU 把运算、带宽、开销三类等待折合到一个指标。注意力虽小但与通信混杂，重算与显存要权衡。本章消掉的是通信的等待（overlap），也埋下了第 21 章的主题。

## 第 9 章 · 账本在哪里会骗人

### 9.1 忽略的中间结果与实际搬运

理论搬运量 B 通常只算了算子的主要输入输出，但实际可能有隐藏的中间结果。比如，矩阵乘后接 softmax，softmax 内部要求 max、sum、norm，这些中间值可能要回写到 HBM（如果片上内存放不下），推高实际 B。[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) 又比如，一个 ReLU 在 backward 时要读 forward 的输出来判断符号，这部分也要算进去。矩阵乘的 backward，梯度对权重的累加（gradient accumulation）也是额外的搬运。

还有一类隐藏成本来自数据格式转换。比如输入是 FP8，计算用 FP32（累积精度），输出再转回 FP8，每个转换操作都是一次搬运。如果融合不彻底，单个转换也能产生 kernel launch 开销。

这就是为什么实测对标很重要。profiler（ncu、nsys）会告诉真实搬运了多少字节，帮你修正理论计算里的漏洞。如果理论与实测差异大（比如差 2 倍），优先怀疑搬运量的定义，其次怀疑算力或带宽的参考值。

### 9.2 缓存失效与带宽降低

理论带宽 3.35 TB/s 是在最好的访问 pattern（连续、合并、缓存热）下的成绩。[[5]](https://www.nvidia.com/en-us/data-center/h100/) 但实际的访问 pattern 可能不连续（gather、scatter、稀疏矩阵乘）、跨 channel（NUMA 跨 socket）、或者涉及多次小的 load（register spilling）。这些都会降低有效带宽。

实际带宽低于规格值。本篇的 Roofline 估算都按 3.35 × 0.9 ≈ 3.02 TB/s 取可达带宽，这个 90% 是假设（见 3.2 节），不是测量值。访问不规则的算子（稀疏矩阵乘、动态 shape 的 gather）能跑到的带宽会低得多，要在真卡上测。

一个容易犯的错误是用 3.35 TB/s 去计算下限，算出来的时间往往太乐观。更稳妥的做法是先用 3.02 TB/s（按 90% 假设）估算下限，实际跑的时候如果更快，那是意外收获；如果更慢，至少不会大惊小怪。

### 9.3 发热与频率抖动

Roofline 假设 GPU 的时钟频率和功耗状态稳定，但实际运行中，发热会导致频率降低（thermal throttling），功耗管理会动态调频。[[5]](https://www.nvidia.com/en-us/data-center/h100/) 所以同一个 kernel 跑两次时间可能不一样。而且 profiler 的采样（ncu、nsys）会改变系统状态：输出的信息少、发热少，影响了 thermal state。

要排除热效应，需要在稳定状态下多次重复测量并取中位数或平均，不要看单次结果。最坏的做法是冷启动机器、跑一个 kernel、立刻看时间，会比热稳定状态快 10～20%。对于持续任务（比如训练循环），第一次迭代往往比后续迭代快，因为系统还没有升温。正确的做法是预热（warm-up）5～10 个 batch，再开始计时，或者直接用 wall-clock time 除以 iterations，让抖动相互抵消。

profiler 时的数字往往比裸跑短 5～15%。这源于 profiler 本身对系统的扰动（中断、cache miss、thermal state 改变）。所以实际性能优化时，profiler 的吞吐数字是下界，不是上界。用 ncu 或 dcgm 看 GPU 的时钟频率，如果稳定且没有 throttle warning，说明系统温度和功耗管理正常；如果频率出现跳跃或抖动，说明有热管理干扰，需要延长预热或调整热管理策略（让 GPU 维持高频率运行）。

### 9.4 通信与计算未能完全 overlap

MFU 的计算假设所有通信都被 overlap 掉，但很难做到 100%。一部分原因是通信发起有延迟（发起 all-reduce 前要等待 kernel 完成），一部分是通信和计算在总线或缓存上竞争资源，彼此互相干扰。[[27]](https://arxiv.org/abs/2402.15627) 在多卡训练里，GPU 必须等待所有卡的梯度都到齐才能继续，所以慢卡会拖累快卡。所以实际的有效吞吐往往比理论 MFU 低几个百分点到 10%+。

要测试真实的 MFU，需要从端到端的训练时间计算（wall-clock time），而不是理论推导。公式是 MFU = (tokens_per_iteration × flops_per_token) / (h100_peak_flops × wall_clock_seconds)。如果实测 MFU 比理论低 10% 以上，说明通信 overlap 还有改进空间，可能需要调整通信与计算的发起顺序、融合通信的粒度、或者用 gradient compression 减少通信字节。

还有一个隐藏的 overlap 难点：forward 和 backward 的计算权重不同。forward 是矩阵乘（算力受限），backward 是矩阵乘加上梯度求和（混合瓶颈）。所以某个 backward 阶段可能在等 HBM 而非在等通信，此时通信早已结束，overlap 因此失效。

> 💡 **常见误区**：
>
> - 理论值与实测只差硬件效率，不会差太多。错。如果算错了搬运量的定义（比如忽略了 backward 的额外读写），差距能是 2～3 倍。
> - 一次测量结果就代表性能。错。需要多次重复、排除缓存热状态和频率抖动、在稳定工作状态下测，否则单次结果可能偏离 10～20%。

> 🔑 **本章要点**：Roofline 账本有四处会骗人：中间结果、非对齐访问、频率抖动、通信 overlap 不完全。都需要在真卡上对标、多次测、排除噪声。本章暴露的等待来自之前三类账本（时间、显存、通信）里不够精准的部分。

## 第 10 章 · Blackwell 时代的拐点：按精度分开算

前 9 章的拐点都以 H100 SXM 为参照：BF16 稠密 989 TFLOPS 除以 3.35 TB/s，约 295 FLOP/B[[5]](https://www.nvidia.com/en-us/data-center/h100/)。换到 Blackwell 之后，常听到一句“拐点大幅右移，访存瓶颈都变成算力瓶颈了”。这句话只对一半：按官网规格算，B200 和 B300 的 BF16 拐点约 281 FLOP/B，和 H100 几乎一样；真正右移的是 FP4 的拐点，而且 B300 比 B200 移得更多。本章先把四张卡、三种精度的拐点算出来，再用一个 decode GEMM 走一遍五步闭环，最后说明 MFU 的分母为什么也要跟着精度换。

### 10.1 · 从官网规格算出单卡数字

NVIDIA 官网给的是 8 卡整板的规格，而且 Tensor Core 数字大多是稀疏口径，页面脚注写明“稠密是稀疏的一半”[[29]](https://www.nvidia.com/en-us/data-center/hgx/)。所以单卡稠密峰值要做两步换算：先除以 2 去掉稀疏，再除以 8 摊到每张卡。HGX B200 和 HGX B300 的 BF16 都标 36 PFLOPS（稀疏），换算后每卡 36 ÷ 2 ÷ 8 = 2.25 PFLOPS；FP8 都标 72 PFLOPS（稀疏），每卡 72 ÷ 2 ÷ 8 = 4.5 PFLOPS。FP4 一栏同时给了稀疏和稠密两个数，稠密值 B200 是 72 PFLOPS、B300 是 108 PFLOPS，每卡分别是 72 ÷ 8 = 9 PFLOPS 和 108 ÷ 8 = 13.5 PFLOPS。同一颗芯片装进 GB200、GB300 超级芯片时功耗上限更高，NVIDIA 技术博客按芯片给的 NVFP4 稠密值是 Blackwell 10 PFLOPS、Blackwell Ultra 15 PFLOPS，FP8 稠密都是 5 PFLOPS[[30]](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。本章统一按 HGX 8 卡板卡口径算；换成超级芯片口径时用同一个算式重算，下面的结论不变。

带宽这一侧，DGX B200 8 卡共 64 TB/s HBM3e，每卡 64 ÷ 8 = 8 TB/s[[15]](https://www.nvidia.com/en-us/data-center/dgx-b200/)；GB300 NVL72 的 72 颗 Blackwell Ultra 合计最高 576 TB/s，每卡 576 ÷ 72 = 8 TB/s，用的仍是 HBM3e[[31]](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)。也就是说，B300 相对 B200 加了显存容量（HGX 整板 2.1 TB 对 1.4 TB）和 FP4 算力，没有加显存带宽。HBM4 要到 Vera Rubin 才出现，见 [GPU-32｜LLM｜Rubin 时代与 Dynamo 推理：Blackwell 之后的时间账（2026）](./32-rubin-dynamo.md)。

拐点 = 稠密峰值 ÷ 显存带宽。把上面的数代进去：

| 单卡（稠密口径） | 显存带宽 | BF16 峰值 / 拐点 | FP8 峰值 / 拐点 | FP4 峰值 / 拐点 |
|---|---|---|---|---|
| H100 SXM | 3.35 TB/s | 989 TFLOPS / 295 | 1,979 TFLOPS / 591 | 无原生 FP4 |
| H200 SXM | 4.8 TB/s | 989 TFLOPS / 206 | 1,979 TFLOPS / 412 | 无原生 FP4 |
| B200 | 8 TB/s | 2.25 PFLOPS / 281 | 4.5 PFLOPS / 563 | 9 PFLOPS / 1,125 |
| B300 | 8 TB/s | 2.25 PFLOPS / 281 | 4.5 PFLOPS / 563 | 13.5 PFLOPS / 1,688 |

表里的拐点单位都是 FLOP/B，算式就是两列相除，例如 B300 FP4：13.5 PFLOPS ÷ 8 TB/s ≈ 1,688 FLOP/B。读这张表有三个结论。同一精度下，B200、B300 的拐点和 H100 基本持平，BF16 是 281 对 295，FP8 是 563 对 591，所以在 H100 上判为访存受限的 BF16 kernel，搬到 B200 上多半还是访存受限，只是两条线都抬高了约 2.3 倍到 2.4 倍。H200 只加带宽不加算力，拐点反而左移到 206，是这几张卡里最“容易吃满算力”的一张。拐点真正大幅右移只发生在 FP4，而且 B300 的 FP4 算力比 B200 多一半、带宽不变，拐点从 1,125 升到 1,688。

![同一精度下拐点几乎不动，右移的是 FP4](assets/02-ledger/fig-09.svg)

### 10.2 · 精度一换，算术强度也跟着换

只看拐点会漏掉另一半：算术强度的分母是字节数，而每个元素占几个字节由精度决定。同一个矩阵乘，FLOP 数不变，权重从 BF16 换成 FP8，字节数减半，算术强度翻倍；FP8 的拐点也恰好约是 BF16 的 2 倍，两边一起翻倍，瓶颈分类不变。这就是为什么“上 FP8 就变成算力瓶颈”通常不成立：FP8 省下的是时间，不是改变了等待的种类。

FP4 要多算一步。NVFP4 每 16 个值共用一个 FP8 缩放因子，折合每个值 4.5 bit，也就是 0.5625 字节，而不是 0.5 字节[[32]](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。以 decode 阶段的权重矩阵乘为例：批大小为 B 时，每个权重元素参与 2B 次浮点运算（每个 token 一次乘加），而权重读一遍即可，KV 和激活在这里先忽略。于是算术强度约为 2B 除以每个权重的字节数：

- BF16 权重：2B ÷ 2 = B FLOP/B。在 H100 上要 B 超过 295、在 B200 上要 B 超过 281，才越过拐点。
- FP8 权重：2B ÷ 1 = 2B FLOP/B。在 H100 上要 2B 超过 591，即 B 超过约 296；在 B200 上 B 超过约 282。
- NVFP4 权重：2B ÷ 0.5625 ≈ 3.56B FLOP/B。在 B200 上要 B 超过 1,125 ÷ 3.56 ≈ 316；在 B300 上要 B 超过 1,688 ÷ 3.56 ≈ 475。

结论很具体：从 H100 BF16 到 B200 FP4，“批大小多大才算力受限”这道门槛一直在约 281 到 316 之间；到了 B300 FP4，门槛跳到约 475。B300 多出来的那一半 FP4 算力，只有在并发足够大时才用得上，小批量 decode 在 B300 上和在 B200 上一样受显存带宽约束。这和官网对 DGX B300 的描述一致：它相对 DGX B200 提高的是稠密 FP4 算力（1.5 倍）和注意力性能（2 倍），没有提显存带宽[[33]](https://www.nvidia.com/en-us/data-center/dgx-b300/)。

> 💡 **常见误区**：拿 FP4 的峰值算力和 FP4 的拐点去判断一个仍在用 BF16 的 kernel。拐点只对“用该精度做计算、用该精度搬数据”的 kernel 有意义；LayerNorm、softmax、残差相加这类逐元素算子多半以 BF16 或 FP32 读写，它们该对照的是 BF16 那一列的 281，而且它们的强度通常只有个位数，换到哪张卡上都是访存受限。

### 10.3 · 案例：把一个 decode 服务从 H100 FP8 迁到 B300 NVFP4

用五步闭环把上面的算式落到一次迁移上。场景是一个在 H100 上用 FP8 权重跑 decode 的服务，稳态并发约 256 个序列，打算迁到 B300 并把权重量化成 NVFP4。

<b>第一步，定天花板。</b>按 10.1 节，B300 每卡 FP4 稠密 13.5 PFLOPS、显存 8 TB/s，FP4 拐点约 1,688 FLOP/B。权重部分的算术强度是 3.56 × 256 ≈ 910 FLOP/B，低于拐点，所以这一步的天花板由带宽决定：每步 decode 至少要把本卡分到的权重读一遍。以一个 70B 参数的稠密模型为例，NVFP4 权重约 70 × 10⁹ × 0.5625 ≈ 39.4 GB，单卡放得下，读一遍的下限约 39.4 GB ÷ 8 TB/s ≈ 4.9 ms。

<b>第二步，测实际。</b>先按 [GPU-16｜单机｜单机验收与调优](./16-node-burnin.md) 的方法在这台机器上实测 HBM 可达带宽，替换掉 8 TB/s 这个标称值；再用 nsys 抓一段稳态 decode，拿每步实际时长和 4.9 ms 的下限比。差得越多，说明权重读取之外还有别的等待。

<b>第三步，定位等待。</b>decode 每步的时间账里，除了权重读取（W_mem），还有 KV 读取、每步几十个小 kernel 的发射间隙（W_host，见 [GPU-15｜单机｜PyTorch 运行时](./15-runtime.md)），多卡时还有 TP 通信（W_comm）。B300 把权重读取缩短了，这些固定开销在一步里的占比就上升。ncu 看单个 GEMM 会显示带宽利用率很高，看不出问题；问题在 nsys 时间线上 kernel 之间的空隙里。

<b>第四步，消除或隐藏。</b>两条路，对应两种等待。想让权重 GEMM 越过拐点，就要把并发从 256 提到约 475 以上，这依赖调度器和 KV 容量，B300 每卡 288 GB HBM3e 在这里有用[[30]](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。想缩短固定开销，就用 CUDA Graph 固化 decode 步、融合逐元素算子，减少 kernel 个数。前者让算力用得上，后者让带宽之外的时间变短，两者不冲突。

<b>第五步，复测。</b>每做一项就回到第二步，用同一套 nsys 区间和同一组请求重新量每步时长，确认省下的时间落在第三步定位的那一项上。如果提并发之后每步时长随批大小线性增加，说明已经越过拐点进入算力受限，这时再加并发只换吞吐、不换时延。

这个案例里所有数字都由官网规格和 NVFP4 的存储格式算出，没有引用任何“实测加速比”。迁移的实际收益取决于第二步量出来的东西，不同模型、不同引擎差别很大，不要拿厂商的整机吞吐对比当作自己服务的预期。

### 10.4 · MFU 的分母要跟着精度换

MFU 的定义没有变：模型实际需要的 FLOPs 除以“时间 × 稠密峰值”，分母不用稀疏峰值（第 8 章）。Blackwell 让分母的选择更容易出错，因为同一张卡同时标着三四种精度的峰值。一个用 BF16 训练的任务在 B200 上，分母应取 BF16 稠密 2.25 PFLOPS；如果误用 FP4 稠密 9 PFLOPS，MFU 会被低估到四分之一；误用 BF16 稀疏 4.5 PFLOPS，会被低估一半。混合精度训练（矩阵乘 FP8、其他 BF16）更要写清分母取哪一档，否则两个团队报出的 MFU 没法比。

反过来，跨代比较 MFU 时也要小心：B200 的 BF16 峰值是 H100 的 2.25 PFLOPS ÷ 989 TFLOPS ≈ 2.3 倍，而带宽只多 8 ÷ 3.35 ≈ 2.4 倍，两者同步增长，所以访存受限的部分在新卡上的占比大体不变，MFU 不会因为换卡自动提高。真正会拉低 MFU 的是那些不随硬件变快的项：kernel 发射、同步、通信等待。它们在新卡上占比更高，这正是后面 [GPU-14｜单机｜性能分析与 Profiling](./14-profiling.md) 到 [GPU-21｜LLM｜训练并行：TP、PP、DP、ZeRO 与 FP8](./21-train-parallel.md) 要逐项处理的等待。

### 10.5 · 拿到一张新卡时的拐点核算步骤

以后遇到 Rubin 或别家的加速卡，按同样的顺序核算，不需要等别人给结论：

- 从官方规格页抄下每种精度的 Tensor Core 峰值，先看脚注确认是稀疏还是稠密口径、是整板还是单卡，统一换成单卡稠密值。
- 抄下显存带宽，同样换成单卡值；再用 [GPU-16｜单机｜单机验收与调优](./16-node-burnin.md) 的方法实测一次，后续计算以实测值为准。
- 按精度分别算拐点，至少算 BF16 和你实际部署的那一种低精度。
- 按实际的存储格式算算术强度，带缩放因子的格式（如 NVFP4 的 4.5 bit）按真实字节数算。
- 对每个主要算子比较强度和对应精度的拐点，标出访存受限还是算力受限，再进入第 6 章的 ncu 验证。

> 🔑 **本章要点**：B200、B300 的 BF16 拐点约 281 FLOP/B，和 H100 的 295 相近，前 9 章的判断在 BF16 上照样成立。拐点大幅右移只发生在 FP4：B200 约 1,125，B300 约 1,688。低精度同时降低每个元素的字节数，所以同一形状的 GEMM 换精度后分类通常不变；只有 B300 FP4 这种“算力加、带宽不加”的组合，才把 decode 越过拐点所需的批大小从约 296 推到约 475。MFU 的分母要取所用精度的稠密峰值。本章没有新增等待项，它校准的是前面每一步“定天花板”用的那把尺子。

## 第 11 章 · 上线前核对清单与自测

到本篇为止，前 10 章分别讲了 Roofline 模型、算术强度、实测拐点、三类瓶颈、算子融合、profiler 工具、LLM 推理和训练的账本、以及 Blackwell 时代的拐点变化。这一章不讲新内容，而是汇总前面的所有知识点，提供一个核对清单和自测题库，帮助你验证是否真正理解了这套性能分析工具。如果清单中的每一项都能独立完成和解释，说明你已经掌握了 GPU 优化的基础工具，可以自信地分析和优化任何单卡 kernel 或训练步骤。

### 📌 上线前核对清单

| 检查项 | 验收标准 |
|---|---|
| 拐点对标 | 理论 Roofline 的拐点用实测峰值和实测带宽重算过吗？给出理论拐点（厂商数据）和实测拐点的对比，差异原因明确吗？ |
| 强度推导 | 给出了具体算子强度的计算推导（F 和 B 的显式公式）吗？至少包含逐元素、矩阵乘、注意力三类。 |
| 代码复现 | 三个小实验（batch 翻倍、融合 4 个算子、kernel launch 开销）都用代码跑过吗？输出与 code_outputs.md 一致吗？ |
| decode 分类 | 判断了 decode 线性层在 1 token、256 token、512 token 时分别属于哪类瓶颈吗？给出具体的强度与拐点对比。 |
| profiler 对标 | 用 ncu 或 nsys 采过一个真实的 kernel，对标过理论值吗？搬运字节、算力吞吐是否吻合？有偏差的原因分析了吗？ |
| MFU 明确 | MFU 计算中明确了是否含重算、是否含通信吗？给出稠密模型和其他模型的 MFU 对比。 |
| 图表清单 | 绘制了至少两张图：1) 全局 Roofline（理论与实测拐点）2) 注意力强度随序列长度变化的曲线。两张图都有标题、副标题、来源标注吗？ |
| 引用齐全 | 正文引用了至少 4 个不同的 refs.json 键，每条引用都用 ref 标签标记吗？（例如 Roofline、H100 规格、强度计算、通信成本） |

### 自测题

<b>1. H100 SXM 的理论拐点是多少？实测拐点呢？为什么不一样？</b>

- 理论（厂商数据）：989 TFLOPS ÷ 3.35 TB/s = 295 FLOP/字节。这是最好情况下的峰值。[[5]](https://www.nvidia.com/en-us/data-center/h100/)
- 实测（MSMF）：786 TFLOPS ÷ 3.02 TB/s = 261 FLOP/字节。这是 MSMF 除以按 90% 假设的可达带宽。本篇 sim_roofline.py ridge 计算。
- 差异的来源：厂商峰值按标称 boost 频率、全部 Tensor Core 每拍满发计算；持续满载时卡在功耗上限，频率降下来，真实 kernel 还有调度和数据搬运开销，所以持续值只有峰值的 79%[[7]](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)。带宽侧的 0.9 是 sim_roofline.py 的取值假设，用实测带宽替换后拐点要重算。

<b>2. batch 为 1 的 decode 线性层（d=4096、f=14336）强度是多少？H100 能跑多快？</b>

- 运算 F = 2 × d × f = 2 × 4096 × 14336 ≈ 117 MFLOP。本篇 code_outputs 给出此值。
- 搬运 B = 2×d（输入）+ d×f×2（权重）+ f×2（输出）≈ 117 MB。
- 强度 I = 117M / 117M = 1 FLOP/字节，远低于拐点 261。
- 下限 = max(F/P, B/W) = max(117M / 989T, 117M / 3.35T) = max(0.1 us, 35 us) = 35 us（带宽受限）。本篇 sim_roofline.py decode 输出 35.1 us。

<b>3. 4 个逐元素算子（ReLU、gelu、dropout、layernorm）分开跑和融合后各搬运多少字节？</b>

- 分开跑：64M 个 BF16 元素占 128 MiB，每个算子读一遍、写一遍共 256 MiB，4 个算子共 1024 MiB。中间张量远大于 50 MB 的 L2，基本要回到 HBM。按 3.35 TB/s 算，下限约 320.5 µs（本篇 sim_regime.py fuse 输出）[[3]](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)。
- 融合后：输入读一遍、结果写一遍，共 256 MiB，下限约 80.1 µs，快 4.0 倍（同一输出）。另外还省掉 3 次 kernel 启动；张量越小，启动开销占比越高，融合的收益越大（sim_regime.py launch 的输出给出了不同元素数下开销的占比）。

<b>4. Roofline 两条线分别由谁决定？</b>

- 算力峰值线由硬件设计决定（多少 Tensor Core、时钟频率、pipeline 宽度）。
- 带宽线由硬件设计决定（HBM 宽度、频率、channel 数）。
- 实际位置（能吃到多少百分比）由编译、运行时和算法优化共同决定。峰值折 86% 是有原因的。

<b>5. 为什么 decode 是 LLM 推理的瓶颈？</b>

- 因为每个 token 的运算很少（1 个线性层 = 117 MFLOP），搬运的参数权重不变（4096 × 14336 字节），强度只有 1 FLOP/字节。
- Roofline 左边（带宽限）的算子每秒出不了多少结果，而且 batch 大小受请求到达速率限制，无法任意增长。
- 所以 GPU 花大部分时间在等数据（等 HBM 返回权重），而不是算数据。[[8]](https://arxiv.org/abs/2211.05102)
- prefill 不同，因为一次输入多个 token，强度翻倍（256 token 时强度 237，接近拐点），所以 prefill 用两倍时间处理 8192 倍多的 token，平均延迟反而短。

<b>6. MFU 50% 需要多少吞吐？怎么从实测时间倒推 MFU？</b>

- 对 8.03B 稠密模型，每 token 运算 48.2 GFLOP，MFU 50% 需要每卡每秒 8098 个 token。本篇 sim_roofline.py mfu 计算。
- 反推：如果实测每卡每秒 9000 token，对应 MFU = (9000 × 48.2G) / (989T × 3600s) = 55.6%。
- 注意这里的 989T 是峰值算力；从端到端吞吐倒推出的 MFU，还会被通信、流水线气泡和数据加载进一步拉低。[[26]](https://arxiv.org/abs/2104.04473) 和 [[34]](https://arxiv.org/abs/2402.16363) 的训练和推理报告都展示了这种倒推方法如何从端到端吞吐数据反向校验系统设计的合理性。

### 瓶颈转移与下一步

到 GPU-02 为止，我们建立了性能账本和坐标系。给定一个 kernel 或训练步，能估出理论时间、判断瓶颈类别（开销、带宽还是算力），还能读懂 profiler 数字背后的真实情况。这套方法适用于单个 GPU 上的单个 kernel 或单个训练迭代。Roofline 的力量在于它的通用性：无论是 A100、H100 还是 B200，无论是矩阵乘还是注意力，同一套账本体系都能适用，区别只是拐点位置和三根线的斜率不同。这也是为什么本篇花了这么多篇幅在实测数字上；一旦你用真实的峰值和带宽重算拐点，后续的所有判断就自动校正了。

实战中的常用命令：[[19]](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 和 [[20]](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 是两个必备工具。ncu 看单个 kernel 的 FMA count、搬运字节、Tensor Core 利用率；nsys 看时间线，找 kernel 间隙和 PCIe 传输。用 `ncu -k regex:目标kernel名 --metrics dram__bytes_read.sum 程序` 可以只抓目标 kernel，读出它实际从显存读了多少字节[[21]](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)；用 `nsys profile -o report 程序` 可以汇总所有 kernel 的时间分布。

但还没说怎么让主机和 GPU 并行工作，避免一个等另一个。主机侧的 Python GIL、框架调度、数据加载、PCIe 握手产生的等待，都藏在 W_host 这一项。如果 GPU 等了主机 1 秒，这 1 秒无法用优化 kernel 的方式消除，只能用编程模型上的异步来隐藏。GPU-03 会讲 CUDA 编程模型里的 stream 和 event，以及怎么用异步拷贝和 kernel 重叠来把 W_host 从关键路径上挪下来，就像我们在第 8 章对 W_comm 做的一样。[[16]](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

> 🔑 **本章要点**：核对清单和自测题帮助你验收这一章的学习。重点是拐点对标、强度推导、三个小实验、decode 分类、profiler 对标、MFU 明确。所有数字都要能从代码复现或理论推导，不编造。核对清单中 8 个检查项分别应对了本篇的核心论证：从理论拐点出发，用实测数据修正，再用三种瓶颈的判断方法贯穿整个分析流程。如果能独立完成这 8 项验证，就说明你已经掌握了 GPU 性能分析的基础工具；否则，回到对应章节重新读一遍，保证理解透彻。

## 参考文献

按正文首次引用顺序编号，资料截至 2026 年 11 月中。

[1] Williams et al. Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures. UC Berkeley EECS-2008-134 (2008) [www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)

[2] He. Making Deep Learning Go Brrrr From First Principles (2022) [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)

[3] NVIDIA. GPU Performance Background User’s Guide (2023) [docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

[4] Austin et al. How To Scale Your Model (2025) [jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)

[5] NVIDIA. H100 Tensor Core GPU 产品规格 (2023) [www.nvidia.com/en-us/data-center/h100/](https://www.nvidia.com/en-us/data-center/h100/)

[6] Ivanov et al. Data Movement Is All You Need: A Case Study on Optimizing Transformers. MLSys (2021) [arxiv.org/abs/2007.00072](https://arxiv.org/abs/2007.00072)

[7] Bekman. Machine Learning Engineering Open Book, Accelerators 章节 (2026) [github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)

[8] Pope et al. Efficiently Scaling Transformer Inference. MLSys (2023) [arxiv.org/abs/2211.05102](https://arxiv.org/abs/2211.05102)

[9] Chowdhery et al. PaLM: Scaling Language Modeling with Pathways. JMLR (2023) [arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311)

[10] Yang et al. Hierarchical Roofline Performance Analysis for Deep Learning Applications. arXiv (2020) [arxiv.org/abs/2009.05257](https://arxiv.org/abs/2009.05257)

[11] NVIDIA. Matrix Multiplication Background User’s Guide (2023) [docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

[12] Boehm. How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog (2022) [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)

[13] NVIDIA. CUDA C++ Best Practices Guide (2026) [docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

[14] Luo et al. Benchmarking and Dissecting the Nvidia Hopper GPU Architecture. arXiv (2024) [arxiv.org/abs/2402.13499](https://arxiv.org/abs/2402.13499)

[15] NVIDIA. DGX B200 产品规格 (2025) [www.nvidia.com/en-us/data-center/dgx-b200/](https://www.nvidia.com/en-us/data-center/dgx-b200/)

[16] PyTorch Blog. Accelerating PyTorch with CUDA Graphs (2021) [pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

[17] Ansel et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. ASPLOS (2024) [pytorch.org/assets/pytorch2-2.pdf](https://pytorch.org/assets/pytorch2-2.pdf)

[18] Tillet et al. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL (2019) [www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

[19] NVIDIA. Nsight Compute Profiling Guide (2026) [docs.nvidia.com/nsight-compute/ProfilingGuide/index.html](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

[20] NVIDIA. Nsight Systems User Guide (2026) [docs.nvidia.com/nsight-systems/UserGuide/index.html](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

[21] Devaramani, Bentz, Scudiero. Unlock GPU Performance: Global Memory Access in CUDA. NVIDIA Technical Blog (2025) [developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)

[22] Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv (2023) [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

[23] Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP (2023) [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

[24] Grattafiori et al. The Llama 3 Herd of Models. arXiv (2024) [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

[25] Liang et al. TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training. arXiv (2024) [arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511)

[26] Narayanan et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC (2021) [arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)

[27] Jiang et al. MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs. NSDI (2024) [arxiv.org/abs/2402.15627](https://arxiv.org/abs/2402.15627)

[28] Korthikanti et al. Reducing Activation Recomputation in Large Transformer Models. MLSys (2023) [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198)

[29] NVIDIA. HGX Platform 规格：HGX B300 与 HGX B200 (2026) [www.nvidia.com/en-us/data-center/hgx/](https://www.nvidia.com/en-us/data-center/hgx/)

[30] NVIDIA Technical Blog. Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era (2025) [developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

[31] NVIDIA. GB300 NVL72 产品规格 (2026) [www.nvidia.com/en-us/data-center/gb300-nvl72/](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)

[32] NVIDIA Technical Blog. Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (2025) [developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

[33] NVIDIA. DGX B300 产品页 (2026) [www.nvidia.com/en-us/data-center/dgx-b300/](https://www.nvidia.com/en-us/data-center/dgx-b300/)

[34] Yuan et al. LLM Inference Unveiled: Survey and Roofline Model Insights. arXiv (2024) [arxiv.org/abs/2402.16363](https://arxiv.org/abs/2402.16363)

---

[在 Colab 里跑本章 CPU 练习](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/02-ledger.ipynb) · 上一章：[GPU-01｜硬件｜GPU 架构：SM、存储层级与延迟隐藏](./01-arch.md) · [目录](./README.md) · 下一章：[GPU-03｜硬件｜CUDA 编程模型：线程层级、stream 与编译链](./03-cuda.md)
