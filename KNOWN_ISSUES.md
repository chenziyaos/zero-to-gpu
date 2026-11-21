# 已知问题

配套代码里有一批模拟器、测试和 README 与公开资料不符。系列文档在每篇的核对清单一章逐条写明了问题和正确的数字；代码本身保持原样，可以当作找错练习：先跑命令看输出，再对照文档找出哪一步的模型或假设错了。

几条通用的读法：

- 数字以文档为准。文档里的数字都标了出处，模拟器的输出只用来理解趋势和练习推导。
- 测试全部通过，只说明代码和它自己的假设一致。很多测试只检查数值落在宽松的区间，不检查方向和量级。

下表按目录列出问题，最后一列是文档里逐条说明的位置。GPU-01、GPU-02 的文档没有记录代码问题。

| 目录 | 问题概要 | 文档位置 |
|---|---|---|
| `03-cuda/` | `sim_launch.py xfer` 的毫秒数单位错、PCIe 带宽偏大；`overlap` 各场景都输出 3.00 | [GPU-03](book/03-cuda.md) 第 11 章 |
| `04-memory/` | `sim_memory.py coalesce --pattern col` 的 sector 数；`transpose`、`reduce` 的带宽不是真实硬件的数字 | [GPU-04](book/04-memory.md) 第 11 章 |
| `05-gemm/` | `sim_gemm.py blocked` 把访存下限当成分块的实际访存；`sim_occupancy.py` 把每 SM 最大 warp 数记成 32 | [GPU-05](book/05-gemm.md) 第 2、5、8 章 |
| `06-tensor-core/` | `sim_tc.py fragment` 的“寄存器”一栏是每线程元素数；`pipeline` 的 600 周期延迟没有出处；`sim_quantize.py format` 的 E4M3 最小值 | [GPU-06](book/06-tensor-core.md) 第 8 章 |
| `07-dsl/` | H100 共享内存上限写成 96 KB；“L2 改善 88.9%”；持久化网格大小；swizzle“冲突减少 87.5%” | [GPU-07](book/07-dsl.md) 第 10 章 |
| `08-attention/` | 算力下限少算几十倍；96 KB 共享内存假设；IO 下界；FA3 数字应为 740 TFLOPs/s、75% | [GPU-08](book/08-attention.md) 第 11 章 |
| `09-driver/` | 兼容类型命名；R515 与 R550 两个版本组合；Xid 日志格式和处置建议 | [GPU-09](book/09-driver.md) 第 9 章 |
| `10-memory-mgmt/` | 正好 1 MB 的请求应进 large 池；显存预算的激活和优化器状态偏小 | [GPU-10](book/10-memory-mgmt.md) 第 10 章 |
| `11-pcie-gdr/` | H2D 扣减比例没有出处；拓扑用数字距离而非 topo -m 的符号；测试断言“Gen4 用 8b/10b” | [GPU-11](book/11-pcie-gdr.md) 第 9 章 |
| `12-sharing/` | MIG profile 名字用的是 A100-40GB；放置判定；MPS chunk 定义；吞吐模型没有出处 | [GPU-12](book/12-sharing.md) 第 9 章 |
| `13-nvlink/` | NVLink 6 规格；拓扑样例用数字距离；nccl-tests 日志是合成的 | [GPU-13](book/13-nvlink.md) 第 10 章 |
| `14-profiling/` | 样例数据时长量级不对；“kernel 间隙”不是时间上的空隙；五分类判定自相矛盾；`gpu_workload.py` 不启动 kernel | [GPU-14](book/14-profiling.md) 第 10 章 |
| `15-runtime/` | `sim_compile.py` 漏算访存时间；三段流水忽略填充排空；测试引用的 arXiv 编号是无关论文 | [GPU-15](book/15-runtime.md) 第 10 章 |
| `16-node-burnin/` | README 的 BF16 合格线 890 与代码的 700 不一致；CONDITIONAL 的退出码；部分致命 Xid 的处置与目录不一致 | [GPU-16](book/16-node-burnin.md) 第 10 章 |
| `17-rdma/` | `test_ecmp.py` 打印出方向相反的结果仍通过；`gpu_perftest_bw.sh` 的工具和参数与 README、perftest 不符 | [GPU-17](book/17-rdma.md) 第 10 章 |
| `18-nccl/` | README 的期望输出和协议效率写错；`sim_overlap.py` 的重叠模型有误 | [GPU-18](book/18-nccl.md) 第 10 章 |
| `19-k8s/` | 机内拓扑建成双环；碎片率与策略无关；`sim_gang.py` 的设定不可能死锁；节点标签不存在 | [GPU-19](book/19-k8s.md) 第 10 章 |
| `20-reliability/` | Daly 公式写错；README 的 Daly 引用编号不对；期望耗时漏了重算；“Llama 3 复现”不成立 | [GPU-20](book/20-reliability.md) 第 10 章 |
| `21-train-parallel/` | 两个文件的优化器状态口径矛盾；激活估计小两个数量级；两套 TP 通信模型；方案搜索不计激活 | [GPU-21](book/21-train-parallel.md) 第 10 章 |
| `22-moe-ep/` | 延迟把总字节除以一条链路；MiB 标成 MB；测试注释里的 DeepSeek-V3 规格；负载和均衡模型没有依据 | [GPU-22](book/22-moe-ep.md) 第 10 章 |
| `23-inference/` | 并发数少除了张量并行度；不检查容量；MHA 基准漏乘层数；预填充按带宽算时间；goodput 定义 | [GPU-23](book/23-inference.md) 第 10 章 |
| `24-capstone/` | 检查点开销固定为一半；时间账比例写死；推理计划的常数没有依据；校准表和 README 示例与实际输出不符 | [GPU-24](book/24-capstone.md) 第 10 章 |

## 已在仓库修复、文档仍按旧版描述的问题

文档的核对是对着修复前的代码做的。下面几处在本仓库已经改好，文档里相应的句子可以忽略：

- `09-driver/sim_xid.py` 已导入 `Tuple`，在 Python 3.9 上也能导入，`test_driver.py` 通过。
- `17-rdma/sim_ecmp.py`、`21-train-parallel/sim_parallel.py` 的 f-string 不再嵌套双引号，不需要 Python 3.12，`test_ecmp.py`、`test_parallel.py` 在 Python 3.9 上通过。
- `22-moe-ep/` 的两个测试不再写死工作目录，换一台机器也能跑。

全部 27 个测试文件在 Python 3.9.6 上运行通过（2026-11-21）。
