# 23-inference

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-gpu/blob/main/colab/23-inference.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

> 已知问题：本目录部分输出与公开资料不符，代码保持原样作为找错练习。逐条说明见 [GPU-23](../book/23-inference.md) 第 10 章，汇总见 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)。

GPU-23 推理系统的配套练习。KV cache 显存账本、Decode 延迟与吞吐、Prefill/Decode 分离配比。

| 文件 | 环境 | 做什么 | 对应文档章节 |
|---|---|---|---|
| `sim_kv.py` | CPU | 按模型配置计算 KV 字节、给定显存的最大并发、对比 GQA/MLA 压缩效果、扫 batch 看 decode 延迟 | 第 2-3 章 |
| `sim_decode.py` | CPU | Decode 阶段延迟与吞吐扫描、不同模型对比、Roofline 分析、batch 吞吐饱和点 | 第 4 章 |
| `sim_pd.py` | CPU | Prefill/Decode 分离决策、资源配比、KV 跨卡传输带宽需求、SLO 下的 goodput 估算 | 第 5-7 章 |
| `test_kv.py` | CPU | 复现 KV cache 账本公式、GQA 压缩比、vLLM 论文的显存改进定性 | 全章 |

## 运行

```bash
cd zero-to-gpu/23-inference

# 测试
python3 test_kv.py

# KV cache 账本（70B with TP=8, 8B with TP=1）
python3 sim_kv.py config --model llama-70b-bf16 --tp 8
python3 sim_kv.py config --model llama-8b-bf16 --tp 1
python3 sim_kv.py capacity --model llama-70b-bf16 --gpu h100-sxm --tp 8 --batch 256

# Decode 延迟与吞吐
python3 sim_decode.py latency --model llama-70b-bf16 --gpu h100-sxm --tp 8
python3 sim_decode.py latency --model llama-8b-bf16 --gpu h100-sxm --tp 1
python3 sim_decode.py compare --gpu h100-sxm

# Prefill/Decode 分离
python3 sim_pd.py prefill_roofline --model llama-70b-bf16 --gpu h100-sxm --tp 8
python3 sim_pd.py pd_split --model llama-70b-bf16 --total-gpus 8 --tp 8
python3 sim_pd.py goodput --model llama-70b-bf16 --gpu h100-sxm --tp 8
```

## 期望输出

`test_kv.py`：7 条测试全部通过（Llama 70B 327680 bytes/token，Llama 8B GQA 4x，OPT-13B FP32 1600 KB/token）。

`sim_kv.py config --model llama-70b-bf16 --tp 8`：Llama 70B (TP=8) 权重 141.2 GB，per-GPU 17.65 GB，KV/token 327680 bytes，seq_len 2K-32K 对应 KV 671.1-10737.4 MB/请求。

`sim_decode.py latency --model llama-70b-bf16 --gpu h100-sxm --tp 8`：H100 上 Llama 70B (TP=8)，seq_len=16K，batch 1-256 TPOT 6.7-57.8 ms，吞吐 148-4425 tok/s。

`test_kv.py` 中 Llama 70B 16K 上下文 KV 占用 5.37 GB，Llama 8B batch=256 最多可并发 28 请求。

## 版本

CUDA 13.4 Update 2，PyTorch 2.15.0。
