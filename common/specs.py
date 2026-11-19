"""GPU 规格表，全系列练习共用。

数字取自厂商公开规格页，口径约定：
  *_tflops    稠密峰值（不含 2:4 稀疏加速），单位 TFLOPS；厂商未单列的填 0
  hbm_tbs     单卡显存带宽，TB/s
  nvlink_gbs  单卡 NVLink 总带宽（双向合计），GB/s；单向要除以 2
  pcie_gbs    单卡 PCIe 总带宽（双向合计），GB/s
整机规格（DGX/HGX）一律除以卡数后再填。新增条目时在注释里写清出处和日期。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    arch: str
    sm: int               # SM 个数
    bf16_tflops: float
    fp8_tflops: float
    fp4_tflops: float
    hbm_gb: float
    hbm_tbs: float
    nvlink_gbs: float
    pcie_gbs: float
    tdp_w: float

    def ridge(self, dtype='bf16'):
        """拐点：峰值算力 / 显存带宽，单位 FLOP/字节。"""
        peak = {'bf16': self.bf16_tflops, 'fp8': self.fp8_tflops, 'fp4': self.fp4_tflops}[dtype]
        return peak * 1e12 / (self.hbm_tbs * 1e12) if peak else float('nan')


GPUS = {
    # NVIDIA A100 产品页，80GB SXM 列（核对于 2026-11-02）
    'a100-sxm': GPU('A100 80GB SXM', 'Ampere', 108, 312, 0, 0, 80, 2.039, 600, 64, 400),
    # NVIDIA H100 产品页，SXM 列；FP8 稀疏 3958，稠密取一半
    'h100-sxm': GPU('H100 SXM', 'Hopper', 132, 989, 1979, 0, 80, 3.35, 900, 128, 700),
    # H800 SXM：算力与显存同 H100 SXM，NVLink 降为 400 GB/s（双向合计），DeepSeek-V3 报告用的就是这款卡
    'h800': GPU('H800 SXM', 'Hopper', 132, 989, 1979, 0, 80, 3.35, 400, 128, 700),
    # NVIDIA DGX B200 产品页整机数字除以 8：HBM 1440 GB / 64 TB/s，FP8 稀疏 72 PF，FP4 稠密 72 PF，NVLink 14.4 TB/s
    # BF16 稠密 2250 取 stas00/ml-engineering 加速器表的理论值列（核对于 2026-11-02）
    'b200': GPU('B200（DGX B200 / 8）', 'Blackwell', 0, 2250, 4500, 9000, 180, 8.0, 1800, 128, 0),
    # NVIDIA Technical Blog, Inside NVIDIA Rubin GPU Architecture (2026-07)：224 SM，NVFP4 50 PF，HBM4 288 GB / 22 TB/s，NVLink 6 3.6 TB/s
    'rubin': GPU('Rubin', 'Rubin', 224, 0, 0, 50000, 288, 22.0, 3600, 0, 0),
}


def get(name):
    if name not in GPUS:
        raise SystemExit(f'未知 GPU {name}，可选：{", ".join(GPUS)}')
    return GPUS[name]


if __name__ == '__main__':
    print(f'{"key":10} {"name":22} {"BF16":>7} {"FP8":>7} {"FP4":>7} {"HBM TB/s":>9} {"ridge(bf16)":>12}')
    for k, g in GPUS.items():
        print(f'{k:10} {g.name:22} {g.bf16_tflops:7g} {g.fp8_tflops:7g} {g.fp4_tflops:7g} {g.hbm_tbs:9g} {g.ridge():12.0f}')
