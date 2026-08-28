---
title: 云算力资源
description: AMD GPU 云算力平台汇总与使用指南
---

# ☁️ AMD 云算力资源

> 无需本地环境，浏览器登录即可上手 AMD GPU 实战。

本节汇总两个 AMD GPU 云算力平台，覆盖数据中心级专业显卡与端侧 APU 两类硬件场景，均提供开箱即用的开发环境。

## 平台概览

| 平台 | 硬件 | 定位 | 入口 |
|------|------|------|------|
| **AUP Learning Cloud** | AMD Ryzen™ AI APU（Strix Halo / Strix Point） | 端侧 APU 远程 JupyterHub / Code Server 环境（内测阶段） | [查看详情](./aup-learning-cloud/) |
| **AMD Radeon Cloud** | AMD Radeon PRO W7900D 等专业级 GPU | 中国区官方云算力平台，注册即送 100 小时算力 | [查看详情](./amd-radeon-cloud/) |


## 快速选择

- **想了解端侧 APU（CPU/GPU/NPU 集成）开发**：推荐 [AUP Learning Cloud](./aup-learning-cloud/)，基于 Ryzen™ AI 构建
- **想快速体验 GPU 在 ROCm + PyTorch 训练推理**：推荐 [AMD Radeon Cloud](./amd-radeon-cloud/)，提供 W7900D(48G) 云端GPU资源，免费 100 小时起步


## 配套教程对应关系

本仓库的算子优化教程可在云算力平台上运行，建议选择内置 ROCm + HIP 开发环境的工作区模板。

---

**导航**：
- [AUP Learning Cloud 使用教程](./aup-learning-cloud/)
- [AMD Radeon Cloud 使用教程](./amd-radeon-cloud/)
- [← 返回首页](/)
