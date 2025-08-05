---

# Tencent Ads Generative Recommendation Competition

 [腾讯广告生成式推荐比赛](https://algo.tencent.com/contest/...) ，旨在通过生成式模型提升广告推荐的相关性和点击率，探索大模型在推荐场景下的创新应用。

## 项目简介

本仓库包含了参赛方案的完整实现流程，包括数据预处理、特征工程、模型训练、推理与评测等模块。我们将采用了先进的生成式推荐方法（如LLM/Transformer/VAE等），设计高效的推荐系统解决方案。

## 目录结构

```
.
├── data/               # 数据处理与存放
├── src/                # 源代码（模型、训练、推理等）
├── configs/            # 配置文件
├── notebooks/          # 相关分析与实验 notebook
├── results/            # 结果与日志
└── README.md           # 项目说明
```

## 快速开始

1. 克隆仓库并安装依赖
    ```bash
    git clone https://github.com/liushuai35/tencent-ads-genrec.git
    cd tencent-ads-genrec
    pip install -r requirements.txt
    ```

2. 数据准备  
   下载官方比赛数据，放置于 `data/` 目录下。

3. 运行训练脚本
    ```bash
    python src/train.py --config configs/default.yaml
    ```

4. 生成推荐结果
    ```bash
    python src/inference.py --config configs/default.yaml
    ```

## 主要特性

- 支持多种生成式推荐模型（如GPT/Transformer/VAE等）
- 高效的数据预处理与特征工程
- 灵活的配置管理，便于实验复现
- 包含完整的评测与可视化工具

## 贡献者

- liushuai35 (shuai_liu@tju.edu.cn)
- 欢迎更多同学提交PR和issue！

---
