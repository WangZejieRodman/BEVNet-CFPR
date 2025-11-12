# BEVNet-CFPR

3D点云场景识别系统，结合BEV特征提取与CFPR全局描述符生成。

## 特性

- **两阶段训练**: Stage1提取点级BEV特征，Stage2生成全局描述符
- **点级监督**: Circle loss + 高度预测 + 关键点检测
- **场景级匹配**: Triplet loss + hard mining
- **Coarse-to-Fine**: 全局检索 + overlap重排序

## 环境配置
```bash
conda create -n bevnet python=3.8
conda activate bevnet
pip install torch==1.12.0 torchvision torchaudio
pip install spconv-cu113  # 根据CUDA版本选择
pip install open3d loguru tensorboard pyyaml scikit-learn tqdm
```

## 数据准备

KITTI Odometry格式:
```
sequences/
├── 00/
│   ├── velodyne/*.bin
│   └── poses.txt
├── 01/
...
```

修改 `config/config.yml`:
```yaml
data_root:
  data_root_folder: "/path/to/sequences"
```

## 训练

### Stage1: Backbone + OverlapHead
```bash
python train/train_stage1_backbone.py
```
- 时间: 3-5天 (300k iterations)
- 输出: `outputs/stage1/backbone_final.ckpt`, `overlap_final.ckpt`

### Stage2: AttnVLAD
```bash
python train/train_stage2_vlad.py
```
- 自动提取BEV特征（如不存在）
- 时间: 1-2天 (100k iterations)
- 输出: `outputs/stage2/attnvlad_final.ckpt`

## 评估
```bash
python evaluate/evaluate.py
```

输出: Recall@1, Recall@5, Recall@1% 等指标

## 配置说明

`config/config.yml`:
```yaml
stage1_training_config:
  pos_threshold_min: 0       # 正样本最小距离
  pos_threshold_max: 60      # 正样本最大距离
  coords_range_xyz: [-50, -50, -4, 50, 50, 3]  # 体素化范围
  div_n: [256, 256, 32]      # 体素网格分辨率
  num_iter: 300000           # 训练迭代次数

stage2_training_config:
  pos_threshold: 10          # 场景级正样本距离
  neg_threshold: 50          # 场景级负样本距离
  epoch: 100000              # 训练epoch数
```

## 测试工具
```bash
# 测试数据加载
python tools/utils/test_dataloader.py

# 测试单batch训练
python train/test_single_batch.py
```

## 项目结构
```
.
├── config/             # 配置文件
├── modules/            # 网络模块和损失函数
├── tools/              # 数据处理工具
├── train/              # 训练脚本
├── evaluate/           # 评估脚本
└── outputs/            # 训练输出
```

## 引用

如使用本代码，请引用相关论文：
- BEVNet (点级监督)
- CFPR (全局描述符)
