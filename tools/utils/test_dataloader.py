import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
from torch.utils.data import DataLoader
from tools.database import KITTIDatasetOverlap, KittiDataset


def test_stage1_dataloader():
    """测试Stage1数据加载器"""
    print("=" * 50)
    print("Testing Stage1 DataLoader (KITTIDatasetOverlap)")
    print("=" * 50)

    # 加载配置
    config = yaml.safe_load(open('/home/wzj/pan1/CFPR-master/config/config.yml'))
    stage1_config = config['stage1_training_config']

    # 创建数据集
    dataset = KITTIDatasetOverlap(
        sequs=stage1_config['training_seqs'][:1],  # 只测试一个序列
        root=config['data_root']['data_root_folder'],
        pos_threshold_min=stage1_config['pos_threshold_min'],
        pos_threshold_max=stage1_config['pos_threshold_max'],
        neg_thresgold=stage1_config['neg_threshold'],
        coords_range_xyz=stage1_config['coords_range_xyz'],
        div_n=stage1_config['div_n'],
        random_rotation=stage1_config['random_rotation'],
        random_occ=stage1_config['random_occlusion'],
        num_iter=100  # 只测试100个样本
    )

    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=stage1_config['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # 测试加载
    print(f"Dataset size: {len(dataset)}")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  voxel0 shape: {batch['voxel0'].shape}")
        print(f"  voxel1 shape: {batch['voxel1'].shape}")
        print(f"  trans0 shape: {batch['trans0'].shape}")
        print(f"  trans1 shape: {batch['trans1'].shape}")
        print(f"  points0 shape: {batch['points0'].shape}")
        print(f"  points1 shape: {batch['points1'].shape}")
        print(f"  points_xy0 shape: {batch['points_xy0'].shape}")
        print(f"  points_xy1 shape: {batch['points_xy1'].shape}")

        if i >= 2:  # 只测试3个batch
            break

    print("\n✓ Stage1 dataloader test passed!")


def test_stage2_dataloader():
    """测试Stage2数据加载器"""
    print("\n" + "=" * 50)
    print("Testing Stage2 DataLoader (KittiDataset)")
    print("=" * 50)

    # 加载配置
    config = yaml.safe_load(open('/home/wzj/pan1/CFPR-master/config/config.yml'))
    stage2_config = config['stage2_training_config']

    # 创建数据集
    dataset = KittiDataset(
        root=config['data_root']['data_root_folder'],
        seqs=stage2_config['training_seqs'][:1],  # 只测试一个序列
        pos_threshold=stage2_config['pos_threshold'],
        neg_threshold=stage2_config['neg_threshold']
    )

    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=stage2_config['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # 测试加载
    print(f"Dataset size: {len(dataset)}")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  id: {batch['id']}")
        print(f"  query_desc shape: {batch['query_desc'].shape}")
        print(f"  pos_desc shape: {batch['pos_desc'].shape}")
        print(f"  neg_desc shape: {batch['neg_desc'].shape}")

        if i >= 2:  # 只测试3个batch
            break

    print("\n✓ Stage2 dataloader test passed!")


if __name__ == "__main__":
    try:
        test_stage1_dataloader()
        test_stage2_dataloader()
        print("\n" + "=" * 50)
        print("All dataloader tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()