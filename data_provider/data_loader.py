"""
Online Time Series Forecasting — Data Loader
====================================================
Dataset_Custom: 严格按时间顺序滑动的流式数据集基类。
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    """
    针对连续流式预测定制的 Dataset。
    
    核心设计原则：
    1. 禁用全局 StandardScaler：在非平稳场景（Actual Drift）下，全局统计量不仅无效，
       还会导致 Data Leakage。归一化统一交由模型内部的 RevIN (encode_local) 动态处理。
    2. 严格的滑动窗口：确保输入输出对 (X, Y) 严格遵循物理时间的因果因果关系。
    """
    # 修改前：
    # def __init__(self, root_path, data_path, seq_len, pred_len, target='target'):
    
    # 修改后：
    def __init__(self, root_path, data_path, seq_len, pred_len, features='M', target='target', scaler_fit_ratio=0.6):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features   # 新增这一行
        self.target = target
        self.scaler_fit_ratio = scaler_fit_ratio
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()

    def __read_data__(self):
        target_file = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"[Data Loader Error] 数据集路径无效，请检查: {target_file}")
            
        df_raw = pd.read_csv(target_file)
        
        # 1. 任务类型解析 (使用正确的 self.features，彻底告别 self.args 报错)
        features = self.features
        
        if features == 'M' or features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif features == 'S':
            df_data = df_raw[[self.target]]
            
        # 2. 防御性类型转换与缺失值插补
        df_data = df_data.astype(np.float32)
        df_data.interpolate(method='linear', limit_direction='both', inplace=True)
        df_data.fillna(0, inplace=True)
        
        # ====================================================================
        # 🚀 核心逻辑：引入基于前 60% 历史数据的全局缩放 (对齐 SOTA 论文的评测标准)
        # ====================================================================
        self.scaler = StandardScaler()
        self.total_rows = len(df_data)  # 保存总行数，供下游 split 计算使用
        # 假设前 60% 为历史可见的 Train Set
        num_train = int(len(df_data) * self.scaler_fit_ratio) 
        
        # 严格防泄露：只使用前 60% fit (拟合)
        self.scaler.fit(df_data.values[:num_train])
        
        # 转换全量数据为标准正态分布 N(0, 1)
        data_scaled = self.scaler.transform(df_data.values)
        
        self.data_x = data_scaled
        
        # 3. Y 标签的通道选择
        if features == 'MS':
            target_idx = df_data.columns.get_loc(self.target)
            self.data_y = data_scaled[:, target_idx:target_idx+1]
        else:
            self.data_y = data_scaled
        
    def __getitem__(self, index):
        """
        获取当前时间步 t 的输入窗口与对应的未来真实标签。
        注意：这里的 y 返回的是未来标签，在 streaming_env.py 中会被放入延迟队列。
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        # 减去 seq_len 和 pred_len 加上 1，确保最后一个窗口能够取到完整的预测标签
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def data_provider(args):
    """
    返回底层 Dataset 与 DataLoader。
    该 DataLoader 后续将被送入 StreamingEnvironment 进行包装。
    """
    scaler_fit_ratio = getattr(args, 'scaler_fit_ratio', 0.6)
    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        features=args.features, 
        target=args.target,
        scaler_fit_ratio=scaler_fit_ratio
    )
    
    # 极度重要：在 Online TSF 中，batch_size 必须为 1，且绝对不能 shuffle
    # 因为物理世界的时间是一步一步流逝的
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,      # 必须为 False
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    
    return dataset, dataloader