"""
Online Time Series Forecasting — Data Loader
====================================================
Dataset_Custom: 严格按时间顺序滑动的流式数据集基类。

因果协议 (Strict Causal Protocol)
----------------------------------
1. 先 split 后插值：全量数据先按原始行数切分为 train/val/test 三段，
   每段独立插值，杜绝跨段 forward-fill 泄露。

2. Scaler 只 fit 训练段原始行：raw_train_end = int(N * train_ratio)，
   只对 data[:raw_train_end] 做 scaler.fit()，再 transform 全段。

3. 窗口归属按 label 时间戳判断：
   - window t 的 label = raw[t+seq_len : t+seq_len+pred_len]
   - train window: label 结束行 <= raw_train_end
   - val   window: label 落在 [raw_train_end, raw_val_end)
   - test  window: label 开始行 >= raw_val_end

   暴露属性：
     dataset.train_size  — 纯训练窗口数，索引范围 [0, train_size)
     dataset.val_start   — 第一个 val 窗口的索引
     dataset.test_start  — 第一个 test 窗口的索引
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
    1. 先 split 后插值：每段独立处理，无跨段信息泄露。
    2. Scaler 只 fit 训练段原始行。
    3. 窗口归属按 label 时间戳判断。
    4. 严格滑动窗口无重叠：x=[t,t+seq_len), y=[t+seq_len,t+seq_len+pred_len)。
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        seq_len: int,
        pred_len: int,
        features: str = 'M',
        target: str = 'target',
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        # backward-compat alias: old callers pass scaler_fit_ratio
        scaler_fit_ratio: float = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        # scaler_fit_ratio is a legacy alias for train_ratio
        self.train_ratio = train_ratio if scaler_fit_ratio is None else scaler_fit_ratio
        self.val_ratio = val_ratio
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_segment(df_seg: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate a single data segment in isolation.
        Forward-fill first (causal), then backward-fill leading NaN
        (segment-local only), then fill remainder with 0.
        Cross-segment values are never used.
        """
        df = df_seg.copy()
        df = df.interpolate(method='linear', limit_direction='forward')
        df = df.interpolate(method='linear', limit_direction='backward')
        df = df.fillna(0)
        return df

    def __read_data__(self):
        target_file = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(target_file):
            raise FileNotFoundError(
                f"[DataLoader] 数据集路径无效: {target_file}"
            )

        df_raw = pd.read_csv(target_file)

        # Feature column selection
        if self.features in ('M', 'MS'):
            cols_data = df_raw.columns[1:]      # drop date/index column
            df_data = df_raw[cols_data]
        else:  # 'S'
            df_data = df_raw[[self.target]]

        df_data = df_data.astype(np.float32)
        N = len(df_data)

        # ── Step 1: Raw-row split boundaries ─────────────────────────
        raw_train_end = int(N * self.train_ratio)
        raw_val_end   = int(N * (self.train_ratio + self.val_ratio))
        raw_train_end = max(1, min(raw_train_end, N))
        raw_val_end   = max(raw_train_end, min(raw_val_end, N))

        self.raw_train_end = raw_train_end
        self.raw_val_end   = raw_val_end
        self.total_rows    = N      # backward compat

        # ── Step 2: Split THEN interpolate (no cross-segment fill) ───
        seg_train = self._interpolate_segment(df_data.iloc[:raw_train_end])
        seg_val   = self._interpolate_segment(df_data.iloc[raw_train_end:raw_val_end])
        seg_test  = self._interpolate_segment(df_data.iloc[raw_val_end:])

        # ── Step 3: Scaler fits ONLY on train raw rows ───────────────
        self.scaler = StandardScaler()
        self.scaler.fit(seg_train.values)

        C = seg_train.shape[1]
        train_s = self.scaler.transform(seg_train.values).astype(np.float32)
        val_s   = (self.scaler.transform(seg_val.values).astype(np.float32)
                   if len(seg_val) > 0 else np.zeros((0, C), dtype=np.float32))
        test_s  = (self.scaler.transform(seg_test.values).astype(np.float32)
                   if len(seg_test) > 0 else np.zeros((0, C), dtype=np.float32))

        # Concatenate in original row order
        data_scaled = np.concatenate([train_s, val_s, test_s], axis=0)
        assert len(data_scaled) == N, "scale concat length mismatch"

        self.data_x = data_scaled

        if self.features == 'MS':
            target_idx = df_data.columns.get_loc(self.target)
            self.data_y = data_scaled[:, target_idx: target_idx + 1]
        else:
            self.data_y = data_scaled

        # ── Step 4: Window membership by LABEL timestamp ─────────────
        #
        # window t:  x = data[t : t+seq_len]
        #            y = data[t+seq_len : t+seq_len+pred_len]
        #
        # label_end(t)   = t + seq_len + pred_len
        # label_start(t) = t + seq_len
        #
        # Train: label_end <= raw_train_end
        #   → t <= raw_train_end - seq_len - pred_len
        #   train_size = max(0, raw_train_end - seq_len - pred_len + 1)
        #
        # Val: label_start >= raw_train_end AND label_end <= raw_val_end
        #   → t >= raw_train_end - seq_len   (label starts in val zone)
        #   val_start = max(0, raw_train_end - seq_len)
        #
        # Test: label_start >= raw_val_end
        #   → t >= raw_val_end - seq_len
        #   test_start = max(0, raw_val_end - seq_len)
        #
        # Windows straddling a boundary are excluded from all metric zones.

        self.train_size  = max(0, raw_train_end - self.seq_len - self.pred_len + 1)
        self.val_start   = max(0, raw_train_end - self.seq_len)
        self.test_start  = max(0, raw_val_end   - self.seq_len)
        self.n_windows   = max(0, N - self.seq_len - self.pred_len + 1)

        print(
            f"[DataLoader] {self.data_path} | N={N} raw rows | "
            f"raw_train={raw_train_end} ({self.train_ratio:.0%}) | "
            f"raw_val={raw_val_end} ({self.train_ratio + self.val_ratio:.0%})"
        )
        print(
            f"[DataLoader] label-ts windows: "
            f"train=[0,{self.train_size}) | "
            f"val=[{self.val_start},{self.test_start}) | "
            f"test=[{self.test_start},{self.n_windows})"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        """
        Strictly causal sliding window — zero overlap between x and y:
            x = data_x[t : t+seq_len]
            y = data_y[t+seq_len : t+seq_len+pred_len]
        """
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end                         # label starts right after x
        r_end   = r_begin + self.pred_len

        return (
            torch.tensor(self.data_x[s_begin:s_end], dtype=torch.float32),
            torch.tensor(self.data_y[r_begin:r_end], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self.n_windows


# ============================================================================
# data_provider
# ============================================================================

def data_provider(args):
    """
    Returns (Dataset_Custom, DataLoader).

    Reads from args:
      args.train_ratio     (default 0.6) — fraction for backbone/PZ training
      args.val_ratio       (default 0.1) — fraction for validation
      args.scaler_fit_ratio (legacy alias for train_ratio)

    Key attributes on returned dataset:
      dataset.train_size   — number of pure-train windows (label in train zone)
      dataset.val_start    — first val window index
      dataset.test_start   — first test window index
    """
    train_ratio      = getattr(args, 'train_ratio', None)
    val_ratio        = getattr(args, 'val_ratio', 0.1)
    scaler_fit_ratio = getattr(args, 'scaler_fit_ratio', None)

    if train_ratio is None:
        train_ratio = scaler_fit_ratio if scaler_fit_ratio is not None else 0.6

    dataset = Dataset_Custom(
        root_path   = args.root_path,
        data_path   = args.data_path,
        seq_len     = args.seq_len,
        pred_len    = args.pred_len,
        features    = args.features,
        target      = getattr(args, 'target', 'OT'),
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
    )

    dataloader = DataLoader(
        dataset,
        batch_size         = 1,
        shuffle            = False,          # streaming order must be preserved
        num_workers        = getattr(args, 'num_workers', 0),
        drop_last          = False,
        pin_memory         = torch.cuda.is_available(),
        persistent_workers = (getattr(args, 'num_workers', 0) > 0),
    )

    return dataset, dataloader

