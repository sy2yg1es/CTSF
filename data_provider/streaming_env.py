# streaming_env.py
import torch
from collections import deque

class StreamingEnvironment:
    """
    流式数据环境模拟器 (Streaming Environment Simulator)
    
    作用：将标准的 (X, Y_future) Dataloader 转换为严格的流式发生器。
    在时刻 t，只吐出当前的输入 X_t，以及刚刚到达的延迟标签 Y_{t-H}。
    """
    def __init__(self, dataloader, forecast_H: int):
        self.dataloader = dataloader
        self.forecast_H = forecast_H
        # 用于暂存未来标签的延迟队列
        self.label_queue = deque()
        
    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        self.label_queue.clear()
        return self
        
    def __next__(self):
        # 1. 从底层 dataloader 获取下一个时间步的 (输入, 未来标签)
        try:
            batch_x, batch_y = next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration
            
        # 2. 将未来标签压入延迟队列
        self.label_queue.append(batch_y)
        
        # 3. 模拟延迟反馈：如果队列长度还没达到 H，说明历史预测的标签还没到
        if len(self.label_queue) <= self.forecast_H:
            # 此时没有标签到达，返回 None 作为占位符
            Y_arrived = None
        else:
            # 队列长度超过 H，最老的那个标签终于到达了当前时刻！
            Y_arrived = self.label_queue.popleft()
            
        # 严格返回：当前只能看到 X_t，和刚刚延迟到达的 Y_arrived
        return batch_x, Y_arrived

# ==========================================
# 在 main.py 中的使用方法演示：
# base_loader = DataLoader(Dataset_Custom(...), batch_size=1, shuffle=False)
# streaming_loader = StreamingEnvironment(base_loader, forecast_H=24)
# ==========================================