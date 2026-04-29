# -*- coding: utf-8 -*-
"""
基准测试与性能统计工具
"""
import time
from typing import Any, Callable, Dict

class Benchmark:
    """
    用于统计各阶段耗时、内存、显存等性能指标
    """
    def __init__(self):
        self.records = {}

    def mark(self, name: str):
        self.records[name] = time.time()

    def elapsed(self, start: str, end: str) -> float:
        return self.records[end] - self.records[start]

    def summary(self) -> Dict[str, float]:
        keys = list(self.records.keys())
        result = {}
        for i in range(1, len(keys)):
            result[f"{keys[i-1]}->{keys[i]}"] = self.records[keys[i]] - self.records[keys[i-1]]
        return result

    def print_summary(self):
        s = self.summary()
        print("\n[Benchmark] 阶段耗时统计：")
        for k, v in s.items():
            print(f"  {k}: {v:.2f}s")

benchmark = Benchmark()
