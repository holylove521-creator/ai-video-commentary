"""
pipeline 包 - AI 视频解说生成四阶段流水线

Stage 1: 视频理解  (stage1_understanding.py)
Stage 2: 脚本生成  (stage2_scriptgen.py)
Stage 3: 语音合成  (stage3_tts.py)
Stage 4: 剪辑合成  (stage4_editing.py)
"""

# Batch 1 工具模块导入
from . import benchmark, media_probe, asr_stage, schema
