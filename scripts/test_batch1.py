# -*- coding: utf-8 -*-
"""
Batch 1 工具链测试脚本
"""
from pipeline import benchmark, media_probe, asr_stage, schema

if __name__ == "__main__":
    # 1. Benchmark 测试
    benchmark.benchmark.mark("start")
    import time; time.sleep(0.1)
    benchmark.benchmark.mark("mid")
    time.sleep(0.2)
    benchmark.benchmark.mark("end")
    benchmark.benchmark.print_summary()

    # 2. 媒体探测
    print("\n[MediaProbe] 测试: probe_video")
    try:
        info = media_probe.probe_video("test.mp4")
        print(info)
    except Exception as e:
        print("[MediaProbe] 失败: ", e)

    # 3. ASR 占位测试
    asr = asr_stage.ASRStage()
    print("\n[ASR] 测试: transcribe")
    print(asr.transcribe("test.wav"))

    # 4. Schema 测试
    eb = schema.EventBlock(0, 10, "动作", "主角奔跑", ["主角"], "跑步对白", ["夜景"], {})
    print("\n[Schema] EventBlock:")
    print(schema.schema_to_json(eb))
