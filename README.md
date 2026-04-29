# 🎬 AI 智能视频解说生成系统

基于 **llama.cpp** 推理框架的本地全链路 AI 智能剪辑与解说视频生成系统，支持游戏、体育、Vlog、纪录片、搞笑吐槽等多种风格，全程本地推理，无需联网。

---

## 架构图

```
原始素材视频
      │
      ▼
┌─────────────────────────────────────────┐
│         Stage 1: 视频理解与分析          │
│  视频抽帧 → 多模态视觉理解 → 场景分割   │
│  (Qwen2.5-VL-32B via llama-server)      │
└─────────────────┬───────────────────────┘
                  │ JSON 场景描述 + 亮点分
                  ▼
┌─────────────────────────────────────────┐
│         Stage 2: 解说脚本生成           │
│  LLM 脚本创作 → 风格控制 → 时间轴对齐  │
│  (Qwen2.5-32B via llama-server)         │
└─────────────────┬───────────────────────┘
                  │ 结构化脚本 [{start,end,text,emotion}]
                  ▼
┌─────────────────────────────────────────┐
│         Stage 3: 语音合成               │
│  TTS 配音 → 情感标签控制 → 时长对齐    │
│  (CosyVoice2-0.5B)                     │
└─────────────────┬───────────────────────┘
                  │ 解说音频轨 (.wav)
                  ▼
┌─────────────────────────────────────────┐
│         Stage 4: 智能剪辑合成           │
│  片段截取 → 音画对齐 → 字幕渲染 → 输出  │
│  (MoviePy + FFmpeg h264_nvenc)          │
└─────────────────┬───────────────────────┘
                  ▼
            🎬 成品解说视频 (MP4)
```

---

## 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU 显存 | 24 GB | **48 GB**（满血运行双服务） |
| 算力 | 80 TOPS | **160 TOPS** |
| 系统内存 | 32 GB | 64 GB |
| 存储 | 100 GB | 200 GB（存放模型） |
| CUDA | 11.8+ | **12.x** |

> 本系统默认针对 48GB / 160 TOPS 硬件优化，所有模型层数全卸载 GPU（n_gpu_layers=999）。

---

## ⚙️ 快速配置本地路径

编辑 `config/model_config.yaml`，将以下字段改为你本地的实际路径：

| 字段 | 说明 | 示例 |
|------|------|------|
| `llamacpp.server_bin` | llama-server 可执行文件路径 | `/home/user/llama.cpp/build/bin/llama-server` |
| `vl_server.model_path` | 视觉理解模型 GGUF 路径 | `/data/models/Qwen2.5-VL-32B-Q5_K_M.gguf` |
| `vl_server.mmproj_path` | 视觉投影矩阵路径 | `/data/models/mmproj-Qwen2.5-VL-32B-f16.gguf` |
| `script_server.model_path` | 脚本生成模型 GGUF 路径 | `/data/models/Qwen2.5-32B-Q5_K_M.gguf` |
| `tts.model_path` | CosyVoice2 模型目录 | `/data/models/CosyVoice2-0.5B` |

配置完成后直接执行：
```bash
bash start_services.sh   # 启动推理服务
python main.py --input your_video.mp4 --output result.mp4 --style game
```

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/holylove521-creator/ai-video-commentary.git
cd ai-video-commentary

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 配置本地模型路径（编辑此文件）
nano config/model_config.yaml

# 4. 启动推理服务
bash start_services.sh

# 5. 运行（命令行）
python main.py --input your_video.mp4 --output result.mp4 --style game

# 或启动 Web UI
python web_ui/app.py
```

---

## 配置说明

所有配置集中在 `config/model_config.yaml`：

```yaml
llamacpp:
  server_bin: /usr/local/bin/llama-server  # llama-server 可执行文件路径

vl_server:
  model_path: /path/to/your/models/Qwen2.5-VL-32B-Q5_K_M.gguf  # 视觉模型路径
  port: 8001
  n_gpu_layers: 999   # 全部卸载 GPU

script_server:
  model_path: /path/to/your/models/Qwen2.5-32B-Q5_K_M.gguf  # 文本模型路径
  port: 8002
  n_gpu_layers: 999

video:
  fps_sample: 1.0      # 抽帧频率（帧/秒）
  output_codec: h264_nvenc  # NVIDIA GPU 硬件编码
```

### 解说风格模板

风格模板位于 `config/style_templates/`，支持：

| 风格 | 文件 | 适用场景 |
|------|------|---------|
| `game` | game.yaml | 游戏录屏、电竞比赛 |
| `sports` | sports.yaml | 体育赛事、运动视频 |
| `vlog` | vlog.yaml | 日常 Vlog、旅行视频 |
| `doc` | doc.yaml | 纪录片、科普内容 |
| `comedy` | comedy.yaml | 搞笑视频、吐槽剪辑 |

自定义风格：复制任意模板 YAML，修改 `system_prompt` 即可。

---

## 项目结构

```
ai-video-commentary/
├── README.md
├── requirements.txt
├── start_services.sh          # 启动 llama-server 服务
├── stop_services.sh           # 停止服务并清理临时文件
├── main.py                    # 命令行主入口
├── config/
│   ├── model_config.yaml      # 全局配置
│   └── style_templates/       # 解说风格模板
│       ├── game.yaml
│       ├── sports.yaml
│       ├── vlog.yaml
│       ├── doc.yaml
│       └── comedy.yaml
├── pipeline/
│   ├── stage1_understanding.py  # 视频理解（VL 模型）
│   ├── stage2_scriptgen.py      # 脚本生成（LLM）
│   ├── stage3_tts.py            # 语音合成（CosyVoice2）
│   └── stage4_editing.py        # 剪辑合成（MoviePy + FFmpeg）
├── utils/
│   ├── llm_client.py          # llama.cpp 异步 HTTP 客户端
│   ├── vram_manager.py        # 显存监控管理
│   ├── frame_extractor.py     # 视频抽帧工具
│   └── subtitle_renderer.py   # ASS 字幕生成与烧录
├── web_ui/
│   └── app.py                 # Gradio Web 界面
└── scripts/
    ├── build_llamacpp.sh      # 编译 llama.cpp（CUDA）
    └── download_models.sh     # 下载所需模型
```

---

## 常见问题 FAQ

**Q: 显存不足 24GB 怎么办？**
A: 修改 `config/model_config.yaml`，将模型替换为更小量化版本（如 Q4_K_M），或改用 7B/14B 模型。

**Q: llama-server 启动失败？**
A: 检查 `config/model_config.yaml` 中 `llamacpp.server_bin` 路径是否正确，确保 CUDA 环境正常（`nvcc --version`）。

**Q: CosyVoice2 未安装时语音合成怎么办？**
A: Stage 3 会优雅降级，生成静音音轨并提示安装路径。参考 [CosyVoice 官方文档](https://github.com/FunAudioLLM/CosyVoice) 安装。

**Q: 如何使用自己的声音？**
A: 录制 3-10 秒干净人声音频（WAV 格式），通过 `--ref-audio` 参数传入，或在 Web UI 上传。

**Q: 视频处理速度如何提升？**
A: 降低抽帧频率（`--fps 0.5`），或在 `config/model_config.yaml` 中增大 `parallel` 值以提高并发推理数量。

**Q: 支持哪些视频格式？**
A: 支持 FFmpeg 可读的所有格式（MP4、MKV、AVI、MOV 等），输出默认为 MP4（h264_nvenc）。

**Q: Windows 是否支持？**
A: 主要针对 Linux 开发。Windows 用户建议使用 WSL2 + CUDA，或修改 shell 脚本为 PowerShell 等效命令。
