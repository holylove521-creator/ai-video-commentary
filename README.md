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

## 📦 安装

### 1. 克隆仓库

```bash
git clone https://github.com/holylove521-creator/ai-video-commentary.git
cd ai-video-commentary
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> CosyVoice2 需单独安装，参考：https://github.com/FunAudioLLM/CosyVoice

### 3. 配置本地模型路径

编辑 `config/model_config.yaml`，将所有 `/path/to/...` 替换为你本地的实际路径：

| 配置字段 | 说明 |
|----------|------|
| `llamacpp.server_bin` | llama-server 可执行文件绝对路径 |
| `vl_server.model_path` | Qwen2.5-VL GGUF 模型文件路径 |
| `vl_server.mmproj_path` | mmproj 投影文件路径 |
| `script_server.model_path` | Qwen2.5-32B GGUF 模型文件路径 |
| `tts.model_path` | CosyVoice2-0.5B 模型目录路径 |

### 4. 启动推理服务

```bash
bash start_services.sh
```

启动脚本会自动检查所有路径是否正确，并等待服务健康检查通过。

### 5. 运行

```bash
# 命令行方式
python main.py --input your_video.mp4 --output result.mp4 --style game

# Web UI 方式
python web_ui/app.py
```

---

## 快速开始

### 启动后台服务

```bash
bash start_services.sh
```

两个 llama-server 实例将在后台启动：
- **VL Server**（端口 8001）：多模态视觉理解
- **Script Server**（端口 8002）：文本脚本生成

### 生成解说视频

```bash
# 基本用法
python main.py --input my_video.mp4 --output result.mp4 --style game

# 指定参考声音（声音克隆）
python main.py --input my_video.mp4 --output result.mp4 --style sports --ref-audio my_voice.wav

# 不生成字幕
python main.py --input my_video.mp4 --style vlog --no-subtitle

# 提高抽帧密度（适合快节奏内容）
python main.py --input my_video.mp4 --fps 2.0 --style comedy
```

### 启动 Web 界面

```bash
python web_ui/app.py
# 浏览器访问 http://localhost:7860
```

### 停止后台服务

```bash
bash stop_services.sh
```

---

## 配置说明

所有配置集中在 `config/model_config.yaml`：

```yaml
vl_server:
  model_path: models/Qwen2.5-VL-32B-Q5_K_M.gguf  # 视觉模型路径
  port: 8080
  n_gpu_layers: 999   # 全部卸载 GPU

script_server:
  model_path: models/Qwen2.5-32B-Q5_K_M.gguf  # 文本模型路径
  port: 8081
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
| `movie` | movie.yaml | 电影解说博主，深沉有磁性，适合电影/剧情类内容 |

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
│       ├── comedy.yaml
│       └── movie.yaml
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
    ├── build_llamacpp.sh      # 占位脚本（llama.cpp 已在本地就绪）
    └── download_models.sh     # 占位脚本（模型已在本地就绪）
```

---

## 常见问题 FAQ

**Q: 显存不足 24GB 怎么办？**
A: 修改 `config/model_config.yaml`，将模型替换为更小量化版本（如 Q4_K_M），或改用 7B/14B 模型。

**Q: llama-server 启动失败？**
A: 检查 `config/model_config.yaml` 中的 `llamacpp.server_bin` 路径是否指向正确的 llama-server 可执行文件，并确认模型路径也已正确配置。

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
