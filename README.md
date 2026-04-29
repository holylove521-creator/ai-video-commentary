# 🎬 AI 智能视频解说生成系统

> **当前定位：电影解说（谷阿莫风格）**
> 输入一部 4K/2h 电影 + 可选外挂 SRT，输出一段 5–10 分钟（默认 7 分钟）的中文电影解说视频。
> 目标：单部电影端到端处理 ≤ 45 分钟（48GB 显存，160 TOPS）。
>
> 流水线主要组件：
> - **Stage 0**：SRT 优先 / faster-whisper 兜底的对白提取（`pipeline/dialogue_stage.py`）
> - **Stage 1**：ffmpeg NVDEC 缩略图 + PySceneDetect 镜头切分 → VL-7B 粗筛 → VL-32B 精分（注入对白）
> - **Stage 2**：3-stage 脚本生成（D1 抽情节 → D2 章节大纲 → D3 逐章写稿，链式 prev_anchor）
> - **Stage 3**：CosyVoice2 语音合成（按标点切句 + 章节静音 + **以 TTS 实际时长**对齐输出时间轴）
> - **Stage 4**：纯 ffmpeg 流水线（流拷贝切片 + concat + 一次性 nvenc 编码 + ASS 字幕烧录）
>
> 服务编排：`services/server_manager.py` 在 `vl` / `script` 两阶段间切换 llama-server 进程，避免 48GB 显存超额。
>
> 旧的 game/sports/vlog/doc/comedy 风格仍可通过 `--style` 选用，但不再是主线。

基于 **llama.cpp** 推理框架的本地全链路 AI 智能剪辑与解说视频生成系统，全程本地推理，无需联网。

---

## 快速开始（电影解说）

```bash
# 1) 编译 llama.cpp 与下载模型
bash scripts/build_llamacpp.sh
bash scripts/download_models.sh

# 2) 安装 Python 依赖
pip install -r requirements.txt

# 3) 一键运行（带 SRT）
python main.py \
    --input /path/to/movie.mkv \
    --srt /path/to/movie.zh.srt \
    --movie-name "盗梦空间" \
    --target-duration 420 \
    --output outputs/movie_commentary.mp4

# 不带 SRT（自动 Whisper）
python main.py --input movie.mp4 --movie-name "片名" --output out.mp4
```

`config/model_config.yaml` 中 `phase_swap.enabled=true` 时，main.py 会自动按阶段启停 llama-server，无需手动运行 `start_services.sh`。


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

## 安装步骤

### 1. 编译 llama.cpp

```bash
bash scripts/build_llamacpp.sh
```

脚本会自动：
- 检测 CUDA 环境
- 克隆 llama.cpp 仓库
- 使用 `GGML_CUDA=ON` 编译
- 自动检测 GPU 架构（sm_80/sm_86/sm_89/sm_90）
- 验证 `llama-server` 二进制文件

### 2. 下载模型

```bash
bash scripts/download_models.sh
```

将自动下载：
- `Qwen2.5-VL-32B-Instruct-Q5_K_M.gguf`（视觉理解）+ mmproj
- `Qwen2.5-32B-Instruct-Q5_K_M.gguf`（脚本生成）
- `CosyVoice2-0.5B`（语音合成）

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 启动后台服务

```bash
bash start_services.sh
```

两个 llama-server 实例将在后台启动：
- **VL Server**（端口 8080）：多模态视觉理解
- **Script Server**（端口 8081）：文本脚本生成

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
A: 检查模型文件路径是否正确，运行 `bash scripts/build_llamacpp.sh` 重新编译，确保 CUDA 环境正常（`nvcc --version`）。

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
