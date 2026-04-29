#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   🎬 AI 视频解说生成系统 - 启动推理服务       ${NC}"
echo -e "${BLUE}================================================${NC}"

CONFIG_FILE="config/model_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 找不到配置文件: $CONFIG_FILE${NC}"
    exit 1
fi

# 从 YAML 读取配置
_yaml() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print($1)"; }

SERVER_BIN=$(_yaml "c['llamacpp']['server_bin']")
VL_MODEL=$(_yaml "c['vl_server']['model_path']")
VL_MMPROJ=$(_yaml "c['vl_server']['mmproj_path']")
VL_PORT=$(_yaml "c['vl_server']['port']")
VL_CTX=$(_yaml "c['vl_server']['ctx_size']")
VL_BATCH=$(_yaml "c['vl_server']['batch_size']")
VL_PARALLEL=$(_yaml "c['vl_server']['parallel']")

SCRIPT_MODEL=$(_yaml "c['script_server']['model_path']")
SCRIPT_PORT=$(_yaml "c['script_server']['port']")
SCRIPT_CTX=$(_yaml "c['script_server']['ctx_size']")
SCRIPT_BATCH=$(_yaml "c['script_server']['batch_size']")
SCRIPT_PARALLEL=$(_yaml "c['script_server']['parallel']")

echo -e "${YELLOW}🔍 检查路径配置...${NC}"

# 检查 llama-server
if [ ! -f "$SERVER_BIN" ]; then
    echo -e "${RED}❌ 找不到 llama-server 可执行文件: $SERVER_BIN${NC}"
    echo -e "${YELLOW}   → 请修改 config/model_config.yaml 中的 llamacpp.server_bin 字段${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ llama-server: $SERVER_BIN${NC}"

# 检查视觉模型
if [ ! -f "$VL_MODEL" ]; then
    echo -e "${RED}❌ 找不到视觉模型: $VL_MODEL${NC}"
    echo -e "${YELLOW}   → 请修改 config/model_config.yaml 中的 vl_server.model_path 字段${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ 视觉模型: $VL_MODEL${NC}"

# 检查 mmproj
if [ ! -f "$VL_MMPROJ" ]; then
    echo -e "${RED}❌ 找不到 mmproj 文件: $VL_MMPROJ${NC}"
    echo -e "${YELLOW}   → 请修改 config/model_config.yaml 中的 vl_server.mmproj_path 字段${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ mmproj: $VL_MMPROJ${NC}"

# 检查脚本模型
if [ ! -f "$SCRIPT_MODEL" ]; then
    echo -e "${RED}❌ 找不到脚本生成模型: $SCRIPT_MODEL${NC}"
    echo -e "${YELLOW}   → 请修改 config/model_config.yaml 中的 script_server.model_path 字段${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ 脚本模型: $SCRIPT_MODEL${NC}"

echo ""
echo -e "${YELLOW}⏳ 启动视觉理解服务 (端口 $VL_PORT)...${NC}"
"$SERVER_BIN" \
  --model "$VL_MODEL" \
  --mmproj "$VL_MMPROJ" \
  --host 0.0.0.0 \
  --port "$VL_PORT" \
  --n-gpu-layers 999 \
  --ctx-size "$VL_CTX" \
  --batch-size "$VL_BATCH" \
  --parallel "$VL_PARALLEL" \
  --log-disable &
VL_PID=$!

echo -e "${YELLOW}⏳ 启动脚本生成服务 (端口 $SCRIPT_PORT)...${NC}"
"$SERVER_BIN" \
  --model "$SCRIPT_MODEL" \
  --host 0.0.0.0 \
  --port "$SCRIPT_PORT" \
  --n-gpu-layers 999 \
  --ctx-size "$SCRIPT_CTX" \
  --batch-size "$SCRIPT_BATCH" \
  --parallel "$SCRIPT_PARALLEL" \
  --mlock \
  --log-disable &
SCRIPT_PID=$!

echo "$VL_PID $SCRIPT_PID" > .pids
echo -e "${GREEN}📝 进程 PID 已保存到 .pids (VL=$VL_PID, Script=$SCRIPT_PID)${NC}"
echo ""

# ---------- 启动快速粗筛服务（VL-7B，如果启用场景检测）----------
SCENE_DETECTION=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG_FILE'))
print(c.get('scene_detection', {}).get('enabled', False))
" 2>/dev/null || echo "False")

FAST_PID=""
FAST_PORT=""
if [ "$SCENE_DETECTION" = "True" ]; then
    FAST_MODEL=$(_yaml "c['vl_server_fast']['model_path']")
    FAST_MMPROJ=$(_yaml "c['vl_server_fast']['mmproj_path']")
    FAST_PORT=$(_yaml "c['vl_server_fast']['port']")
    FAST_CTX=$(_yaml "c['vl_server_fast']['ctx_size']")
    FAST_PARALLEL=$(_yaml "c['vl_server_fast']['parallel']")

    if [ ! -f "$FAST_MODEL" ]; then
        echo -e "${RED}❌ 找不到 VL-7B 模型: $FAST_MODEL${NC}"
        echo -e "${YELLOW}   → 请修改 config/model_config.yaml 中的 vl_server_fast.model_path${NC}"
        echo -e "${YELLOW}   → 或将 scene_detection.enabled 设为 false 禁用两阶段模式${NC}"
        exit 1
    fi
    if [ ! -f "$FAST_MMPROJ" ]; then
        echo -e "${RED}❌ 找不到 VL-7B mmproj: $FAST_MMPROJ${NC}"
        echo -e "${YELLOW}   → 请修改 config/model_config.yaml 中的 vl_server_fast.mmproj_path${NC}"
        echo -e "${YELLOW}   → 或将 scene_detection.enabled 设为 false 禁用两阶段模式${NC}"
        exit 1
    fi

    echo -e "${YELLOW}⏳ 启动快速粗筛服务 VL-7B (端口 $FAST_PORT, 并发 $FAST_PARALLEL)...${NC}"
    "$SERVER_BIN" \
      --model "$FAST_MODEL" \
      --mmproj "$FAST_MMPROJ" \
      --host 0.0.0.0 \
      --port "$FAST_PORT" \
      --n-gpu-layers 999 \
      --ctx-size "$FAST_CTX" \
      --batch-size 512 \
      --parallel "$FAST_PARALLEL" \
      --log-disable \
      > /tmp/ai_video_fast_server.log 2>&1 &
    FAST_PID=$!
    echo "$VL_PID $SCRIPT_PID $FAST_PID" > .pids
    echo -e "${GREEN}📝 进程 PID 已更新到 .pids (Fast=$FAST_PID)${NC}"
fi

# ---------- 健康检查（最多等待 60 秒）----------
echo -e "${YELLOW}⏳ 等待服务就绪（最多 60 秒）...${NC}"

check_health() {
    local PORT="$1"
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000"
}

TIMEOUT=60
ELAPSED=0
VL_READY=false
SCRIPT_READY=false
FAST_READY=false

# 若未启用场景检测，跳过 fast server 检查
if [ -z "$FAST_PORT" ]; then
    FAST_READY=true
fi

while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! $VL_READY && [ "$(check_health "$VL_PORT")" = "200" ]; then
        echo -e "${GREEN}  ✅ 视觉服务就绪 (http://localhost:$VL_PORT)${NC}"
        VL_READY=true
    fi
    if ! $SCRIPT_READY && [ "$(check_health "$SCRIPT_PORT")" = "200" ]; then
        echo -e "${GREEN}  ✅ 脚本服务就绪 (http://localhost:$SCRIPT_PORT)${NC}"
        SCRIPT_READY=true
    fi
    if ! $FAST_READY && [ "$(check_health "$FAST_PORT")" = "200" ]; then
        echo -e "${GREEN}  ✅ 快速粗筛服务就绪 (http://localhost:$FAST_PORT)${NC}"
        FAST_READY=true
    fi
    if $VL_READY && $SCRIPT_READY && $FAST_READY; then
        break
    fi
    printf "."
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo ""
if $VL_READY && $SCRIPT_READY && $FAST_READY; then
    echo -e "${GREEN}✅ 所有推理服务已就绪！${NC}"
    echo -e "${BLUE}   视觉服务: http://localhost:$VL_PORT${NC}"
    echo -e "${BLUE}   脚本服务: http://localhost:$SCRIPT_PORT${NC}"
    if [ -n "$FAST_PORT" ]; then
        echo -e "${BLUE}   快速粗筛服务: http://localhost:$FAST_PORT${NC}"
    fi
    echo ""
    echo -e "${GREEN}现在可以运行:${NC}"
    echo -e "  python main.py --input your_video.mp4 --output result.mp4 --style game"
    echo -e "  python web_ui/app.py"
    exit 0
else
    echo -e "${RED}❌ 服务启动超时，请检查：${NC}"
    echo -e "${YELLOW}   1. 模型路径是否正确${NC}"
    echo -e "${YELLOW}   2. 显存是否充足（建议 48GB）${NC}"
    echo -e "${YELLOW}   3. 查看进程日志排查错误${NC}"
    if [ -n "$FAST_PORT" ]; then
        echo -e "${YELLOW}   4. 快速粗筛服务日志: /tmp/ai_video_fast_server.log${NC}"
    fi
    exit 1
fi
