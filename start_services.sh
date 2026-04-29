#!/bin/bash
# ============================================================
# start_services.sh - 启动 llama.cpp VL 服务器和脚本生成服务器
# ============================================================
# 注意：当 config/model_config.yaml 中 phase_swap.enabled=true 时，
# main.py 会通过 services.server_manager 自动按阶段启停 llama-server。
# 此脚本仅用于手动调试或 phase_swap=false 的常驻模式。
# 在 48GB 显存上同时启动 VL-32B + Script-32B + VL-7B 会显存超额。
set -e

# ---------- 颜色定义 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # 无颜色

log_info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------- 读取配置 ----------
CONFIG_FILE="config/model_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 用 Python 解析 YAML，提取服务器参数
read_yaml() {
    python3 - "$CONFIG_FILE" "$1" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
keys = sys.argv[2].split(".")
v = cfg
for k in keys:
    v = v[k]
print(v)
EOF
}

VL_MODEL=$(read_yaml "vl_server.model_path")
VL_MMPROJ=$(read_yaml "vl_server.mmproj_path")
VL_HOST=$(read_yaml "vl_server.host")
VL_PORT=$(read_yaml "vl_server.port")
VL_GPU_LAYERS=$(read_yaml "vl_server.n_gpu_layers")
VL_CTX=$(read_yaml "vl_server.ctx_size")
VL_BATCH=$(read_yaml "vl_server.batch_size")
VL_PARALLEL=$(read_yaml "vl_server.parallel")

SCRIPT_MODEL=$(read_yaml "script_server.model_path")
SCRIPT_HOST=$(read_yaml "script_server.host")
SCRIPT_PORT=$(read_yaml "script_server.port")
SCRIPT_GPU_LAYERS=$(read_yaml "script_server.n_gpu_layers")
SCRIPT_CTX=$(read_yaml "script_server.ctx_size")
SCRIPT_BATCH=$(read_yaml "script_server.batch_size")
SCRIPT_PARALLEL=$(read_yaml "script_server.parallel")

# ---------- 检查 llama-server ----------
if ! command -v llama-server &>/dev/null; then
    # 尝试本地编译路径
    if [ -f "llama.cpp/build/bin/llama-server" ]; then
        export PATH="$PWD/llama.cpp/build/bin:$PATH"
        log_info "使用本地编译的 llama-server: llama.cpp/build/bin/llama-server"
    else
        log_error "llama-server 未找到，请先运行: bash scripts/build_llamacpp.sh"
        exit 1
    fi
fi

# ---------- 检查模型文件 ----------
for MODEL_FILE in "$VL_MODEL" "$VL_MMPROJ" "$SCRIPT_MODEL"; do
    if [ ! -f "$MODEL_FILE" ]; then
        log_warn "模型文件不存在: $MODEL_FILE，请先运行: bash scripts/download_models.sh"
    fi
done

PID_FILE=".pids"
> "$PID_FILE"

# ---------- 启动 VL 服务器 ----------
log_info "启动 VL 服务器（多模态视觉理解）..."
log_info "  模型: $VL_MODEL"
log_info "  地址: $VL_HOST:$VL_PORT"

llama-server \
    --model "$VL_MODEL" \
    --mmproj "$VL_MMPROJ" \
    --host "$VL_HOST" \
    --port "$VL_PORT" \
    --n-gpu-layers "$VL_GPU_LAYERS" \
    --ctx-size "$VL_CTX" \
    --batch-size "$VL_BATCH" \
    --parallel "$VL_PARALLEL" \
    --log-disable \
    > /tmp/ai_video_vl_server.log 2>&1 &

VL_PID=$!
echo "vl_server $VL_PID" >> "$PID_FILE"
log_info "VL 服务器 PID: $VL_PID"

# ---------- 启动脚本生成服务器 ----------
log_info "启动脚本生成服务器（文本大模型）..."
log_info "  模型: $SCRIPT_MODEL"
log_info "  地址: $SCRIPT_HOST:$SCRIPT_PORT"

llama-server \
    --model "$SCRIPT_MODEL" \
    --host "$SCRIPT_HOST" \
    --port "$SCRIPT_PORT" \
    --n-gpu-layers "$SCRIPT_GPU_LAYERS" \
    --ctx-size "$SCRIPT_CTX" \
    --batch-size "$SCRIPT_BATCH" \
    --parallel "$SCRIPT_PARALLEL" \
    --mlock \
    --log-disable \
    > /tmp/ai_video_script_server.log 2>&1 &

SCRIPT_PID=$!
echo "script_server $SCRIPT_PID" >> "$PID_FILE"
log_info "脚本服务器 PID: $SCRIPT_PID"

# ---------- 启动快速粗筛服务（VL-7B，如果启用场景检测）----------
SCENE_DETECTION=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG_FILE'))
print(c.get('scene_detection', {}).get('enabled', False))
" 2>/dev/null || echo "False")

FAST_PID=""
FAST_PORT=""
if [ "$SCENE_DETECTION" = "True" ]; then
    FAST_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server_fast']['model_path'])")
    FAST_MMPROJ=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server_fast']['mmproj_path'])")
    FAST_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server_fast']['port'])")
    FAST_CTX=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server_fast']['ctx_size'])")
    FAST_PARALLEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server_fast']['parallel'])")

    if [ ! -f "$FAST_MODEL" ]; then
        log_error "找不到 VL-7B 模型: $FAST_MODEL"
        log_warn "   → 请修改 config/model_config.yaml 中的 vl_server_fast.model_path"
        log_warn "   → 或将 scene_detection.enabled 设为 false 禁用两阶段模式"
        exit 1
    fi
    if [ ! -f "$FAST_MMPROJ" ]; then
        log_error "找不到 VL-7B mmproj: $FAST_MMPROJ"
        log_warn "   → 请修改 config/model_config.yaml 中的 vl_server_fast.mmproj_path"
        log_warn "   → 或将 scene_detection.enabled 设为 false 禁用两阶段模式"
        exit 1
    fi

    log_info "启动快速粗筛服务 VL-7B (端口 $FAST_PORT, 并发 $FAST_PARALLEL)..."
    llama-server \
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
    echo "fast_server $FAST_PID" >> "$PID_FILE"
    log_info "快速粗筛服务 PID: $FAST_PID"
fi

# ---------- 健康检查（最多等待 60 秒）----------
log_info "等待服务就绪（最多 60 秒）..."

check_health() {
    local URL="http://localhost:$1/health"
    curl -sf "$URL" -o /dev/null 2>/dev/null
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
    if ! $VL_READY && check_health "$VL_PORT"; then
        log_success "VL 服务器就绪 (http://$VL_HOST:$VL_PORT)"
        VL_READY=true
    fi
    if ! $SCRIPT_READY && check_health "$SCRIPT_PORT"; then
        log_success "脚本服务器就绪 (http://$SCRIPT_HOST:$SCRIPT_PORT)"
        SCRIPT_READY=true
    fi
    if ! $FAST_READY && check_health "$FAST_PORT"; then
        log_success "快速粗筛服务就绪 (http://localhost:$FAST_PORT)"
        FAST_READY=true
    fi
    if $VL_READY && $SCRIPT_READY && $FAST_READY; then
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

echo ""
if $VL_READY && $SCRIPT_READY && $FAST_READY; then
    log_success "╔══════════════════════════════════════╗"
    log_success "║   所有服务已成功启动！               ║"
    log_success "║   VL Server:     http://$VL_HOST:$VL_PORT     ║"
    log_success "║   Script Server: http://$SCRIPT_HOST:$SCRIPT_PORT    ║"
    if [ -n "$FAST_PORT" ]; then
        log_success "║   Fast Server:   http://localhost:$FAST_PORT  ║"
    fi
    log_success "╚══════════════════════════════════════╝"
    log_info "使用 'bash stop_services.sh' 停止服务"
else
    log_warn "部分服务可能未完全就绪，请检查日志："
    log_warn "  VL 服务器:     /tmp/ai_video_vl_server.log"
    log_warn "  脚本服务器:    /tmp/ai_video_script_server.log"
    if [ -n "$FAST_PORT" ]; then
        log_warn "  快速粗筛服务:  /tmp/ai_video_fast_server.log"
    fi
fi
