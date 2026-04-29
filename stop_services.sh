#!/bin/bash
# ============================================================
# stop_services.sh - 停止 llama.cpp 服务并清理临时文件
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "\033[0;36m[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

PID_FILE=".pids"

# ---------- 读取并终止进程 ----------
if [ -f "$PID_FILE" ]; then
    while IFS=' ' read -r name pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && log_success "已停止 $name (PID: $pid)"
        else
            log_warn "$name (PID: $pid) 进程不存在或已退出"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    log_info "已清除 PID 文件"
else
    log_warn "未找到 PID 文件 ($PID_FILE)，尝试按端口查找进程..."
    for PORT in 8080 8081; do
        PID=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
        if [ -n "$PID" ]; then
            kill "$PID" 2>/dev/null && log_success "已终止占用端口 $PORT 的进程 (PID: $PID)"
        fi
    done
fi

# ---------- 清理临时目录 ----------
TMP_DIRS=(
    "/tmp/ai_video_tmp"
    "/tmp/ai_video_frames"
)

for DIR in "${TMP_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        rm -rf "$DIR"
        log_info "已删除临时目录: $DIR"
    fi
done

# 清理服务日志
for LOG in /tmp/ai_video_vl_server.log /tmp/ai_video_script_server.log; do
    if [ -f "$LOG" ]; then
        rm -f "$LOG"
        log_info "已删除日志文件: $LOG"
    fi
done

echo ""
log_success "所有服务已停止，临时文件已清理。"
