#!/bin/bash
# ============================================================
# scripts/download_models.sh - 下载所有必要模型
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

MODELS_DIR="${MODELS_DIR:-models}"

# ---------- 检查 huggingface-cli ----------
if ! command -v huggingface-cli &>/dev/null; then
    log_warn "huggingface-cli 未安装，尝试自动安装..."
    pip install -q huggingface_hub[cli] || log_error "安装 huggingface_hub 失败，请手动运行: pip install huggingface_hub[cli]"
fi
log_success "huggingface-cli: $(huggingface-cli version 2>/dev/null || echo 'OK')"

# ---------- 创建目录结构 ----------
mkdir -p "$MODELS_DIR"
log_info "模型存放目录: $(realpath "$MODELS_DIR")"

# ---------- 下载函数 ----------
download_gguf() {
    local REPO="$1"
    local FILENAME="$2"
    local DESC="$3"
    local APPROX_SIZE="$4"
    local OUT_PATH="$MODELS_DIR/$FILENAME"

    if [ -f "$OUT_PATH" ]; then
        log_success "$DESC 已存在，跳过下载: $OUT_PATH"
        return 0
    fi

    log_info "下载 $DESC (~$APPROX_SIZE)..."
    log_info "  来源: https://huggingface.co/$REPO"
    log_info "  文件: $FILENAME"

    huggingface-cli download \
        "$REPO" \
        "$FILENAME" \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False

    if [ -f "$OUT_PATH" ]; then
        ACTUAL_SIZE=$(du -sh "$OUT_PATH" | cut -f1)
        log_success "$DESC 下载完成  实际大小: $ACTUAL_SIZE"
    else
        log_error "$DESC 下载失败，请检查网络连接或手动下载"
    fi
}

download_repo() {
    local REPO="$1"
    local LOCAL_DIR="$MODELS_DIR/$2"
    local DESC="$3"

    if [ -d "$LOCAL_DIR" ] && [ "$(ls -A "$LOCAL_DIR" 2>/dev/null)" ]; then
        log_success "$DESC 已存在，跳过: $LOCAL_DIR"
        return 0
    fi

    log_info "下载 $DESC..."
    mkdir -p "$LOCAL_DIR"
    huggingface-cli download \
        "$REPO" \
        --local-dir "$LOCAL_DIR" \
        --local-dir-use-symlinks False

    if [ -d "$LOCAL_DIR" ]; then
        log_success "$DESC 下载完成: $LOCAL_DIR"
    else
        log_error "$DESC 下载失败"
    fi
}

# ---------- 模型下载列表 ----------
echo ""
log_info "=========================================="
log_info "开始下载所需模型（共 3 个）"
log_info "=========================================="
echo ""

# 1. Qwen2.5-VL-32B 视觉模型 GGUF（Q5_K_M 量化，~22GB）
log_info "【1/3】Qwen2.5-VL-32B 视觉理解模型（Stage 1）"
download_gguf \
    "bartowski/Qwen2.5-VL-32B-Instruct-GGUF" \
    "Qwen2.5-VL-32B-Instruct-Q5_K_M.gguf" \
    "Qwen2.5-VL-32B Q5_K_M GGUF" \
    "~22GB"

# mmproj（多模态投影头，~1GB）
download_gguf \
    "bartowski/Qwen2.5-VL-32B-Instruct-GGUF" \
    "mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf" \
    "Qwen2.5-VL-32B mmproj" \
    "~1GB"

echo ""

# 2. Qwen2.5-32B 文本模型 GGUF（Q5_K_M 量化，~22GB）
log_info "【2/3】Qwen2.5-32B 脚本生成模型（Stage 2）"
download_gguf \
    "bartowski/Qwen2.5-32B-Instruct-GGUF" \
    "Qwen2.5-32B-Instruct-Q5_K_M.gguf" \
    "Qwen2.5-32B Q5_K_M GGUF" \
    "~22GB"

echo ""

# 3. CosyVoice2-0.5B TTS 模型（~2GB）
log_info "【3/3】CosyVoice2-0.5B 语音合成模型（Stage 3）"
download_repo \
    "FunAudioLLM/CosyVoice2-0.5B" \
    "CosyVoice2-0.5B" \
    "CosyVoice2-0.5B"

echo ""

# ---------- 验证下载 ----------
log_info "=========================================="
log_info "验证下载结果："
MISSING=0
for FILE in \
    "$MODELS_DIR/Qwen2.5-VL-32B-Instruct-Q5_K_M.gguf" \
    "$MODELS_DIR/mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf" \
    "$MODELS_DIR/Qwen2.5-32B-Instruct-Q5_K_M.gguf" \
    "$MODELS_DIR/CosyVoice2-0.5B"; do
    if [ -e "$FILE" ]; then
        SIZE=$(du -sh "$FILE" 2>/dev/null | cut -f1)
        log_success "  ✓ $FILE  ($SIZE)"
    else
        log_warn "  ✗ $FILE  【缺失】"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
if [ "$MISSING" -eq 0 ]; then
    log_success "所有模型下载验证通过！"
    log_info "下一步: bash start_services.sh"
else
    log_warn "有 $MISSING 个模型未下载成功，请检查网络或手动下载"
fi
