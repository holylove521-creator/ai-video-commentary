#!/bin/bash
# ============================================================
# scripts/build_llamacpp.sh - 编译 llama.cpp（CUDA 支持）
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

BUILD_DIR="${1:-llama.cpp}"

# ---------- 检查依赖 ----------
log_info "检查编译依赖..."

if ! command -v cmake &>/dev/null; then
    log_error "cmake 未安装，请运行: apt-get install cmake 或 brew install cmake"
fi
log_success "cmake: $(cmake --version | head -1)"

if ! command -v git &>/dev/null; then
    log_error "git 未安装"
fi

# ---------- 检查 CUDA ----------
CUDA_AVAILABLE=false
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    log_success "CUDA 检测到: $CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    log_warn "nvcc 未找到，将编译 CPU-only 版本（推理速度将大幅降低）"
fi

# ---------- 自动检测 GPU 架构 ----------
CMAKE_CUDA_ARCHS=""
if $CUDA_AVAILABLE && command -v nvidia-smi &>/dev/null; then
    # 获取 compute capability（如 8.9 → sm_89）
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$GPU_CC" ]; then
        CMAKE_CUDA_ARCHS="$GPU_CC"
        log_success "GPU 架构: sm_${GPU_CC}"
    else
        log_warn "未能自动检测 GPU 架构，使用默认值（sm_80;sm_86;sm_89;sm_90）"
        CMAKE_CUDA_ARCHS="80;86;89;90"
    fi
fi

# ---------- 克隆或更新仓库 ----------
if [ -d "$BUILD_DIR/.git" ]; then
    log_info "llama.cpp 目录已存在，拉取最新代码..."
    (cd "$BUILD_DIR" && git pull --ff-only)
else
    log_info "克隆 llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$BUILD_DIR"
fi

# ---------- 编译 ----------
log_info "开始编译 llama.cpp..."
cd "$BUILD_DIR"
mkdir -p build
cd build

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DLLAMA_BUILD_SERVER=ON
)

if $CUDA_AVAILABLE; then
    CMAKE_ARGS+=(
        -DGGML_CUDA=ON
    )
    if [ -n "$CMAKE_CUDA_ARCHS" ]; then
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHS}")
    fi
fi

cmake .. "${CMAKE_ARGS[@]}"

CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
log_info "使用 $CPUS 个 CPU 核心并行编译..."
cmake --build . --config Release -j "$CPUS"

# ---------- 验证 ----------
cd ../..
LLAMA_SERVER_BIN="$BUILD_DIR/build/bin/llama-server"
if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    log_error "编译失败：llama-server 二进制未找到于 $LLAMA_SERVER_BIN"
fi

log_success "编译成功！"
log_success "llama-server 路径: $(realpath "$LLAMA_SERVER_BIN")"
echo ""
log_info "将以下路径加入 PATH 以全局使用："
log_info "  export PATH=\"\$PATH:$(realpath "$BUILD_DIR/build/bin")\""
log_info "或直接运行: $LLAMA_SERVER_BIN --help"
