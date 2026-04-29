#!/bin/bash
set -e

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 启动 AI 视频解说服务...${NC}"

# 用 Python 解析 YAML 配置
CONFIG_FILE="config/model_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 找不到配置文件: $CONFIG_FILE${NC}"
    exit 1
fi

SERVER_BIN=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['llamacpp']['server_bin'])")
VL_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server']['model_path'])")
VL_MMPROJ=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server']['mmproj_path'])")
SCRIPT_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['script_server']['model_path'])")
VL_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server']['port'])")
SCRIPT_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['script_server']['port'])")
VL_CTX=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['vl_server']['ctx_size'])")
SCRIPT_CTX=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['script_server']['ctx_size'])")

# 检查 llama-server 可执行文件
if [ ! -f "$SERVER_BIN" ]; then
    echo -e "${RED}❌ 找不到 llama-server: $SERVER_BIN${NC}"
    echo -e "${YELLOW}   请修改 config/model_config.yaml 中的 llamacpp.server_bin 字段${NC}"
    exit 1
fi

# 检查模型文件
if [ ! -f "$VL_MODEL" ]; then
    echo -e "${RED}❌ 找不到视觉模型: $VL_MODEL${NC}"
    echo -e "${YELLOW}   请修改 config/model_config.yaml 中的 vl_server.model_path 字段${NC}"
    exit 1
fi

if [ ! -f "$VL_MMPROJ" ]; then
    echo -e "${RED}❌ 找不到视觉投影矩阵: $VL_MMPROJ${NC}"
    echo -e "${YELLOW}   请修改 config/model_config.yaml 中的 vl_server.mmproj_path 字段${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_MODEL" ]; then
    echo -e "${RED}❌ 找不到脚本模型: $SCRIPT_MODEL${NC}"
    echo -e "${YELLOW}   请修改 config/model_config.yaml 中的 script_server.model_path 字段${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 路径检查通过${NC}"

# 启动视觉理解服务
echo -e "${YELLOW}⏳ 启动视觉理解服务 (端口 $VL_PORT)...${NC}"
"$SERVER_BIN" \
  --model "$VL_MODEL" \
  --mmproj "$VL_MMPROJ" \
  --host 0.0.0.0 \
  --port "$VL_PORT" \
  --n-gpu-layers 999 \
  --ctx-size "$VL_CTX" \
  --batch-size 512 \
  --parallel 2 \
  --log-disable &
VL_PID=$!

# 启动脚本生成服务
echo -e "${YELLOW}⏳ 启动脚本生成服务 (端口 $SCRIPT_PORT)...${NC}"
"$SERVER_BIN" \
  --model "$SCRIPT_MODEL" \
  --host 0.0.0.0 \
  --port "$SCRIPT_PORT" \
  --n-gpu-layers 999 \
  --ctx-size "$SCRIPT_CTX" \
  --batch-size 1024 \
  --parallel 4 \
  --mlock \
  --log-disable &
SCRIPT_PID=$!

# 保存 PID
echo "$VL_PID $SCRIPT_PID" > .pids
echo -e "${GREEN}📝 PID 已保存: VL=$VL_PID, Script=$SCRIPT_PID${NC}"

# 健康检查（最多等 60 秒）
echo -e "${YELLOW}⏳ 等待服务就绪...${NC}"
for i in $(seq 1 60); do
    VL_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$VL_PORT/health 2>/dev/null || echo "000")
    SC_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SCRIPT_PORT/health 2>/dev/null || echo "000")
    if [ "$VL_OK" = "200" ] && [ "$SC_OK" = "200" ]; then
        echo -e "${GREEN}✅ 所有服务已就绪！${NC}"
        exit 0
    fi
    sleep 1
done

echo -e "${RED}❌ 服务启动超时，请检查模型路径和显存是否充足${NC}"
exit 1
