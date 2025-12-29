#!/bin/bash
# Получаем абсолютный путь к текущей директории
CURRENT_DIR=$(pwd)

echo "Starting Ascend C container..."
echo "Host directory: $CURRENT_DIR"
echo "Container directory: /workspace"
echo ""
echo "Press Ctrl+D or type 'exit' to stop container."

# Запускаем контейнер в интерактивном режиме
docker run -it --rm \
  --name ascendc-workspace \
  -v "$CURRENT_DIR:/workspace" \
  -w /workspace \
  ascendc-dev