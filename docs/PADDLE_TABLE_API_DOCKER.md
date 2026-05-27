# Paddle Table API Docker

当前推荐使用项目根目录的 `docker/Dockerfile` 构建一体化 `mineru-operator` 镜像。这个镜像已经同时包含：

- MinerU API
- Paddle Table API
- `extract_pdf_file`
- `extract_pdf_dir`
- Daft 批处理入口

因此一般不需要单独部署 `paddle-table-api` 容器。单独 Paddle API 容器只保留给调试使用。

## 推荐方式

```bash
mkdir -p data/input output logs run .cache/mineru-operator
docker compose up -d --build
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8200/health
```

Paddle 批处理：

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input \
  --output-dir /workspace/output/paddle \
  --table-engine paddle \
  --concurrency 2 \
  --overwrite
```

## 单独 Paddle API 调试

```bash
bash scripts/start-paddle-table-api-docker.sh
curl http://127.0.0.1:8200/health
```

注意：单独 Paddle API 仍然要求 API 容器能看到调用方传入的 PDF 和表格裁剪图路径。生产使用时不要跨服务器传本机路径；应在目标服务器本地启动完整 `mineru-operator` 容器。
