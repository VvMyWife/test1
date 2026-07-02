# MinerU Server Operation Guide

This repository is deployed on `10.20.0.15`.

The server-side project path is:

```bash
/home/ubuntu/mineru_workspace
```

The container now uses source bind mount:

```bash
/home/ubuntu/mineru_workspace -> /opt/mineru_workspace
```

If that bind mount is present, ordinary code changes usually only need a container restart.
If the bind mount is absent, or Dockerfile / dependency inputs changed, you must rebuild the image.

## 1. Connect To The Server

You must log in with `ubuntu` first, then switch to `root`.

```bash
ssh ubuntu@10.20.0.15
```

After login:

```bash
su - root
```

## 2. Go To The Project Directory

```bash
cd /home/ubuntu/mineru_workspace
```

## 3. Check Current Git State

```bash
git status --short --branch
git rev-parse HEAD
git log --oneline --decorate -n 5
```

## 4. Switch To A Specific Commit

Example: switch to `862d9472af530e7368b22a1a8d1cdabc8dec8458`

```bash
cd /home/ubuntu/mineru_workspace
git checkout 862d9472af530e7368b22a1a8d1cdabc8dec8458
```

If you want to go back to the main branch later:

```bash
git checkout main
```

## 5. Start The Container For The First Time

Only needed when:

- the image does not exist yet
- Dockerfile changed
- Python dependencies changed
- system packages changed

```bash
cd /home/ubuntu/mineru_workspace
docker compose up -d --build
```

## 6. Normal Code Update Flow

Because the source directory is now mounted into the container, ordinary code changes only need restart:

```bash
cd /home/ubuntu/mineru_workspace
docker compose restart mineru-operator
```

If you want to force recreate the container:

```bash
cd /home/ubuntu/mineru_workspace
docker compose up -d --force-recreate mineru-operator
```

## 7. Verify Service Status

Check container status:

```bash
cd /home/ubuntu/mineru_workspace
docker compose ps
docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep mineru-operator
```

Check health endpoints:

```bash
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18200/health
```

Check recent logs:

```bash
docker logs --tail 100 mineru-operator
```

Follow logs continuously:

```bash
docker logs -f mineru-operator
```

## 8. Confirm Source Bind Mount

```bash
docker inspect mineru-operator --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

You should see:

```bash
/home/ubuntu/mineru_workspace -> /opt/mineru_workspace
```

## 9. Rebuild Only When Dependencies Change

If you changed files like these, use rebuild instead of plain restart:

- `docker/Dockerfile`
- `docker-compose.yml`
- `backend/foundation/pyproject.toml`
- any dependency lock or install source

Command:

```bash
cd /home/ubuntu/mineru_workspace
docker compose up -d --build mineru-operator
```

## 10. Delivery Discipline For Server-Side Model Changes

Do not treat local regression as completion. For Paddle / MinerU / OCR serving changes, finish with this checklist:

```bash
cd /home/ubuntu/mineru_workspace

# 1) confirm server workspace code is updated
git status --short
git rev-parse HEAD

# 2) confirm whether the container uses source bind mount
docker inspect mineru-operator --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

# 3) restart or rebuild depending on runtime wiring
docker compose restart mineru-operator
# or, when image contents / dependencies changed
docker compose up -d --build mineru-operator

# 4) confirm container is running the expected code and services
docker compose ps
docker logs --tail 100 mineru-operator

# 5) verify runtime health from inside the server
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8200/health
```

If the target capability is VL, do not accept fallback-only success. `paddle_vl.json` must contain non-empty `pages`, otherwise the fix is incomplete.

## 11. Model Cache And Download Discipline

Prefer existing persistent caches before downloading again. On the server, check whether model files already exist in mounted or durable locations before triggering another download.

Do not leave required model assets only in temporary locations. If a temporary download proves to be the needed fix, move it into the persistent cache path or reinstall it in the persistent runtime so the next restart can reuse it directly.

Before closing the task, verify:

```bash
docker inspect mineru-operator --format '{{json .Mounts}}'
docker exec mineru-operator sh -lc 'echo $HOME && ls -la /workspace/.cache || true'
```

Confirm the models used by the fix live in a durable path, not just `/tmp` or another transient directory. For the current Paddle stack, persistent paths should include:

```bash
/workspace/.cache/modelscope
/home/ubuntu/mineru_workspace/.cache/modelscope
/home/ubuntu/mineru_workspace/.cache/paddlex
```

## 12. Full Example: Switch Commit And Reload Code

```bash
ssh ubuntu@10.20.0.15
su - root
cd /home/ubuntu/mineru_workspace
git checkout 862d9472af530e7368b22a1a8d1cdabc8dec8458
docker compose restart mineru-operator
docker compose ps
docker logs --tail 50 mineru-operator
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18200/health
```

## 13. Full Example: Dependency Change Or First-Time Build

```bash
ssh ubuntu@10.20.0.15
su - root
cd /home/ubuntu/mineru_workspace
git checkout main
docker compose up -d --build mineru-operator
docker compose ps
docker logs --tail 50 mineru-operator
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18200/health
```

## 14. Current Remote Change Made In This Session

The following server file was changed:

```bash
/home/ubuntu/mineru_workspace/docker-compose.yml
```

Added bind mount:

```yaml
- .:/opt/mineru_workspace
```

The original compose file was backed up on the server as:

```bash
docker-compose.yml.bak.YYYYMMDDHHMMSS
```
