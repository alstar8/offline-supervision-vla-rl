# Run Proxy Planner And Replay

Эта инструкция покрывает только текущий рабочий pipeline:

- запуск `proxy planner`
- сохранение `rl4vla_raw_episode*.npz`
- inspection/statistics для `.npz`
- replay сохранённой траектории

## 1. Подготовить runtime

Из корня репозитория:

```bash
make config
make build
make up
```

Для быстрой проверки после подъёма контейнера полезно прогнать:

```bash
make test
```

## 2. Запустить planner

Planner с визуализацией:

```bash
make planner
```

Пример с другим объектом:

```bash
make planner TASK_OBJECT=yellow_cube_ext
```

Если нужен headless-run, используй:

```bash
make planner-headless
```

Прямой запуск с хоста через Docker container:

```bash
docker exec -it sim2real-simulation bash -lc '
cd /app &&
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py \
  --motion_backend proxy_ee_delta \
  --config_path config/config_debug.yaml \
  --key airi_table_new_empty3_image \
  --scene assets/scenes/airi_table_new_empty3_image/simulation/scene.json \
  --task_type pick_up
'
```

Пример с другим объектом:

```bash
docker exec -it sim2real-simulation bash -lc '
cd /app &&
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py \
  --motion_backend proxy_ee_delta \
  --config_path config/config_debug.yaml \
  --key airi_table_new_empty3_image \
  --scene assets/scenes/airi_table_new_empty3_image/simulation/scene.json \
  --task_type pick_up \
  --task_object_id yellow_cube_ext
'
```

Прямой запуск внутри контейнера без `make`:

```bash
cd /app
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py \
  --motion_backend proxy_ee_delta \
  --config_path config/config_debug.yaml \
  --key airi_table_new_empty3_image \
  --scene assets/scenes/airi_table_new_empty3_image/simulation/scene.json \
  --task_type pick_up
```

Пример с другим объектом:

```bash
cd /app
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py \
  --motion_backend proxy_ee_delta \
  --config_path config/config_debug.yaml \
  --key airi_table_new_empty3_image \
  --scene assets/scenes/airi_table_new_empty3_image/simulation/scene.json \
  --task_type pick_up \
  --task_object_id yellow_cube_ext
```

## 3. Что сохраняется после planner-run

По умолчанию planner создаёт run directory в:

- `runs/manual/...`

И сохраняет:

- `execution_summary.json`
- `runtime_request.json`
- `runtime_config.yaml`
- `rl4vla_raw_episode_success.npz` или `rl4vla_raw_episode_fail.npz`
- `debug_video_success.gif` или `debug_video_fail.gif`

`rl4vla_raw_episode*.npz` дополнительно содержит embedded:

- `runtime_config.yaml`
- `runtime_request.json`

## 4. Посмотреть `.npz` без viewer

Показать summary/statistics:

```bash
make stats NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Экспортировать actions в CSV и embedded runtime sidecars:

```bash
make csv NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Экспортировать в пользовательские пути:

```bash
make csv \
  NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz \
  OUT_CSV=/tmp/actions.csv \
  OUT_YAML=/tmp/runtime_config.yaml \
  OUT_JSON=/tmp/runtime_request.json
```

Dry-run:

```bash
make replay-dry NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Прямой запуск с хоста через Docker container:

```bash
docker exec sim2real-simulation bash -lc '
cd /app &&
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/summarize_npz_actions.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
'
```

```bash
docker exec sim2real-simulation bash -lc '
cd /app &&
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/export_npz_actions_to_csv.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
'
```

```bash
docker exec sim2real-simulation bash -lc '
cd /app &&
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz \
  --dry_run
'
```

Прямые команды внутри контейнера:

```bash
cd /app
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/summarize_npz_actions.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

```bash
cd /app
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/export_npz_actions_to_csv.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

```bash
cd /app
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz \
  --dry_run
```

## 5. Запустить replay

```bash
make replay NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Прямой запуск с хоста через Docker container:

```bash
docker exec -it sim2real-simulation bash -lc '
cd /app &&
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
'
```

Прямой запуск внутри контейнера:

```bash
cd /app
PYTHONPATH=/app /opt/conda/bin/python openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

## 6. Остальные команды

Полный список команд:

```bash
make help
```
