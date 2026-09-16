# Sim2Real

Минимальный рабочий репозиторий для текущего `proxy planner` и `replay player`.

## Цель проекта

Репозиторий нужен для двух основных задач:

- запуск планировщика с визуализацией
- сохранение траектории в `rl4vla_raw_episode*.npz`
- повторное воспроизведение сохранённой траектории через player

## Что сделано

В репозитории находится актуальный runtime для текущего пайплайна:

- planner runtime
- replay player
- scene assets и object assets
- robot assets и runtime-конфиги
- docker-окружение

## Общая структура пайплайна

Пайплайн устроен просто:

1. planner загружает сцену, робот, object assets и runtime config
2. planner строит и исполняет траекторию
3. результат сохраняется в `runs/manual/...`
4. основной артефакт это `rl4vla_raw_episode_success.npz`
5. player загружает этот `.npz` и воспроизводит траекторию

## Где что лежит

- scene assets: `assets/scenes/`
- reusable object assets: `assets/object_bank/`
- scene/runtime config: `config/config_debug.yaml`
- lighting profiles: `config/lighting_profiles.yaml`
- robot runtime code: `openreal2sim/simulation/maniskill/`
- robot URDF и meshes: `openreal2sim/simulation/maniskill/robot_assets/`
- vendored ManiSkill/robot/controller stack: `mani_skill/`

Все основные настройки сцены и объектов находятся в:

- `config/config_debug.yaml`

## Ключевые скрипты

- планировщик с визуализацией:
  `openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py`
- плейер:
  `openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py`

## Доступные сцены (KEY)

Сцены настроены в `config/config_debug.yaml`:

| KEY | Сцена |
| --- | --- |
| `airy_table_scene14sep26_left_image` | Новая сцена левой камеры, по умолчанию в `release.v4` |
| `airi_table_new_empty3_image` | Предыдущая сцена из `release.v3`, сохранена без изменений |
| `airy_table_scene14sep26_left_image_metric` | Новая сцена с метрической коррекцией масштаба (×1.0915 вокруг камеры) для переноса на реальный робот, см. `assets/scenes/airy_table_scene14sep26_left_image_metric/README.md` |

Новая сцена использует тот же набор из восьми активных объектов и их расстановку
относительно базы робота, что и предыдущая. Фон, камера и калибровка базы/высоты
стола у сцен различаются. Результаты проверки: [release.v4](docs/release_v4_scene.md).

Команды Make выполняются **на хосте из каталога `sim2real`**, не внутри контейнера.
Если настроенный контейнер уже существует, достаточно запустить его без пересборки:

```bash
docker start sim2real-simulation

# Viewer: новая сцена (также просто make planner)
make planner KEY=airy_table_scene14sep26_left_image

# Viewer: предыдущая сцена
make planner KEY=airi_table_new_empty3_image

# Без viewer: новая и предыдущая сцены
make planner-headless KEY=airy_table_scene14sep26_left_image
make planner-headless KEY=airi_table_new_empty3_image

# Метрически скорректированная новая сцена (для переноса на реальный робот)
make planner KEY=airy_table_scene14sep26_left_image_metric
make planner-headless KEY=airy_table_scene14sep26_left_image_metric

# Выбор объекта независимо от KEY
make planner KEY=airy_table_scene14sep26_left_image TASK_OBJECT=yellow_cube_ext
```

Это запуск **proxy planner**, не MPlib. Текущий runner автоматически запускает
макрос захвата. Без `TASK_OBJECT` цель берётся из `manip_object_id` выбранной
секции конфига; сейчас для обеих сцен это `blue_cube_ext`.

`SCENE` автоматически определяется как `assets/scenes/$(KEY)/simulation/scene.json`.
Менять его вручную при выборе KEY не нужно. Явный `SCENE=...` имеет приоритет;
при таком переопределении соответствие ассетов и калибровки KEY проверяйте отдельно.
`KEY` выбирает сцену для `planner`/`planner-headless`; `replay` использует данные
сохранённого NPZ, а не текущий KEY из Makefile.

Материалы кисти для всех KEY задаются в `config/config_debug.yaml`:
`hand_visual_profiles.rc5_real_matte_v1`, включение через
`global.simulation.hand_visual_profile: *rc5_real_matte_v1`.
`plastic`, `pad`, `adapter_green` задают цвет (`base_color`, линейный RGB),
матовость (`roughness`) и блики (`specular`). Свет и физика от этого не меняются.
Камера на креплении разделена на `camera_panel` (тёмная передняя панель) и
`camera_metal` (серебристо-серый корпус, `metallic: 0.8`, `roughness: 0.25`).
Граница панели задана в `parts.prehand[3].regions` в координатах исходного меша
до масштаба URDF (мм); количество треугольников проверяется, без fallback.
`hand_visual_profile: null` возвращает прежний вид; неизвестные звенья или
ошибочные параметры вызывают явную ошибку, без подмены профиля.
Новые NPZ сохраняют профиль в runtime-конфиге; replay использует его из записи.
Старые записи без профиля сохраняют прежний вид кисти.
Это приближение материалов, не фотореалистичная текстура винтов, тяг и проводов.

## Быстрый запуск

Для первого развёртывания (если образ/контейнер ещё не подготовлены):

0. Клонировать репозиторий

```bash
git clone https://github.com/RonWise/sim2real
```

1. Перейти в папку репозитория

```bash
cd sim2real
```

2. Собрать docker image

```bash
make build
```

3. Поднять контейнер

```bash
make up
```

4. Запустить планировщик с визуализацией и сохранением траектории

```bash
make planner
```

Пример с выбором конкретного объекта:

```bash
make planner TASK_OBJECT=yellow_cube_ext
```

5. Запустить плейер

```bash
make replay NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

6. Посмотреть статистику по сохранённому `.npz`

```bash
make stats NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

7. Выгрузить actions в CSV и, если они встроены в `.npz`, `runtime_config.yaml` и `runtime_request.json`

```bash
make csv NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

8. Сжать сохранённую траекторию в более короткий `.npz`

```bash
make compress NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

По умолчанию скрипт:

- создаёт рядом `rl4vla_raw_episode_success.compressed.npz`
- создаёт planner-style GIF sidecar `rl4vla_raw_episode_success.compressed.gif`
- консервативно merge-ит только безопасные delta-action участки
- по умолчанию не merge-ит шаги с отрицательным `dz`, чтобы не ломать `descent`

9. Для проверки корректности пайплайна полезно прогнать pytest-набор

```bash
make test
```

После planner-run артефакты лежат в `runs/manual/<run>/`, включая:

- `rl4vla_raw_episode_success.npz`
- `runtime_request.json`
- `runtime_config.yaml`
- `debug_video_success.gif`

Остальные команды смотри через:

```bash
make help
```

## Сжатие RL4VLA `.npz`

Скрипт:

- `openreal2sim/simulation/maniskill/scripts/compress_npz_actions.py`

Нужен для того, чтобы уменьшать число action-точек в `rl4vla_raw_episode*.npz`, не переписывая сам planner runtime.

Базовый запуск через `Makefile`:

```bash
make compress NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Эквивалентный прямой запуск скрипта:

```bash
docker exec sim2real-simulation python /app/openreal2sim/simulation/maniskill/scripts/compress_npz_actions.py \
  --npz_path /app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Что делает скрипт:

- читает исходный `rl4vla_raw_episode*.npz`
- строит новый, более короткий `...compressed.npz`
- сохраняет рядом `...compressed.gif`
- сохраняет embedded runtime bundle внутри нового `.npz`

Важно:

- это не lossless-перекодирование, а приближённое сжатие delta-action траектории
- на free-space участках оно обычно работает хорошо
- на `descent`, `close` и `lift` слишком агрессивное merge может менять replay-поведение

### Параметры `make compress`

- `NPZ`
  Обязательный путь к исходному `rl4vla_raw_episode*.npz`.
- `COMPRESS_OUTPUT_NPZ`
  Необязательный override для пути выходного `.npz`.
- `COMPRESS_OUTPUT_GIF`
  Необязательный override для пути выходного `.gif`.
- `COMPRESS_ARGS`
  Необязательная строка с дополнительными аргументами, которые будут напрямую прокинуты в `compress_npz_actions.py`.

### Основные параметры

- `--npz_path`
  Путь к исходному `rl4vla_raw_episode*.npz`.
- `--output_npz`
  Явный путь для нового сжатого `.npz`. Если не задан, используется `...compressed.npz`.
- `--output_gif`
  Явный путь для GIF sidecar. Если не задан, используется `...compressed.gif`.
- `--save_gif` / `--no-save_gif`
  Сохранять или не сохранять GIF sidecar после конвертации.
- `--max_steps_per_chunk`
  Максимум исходных action-шагов, которые можно схлопнуть в один merged chunk.
- `--max_translation_norm`
  Верхний предел на суммарную норму перевода внутри одного merged chunk.
- `--translation_eps`
  Порог, ниже которого translational delta считается нулевой.
- `--rotation_eps`
  Порог, ниже которого rotational delta считается нулевой.
- `--allow_nonzero_gripper`
  Разрешает merge шагов с ненулевым сигналом `gripper`. По умолчанию такие шаги не merge-ятся.
- `--allow_negative_dz_merge`
  Разрешает merge шагов с отрицательным `dz`, то есть части стадии спуска. По умолчанию выключено.
- `--negative_dz_keep_tail_ratio`
  Работает только вместе с `--allow_negative_dz_merge`. Оставляет хвост каждого непрерывного участка спуска несжатым.
  Пример: `0.2` означает “сжимать только первые 80% спуска, последние 20% оставить как есть”.
- `--max_merged_negative_dz`
  Работает только вместе с `--allow_negative_dz_merge`. Ограничивает суммарный отрицательный `dz` внутри одного merged chunk.
  Это полезно, чтобы даже в сжатой части `descent` не появлялись слишком крупные прыжки вниз.

### Примеры

Только консервативное сжатие free-space участков:

```bash
make compress NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Сжать начало `descent`, но оставить последние `20%` спуска несжатыми:

```bash
make compress \
  NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz \
  COMPRESS_ARGS='--allow_negative_dz_merge --negative_dz_keep_tail_ratio 0.2 --max_merged_negative_dz 0.004'
```

Однострочник:

```bash
make compress NPZ=/app/runs/manual/proxy_ee_delta_viewer_airi_table_new_empty3_image_pick_up_blue_cube_ext_20260527_194641/rl4vla_raw_episode_success.npz COMPRESS_ARGS='--allow_negative_dz_merge --negative_dz_keep_tail_ratio 0.2 --max_merged_negative_dz 0.004'
```

Более радикальное сжатие:

```bash
make compress NPZ=/app/runs/manual/proxy_ee_delta_viewer_airi_table_new_empty3_image_pick_up_blue_cube_ext_20260527_194641/rl4vla_raw_episode_success.npz COMPRESS_ARGS='--allow_negative_dz_merge --negative_dz_keep_tail_ratio 0.1 --max_merged_negative_dz 0.012'
```

**ВАЖНО:**
**После сжатия сгенерируется гифка в которой показано как в .npz сохранены кадры
Но нет гарантии что при проигрывании сжатой траектории в плейере будет корректно захватываться обьект. Т.е. проиграть для проверки в плейере полезно, но поведение может немного отличаться от кадров, сохраненных в сжатом .npz**

Сжать траекторию и явно задать имена выходных файлов:

```bash
make compress \
  NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz \
  COMPRESS_OUTPUT_NPZ=/app/runs/manual/<run>/episode_short.npz \
  COMPRESS_OUTPUT_GIF=/app/runs/manual/<run>/episode_short.gif
```

Проверить результат через replay:

```bash
make replay NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.compressed.npz
```
