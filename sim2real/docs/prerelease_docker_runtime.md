# Prerelease Docker Runtime

Этот документ описывает минимальный Docker workflow для `sim2real`.

## Что делает Docker-слой

Docker-слой нужен для:

- сборки runtime image
- запуска persistent simulation container
- работы planner viewer и replay viewer с GPU

## Требования к хосту

На хосте должны быть:

- Docker
- Docker Compose v2
- NVIDIA Container Toolkit
- рабочий X11 display

Полезные проверки:

```bash
docker --version
docker compose version
nvidia-smi
```

## Сборка и запуск

Из корня репозитория:

```bash
make build
make up
```

Контейнер поднимается как:

- `sim2real-simulation`

Рабочая директория внутри контейнера:

- `/app`

Репозиторий монтируется внутрь контейнера в:

- `/app`

## Полезные команды

Проверить compose config:

```bash
make config
```

Остановить контейнер:

```bash
make stop
```

Удалить контейнер и compose-ресурсы:

```bash
make down
```

Открыть shell внутри контейнера:

```bash
make shell
```

## Проверка GPU

Если `glxinfo` доступен, полезно проверить renderer:

```bash
docker exec sim2real-simulation bash -lc 'glxinfo | grep "OpenGL renderer"'
```

Ожидается renderer вида:

- `NVIDIA ...`

Если viewer не открывается или рендер идёт через CPU, сначала проверь:

- `DISPLAY`
- mount `/tmp/.X11-unix`
- GPU visibility в compose
- `/dev/dri:/dev/dri`

## Проверка пайплайна

После сборки и запуска полезно прогнать pytest suite:

```bash
make test
```

## Planner и replay

Основные команды:

```bash
make planner
make replay NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz
```

Все остальные команды можно посмотреть через:

```bash
make help
```
