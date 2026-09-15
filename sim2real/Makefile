SHELL := /bin/bash
.DEFAULT_GOAL := help

DOCKER_BUILDKIT ?= 1
COMPOSE_DOCKER_CLI_BUILD ?= 1

COMPOSE := DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) COMPOSE_DOCKER_CLI_BUILD=$(COMPOSE_DOCKER_CLI_BUILD) docker compose -f docker/compose.yml
CONTAINER := sim2real-simulation
PYTHON := /opt/conda/bin/python

KEY ?= airy_table_scene14sep26_left_image
TASK ?= pick_up
TASK_OBJECT ?=
CONFIG ?= config/config_debug.yaml
SCENE ?= assets/scenes/$(KEY)/simulation/scene.json
NPZ ?=
OUT_CSV ?=
OUT_YAML ?=
OUT_JSON ?=
COMPRESS_OUTPUT_NPZ ?=
COMPRESS_OUTPUT_GIF ?=
COMPRESS_ARGS ?=
TASK_OBJECT_ARG := $(if $(strip $(TASK_OBJECT)),--task_object_id $(TASK_OBJECT),)
OUT_CSV_ARG := $(if $(strip $(OUT_CSV)),--output_csv $(OUT_CSV),)
OUT_YAML_ARG := $(if $(strip $(OUT_YAML)),--output_runtime_config_yaml $(OUT_YAML),)
OUT_JSON_ARG := $(if $(strip $(OUT_JSON)),--output_runtime_request_json $(OUT_JSON),)
COMPRESS_OUTPUT_NPZ_ARG := $(if $(strip $(COMPRESS_OUTPUT_NPZ)),--output_npz $(COMPRESS_OUTPUT_NPZ),)
COMPRESS_OUTPUT_GIF_ARG := $(if $(strip $(COMPRESS_OUTPUT_GIF)),--output_gif $(COMPRESS_OUTPUT_GIF),)

BLUE := \033[1;34m
CYAN := \033[1;36m
GREEN := \033[1;32m
YELLOW := \033[1;33m
RESET := \033[0m

.PHONY: help config build up stop down restart ps logs shell planner planner-headless replay replay-dry npz-stats stats csv compress test

help: ## Показать доступные команды Make и основные override-параметры
	@printf "$(BLUE)Команды Make для Sim2Real$(RESET)\n\n"
	@printf "$(YELLOW)Переопределяемые переменные$(RESET)\n"
	@printf "  $(CYAN)KEY$(RESET)=%s\n" "$(KEY)"
	@printf "  $(CYAN)TASK$(RESET)=%s\n" "$(TASK)"
	@printf "  $(CYAN)TASK_OBJECT$(RESET)=%s\n" "$(TASK_OBJECT)"
	@printf "  $(CYAN)CONFIG$(RESET)=%s\n" "$(CONFIG)"
	@printf "  $(CYAN)SCENE$(RESET)=%s\n" "$(SCENE)"
	@printf "  $(CYAN)NPZ$(RESET)=/app/runs/.../rl4vla_raw_episode_success.npz\n\n"
	@printf "  $(CYAN)OUT_CSV$(RESET)=/tmp/actions.csv\n"
	@printf "  $(CYAN)OUT_YAML$(RESET)=/tmp/runtime_config.yaml\n"
	@printf "  $(CYAN)OUT_JSON$(RESET)=/tmp/runtime_request.json\n"
	@printf "  $(CYAN)COMPRESS_OUTPUT_NPZ$(RESET)=/app/runs/.../episode.compressed.npz\n"
	@printf "  $(CYAN)COMPRESS_OUTPUT_GIF$(RESET)=/app/runs/.../episode.compressed.gif\n"
	@printf "  $(CYAN)COMPRESS_ARGS$(RESET)=--allow_negative_dz_merge --negative_dz_keep_tail_ratio 0.2 --max_merged_negative_dz 0.004\n\n"
	@printf "$(YELLOW)Примеры$(RESET)\n"
	@printf "  make build\n"
	@printf "  make up\n"
	@printf "  make planner\n"
	@printf "  make planner TASK_OBJECT=yellow_cube_ext\n"
	@printf "  make planner KEY=airy_table_scene14sep26_left_image TASK=pick_up\n"
	@printf "  make planner KEY=airy_table_scene14sep26_left_image TASK=pick_up TASK_OBJECT=yellow_cube_ext\n"
	@printf "  make planner KEY=airy_table_scene14sep26_left_image TASK=pick_up SCENE=assets/scenes/airy_table_scene14sep26_left_image/simulation/scene.json\n"
	@printf "  make test\n"
	@printf "  make stats NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz\n"
	@printf "  make csv NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz\n"
	@printf "  make csv NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz OUT_CSV=/tmp/actions.csv OUT_YAML=/tmp/runtime_config.yaml OUT_JSON=/tmp/runtime_request.json\n"
	@printf "  make compress NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz\n"
	@printf "  make compress NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz COMPRESS_ARGS='--allow_negative_dz_merge --negative_dz_keep_tail_ratio 0.2 --max_merged_negative_dz 0.004'\n"
	@printf "  make replay NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz\n"
	@printf "  make npz-stats NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz\n"
	@printf "  make replay-dry NPZ=/app/runs/manual/<run>/rl4vla_raw_episode_success.npz\n\n"
	@printf "$(YELLOW)Команды$(RESET)\n"
	@awk 'BEGIN {FS = ":.*## ";} /^[a-zA-Z0-9_.-]+:.*## / {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

config: ## Проверить и вывести итоговую конфигурацию docker compose
	$(COMPOSE) config

build: ## Собрать prerelease Docker image
	$(COMPOSE) build

up: ## Поднять persistent simulation container в фоне
	$(COMPOSE) up -d

stop: ## Остановить simulation container без удаления
	$(COMPOSE) stop

down: ## Удалить simulation container и compose-ресурсы
	$(COMPOSE) down

restart: ## Пересоздать simulation container
	$(COMPOSE) down
	$(COMPOSE) up -d

ps: ## Показать статус compose-сервиса
	$(COMPOSE) ps

logs: ## Показать логи simulation container
	$(COMPOSE) logs -f simulation

shell: ## Открыть интерактивный shell внутри simulation container
	docker exec -it $(CONTAINER) bash -lc 'cd /app && exec bash'

planner: ## Запустить proxy planner с визуализацией
	docker exec -it $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py --motion_backend proxy_ee_delta --config_path $(CONFIG) --key $(KEY) --scene $(SCENE) --task_type $(TASK) $(TASK_OBJECT_ARG)'

planner-headless: ## Запустить proxy planner в headless-режиме
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py --motion_backend proxy_ee_delta --config_path $(CONFIG) --key $(KEY) --scene $(SCENE) --task_type $(TASK) $(TASK_OBJECT_ARG) --headless'

replay: ## Запустить replay для сохранённого RL4VLA raw episode; нужен NPZ=/app/...
	@if [ -z "$(NPZ)" ]; then echo "Set NPZ=/app/runs/.../rl4vla_raw_episode_success.npz"; exit 1; fi
	docker exec -it $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py --npz_path $(NPZ)'

replay-dry: ## Проверить saved RL4VLA raw episode без открытия viewer
	@if [ -z "$(NPZ)" ]; then echo "Set NPZ=/app/runs/.../rl4vla_raw_episode_success.npz"; exit 1; fi
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py --npz_path $(NPZ) --dry_run'

npz-stats: ## Показать summary/statistics для NPZ без открытия viewer
	@if [ -z "$(NPZ)" ]; then echo "Set NPZ=/app/runs/.../rl4vla_raw_episode_success.npz"; exit 1; fi
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py --npz_path $(NPZ) --dry_run'

stats: ## Показать статистику action-каналов для RL4VLA raw episode
	@if [ -z "$(NPZ)" ]; then echo "Set NPZ=/app/runs/.../rl4vla_raw_episode_success.npz"; exit 1; fi
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/summarize_npz_actions.py --npz_path $(NPZ)'

csv: ## Экспортировать actions в CSV и, если доступны, embedded runtime YAML/JSON
	@if [ -z "$(NPZ)" ]; then echo "Set NPZ=/app/runs/.../rl4vla_raw_episode_success.npz"; exit 1; fi
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/export_npz_actions_to_csv.py --npz_path $(NPZ) $(OUT_CSV_ARG) $(OUT_YAML_ARG) $(OUT_JSON_ARG)'

compress: ## Сжать RL4VLA raw episode в более короткий NPZ и сохранить GIF sidecar
	@if [ -z "$(NPZ)" ]; then echo "Set NPZ=/app/runs/.../rl4vla_raw_episode_success.npz"; exit 1; fi
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) openreal2sim/simulation/maniskill/scripts/compress_npz_actions.py --npz_path $(NPZ) $(COMPRESS_OUTPUT_NPZ_ARG) $(COMPRESS_OUTPUT_GIF_ARG) $(COMPRESS_ARGS)'

test: ## Запустить planner pytest suite внутри simulation container
	docker exec $(CONTAINER) bash -lc 'cd /app && PYTHONPATH=/app $(PYTHON) -m pytest tests/planner -q'
