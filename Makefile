PYTHON ?= $(shell if [ -x /opt/homebrew/bin/python3.12 ]; then echo /opt/homebrew/bin/python3.12; else echo python3; fi)
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_MJPY := $(VENV)/bin/mjpython
GUI_PY := $(if $(wildcard $(VENV_MJPY)),$(VENV_MJPY),$(VENV_PY))

ARTIFACTS_DIR := artifacts
TRAIN_DIR := $(ARTIFACTS_DIR)/trained
POLICY := $(TRAIN_DIR)/routine.onnx
DEMO_DIR := $(ARTIFACTS_DIR)/demo
DEMO_VIDEO := $(DEMO_DIR)/scripted_pick_place.mp4

DEVICE ?= mps
IMAGE_SIZE ?= 64
MAX_STEPS ?= 220
SEED ?= 0
RUNS ?= 5

# ---- Demo (train-teacher-high / run-*) spawn range ----
# Narrow band where the scripted teacher is reliable. Matches what the
# committed routine.onnx is trained on, so `make run-headless` works
# out-of-the-box. Override at the CLI if you want a wider eval range.
DEMO_SPAWN_X_MIN ?= 0.140
DEMO_SPAWN_X_MAX ?= 0.141
DEMO_SPAWN_Y_MIN ?= -0.045
DEMO_SPAWN_Y_MAX ?= -0.044
DEMO_SPAWN_ARGS := \
  --block-spawn-x-min $(DEMO_SPAWN_X_MIN) --block-spawn-x-max $(DEMO_SPAWN_X_MAX) \
  --block-spawn-y-min $(DEMO_SPAWN_Y_MIN) --block-spawn-y-max $(DEMO_SPAWN_Y_MAX)

# ---- Pure-RL profile for `make train` (option C, no teacher) ----
NUM_ENVS ?= 8
HORIZON ?= 256
UPDATES ?= 600
MINIBATCH_SIZE ?= 256
UPDATE_EPOCHS ?= 4
LEARNING_RATE ?= 3e-4
TRAIN_RESET_SEED ?=
TRAIN_RESET_SEED_ARG := $(if $(strip $(TRAIN_RESET_SEED)),--train-reset-seed $(TRAIN_RESET_SEED),)
SUCCESS_WINDOW ?= 20
STOP_SUCCESS ?= 0.7
SAVE_EVERY ?= 25

# ---- BC + PPO fine-tune profile for `make train-teacher-medium` (option B) ----
MEDIUM_COLLECT_SUCCESSES ?= 12
MEDIUM_MAX_EPISODES ?= 60
MEDIUM_BC_EPOCHS ?= 120
MEDIUM_SEEDS ?= 0,1,2,3,4,5,6,7
MEDIUM_REPEAT_PER_SEED ?= 2
MEDIUM_UPDATES ?= 200
MEDIUM_DEMO_UPDATES ?= 60

# ---- Heavy BC profile for `make train-teacher-high` (option A, ~5 min CPU) ----
DEMO_COLLECT_SUCCESSES ?= 8
DEMO_MAX_EPISODES ?= 12
DEMO_BC_EPOCHS ?= 600
DEMO_SEEDS ?= 0,1,2,3
DEMO_REPEAT_PER_SEED ?= 2

.PHONY: help venv install train train-teacher-medium train-teacher-high run-headless run-headful demo demo-viewer record-demo syntax clean

help:
	@printf "%s\n" \
		"make install                         Create the venv and install dependencies" \
		"make train-teacher-high              (A) Heavy teacher BC, narrow spawn, ~5 min CPU — produces committed routine.onnx" \
		"make train-teacher-medium            (B) BC warm-start + PPO fine-tune, full spawn range" \
		"make train                           (C) Pure-RL PPO from reward only, no teacher, full spawn range" \
		"make run-headless RUNS=100           Evaluate routine.onnx on the demo spawn range and print success counts" \
		"make run-headful                     Open the passive MuJoCo viewer and run routine.onnx" \
		"make demo                            Run the scripted teacher headlessly (narrow spawn)" \
		"make demo-viewer                     Show the scripted teacher in the MuJoCo viewer" \
		"make record-demo                     Export a multi-camera MP4 of the scripted teacher" \
		"make syntax                          Run Python syntax checks" \
		"" \
		"Useful overrides:" \
		"  make train DEVICE=cpu" \
		"  make run-headless RUNS=20 POLICY=artifacts/other/routine.onnx" \
		"  make run-headless DEMO_SPAWN_X_MIN=-0.02 DEMO_SPAWN_X_MAX=0.18   # widen eval"

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip setuptools wheel

$(ARTIFACTS_DIR):
	mkdir -p $(ARTIFACTS_DIR)

venv: $(VENV_PY)

install: $(VENV_PY) requirements.txt
	$(VENV_PIP) install -r requirements.txt

# (C) Pure RL: no teacher, full spawn range — long-running.
train: install $(ARTIFACTS_DIR)
	mkdir -p $(TRAIN_DIR)
	$(VENV_PY) demo.py train \
		--device $(DEVICE) \
		--num-envs $(NUM_ENVS) \
		--horizon $(HORIZON) \
		--updates $(UPDATES) \
		--minibatch-size $(MINIBATCH_SIZE) \
		--update-epochs $(UPDATE_EPOCHS) \
		--learning-rate $(LEARNING_RATE) \
		--image-size $(IMAGE_SIZE) \
		--max-steps $(MAX_STEPS) \
		--seed $(SEED) \
		--save-every $(SAVE_EVERY) \
		$(TRAIN_RESET_SEED_ARG) \
		--success-window $(SUCCESS_WINDOW) \
		--stop-success $(STOP_SUCCESS) \
		--save-dir $(TRAIN_DIR)

# (B) BC warm-start + PPO fine-tune over full spawn range.
train-teacher-medium: install $(ARTIFACTS_DIR)
	rm -rf $(TRAIN_DIR)
	mkdir -p $(TRAIN_DIR)
	$(VENV_PY) demo.py train \
		--device $(DEVICE) \
		--num-envs $(NUM_ENVS) \
		--horizon $(HORIZON) \
		--updates $(MEDIUM_UPDATES) \
		--minibatch-size $(MINIBATCH_SIZE) \
		--update-epochs $(UPDATE_EPOCHS) \
		--learning-rate $(LEARNING_RATE) \
		--image-size $(IMAGE_SIZE) \
		--max-steps $(MAX_STEPS) \
		--seed $(SEED) \
		--save-every $(SAVE_EVERY) \
		--demo \
		--demo-updates $(MEDIUM_DEMO_UPDATES) \
		--demo-collect-successes $(MEDIUM_COLLECT_SUCCESSES) \
		--demo-max-episodes $(MEDIUM_MAX_EPISODES) \
		--demo-bc-epochs $(MEDIUM_BC_EPOCHS) \
		--demo-seeds $(MEDIUM_SEEDS) \
		--demo-repeat-per-seed $(MEDIUM_REPEAT_PER_SEED) \
		--success-window $(SUCCESS_WINDOW) \
		--stop-success $(STOP_SUCCESS) \
		--save-dir $(TRAIN_DIR)

# (A) Heavy teacher BC over narrow spawn — ~5 min CPU, committed artifact.
train-teacher-high: install $(ARTIFACTS_DIR)
	rm -rf $(TRAIN_DIR)
	mkdir -p $(TRAIN_DIR)
	$(VENV_PY) demo.py train \
		--device $(DEVICE) \
		--num-envs 1 \
		--horizon 1 \
		--updates 0 \
		--minibatch-size 256 \
		--update-epochs 1 \
		--learning-rate 3e-4 \
		--image-size $(IMAGE_SIZE) \
		--max-steps $(MAX_STEPS) \
		--seed $(SEED) \
		--save-every 1 \
		--demo \
		--demo-updates 0 \
		--demo-collect-successes $(DEMO_COLLECT_SUCCESSES) \
		--demo-max-episodes $(DEMO_MAX_EPISODES) \
		--demo-bc-epochs $(DEMO_BC_EPOCHS) \
		--demo-seeds $(DEMO_SEEDS) \
		--demo-repeat-per-seed $(DEMO_REPEAT_PER_SEED) \
		--demo-aux-weight 0.0 \
		--success-window 1 \
		--stop-success 2.0 \
		$(DEMO_SPAWN_ARGS) \
		--save-dir $(TRAIN_DIR)

run-headless: install
	$(VENV_PY) sim.py \
		--policy $(POLICY) \
		--runs $(RUNS) \
		--image-size $(IMAGE_SIZE) \
		--max-steps $(MAX_STEPS) \
		--seed $(SEED) \
		$(DEMO_SPAWN_ARGS) \
		--summary-only

run-headful: install
	$(GUI_PY) sim.py \
		--policy $(POLICY) \
		--image-size $(IMAGE_SIZE) \
		--max-steps $(MAX_STEPS) \
		--seed $(SEED) \
		$(DEMO_SPAWN_ARGS) \
		--viewer

demo: install
	$(VENV_PY) demo.py run \
		--episodes 1 \
		--image-size $(IMAGE_SIZE) \
		--max-steps 700 \
		--seed $(SEED) \
		$(DEMO_SPAWN_ARGS)

demo-viewer: install
	$(GUI_PY) demo.py run \
		--episodes 1 \
		--image-size $(IMAGE_SIZE) \
		--max-steps 700 \
		--seed $(SEED) \
		$(DEMO_SPAWN_ARGS) \
		--viewer

record-demo: install $(ARTIFACTS_DIR)
	mkdir -p $(DEMO_DIR)
	$(VENV_PY) demo.py run \
		--episodes 1 \
		--image-size $(IMAGE_SIZE) \
		--max-steps 700 \
		--seed $(SEED) \
		$(DEMO_SPAWN_ARGS) \
		--record $(DEMO_VIDEO)

syntax: install
	$(VENV_PY) -m py_compile demo.py sim.py train.py run.py

clean:
	rm -rf $(ARTIFACTS_DIR) __pycache__
