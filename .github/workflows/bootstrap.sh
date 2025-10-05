#!/usr/bin/env bash
# scripts/bootstrap.sh
set -euo pipefail

# ======== CONFIG ========
APP_USER="${APP_USER:-student-admin}"
APP_HOME="/home/${APP_USER}"
APP_DIR="${APP_DIR:-$APP_HOME/Case_study1}"
ENV_DIR="${ENV_DIR:-$APP_HOME/conda/envs/app311}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/Juju519/Case_study1.git}"
BRANCH="${BRANCH:-main}"
ENV_FILE="/etc/yourapp.env"
UNIT="/etc/systemd/system/gompei.service"
HEALTH_SVC="/etc/systemd/system/gompei-health.service"
HEALTH_TMR="/etc/systemd/system/gompei-health.timer"
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"
# ========================

echo "[bootstrap] starting"

# 0) user
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd -m "$APP_USER"
fi

# 1) Miniconda
if ! [ -x /opt/miniconda/bin/conda ]; then
  echo "[bootstrap] installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/m.sh
  bash /tmp/m.sh -b -p /opt/miniconda
fi
# shellcheck disable=SC1091
source /opt/miniconda/etc/profile.d/conda.sh

# 2) Repo
if [ ! -d "$APP_DIR/.git" ]; then
  echo "[bootstrap] cloning $GITHUB_REPO"
  sudo -u "$APP_USER" git clone -b "$BRANCH" "$GITHUB_REPO" "$APP_DIR"
else
  echo "[bootstrap] updating repo"
  pushd "$APP_DIR" >/dev/null
  sudo -u "$APP_USER" git fetch --all --prune
  sudo -u "$APP_USER" git reset --hard "origin/$BRANCH"
  popd >/dev/null
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# 3) Conda env
if ! [ -x "$PY" ]; then
  echo "[bootstrap] creating conda env at $ENV_DIR"
  conda create -y -p "$ENV_DIR" python=3.11
fi

# 4) Deps
echo "[bootstrap] installing deps"
"$PIP" install -U pip
[ -f "$APP_DIR/requirements.txt" ] && "$PIP" install -r "$APP_DIR/requirements.txt" || true
"$PIP" install "gradio==4.44.1" "huggingface_hub>=0.25" requests

# 5) Runtime env (secrets)
if [ ! -f "$ENV_FILE" ]; then
  echo "[bootstrap] writing $ENV_FILE (fill in secrets!)"
  cat >"$ENV_FILE" <<'EOF'
# === runtime config (edit values) ===
API_PROVIDER=nebius
NEBIUS_API_KEY=sk_fill_me
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1

# (Optional HF route)
# API_PROVIDER=hf
# HF_TOKEN=hf_fill_me
# HF_MODEL_ID=sshleifer/tiny-gpt2

GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
ENABLE_OAUTH=0
SKIP_UI_ON_IMPORT=0
EOF
fi
chmod 600 "$ENV_FILE"
chown root:root "$ENV_FILE"

# 6) systemd unit
echo "[bootstrap] installing systemd unit"
cat >"$UNIT" <<EOF
[Unit]
Description=Chat with Gompei (Gradio) service
Wants=network-online.target
After=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$PY $APP_DIR/app.py
Restart=always
RestartSec=5
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=false
PrivateTmp=true
KillSignal=SIGTERM
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF

# 7) health timer (self-restart if UI goes down)
cat >"$HEALTH_SVC" <<'EOF'
[Unit]
Description=Gompei healthcheck

[Service]
Type=oneshot
EnvironmentFile=/etc/yourapp.env
ExecStart=/bin/bash -lc 'curl -fsS --max-time 5 "http://127.0.0.1:${GRADIO_SERVER_PORT:-7860}/" >/dev/null || systemctl restart gompei.service'
EOF

cat >"$HEALTH_TMR" <<'EOF'
[Unit]
Description=Run Gompei healthcheck every minute
[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
AccuracySec=10s
Unit=gompei-health.service
[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable gompei.service gompei-health.timer
systemctl restart gompei.service
systemctl start gompei-health.timer

echo "[bootstrap] done"
systemctl --no-pager --full status gompei.service || true
