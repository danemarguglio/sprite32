#!/usr/bin/env bash
# Run scripts/sprite_web.py in a tmux session.
#   scripts/sprite_web.sh start [extra sprite_web.py args] | stop | restart | status | log
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=sprite_web
PORT=${SPRITE_WEB_PORT:-8077}
PY=${SPRITE_WEB_PY:-python}  # the repository's .venv when there is one
if [ -z "${SPRITE_WEB_PY:-}" ] && [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
LOG=outputs_diffusers/web_samples/server.log

case "${1:-status}" in
  start)
    shift || true
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running (tmux session $SESSION)"; exit 0; fi
    mkdir -p outputs_diffusers/web_samples
    tmux new-session -d -s "$SESSION" \
      "$PY scripts/sprite_web.py --port $PORT $* 2>&1 | tee -a $LOG"
    echo "started on port $PORT (tmux session $SESSION, log $LOG)"
    ;;
  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo stopped || echo "not running"
    ;;
  restart)
    shift || true
    "$0" stop
    sleep 2
    "$0" start "$@"
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "running (tmux session $SESSION)"
      curl -s --max-time 3 "http://127.0.0.1:$PORT/api/status" && echo || echo "not answering yet on port $PORT"
    else
      echo "not running"
    fi
    ;;
  log)
    tail -n 40 "$LOG"
    ;;
  *) echo "usage: $0 start|stop|restart|status|log"; exit 2 ;;
esac
