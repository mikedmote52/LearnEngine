#!/bin/bash
# LearnEngine daily reminder
# Reads /api/sync-due-count from Vercel; if >=5 due and freshness <=7 days, sends iMessage.
# Otherwise silent skip.

set -e

API="https://learnengine-eta.vercel.app/api/sync-due-count"
THRESHOLD=5
PHONE="+17146120723"
URL="https://mikedmote52.github.io/LearnEngine/#review"
LOG="$HOME/Library/Application Support/LearnEngine/cron/reminder.log"

mkdir -p "$(dirname "$LOG")"

ts_now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts_now)] $*" >> "$LOG"; }

PAYLOAD=$(curl -s -m 15 "$API" || echo '{"error":"fetch_failed"}')
COUNT=$(echo "$PAYLOAD" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('count') or 0)" 2>/dev/null || echo 0)
TS=$(echo "$PAYLOAD" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('stored_at') or 0)" 2>/dev/null || echo 0)

NOW_EPOCH=$(date +%s)
AGE_DAYS=$(python3 -c "import sys; ts=float('$TS' or 0); now=$NOW_EPOCH; print(round((now-ts)/86400,1) if ts else 999)")

log "fetched count=$COUNT age=${AGE_DAYS}d threshold=$THRESHOLD"

# Silent skip: under threshold
if [ "$COUNT" -lt "$THRESHOLD" ]; then
  log "skip: count under threshold"
  exit 0
fi

# Silent skip: stale data (>7d)
STALE=$(python3 -c "print(1 if float('$AGE_DAYS') > 7 else 0)")
if [ "$STALE" -eq 1 ]; then
  log "skip: data stale (${AGE_DAYS}d)"
  exit 0
fi

# Allow override args: --dry-run, --force
if [ "${1:-}" = "--dry-run" ]; then
  log "dry-run: would send 'Daily review · $COUNT cards due · $URL'"
  echo "DRY-RUN: would send iMessage with count=$COUNT"
  exit 0
fi

MSG="Daily review · $COUNT cards due · $URL"
log "send iMessage: $MSG"

osascript -e "tell application \"Messages\"
  set targetService to 1st service whose service type = iMessage
  set targetBuddy to buddy \"$PHONE\" of targetService
  send \"$MSG\" to targetBuddy
end tell" 2>>"$LOG" && log "iMessage sent" || log "iMessage send failed"
