#!/usr/bin/env bash
set -u

SERVICE_NAME="${1:-scanner-backend.service}"
LOCAL_URL="${2:-http://127.0.0.1:8000/api/system/mode}"
LAN_URL="${3:-}"

ok() { printf 'OK   %s\n' "$1"; }
warn() { printf 'WARN %s\n' "$1"; }
fail() { printf 'FAIL %s\n' "$1"; }

echo "== Scanner backend recovery/check =="
echo "Service : ${SERVICE_NAME}"
echo "Local   : ${LOCAL_URL}"
if [ -n "${LAN_URL}" ]; then
  echo "LAN URL : ${LAN_URL}"
else
  echo "LAN URL : (not provided, LAN probe skipped)"
fi

if systemctl is-active --quiet "${SERVICE_NAME}"; then
  ok "systemd service is active"
else
  fail "systemd service is NOT active"
  systemctl --no-pager --full status "${SERVICE_NAME}" | sed -n '1,20p'
fi

local_code="$(curl -sS -o /dev/null -m 8 -w '%{http_code}' "${LOCAL_URL}" 2>/dev/null || true)"
if [ "${local_code}" = "200" ]; then
  ok "local endpoint health is HTTP 200"
else
  fail "local endpoint health failed (HTTP ${local_code:-N/A})"
fi

if [ -n "${LAN_URL}" ]; then
  lan_code="$(curl -sS -o /dev/null -m 8 -w '%{http_code}' "${LAN_URL}" 2>/dev/null || true)"
  if [ "${lan_code}" = "200" ]; then
    ok "LAN URL reachable from Pi shell (best-effort check)"
  else
    warn "LAN URL check did not return 200 (HTTP ${lan_code:-N/A}); verify from another LAN device/browser"
  fi
fi

echo
echo "Recent logs:"
journalctl -u "${SERVICE_NAME}" -n 20 --no-pager || true

