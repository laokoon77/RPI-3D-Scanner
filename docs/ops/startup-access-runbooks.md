# Raspberry Pi Scanner Backend Startup + Access Runbooks

Scope: production-grade startup/access behavior with minimal and reversible changes.

## 1) Stable product path (recommended): systemd on boot

Service template file: `docs/ops/scanner-backend.service`

### 1.1 Install/update service on Pi

From Windows (PowerShell/cmd with PuTTY tools on PATH), using host-key pinning:

```powershell
pscp -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" .\docs\ops\scanner-backend.service user@100.106.34.100:/tmp/scanner-backend.service
```

```powershell
plink -ssh -batch -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" user@100.106.34.100 "sudo install -m 644 /tmp/scanner-backend.service /etc/systemd/system/scanner-backend.service && sudo systemctl daemon-reload && sudo systemctl enable scanner-backend.service && sudo systemctl restart scanner-backend.service"
```

### 1.2 Verify service health on Pi

```bash
sudo systemctl is-active scanner-backend.service
sudo systemctl status scanner-backend.service --no-pager
curl -sS -m 8 http://127.0.0.1:8000/api/system/mode
journalctl -u scanner-backend.service -n 50 --no-pager
```

Expected patterns:
- `is-active` returns `active`
- local endpoint returns JSON with `"ok": true`
- journal shows app startup and request logs

### 1.3 LAN access behavior

Service binds to `0.0.0.0:8000`, so browser clients on same LAN can open:

`http://<pi-lan-ip>:8000/`

## 2) Temporary developer path (non-boot-managed)

Use only during development/troubleshooting.

### 2.1 Start temporary process over SSH

```powershell
plink -ssh -batch -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" user@100.106.34.100 "cd '/home/user/RPI 3D Scanner' && pkill -f 'uvicorn webapp:app' 2>/dev/null || true && nohup ./.venv/bin/python -m uvicorn webapp:app --host 0.0.0.0 --port 8000 >/tmp/scanner_uvicorn.log 2>&1 < /dev/null & sleep 3 && curl -sS -m 8 http://127.0.0.1:8000/api/system/mode"
```

### 2.2 Temporary-path notes

- Not persistent across reboot.
- Not supervised by `systemd`.
- Log file is `/tmp/scanner_uvicorn.log`.

## 3) Operator runbook (non-technical)

1. Power on Raspberry Pi and wait ~1 minute.
2. On another device in same LAN, open browser:
   - `http://<pi-lan-ip>:8000/`
3. If page does not load, ask support to run on Pi:
   - `tools/pi_service_check.sh scanner-backend.service http://127.0.0.1:8000/api/system/mode http://<pi-lan-ip>:8000/api/system/mode`
4. If check reports service not active, support runs:
   - `sudo systemctl restart scanner-backend.service`
5. Re-open browser and confirm UI loads.

## 4) Developer Windows -> Pi update runbook (exact flow)

Always use host-key pinning (`-hostkey ...`) in `plink`/`pscp` commands.

### 4.0 One-time host key pinning (verification)

On trusted channel, confirm Pi fingerprint is:

`ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4`

Then use this exact value in every command below.

### 4.1 Local edit

Edit file(s) in workspace.

### 4.2 Local syntax check

```powershell
python -m py_compile webapp.py calibration_intrinsics.py
```

If only one file changed, compile that file accordingly.

### 4.3 Copy file(s) to Pi

```powershell
pscp -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" webapp.py user@100.106.34.100:"/home/user/RPI 3D Scanner/webapp.py"
pscp -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" calibration_intrinsics.py user@100.106.34.100:"/home/user/RPI 3D Scanner/calibration_intrinsics.py"
```

### 4.4 Remote syntax check

```powershell
plink -ssh -batch -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" user@100.106.34.100 "cd '/home/user/RPI 3D Scanner' && ./.venv/bin/python -m py_compile webapp.py calibration_intrinsics.py && echo REMOTE_COMPILE_OK"
```

### 4.5 Restart service (stable path)

```powershell
plink -ssh -batch -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" user@100.106.34.100 "sudo systemctl restart scanner-backend.service && sudo systemctl is-active scanner-backend.service"
```

### 4.6 Verify health + key APIs

```powershell
plink -ssh -batch -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" user@100.106.34.100 "curl -sS -m 8 http://127.0.0.1:8000/api/system/mode && echo && curl -sS -m 8 http://127.0.0.1:8000/api/calibration/status && echo"
```

From Windows LAN side (replace IP):

```powershell
curl -sS -m 8 http://<pi-lan-ip>:8000/api/system/mode
```

Optional current known calibration API check:

```powershell
plink -ssh -batch -hostkey "ssh-ed25519 255 SHA256:prFHITHc4T05mFXEmwTCM11o6k5QV+y9wDAZ7RjsFK4" user@100.106.34.100 "curl -sS -m 8 http://127.0.0.1:8000/api/calibration/intrinsics/charuco-manual/status && echo"
```

## 5) Rollback instructions

### 5.1 Rollback to previous app file(s)

1. Copy backed-up file(s) back to `/home/user/RPI 3D Scanner/`.
2. Run remote compile check.
3. Restart stable service:
   - `sudo systemctl restart scanner-backend.service`

### 5.2 Rollback service configuration

If new service file is problematic:

```bash
sudo cp /etc/systemd/system/scanner-backend.service /etc/systemd/system/scanner-backend.service.bak.$(date +%Y%m%d%H%M%S)
# restore older known-good unit file
sudo systemctl daemon-reload
sudo systemctl restart scanner-backend.service
sudo systemctl status scanner-backend.service --no-pager
```

### 5.3 Emergency temporary recovery path

If systemd path fails and immediate access is required, use temporary path command from section 2, then repair `systemd` configuration.

## 6) Smoke-test results (actual evidence, 2026-02-22)

Scope for this evidence run: startup/access verification only, executed from Windows to Pi with host-key pinning in PuTTY commands.

### 6.1 Pass/fail summary

| Verification item | Verdict | Evidence summary |
|---|---|---|
| Service file install + daemon-reload | PASS | `pscp` copy to `/tmp`, `sudo install -m 644`, `sudo systemctl daemon-reload` completed without install/reload errors. |
| Service enabled on boot (`enable`, `is-enabled`) | PASS | `systemctl enable scanner-backend.service` and `systemctl is-enabled` returned `enabled`. |
| Service active after restart (`restart`, `is-active`) | PASS | `systemctl restart` returned cleanly and `systemctl is-active scanner-backend.service` returned `active`. |
| Reboot-start behavior | LIMITED (NOT EXECUTED) | Reboot test intentionally deferred to avoid disrupting active access. Non-reboot startup evidence is strong: `UnitFileState=enabled`, `is-enabled=enabled`, and current runtime is healthy (`is-active=active`). |
| Health endpoint on Pi (`curl localhost`) | PASS | Pi-local checks returned JSON on `http://127.0.0.1:8000/api/system/mode` and `http://127.0.0.1:8000/api/calibration/status`. |
| Key API endpoint (`/api/system/mode`) | PASS | Endpoint returned `{"ok":true,...}` after service restart. |
| Calibration status endpoint (`/api/calibration/status`) | PASS | Endpoint returned `{"ok":true,...}` after service restart. |
| LAN path from Windows (`http://<pi-ip>:8000/...`) | PASS | Windows LAN check to `http://100.106.34.100:8000/` returned HTTP `200`. |
| Developer update cycle (local compile -> copy -> remote compile -> restart -> verify endpoint) | PASS | Compile/copy/remote compile/restart succeeded and endpoint verification returned JSON. |

### 6.2 Command/output snippets captured

```text
# Enable + state
EVIDENCE_ENABLE_CHECK
enabled
EVIDENCE_UNITFILESTATE
UnitFileState=enabled
```

```text
# Runtime and endpoint checks after fixed unit deployment
EVIDENCE_IS_ACTIVE
active
EVIDENCE_MODE
{"ok":true,"mock_hw":false,...}
EVIDENCE_CAL
{"ok":true,...}
```

```text
# Windows LAN checks
EVIDENCE_LAN_ROOT_HTTP
200
```

```text
# Developer cycle evidence
DEV_LOCAL_COMPILE_OK
DEV_COPY_OK
DEV_REMOTE_COMPILE_OK
active
DEV_ENDPOINT
{"ok":true,"mock_hw":false,...}
```

### 6.3 Reboot verification note

- Reboot test was **not executed** in this evidence run to avoid disrupting access during active troubleshooting.
- Strongest non-reboot startup evidence was collected instead: enable state and unit-file persistence are correct, and runtime is currently healthy (`is-active=active`) with successful local/LAN endpoint checks.

