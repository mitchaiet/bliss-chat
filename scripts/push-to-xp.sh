#!/bin/bash
# Push built artifacts to the live XP box at C:\xp-llm\.
# Assumes:
#   - HTTP server (python3 -m http.server 8088) running locally that serves $HERE/build/
#   - XP machine has telnet (xpt/xpt-dimension), get.vbs and XPGET.EXE in C:\xp-llm
#
# Usage: push-to-xp.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$HERE/build"
DEPLOY="$BUILD/deploy"

XP_HOST="${XP_HOST:-192.168.1.31}"
XP_USER="${XP_USER:-xpt}"
XP_PASS="${XP_PASS:-xpt-dimension}"
MAC_IP="${MAC_IP:-192.168.1.43}"
HTTP_PORT="${HTTP_PORT:-8088}"

# Stage files to a single directory the HTTP server already serves
STAGE="${STAGE:-/Users/mitchaiet/VMs/WindowsXP/xp-deploy}"
mkdir -p "$STAGE"
cp -f "$BUILD/NC_RUN.EXE"  "$STAGE/"
cp -f "$BUILD/XPCHAT.EXE"  "$STAGE/"
[ -f "$DEPLOY/MODEL.NCB" ]     && cp -f "$DEPLOY/MODEL.NCB"     "$STAGE/"
[ -f "$DEPLOY/TOKENIZER.NCT" ] && cp -f "$DEPLOY/TOKENIZER.NCT" "$STAGE/"

# Make sure HTTP server is up (background)
if ! curl -sI "http://${MAC_IP}:${HTTP_PORT}/NC_RUN.EXE" >/dev/null 2>&1; then
  (cd "$STAGE" && nohup python3 -m http.server "$HTTP_PORT" --bind "$MAC_IP" > /tmp/xp-http.log 2>&1 &)
  sleep 1
fi

expect <<EOF
set timeout 90
log_user 1
spawn telnet $XP_HOST
expect "password:"; send "\r"
expect "login:"; send "$XP_USER\r"
expect "password:"; send "$XP_PASS\r"
expect ">"
send "cd /D C:\\\\xp-llm\r"; expect ">"
send "taskkill /F /IM XPCHAT.EXE /IM NC_RUN.EXE 2>nul\r"; expect ">"
send "ping -n 2 127.0.0.1 >nul\r"; expect ">"
send "del NC_RUN.EXE XPCHAT.EXE 2>nul\r"; expect ">"
send "cscript //nologo get.vbs http://${MAC_IP}:${HTTP_PORT}/NC_RUN.EXE NC_RUN.EXE\r"
expect { -re "OK NC_RUN.EXE" { expect ">" } timeout { exit 1 } }
send "cscript //nologo get.vbs http://${MAC_IP}:${HTTP_PORT}/XPCHAT.EXE XPCHAT.EXE\r"
expect { -re "OK XPCHAT.EXE" { expect ">" } timeout { exit 1 } }
set timeout 180
send "XPGET.EXE ${MAC_IP} ${HTTP_PORT} /MODEL.NCB MODEL.NCB\r"
expect { -re "xpget: done" { expect ">" } timeout { exit 1 } }
set timeout 60
send "cscript //nologo get.vbs http://${MAC_IP}:${HTTP_PORT}/TOKENIZER.NCT TOKENIZER.NCT\r"
expect { -re "OK TOKENIZER.NCT" { expect ">" } timeout { exit 1 } }
send "dir NC_RUN.EXE XPCHAT.EXE MODEL.NCB TOKENIZER.NCT\r"; expect ">"
send "exit\r"; expect eof
EOF
