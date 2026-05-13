#!/usr/bin/env python3
"""WinXP-themed web chat UI for testing Bliss locally from phone/browser.

Runs a persistent nc_run_native backend on the Mac and serves a tiny
mobile-friendly web UI. Intended for LAN/Tailscale testing before packaging.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "build" / "nc_run_native"
MODEL = ROOT / "build" / "deploy" / "MODEL.NCB"
TOKENIZER = ROOT / "build" / "deploy" / "TOKENIZER.NCT"
SOH = b"\x01"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Bliss Chat XP Web</title>
<style>
  :root{
    --xp-blue:#245edb; --xp-blue2:#0c3aa5; --xp-silver:#ece9d8; --xp-border:#003c74;
    --xp-green:#38b000; --xp-shadow:#6b86d9; --text:#111; --muted:#555;
  }
  *{box-sizing:border-box} html,body{height:100%;margin:0;overflow:hidden}
  body{font:13px Tahoma,Verdana,Arial,sans-serif;color:var(--text);
    background:linear-gradient(180deg,#7db9ff 0,#3f79e8 45%,#245edb 100%);
    padding:env(safe-area-inset-top) 8px env(safe-area-inset-bottom)}
  .window{height:calc(var(--app-height, 100dvh) - env(safe-area-inset-top) - env(safe-area-inset-bottom) - 16px);
    min-height:0;max-width:860px;margin:8px auto;border:1px solid #08216b;border-radius:8px 8px 3px 3px;
    box-shadow:0 12px 30px rgba(0,0,0,.35);display:flex;flex-direction:column;overflow:hidden;background:var(--xp-silver)}
  @supports not (height:100dvh){.window{height:calc(100vh - env(safe-area-inset-top) - env(safe-area-inset-bottom) - 16px)}}
  .titlebar{height:34px;padding:5px 8px;color:#fff;font-weight:bold;display:flex;align-items:center;gap:8px;
    background:linear-gradient(180deg,#2f7df5 0,#245edb 45%,#0b3cae 100%);text-shadow:1px 1px #003;
    border-bottom:1px solid #001b5a}.icon{width:20px;height:20px;border-radius:4px;background:linear-gradient(135deg,#ffff9a,#00a650);border:1px solid #fff}.spacer{flex:1}
  .winbtn{width:22px;height:22px;border:1px solid #fff;border-radius:3px;background:linear-gradient(#ffb3a7,#d93424);box-shadow:inset 0 -1px #901;color:#fff;text-align:center;line-height:19px;font-weight:bold}
  .toolbar{padding:7px 8px;border-bottom:1px solid #aca899;background:linear-gradient(#fff,#ece9d8);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button{font:13px Tahoma,Verdana,sans-serif;padding:4px 10px;border:1px solid #7f9db9;border-radius:3px;background:linear-gradient(#fff,#d8e8ff);min-height:28px}button:active{background:linear-gradient(#c2d9ff,#fff)}button:disabled{color:#888;background:#ddd}
  .status{font-size:12px;color:#333}.badge{background:#fff;border:1px solid #aca899;padding:2px 6px;border-radius:2px}.good{color:#087208}.bad{color:#a00}
  .chat{flex:1;min-height:0;overflow:auto;padding:10px;background:#fff;border:1px solid #7f9db9;margin:8px;box-shadow:inset 1px 1px 2px #999;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth}
  .msg{max-width:92%;border-radius:4px;padding:8px 10px;line-height:1.35;white-space:pre-wrap;word-wrap:break-word;border:1px solid #aaa}.user{align-self:flex-end;background:#d9ecff;border-color:#7bb0e8}.assistant{align-self:flex-start;background:#fffde7;border-color:#d0c36a}.meta{font-size:11px;color:var(--muted);margin-top:4px}.empty{color:#666;text-align:center;margin:auto;font-size:13px}
  .composer{display:flex;gap:8px;padding:0 8px 8px;background:var(--xp-silver);flex-shrink:0}.composer textarea{flex:1;min-height:48px;max-height:120px;resize:vertical;font:16px Tahoma,Verdana,sans-serif;padding:8px;border:1px solid #7f9db9;box-shadow:inset 1px 1px 2px #999}.send{min-width:78px;font-weight:bold}
  @media (max-width:520px){
    body{padding:0;overscroll-behavior:none}
    .window{width:100vw;height:var(--app-height, 100dvh);margin:0;border-radius:0;border-left:0;border-right:0;box-shadow:none}
    @supports not (height:100dvh){.window{height:100vh}}
    .titlebar{height:42px;padding:5px 8px;font-size:16px;gap:7px;border-radius:0}.icon{width:24px;height:24px}.winbtn{width:28px;height:28px;line-height:24px}
    .toolbar{padding:6px 8px;gap:6px;flex-wrap:nowrap;overflow:hidden}.toolbar .status:not(.badge){display:none}.badge{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:calc(100vw - 120px)}
    button{font-size:16px;min-height:38px;padding:6px 12px}.status{font-size:14px}
    .chat{margin:6px;padding:8px;gap:8px}.msg{max-width:96%;font-size:15px}.empty{font-size:15px;line-height:1.3;padding:0 10px}
    .composer{padding:4px 6px calc(8px + env(safe-area-inset-bottom));gap:6px}.composer textarea{font-size:16px;min-height:44px;max-height:86px}.send{min-width:70px}
  }
</style>
</head>
<body>
<div class="window">
  <div class="titlebar"><div class="icon"></div><div>Bliss Chat - Windows XP Web</div><div class="spacer"></div><div class="winbtn">×</div></div>
  <div class="toolbar">
    <button id="resetBtn">New Chat</button>
    <span id="status" class="status badge">Connecting...</span>
    <span class="status">defaults: temp 0.0 · top-p 0.95 · ctx 256 · prompt assist</span>
  </div>
  <div id="chat" class="chat"><div class="empty">Type a message below. This is the Mac-hosted test UI, not a new XP package.</div></div>
  <div class="composer">
    <textarea id="prompt" placeholder="Ask Bliss something..." autocomplete="off"></textarea>
    <button id="sendBtn" class="send">Send</button>
  </div>
</div>
<script>
function syncViewport(){
  const h = window.visualViewport ? window.visualViewport.height : window.innerHeight;
  document.documentElement.style.setProperty('--app-height', h + 'px');
}
syncViewport();
window.addEventListener('resize', syncViewport);
if(window.visualViewport){window.visualViewport.addEventListener('resize', syncViewport);}
const chat = document.getElementById('chat');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');
const statusEl = document.getElementById('status');
function setStatus(txt, ok=true){statusEl.textContent=txt; statusEl.className='status badge '+(ok?'good':'bad')}
function addMsg(role, text, meta=''){
  const empty=chat.querySelector('.empty'); if(empty) empty.remove();
  const d=document.createElement('div'); d.className='msg '+role; d.textContent=text;
  if(meta){ const m=document.createElement('div'); m.className='meta'; m.textContent=meta; d.appendChild(m); }
  chat.appendChild(d); chat.scrollTop=chat.scrollHeight; return d;
}
async function refreshStatus(){
  try{const r=await fetch('/api/status'); const j=await r.json(); setStatus(j.ready?j.model:'not ready', j.ready)}
  catch(e){setStatus('offline', false)}
}
async function send(text){
  text = (text || promptEl.value).trim(); if(!text) return;
  promptEl.value=''; sendBtn.disabled=true; resetBtn.disabled=true;
  addMsg('user', text); const pending=addMsg('assistant','Thinking...'); setStatus('generating...', true);
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:text})});
    const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText);
    pending.firstChild.nodeValue = j.answer || '(empty)';
    const meta=document.createElement('div'); meta.className='meta'; meta.textContent=`${j.tokens} tokens · ${j.tok_per_sec.toFixed(1)} tok/s · ${j.elapsed.toFixed(2)}s`; pending.appendChild(meta);
    setStatus(j.model || 'ready', true);
  }catch(e){pending.firstChild.nodeValue='Error: '+e.message; setStatus('error', false)}
  finally{sendBtn.disabled=false; resetBtn.disabled=false; promptEl.focus()}
}
sendBtn.onclick=()=>send();
resetBtn.onclick=async()=>{await fetch('/api/reset',{method:'POST'}); chat.innerHTML='<div class="empty">New chat started.</div>'; refreshStatus();};
promptEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
refreshStatus(); setInterval(refreshStatus,5000);
</script>
</body>
</html>
"""

class BlissBackend:
    """Simple one-shot backend runner.

    Persistent pipes are faster, but a one-shot process is much more reliable
    for phone testing because every request starts from a clean KV cache and
    cannot be contaminated by earlier bad turns/settings. Startup cost is ~12s
    on this Mac for d12; generation itself is fast.
    """
    def __init__(self, ctx: int, temp: float, top_p: float, seed: int | None = None):
        self.ctx = ctx
        self.temp = temp
        self.top_p = top_p
        self.seed = seed
        self.lock = threading.Lock()
        self.model = "Bliss d12 293M (int8)"
        self.ready = True

    def command(self, line: str):
        if line.strip() == "/reset":
            return {"answer": "", "tokens": 0, "elapsed": 0.0, "tok_per_sec": 0.0, "info": ["one-shot mode: every prompt is already a fresh chat"], "model": self.model}
        shaped_line = shape_prompt(line)
        with self.lock:
            cmd = [str(BACKEND), str(MODEL), str(TOKENIZER), "-c", str(self.ctx), "-t", str(self.temp), "-p", str(self.top_p)]
            if self.seed is not None:
                cmd += ["-s", str(self.seed)]
            start = time.time()
            proc = subprocess.run(cmd, input=(shaped_line.rstrip("\n") + "\n").encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(ROOT), timeout=120)
            elapsed = max(time.time() - start, 1e-6)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace") or f"backend exited {proc.returncode}")
            parsed = parse_backend_output(proc.stdout)
            if parsed["model"]:
                self.model = parsed["model"]
            tokens = parsed["tokens"]
            return {"answer": parsed["answer"].strip(), "tokens": tokens, "elapsed": elapsed, "tok_per_sec": tokens / elapsed, "info": parsed["info"], "model": self.model}

    def stop(self):
        pass

def parse_backend_output(data: bytes):
    text: list[str] = []
    info: list[str] = []
    model = ""
    tokens = 0
    i = 0
    while i < len(data):
        if data[i:i+1] == SOH:
            j = data.find(b"\n", i)
            if j < 0:
                break
            line = data[i+1:j].decode("utf-8", "replace").strip("\r")
            if " " in line:
                kind, payload = line.split(" ", 1)
            else:
                kind, payload = line, ""
            if kind == "INFO":
                info.append(payload)
                if payload.startswith("Bliss"):
                    model = payload
            elif kind == "EOT":
                try:
                    tokens = int(payload.strip() or "0")
                except ValueError:
                    tokens = 0
            i = j + 1
        else:
            text.append(chr(data[i]))
            i += 1
    return {"answer": "".join(text), "tokens": tokens, "info": info, "model": model}

def shape_prompt(prompt: str) -> str:
    """Make prompts easier for the tiny base model without changing the model.

    The d12 model responds much better to completion-style prompts than to
    high-level instruction wording like "please compliment Phil". Keep this
    intentionally small and visible in the UI as "prompt assist".
    """
    import re
    p = " ".join(prompt.strip().split())
    low = p.lower()
    if "compliment" in low:
        name = None
        patterns = [
            r"named\s+([A-Za-z][A-Za-z'-]*)",
            r"compliment\s+(?:for\s+)?(?:my\s+friend\s+)?([A-Za-z][A-Za-z'-]*)",
            r"compliment\s+to\s+(?:my\s+friend\s+)?([A-Za-z][A-Za-z'-]*)",
            r"compliment\s+for\s+(?:somebody|someone|a person)\s+(?:named\s+)?([A-Za-z][A-Za-z'-]*)",
        ]
        for pat in patterns:
            m = re.search(pat, p, re.I)
            if m:
                name = m.group(1)
                break
        if name and name.lower() not in {"for", "my", "friend", "somebody", "someone", "person"}:
            return f"Write exactly: {name} is a great person."
    if "guitar" in low and "fact" in low:
        return "A cool guitar fact:"
    return prompt

backend: BlissBackend | None = None

class Handler(BaseHTTPRequestHandler):
    server_version = "BlissXPWeb/0.1"
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            assert backend is not None
            self.json(200, {"ready": backend.ready, "model": backend.model, "ctx": backend.ctx, "temp": backend.temp, "top_p": backend.top_p})
        else:
            self.json(404, {"error": "not found"})
    def do_POST(self):
        global backend
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        try:
            assert backend is not None
            if path == "/api/chat":
                prompt = str(data.get("prompt", "")).strip()
                if not prompt:
                    self.json(400, {"error": "empty prompt"})
                    return
                self.json(200, backend.command(prompt))
            elif path == "/api/reset":
                self.json(200, backend.command("/reset"))
            else:
                self.json(404, {"error": "not found"})
        except Exception as e:
            self.json(500, {"error": str(e)})

def main() -> int:
    global backend
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    for p in (BACKEND, MODEL, TOKENIZER):
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")
    backend = BlissBackend(args.ctx, args.temp, args.top_p, args.seed)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Bliss XP Web ready on http://{args.host}:{args.port}/ ({backend.model})", flush=True)
    try:
        httpd.serve_forever()
    finally:
        if backend:
            backend.stop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
