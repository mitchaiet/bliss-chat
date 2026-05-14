#!/usr/bin/env python3
"""
nanochat training dashboard.
- Parses ~/nanochat-logs/*.log to extract step/loss/eta
- Queries nvidia-smi for GPU stats
- Serves a single-page web UI at http://localhost:8899/
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

LOG_DIR = Path.home() / "nanochat-logs"
PORT = 8899

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+\(([\d.]+)%\)\s+\|\s+loss:\s+([-\d.nNaN]+)"
    r".*?dt:\s+([\d.]+)ms.*?tok/sec:\s+([\d,]+)"
    r".*?total time:\s+([\d.]+)m.*?eta:\s+([\d.]+)m"
)


def parse_log(path):
    """Return latest training state + history of (step, loss) for last N points."""
    if not path.exists():
        return None
    history = deque(maxlen=600)
    last = None
    try:
        with path.open("r", errors="ignore") as f:
            # read tail efficiently for large log files
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 1024 * 1024), os.SEEK_SET)
            tail = f.read()
        for m in STEP_RE.finditer(tail):
            step = int(m.group(1))
            total = int(m.group(2))
            loss_s = m.group(4)
            loss = float("nan") if loss_s.lower() == "nan" else float(loss_s)
            dt = float(m.group(5))
            tok_per_sec = int(m.group(6).replace(",", ""))
            elapsed_min = float(m.group(7))
            eta_min = float(m.group(8))
            history.append((step, loss))
            last = {
                "step": step,
                "total": total,
                "pct": round(100.0 * step / total, 2) if total else 0,
                "loss": None if (loss != loss) else loss,
                "dt_ms": dt,
                "tok_per_sec": tok_per_sec,
                "elapsed_min": elapsed_min,
                "eta_min": eta_min,
            }
    except Exception as e:
        return {"error": str(e), "history": []}
    return {"last": last, "history": list(history)}


def gpu_stats():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=2).strip().split(",")
        return {
            "util_pct": int(out[0].strip()),
            "mem_used_mb": int(out[1].strip()),
            "mem_total_mb": int(out[2].strip()),
            "power_w": float(out[3].strip()),
            "temp_c": int(out[4].strip()),
        }
    except Exception as e:
        return {"error": str(e)}


def proc_running(pid_file):
    try:
        pid = int(Path(pid_file).read_text().strip())
        return Path(f"/proc/{pid}").exists()
    except Exception:
        return False


def status_json():
    runs = []
    for log in sorted(LOG_DIR.glob("*.log")):
        name = log.stem
        pid_file = LOG_DIR / f"{name}.pid"
        running = proc_running(pid_file) if pid_file.exists() else False
        parsed = parse_log(log)
        runs.append({
            "name": name,
            "log_path": str(log),
            "running": running,
            "size_kb": round(log.stat().st_size / 1024.0, 1),
            "mtime": log.stat().st_mtime,
            "parsed": parsed,
        })
    return {
        "runs": runs,
        "gpu": gpu_stats(),
        "now": time.time(),
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nanochat dashboard</title>
<style>
  :root {
    --bg: #0e1116; --card: #161a22; --fg: #e6e6e6; --muted: #8a93a3; --accent: #4ade80;
    --warn: #facc15; --danger: #f87171; --grid: #2a313c; --bar: #1f2937;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); margin: 0;
         font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 16px 20px; background: linear-gradient(180deg, #1a2030, #161a22);
           border-bottom: 1px solid var(--grid); }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  h2 { margin: 0 0 8px 0; font-size: 13px; font-weight: 600; color: var(--muted);
       text-transform: uppercase; letter-spacing: 0.5px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
          gap: 16px; padding: 20px; }
  .card { background: var(--card); border: 1px solid var(--grid);
          border-radius: 8px; padding: 16px; }
  .card.run { display: flex; flex-direction: column; gap: 12px; }
  .row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
          font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .pill.running { background: rgba(74, 222, 128, 0.15); color: var(--accent); }
  .pill.idle    { background: rgba(138, 147, 163, 0.15); color: var(--muted); }
  .progress { background: var(--bar); border-radius: 6px; height: 10px; overflow: hidden; }
  .progress > .bar { height: 100%; background: var(--accent); transition: width 0.5s ease; }
  .stat { display: flex; flex-direction: column; gap: 2px; }
  .stat .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat .l { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .stats4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .gpu-stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  canvas { background: var(--bar); border-radius: 6px; }
  .muted { color: var(--muted); font-size: 12px; }
  .nan { color: var(--danger); font-weight: 600; }
  footer { padding: 8px 20px; color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>nanochat dashboard <span id="now" class="muted"></span></h1>
</header>
<div id="root" class="grid"></div>
<footer>auto-refresh every 3s · log dir: ~/nanochat-logs</footer>

<script>
function fmt(x, digits) { if (x === null || x === undefined) return '—';
  if (typeof x === 'number' && !isFinite(x)) return '—';
  if (typeof x === 'number') return x.toFixed(digits ?? 0);
  return String(x); }
function fmtTime(min) { if (!min || !isFinite(min)) return '—';
  if (min < 60) return min.toFixed(1) + ' min';
  const h = Math.floor(min/60), m = Math.round(min%60);
  return h+'h '+m+'m'; }

let charts = {};
function ensureCanvas(id) {
  let c = document.getElementById('chart-'+id);
  if (c) return c;
  return null;
}

function drawLoss(canvas, history) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  if (!history || !history.length) return;
  // Filter NaN losses for plotting
  const pts = history.filter(p => p[1] === p[1] && p[1] !== null && isFinite(p[1]));
  if (!pts.length) return;
  const xs = pts.map(p=>p[0]);
  const ys = pts.map(p=>p[1]);
  const xmin = xs[0], xmax = xs[xs.length-1] || xmin+1;
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xScale = x => (x - xmin) / (xmax - xmin || 1) * (w - 30) + 25;
  const yScale = y => h - 20 - (y - ymin) / (ymax - ymin || 1) * (h - 30);
  // grid
  ctx.strokeStyle = '#2a313c'; ctx.lineWidth = 1;
  for (let i=0; i<=4; i++) {
    const y = 5 + i*((h-25)/4);
    ctx.beginPath(); ctx.moveTo(25, y); ctx.lineTo(w-5, y); ctx.stroke();
  }
  // axis labels
  ctx.fillStyle = '#8a93a3'; ctx.font = '10px sans-serif';
  ctx.fillText(ymax.toFixed(2), 2, 12);
  ctx.fillText(ymin.toFixed(2), 2, h-22);
  ctx.fillText('step '+xmin, 25, h-5);
  ctx.fillText('step '+xmax, w-60, h-5);
  // line
  ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = xScale(p[0]), y = yScale(p[1]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function render(data) {
  const root = document.getElementById('root');
  const now = new Date(data.now * 1000);
  document.getElementById('now').textContent = '· ' + now.toLocaleTimeString();

  // GPU card
  const g = data.gpu || {};
  const gpuCard = `
    <section class="card">
      <h2>GPU · CUDA training workstation</h2>
      <div class="gpu-stats">
        <div class="stat"><div class="v">${fmt(g.util_pct)}%</div><div class="l">util</div></div>
        <div class="stat"><div class="v">${fmt((g.mem_used_mb||0)/1024, 1)} GB</div><div class="l">vram used</div></div>
        <div class="stat"><div class="v">${fmt((g.mem_total_mb||0)/1024, 1)} GB</div><div class="l">vram total</div></div>
        <div class="stat"><div class="v">${fmt(g.power_w, 0)} W</div><div class="l">power</div></div>
        <div class="stat"><div class="v">${fmt(g.temp_c, 0)}°C</div><div class="l">temp</div></div>
      </div>
    </section>
  `;

  // Run cards
  const runCards = data.runs.map(r => {
    const last = r.parsed && r.parsed.last;
    const pct = last ? last.pct : 0;
    const lossStr = last && last.loss !== null ? last.loss.toFixed(4) :
                    (last ? '<span class="nan">nan</span>' : '—');
    const status = r.running ? `<span class="pill running">running</span>` :
                               `<span class="pill idle">idle</span>`;
    return `
      <section class="card run">
        <div class="row">
          <h2 style="margin:0">${r.name}</h2>
          ${status}
        </div>
        ${ last ? `
          <div class="progress"><div class="bar" style="width:${pct}%"></div></div>
          <div class="row muted"><span>step ${last.step.toLocaleString()} / ${last.total.toLocaleString()} (${pct}%)</span><span>ETA ${fmtTime(last.eta_min)}</span></div>
          <div class="stats4">
            <div class="stat"><div class="v">${lossStr}</div><div class="l">loss</div></div>
            <div class="stat"><div class="v">${(last.tok_per_sec/1000).toFixed(0)}K</div><div class="l">tok/s</div></div>
            <div class="stat"><div class="v">${last.dt_ms.toFixed(0)} ms</div><div class="l">step time</div></div>
            <div class="stat"><div class="v">${fmtTime(last.elapsed_min)}</div><div class="l">elapsed</div></div>
          </div>
          <canvas id="chart-${r.name}" width="400" height="120"></canvas>
        ` : `<div class="muted">no parsed steps yet (size ${r.size_kb} KB)</div>` }
      </section>
    `;
  }).join('');

  root.innerHTML = gpuCard + runCards;

  data.runs.forEach(r => {
    if (r.parsed && r.parsed.history) {
      const c = document.getElementById('chart-'+r.name);
      if (c) drawLoss(c, r.parsed.history);
    }
  });
}

async function tick() {
  try {
    const res = await fetch('/api/status', { cache: 'no-store' });
    const data = await res.json();
    render(data);
  } catch (e) {
    console.error(e);
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **kw):
        pass  # quiet
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/status":
            data = status_json()
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def daemonize(log_path):
    """Standard double-fork to fully detach from controlling terminal."""
    if os.fork() > 0: os._exit(0)        # exit first parent
    os.setsid()                           # new session
    if os.fork() > 0: os._exit(0)        # exit second parent
    os.umask(0)
    # redirect stdio
    sys.stdout.flush(); sys.stderr.flush()
    with open("/dev/null", "rb") as f: os.dup2(f.fileno(), 0)
    with open(log_path, "ab", buffering=0) as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)


def main():
    if "--daemon" in sys.argv:
        daemonize(os.path.expanduser("~/nc_dashboard.log"))
    server = socketserver.ThreadingTCPServer(("0.0.0.0", PORT), Handler)
    server.allow_reuse_address = True
    sys.stderr.write(f"[dashboard] http://localhost:{PORT}/\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
