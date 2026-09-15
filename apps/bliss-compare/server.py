#!/usr/bin/env python3
"""Private side-by-side chat for the two preserved Bliss native models."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(os.environ.get('BLISS_RUN_ROOT', str(Path.home() / 'bliss-runs/20260915-next')))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench/coherence'))
from evaluate import Backend, visible, EOT

MODELS = {
    'previous': (ROOT/'artifacts/nc_run_native', ROOT/'artifacts/MODEL.NCB', ROOT/'artifacts/TOKENIZER.NCT'),
    'new': (ROOT/'native/build/slm/variants/128699c1e02a1f78/slm_run', ROOT/'native/exports/lfm350-trained-q6/MODEL.SLM', ROOT/'native/exports/lfm350-trained-q6/TOKENIZER.SLT'),
}
SESSIONS = {}
REGISTRY_LOCK = threading.Lock()

class StreamingBackend(Backend):
    callback = None

    def read_until(self, target, started):
        raw = b''
        prior = ''
        while True:
            if time.perf_counter() - started > self.timeout:
                raise TimeoutError('The model took too long. Start a new chat and try a shorter message.')
            for key, _ in self.sel.select(.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    self.sel.unregister(key.fileobj)
                    continue
                if key.data == 'err':
                    self.stderr_file.write(chunk)
                    self.stderr_file.flush()
                    continue
                raw += chunk
                answer = visible(raw)
                if answer != prior and self.callback:
                    self.callback({'type': 'text', 'text': answer})
                prior = answer
            done = b'\x01READY\n' in raw if target == 'ready' else EOT.search(raw)
            if done:
                return {'answer': visible(raw), 'elapsed_s': time.perf_counter() - started}
            if not self.sel.get_map():
                raise RuntimeError('The model stopped. Start a new chat to reconnect.')

class ModelChat:
    def __init__(self, key):
        self.key = key
        self.lock = threading.Lock()
        self.backend = None
        self.temp = None
        self.error_file = None
        self.used = time.monotonic()

    def close(self):
        if self.backend:
            self.backend.close()
            self.backend = None
        if self.error_file:
            self.error_file.close()
            self.error_file = None
        if self.temp:
            self.temp.cleanup()
            self.temp = None

    def answer(self, prompt, emit):
        if not self.lock.acquire(blocking=False):
            emit({'type': 'error', 'message': 'This model is already answering.'})
            return
        try:
            self.used = time.monotonic()
            if self.backend is None:
                emit({'type': 'status', 'text': 'Loading model…'})
                self.temp = tempfile.TemporaryDirectory(prefix='bliss-browser-')
                notes = Path(self.temp.name)/'notes.txt'
                notes.write_text('')
                self.error_file = (Path(self.temp.name)/'stderr.log').open('wb')
                self.backend = StreamingBackend([str(p) for p in MODELS[self.key]] +
                    ['-c', '512', '-t', '0', '-s', '42', '-m', str(notes)], self.error_file, 240)
                self.backend.turn('/maxtok 128')
            emit({'type': 'status', 'text': 'Thinking…'})
            self.backend.callback = emit
            result = self.backend.turn(prompt)
            self.backend.callback = None
            emit({'type': 'done', 'text': result['answer'], 'seconds': result['elapsed_s']})
        except (Exception,) as error:
            self.close()
            try:
                emit({'type': 'error', 'message': str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.used = time.monotonic()
            self.lock.release()

def cleanup():
    while True:
        time.sleep(60)
        with REGISTRY_LOCK:
            for sid, models in list(SESSIONS.items()):
                if all(time.monotonic() - model.used > 1800 for model in models.values()):
                    locks = []
                    for model in models.values():
                        if model.lock.acquire(blocking=False):
                            locks.append(model)
                        else:
                            break
                    if len(locks) == len(models):
                        for model in models.values():
                            model.close()
                        del SESSIONS[sid]
                    for model in locks:
                        model.lock.release()

class Handler(BaseHTTPRequestHandler):
    def json_response(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get('Host', '').split(':')[0] not in ('127.0.0.1', 'localhost'):
            return self.json_response(403, {'error': 'Use the local chat address.'})
        if self.path == '/api/health':
            return self.json_response(200, {'ready': True, 'models': list(MODELS)})
        if self.path not in ('/', '/index.html'):
            return self.json_response(404, {'error': 'Not found'})
        body = Path(__file__).with_name('index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.headers.get('Host', '').split(':')[0] not in ('127.0.0.1', 'localhost'):
            return self.json_response(403, {'error': 'Use the local chat address.'})
        if self.headers.get('Origin') and self.headers['Origin'] != 'http://' + self.headers.get('Host', ''):
            return self.json_response(403, {'error': 'Origin mismatch'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('Message is too large.')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('Expected a message object.')
            sid = data.get('session', '')
            if not re.fullmatch(r'[a-f0-9-]{36}', sid):
                raise ValueError('Invalid chat session. Reload the page.')
            if self.path == '/api/reset':
                with REGISTRY_LOCK:
                    models = SESSIONS.get(sid, {})
                    locked = []
                    for model in models.values():
                        if model.lock.acquire(blocking=False):
                            locked.append(model)
                        else:
                            for held in locked:
                                held.lock.release()
                            return self.json_response(409, {'error': 'Wait for both models to finish.'})
                    for model in models.values():
                        model.close()
                    SESSIONS.pop(sid, None)
                    for held in locked:
                        held.lock.release()
                return self.json_response(200, {'reset': True})
            if self.path != '/api/chat':
                return self.json_response(404, {'error': 'Not found'})
            key = data.get('model')
            message = data.get('message')
            if key not in MODELS or not isinstance(message, str) or not 1 <= len(message.strip()) <= 2000:
                raise ValueError('Enter a message of up to 2,000 characters.')
            prompt = ' '.join(message.split())
            if any(ord(c) < 32 for c in prompt):
                raise ValueError('The message contains an unsupported control character.')
            if prompt.startswith('/'):
                prompt = 'Respond to this user message: ' + prompt
            with REGISTRY_LOCK:
                if sid not in SESSIONS:
                    if len(SESSIONS) >= 8:
                        return self.json_response(503, {'error': 'Too many open chats. Close an older chat using New chat.'})
                    SESSIONS[sid] = {name: ModelChat(name) for name in MODELS}
                model = SESSIONS[sid][key]
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self.json_response(400, {'error': str(error)})
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        def emit(event):
            self.wfile.write((json.dumps(event, ensure_ascii=False) + '\n').encode())
            self.wfile.flush()
        model.answer(prompt, emit)

if __name__ == '__main__':
    threading.Thread(target=cleanup, daemon=True).start()
    print('Bliss Compare listening on 127.0.0.1:8791', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8791), Handler).serve_forever()
