"""Loopback-only external-provider fixture; never forwards network requests."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

COUNTS = {"primary": 0, "fallback": 0}


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps(COUNTS if self.path == '/counts' else {'error': 'catalog unavailable'}).encode()
        self.send_response(200 if self.path == '/counts' else 503)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        primary = self.server.server_port == 9101
        COUNTS['primary' if primary else 'fallback'] += 1
        self.send_response(429 if primary else 200)
        self.send_header('Content-Type', 'application/json' if primary else 'text/event-stream')
        self.end_headers()
        if primary:
            self.wfile.write(b'{"error":{"message":"synthetic rate limit"}}')
            return
        chunk = {'id': 'synthetic-runtime', 'object': 'chat.completion.chunk', 'created': 1,
                 'model': body['model'], 'choices': [{'index': 0, 'delta': {'role': 'assistant',
                 'content': 'Synthetic fallback response'}, 'finish_reason': 'stop'}]}
        self.wfile.write(('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())


for port in (9101, 9102):
    server = ThreadingHTTPServer(('0.0.0.0', port), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
threading.Event().wait()
