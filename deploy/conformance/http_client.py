"""Standard-library HTTP and WebSocket client for the conformance checks (no third-party packages).

The same file ships in WhissleAI/arc_transcriber and WhissleAI/arc-tagger-c (deploy/conformance/).
"""
import base64
import http.client
import json
import os
import socket
import struct
import urllib.parse
import uuid


class Resp:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body

    def json(self):
        return json.loads(self.body)

    @property
    def text(self):
        return self.body.decode("utf-8", "replace")


def request(base, method, path, body=None, headers=None, timeout=600):
    u = urllib.parse.urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    out = Resp(r.status, {k.lower(): v for k, v in r.getheaders()}, r.read())
    c.close()
    return out


def get(base, path, **kw):
    return request(base, "GET", path, **kw)


def post_json(base, path, obj, headers=None, **kw):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    return request(base, "POST", path, json.dumps(obj).encode(), h, **kw)


def multipart(fields, files):
    """fields: {name: str or [str, ...]}; files: {name: (filename, bytes, content_type)} -> (body, content_type)"""
    b = "----arc" + uuid.uuid4().hex
    out = []
    for k, vs in fields.items():
        for v in vs if isinstance(vs, list) else [vs]:
            out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (fn, data, ct) in files.items():
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\nContent-Type: {ct}\r\n\r\n'.encode())
        out.append(data + b"\r\n")
    out.append(f"--{b}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={b}"


def post_multipart(base, path, fields, files, **kw):
    body, ct = multipart(fields, files)
    return request(base, "POST", path, body, {"Content-Type": ct}, **kw)


class WS:
    """A minimal RFC 6455 client: text / binary messages, masking, close."""

    def __init__(self, base, path, timeout=120):
        u = urllib.parse.urlparse(base)
        self.s = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.s.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = self.s.recv(4096)
            if not d:
                raise ConnectionError("closed during handshake")
            buf += d
        head, self.rest = buf.split(b"\r\n\r\n", 1)
        self.status = int(head.split(b" ")[1])
        self.handshake = head.decode("latin-1")
        if self.status != 101:
            raise ConnectionError(f"upgrade refused: {head[:200]!r}")

    def _recv(self, n):
        while len(self.rest) < n:
            d = self.s.recv(65536)
            if not d:
                raise ConnectionError("closed")
            self.rest += d
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def send(self, data, binary=False, op=None):
        if isinstance(data, str):
            data = data.encode()
        op = op or (2 if binary else 1)
        n = len(data)
        h = bytes([0x80 | op])
        if n < 126:
            h += bytes([0x80 | n])
        elif n < 65536:
            h += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            h += bytes([0x80 | 127]) + struct.pack(">Q", n)
        m = os.urandom(4)
        self.s.sendall(h + m + bytes(b ^ m[i & 3] for i, b in enumerate(data)))

    def send_json(self, o):
        self.send(json.dumps(o))

    def recv(self):
        """-> (opcode, bytes); opcode 8 = close"""
        while True:
            h = self._recv(2)
            op, n = h[0] & 0x0F, h[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv(8))[0]
            data = self._recv(n) if n else b""
            if op == 9:
                continue
            return op, data

    def recv_json(self):
        op, d = self.recv()
        if op == 8:
            return None
        return json.loads(d)

    def close(self):
        try:
            self.send(struct.pack(">H", 1000), op=8)
        except OSError:
            pass
        self.s.close()
