/* http.c: see http.h. Part of the Arc deploy service (Whissle Inc.); licensed as the repository it is in. */
#define _GNU_SOURCE   /* strcasestr, M_PI, strdup, strtok_r, clock_gettime on glibc with -std=c11 */
#include "http.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define HDR_MAX (64 * 1024)

static atomic_int g_conns;
static atomic_int g_stop;
static int g_listen_fd = -1;

double http_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

const char* http_reason(int s) {
  switch (s) {
    case 100: return "Continue";
    case 101: return "Switching Protocols";
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 422: return "Unprocessable Entity";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Unknown";
  }
}

static int send_all(int fd, const void* buf, size_t n) {
  const char* p = buf;
  while (n) {
#ifdef MSG_NOSIGNAL
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
#else
    ssize_t k = send(fd, p, n, 0);
#endif
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

/* ------------------------------------------------------------------ connection buffer */
typedef struct {
  int fd;
  char* buf;
  size_t len, cap;
} conn_t;

static int conn_fill(conn_t* c, size_t want_cap) {
  if (c->cap < want_cap) {
    char* nb = realloc(c->buf, want_cap);
    if (!nb) return -1;
    c->buf = nb;
    c->cap = want_cap;
  }
  if (c->len >= c->cap) return -1;
  ssize_t k;
  do k = recv(c->fd, c->buf + c->len, c->cap - c->len, 0);
  while (k < 0 && errno == EINTR);
  if (k <= 0) return -1;
  c->len += (size_t)k;
  return 0;
}

static void conn_consume(conn_t* c, size_t n) {
  memmove(c->buf, c->buf + n, c->len - n);
  c->len -= n;
}

/* ------------------------------------------------------------------ helpers */
static int hexval(int ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  return -1;
}

static void url_decode(const char* s, size_t n, char* out, size_t out_len, int plus_space) {
  size_t o = 0;
  for (size_t i = 0; i < n && o + 1 < out_len; ++i) {
    if (s[i] == '%' && i + 2 < n && hexval(s[i + 1]) >= 0 && hexval(s[i + 2]) >= 0) {
      out[o++] = (char)(hexval(s[i + 1]) * 16 + hexval(s[i + 2]));
      i += 2;
    } else if (plus_space && s[i] == '+') out[o++] = ' ';
    else out[o++] = s[i];
  }
  out[o] = 0;
}

const char* http_header(const http_req* r, const char* name) {
  for (int i = 0; i < r->n_headers; ++i)
    if (!strcasecmp(r->headers[i].name, name)) return r->headers[i].value;
  return NULL;
}

int http_query(const http_req* r, const char* key, char* out, size_t out_len) {
  size_t kl = strlen(key);
  const char* p = r->query;
  while (*p) {
    const char* amp = strchr(p, '&');
    size_t seg = amp ? (size_t)(amp - p) : strlen(p);
    const char* eq = memchr(p, '=', seg);
    size_t nl = eq ? (size_t)(eq - p) : seg;
    char nb[256];
    url_decode(p, nl, nb, sizeof nb, 1);
    if (strlen(nb) == kl && !memcmp(nb, key, kl)) {
      if (eq) url_decode(eq + 1, seg - nl - 1, out, out_len, 1);
      else if (out_len) out[0] = 0;
      return 1;
    }
    p += seg;
    if (*p == '&') ++p;
  }
  return 0;
}

int http_ctype_is(const http_req* r, const char* prefix) {
  const char* ct = http_header(r, "Content-Type");
  if (!ct) return 0;
  while (*ct == ' ') ++ct;
  return !strncasecmp(ct, prefix, strlen(prefix));
}

void http_respond(http_req* r, int status, const char* ctype, const char* body, size_t len) {
  char hdr[512];
  int n = snprintf(hdr, sizeof hdr,
                   "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: %s\r\n"
                   "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
                   status, http_reason(status), ctype ? ctype : "application/octet-stream", len,
                   r->keep_alive ? "keep-alive" : "close");
  r->status = status;
  if (send_all(r->fd, hdr, (size_t)n) == 0 && len && strcmp(r->method, "HEAD")) send_all(r->fd, body, len);
}

void http_respond_json(http_req* r, int status, const char* json) {
  http_respond(r, status, "application/json", json, strlen(json));
}

static size_t json_escape(const char* s, char* out, size_t cap) {
  size_t o = 0;
  for (; *s && o + 7 < cap; ++s) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = (char)c; }
    else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
    else if (c < 0x20) o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c);
    else out[o++] = (char)c;
  }
  out[o] = 0;
  return o;
}

void http_respond_error(http_req* r, int status, const char* type, const char* fmt, ...) {
  char msg[1024], esc[2048], body[2400];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  json_escape(msg, esc, sizeof esc);
  snprintf(body, sizeof body, "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"code\":%d}}", esc,
           type ? type : "invalid_request_error", status);
  http_respond_json(r, status, body);
}

/* ------------------------------------------------------------------ base64 */
static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char* b64_encode(const unsigned char* d, size_t n) {
  char* o = malloc((n + 2) / 3 * 4 + 1);
  if (!o) return NULL;
  size_t j = 0;
  for (size_t i = 0; i < n; i += 3) {
    uint32_t v = (uint32_t)d[i] << 16 | (i + 1 < n ? (uint32_t)d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
    o[j++] = B64[v >> 18 & 63];
    o[j++] = B64[v >> 12 & 63];
    o[j++] = i + 1 < n ? B64[v >> 6 & 63] : '=';
    o[j++] = i + 2 < n ? B64[v & 63] : '=';
  }
  o[j] = 0;
  return o;
}

unsigned char* b64_decode(const char* s, size_t n, size_t* out_len) {
  unsigned char* o = malloc(n / 4 * 3 + 4);
  if (!o) return NULL;
  uint32_t acc = 0;
  int bits = 0;
  size_t j = 0;
  for (size_t i = 0; i < n; ++i) {
    int c = (unsigned char)s[i], v;
    if (c >= 'A' && c <= 'Z') v = c - 'A';
    else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
    else if (c >= '0' && c <= '9') v = c - '0' + 52;
    else if (c == '+' || c == '-') v = 62;
    else if (c == '/' || c == '_') v = 63;
    else if (c == '=') break;
    else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    else { free(o); return NULL; }
    acc = acc << 6 | (uint32_t)v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      o[j++] = (unsigned char)(acc >> bits & 0xff);
    }
  }
  *out_len = j;
  return o;
}

/* ------------------------------------------------------------------ SHA-1 (for the WebSocket handshake only) */
typedef struct {
  uint32_t h[5];
  uint64_t n;
  unsigned char b[64];
  size_t bl;
} sha1_t;

#define ROL(x, k) (((x) << (k)) | ((x) >> (32 - (k))))
static void sha1_block(sha1_t* s, const unsigned char* p) {
  uint32_t w[80];
  for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 80; ++i) w[i] = ROL(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
  uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4];
  for (int i = 0; i < 80; ++i) {
    uint32_t f, k;
    if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
    else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
    else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
    else { f = b ^ c ^ d; k = 0xCA62C1D6; }
    uint32_t t = ROL(a, 5) + f + e + k + w[i];
    e = d; d = c; c = ROL(b, 30); b = a; a = t;
  }
  s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e;
}

static void sha1(const void* data, size_t n, unsigned char out[20]) {
  sha1_t s = {{0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0}, 0, {0}, 0};
  const unsigned char* p = data;
  s.n = (uint64_t)n * 8;
  while (n >= 64) { sha1_block(&s, p); p += 64; n -= 64; }
  unsigned char t[128] = {0};
  memcpy(t, p, n);
  t[n] = 0x80;
  size_t tl = n + 1 + 8 <= 64 ? 64 : 128;
  for (int i = 0; i < 8; ++i) t[tl - 1 - i] = (unsigned char)(s.n >> (8 * i));
  sha1_block(&s, t);
  if (tl == 128) sha1_block(&s, t + 64);
  for (int i = 0; i < 5; ++i) {
    out[4 * i] = (unsigned char)(s.h[i] >> 24);
    out[4 * i + 1] = (unsigned char)(s.h[i] >> 16);
    out[4 * i + 2] = (unsigned char)(s.h[i] >> 8);
    out[4 * i + 3] = (unsigned char)s.h[i];
  }
}

/* ------------------------------------------------------------------ WebSocket */
static int header_has_token(const char* v, const char* tok) {
  if (!v) return 0;
  size_t tl = strlen(tok);
  for (const char* p = v; *p;) {
    while (*p == ' ' || *p == ',') ++p;
    const char* e = p;
    while (*e && *e != ',') ++e;
    const char* t = e;
    while (t > p && t[-1] == ' ') --t;
    if ((size_t)(t - p) == tl && !strncasecmp(p, tok, tl)) return 1;
    p = e;
  }
  return 0;
}

int http_is_ws_upgrade(const http_req* r) {
  return !strcmp(r->method, "GET") && header_has_token(http_header(r, "Upgrade"), "websocket") &&
         header_has_token(http_header(r, "Connection"), "upgrade") && http_header(r, "Sec-WebSocket-Key");
}

int ws_accept(http_req* r) {
  const char* key = http_header(r, "Sec-WebSocket-Key");
  if (!key) return -1;
  char cat[256];
  snprintf(cat, sizeof cat, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
  unsigned char dg[20];
  sha1(cat, strlen(cat), dg);
  char* acc = b64_encode(dg, 20);
  char hdr[512];
  int n = snprintf(hdr, sizeof hdr,
                   "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   "Sec-WebSocket-Accept: %s\r\n\r\n", acc);
  free(acc);
  r->upgraded = 1;
  r->keep_alive = 0;
  r->status = 101;
  struct timeval tv = {600, 0};   /* a realtime session may idle up to 10 min */
  setsockopt(r->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  return send_all(r->fd, hdr, (size_t)n);
}

static int recv_exact(int fd, void* buf, size_t n) {
  char* p = buf;
  while (n) {
    ssize_t k = recv(fd, p, n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

int ws_send(int fd, int op, const void* data, size_t len) {
  unsigned char h[10];
  size_t hl;
  h[0] = (unsigned char)(0x80 | op);
  if (len < 126) { h[1] = (unsigned char)len; hl = 2; }
  else if (len < 65536) { h[1] = 126; h[2] = (unsigned char)(len >> 8); h[3] = (unsigned char)len; hl = 4; }
  else {
    h[1] = 127;
    for (int i = 0; i < 8; ++i) h[2 + i] = (unsigned char)((uint64_t)len >> (56 - 8 * i));
    hl = 10;
  }
  if (send_all(fd, h, hl)) return -1;
  return len ? send_all(fd, data, len) : 0;
}

int ws_send_text(int fd, const char* s) { return ws_send(fd, WS_TEXT, s, strlen(s)); }

void ws_close(int fd, int code, const char* reason) {
  unsigned char b[128];
  size_t rl = reason ? strlen(reason) : 0;
  if (rl > 120) rl = 120;
  b[0] = (unsigned char)(code >> 8);
  b[1] = (unsigned char)code;
  if (rl) memcpy(b + 2, reason, rl);
  ws_send(fd, WS_CLOSE, b, 2 + rl);
}

int ws_recv(int fd, unsigned char** data, size_t* len, size_t max_len) {
  unsigned char* msg = NULL;
  size_t ml = 0;
  int msg_op = 0;
  *data = NULL;
  *len = 0;
  for (;;) {
    unsigned char h[2];
    if (recv_exact(fd, h, 2)) goto fail;
    int fin = h[0] & 0x80, op = h[0] & 0x0f, masked = h[1] & 0x80;
    uint64_t pl = h[1] & 0x7f;
    if (pl == 126) {
      unsigned char e[2];
      if (recv_exact(fd, e, 2)) goto fail;
      pl = (uint64_t)e[0] << 8 | e[1];
    } else if (pl == 127) {
      unsigned char e[8];
      if (recv_exact(fd, e, 8)) goto fail;
      pl = 0;
      for (int i = 0; i < 8; ++i) pl = pl << 8 | e[i];
    }
    unsigned char mk[4] = {0};
    if (masked && recv_exact(fd, mk, 4)) goto fail;
    if (pl > max_len || ml + pl > max_len) goto fail;
    unsigned char* p = malloc((size_t)pl + 1);
    if (!p) goto fail;
    if (pl && recv_exact(fd, p, (size_t)pl)) { free(p); goto fail; }
    if (masked) for (uint64_t i = 0; i < pl; ++i) p[i] ^= mk[i & 3];
    p[pl] = 0;
    if (op == WS_PING) { ws_send(fd, WS_PONG, p, (size_t)pl); free(p); continue; }
    if (op == WS_PONG) { free(p); continue; }
    if (op == WS_CLOSE) {
      ws_send(fd, WS_CLOSE, p, pl >= 2 ? 2 : 0);
      free(p);
      free(msg);
      return WS_CLOSE;
    }
    if (op == WS_TEXT || op == WS_BINARY) {
      if (msg) { free(p); goto fail; }   /* a new message inside a fragmented one */
      msg_op = op;
      msg = p;
      ml = (size_t)pl;
    } else if (op == 0) {
      if (!msg) { free(p); goto fail; }
      unsigned char* nm = realloc(msg, ml + (size_t)pl + 1);
      if (!nm) { free(p); goto fail; }
      msg = nm;
      memcpy(msg + ml, p, (size_t)pl);
      ml += (size_t)pl;
      msg[ml] = 0;
      free(p);
    } else { free(p); goto fail; }
    if (fin) {
      *data = msg;
      *len = ml;
      return msg_op;
    }
  }
fail:
  free(msg);
  return -1;
}

/* ------------------------------------------------------------------ multipart */
static const char* memfind(const char* h, size_t hl, const char* n, size_t nl) {
  if (nl == 0 || hl < nl) return NULL;
  for (size_t i = 0; i + nl <= hl; ++i)
    if (h[i] == n[0] && !memcmp(h + i, n, nl)) return h + i;
  return NULL;
}

/* finds parameter key (key="v" or key=v) in a header value without modifying it: *vs / *vl = the value */
static int find_param(const char* v, const char* key, const char** vs, size_t* vl) {
  size_t kl = strlen(key);
  for (const char* p = v; (p = strcasestr(p, key)) != NULL; p += kl) {
    if (p > v && p[-1] != ';' && p[-1] != ' ' && p[-1] != '\t') continue;
    const char* q = p + kl;
    while (*q == ' ') ++q;
    if (*q != '=') continue;
    ++q;
    while (*q == ' ') ++q;
    if (*q == '"') {
      const char* e = strchr(q + 1, '"');
      if (!e) return 0;
      *vs = q + 1;
      *vl = (size_t)(e - q - 1);
      return 1;
    }
    const char* e = q;
    while (*e && *e != ';' && *e != ' ' && *e != '\r') ++e;
    *vs = q;
    *vl = (size_t)(e - q);
    return 1;
  }
  return 0;
}

int http_multipart(http_req* r, http_part* parts, int max) {
  const char* ct = http_header(r, "Content-Type");
  if (!ct || strncasecmp(ct, "multipart/form-data", 19)) return -1;
  const char* bs;
  size_t bl;
  if (!find_param(ct, "boundary", &bs, &bl) || bl == 0 || bl > 200) return -1;
  char delim[256];
  int dl = snprintf(delim, sizeof delim, "--%.*s", (int)bl, bs);
  char* b = r->body;
  size_t n = r->body_len;
  const char* p = memfind(b, n, delim, (size_t)dl);
  if (!p) return -1;
  int np = 0;
  for (;;) {
    p += dl;
    if ((size_t)(p - b) + 2 <= n && p[0] == '-' && p[1] == '-') break;   /* closing delimiter */
    if ((size_t)(p - b) + 2 > n || p[0] != '\r' || p[1] != '\n') return -1;
    p += 2;
    const char* he = memfind(p, n - (size_t)(p - b), "\r\n\r\n", 4);
    if (!he) return -1;
    char* hdrs = (char*)p;
    hdrs[he - p] = 0;   /* the CR of the blank line */
    const char* data = he + 4;
    char dl2[260];
    int dll = snprintf(dl2, sizeof dl2, "\r\n%s", delim);
    const char* nx = memfind(data, n - (size_t)(data - b), dl2, (size_t)dll);
    if (!nx) return -1;
    if (np < max) {
      http_part* pt = &parts[np];
      memset(pt, 0, sizeof *pt);
      pt->data = data;
      pt->len = (size_t)(nx - data);
      for (char* line = hdrs; line && *line;) {
        char* eol = strstr(line, "\r\n");
        if (eol) *eol = 0;
        if (!strncasecmp(line, "Content-Disposition:", 20)) {
          const char *ns = NULL, *fs = NULL;
          size_t nl = 0, fl = 0;
          int hn = find_param(line + 20, "name", &ns, &nl), hf = find_param(line + 20, "filename", &fs, &fl);
          if (hn) { ((char*)ns)[nl] = 0; pt->name = ns; }      /* terminate after both are found */
          if (hf) { ((char*)fs)[fl] = 0; pt->filename = fs; }
        } else if (!strncasecmp(line, "Content-Type:", 13)) {
          char* v = line + 13;
          while (*v == ' ') ++v;
          pt->content_type = v;
        }
        line = eol ? eol + 2 : NULL;
      }
      ++np;
    }
    p = nx + 2;
  }
  return np;
}

const http_part* http_part_get(const http_part* parts, int n, const char* name) {
  for (int i = 0; i < n; ++i)
    if (parts[i].name && !strcmp(parts[i].name, name)) return &parts[i];
  return NULL;
}

char* http_part_str(const http_part* parts, int n, const char* name) {
  const http_part* p = http_part_get(parts, n, name);
  if (!p) return NULL;
  char* s = malloc(p->len + 1);
  memcpy(s, p->data, p->len);
  s[p->len] = 0;
  return s;
}

/* ------------------------------------------------------------------ request reading */
/* 0: a request in r (body malloc'd); -1: connection closed / error (with status sent when possible) */
static int read_request(conn_t* c, http_req* r, const http_server_config* cfg) {
  size_t hend = 0;
  for (;;) {
    char* e = c->len >= 4 ? (char*)memfind(c->buf, c->len, "\r\n\r\n", 4) : NULL;
    if (e) { hend = (size_t)(e - c->buf) + 4; break; }
    if (c->len >= HDR_MAX) {
      memset(r, 0, sizeof *r);
      r->fd = c->fd;
      strcpy(r->method, "GET");
      http_respond_error(r, 431, "invalid_request_error", "request headers too large");
      return -1;
    }
    if (conn_fill(c, HDR_MAX + 1)) return -1;
  }
  memset(r, 0, sizeof *r);
  r->fd = c->fd;
  r->t_start_ms = http_now_ms();
  char* h = malloc(hend + 1);
  memcpy(h, c->buf, hend);
  h[hend] = 0;
  conn_consume(c, hend);
  /* request line */
  char* eol = strstr(h, "\r\n");
  *eol = 0;
  char target[6144] = {0}, ver[16] = {0};
  if (sscanf(h, "%15s %6143s %15s", r->method, target, ver) != 3) {
    free(h);
    strcpy(r->method, "GET");
    http_respond_error(r, 400, "invalid_request_error", "malformed request line");
    return -1;
  }
  char* q = strchr(target, '?');
  if (q) {
    *q = 0;
    snprintf(r->query, sizeof r->query, "%s", q + 1);
  }
  url_decode(target, strlen(target), r->path, sizeof r->path, 0);
  r->keep_alive = !strcmp(ver, "HTTP/1.1");
  /* headers: the strings live in the malloc'd block h, freed by the caller through headers[0].name's block */
  char* line = eol + 2;
  while (*line && strncmp(line, "\r\n", 2)) {
    char* le = strstr(line, "\r\n");
    if (!le) break;
    *le = 0;
    char* colon = strchr(line, ':');
    if (colon && r->n_headers < HTTP_MAX_HEADERS) {
      *colon = 0;
      char* v = colon + 1;
      while (*v == ' ' || *v == '\t') ++v;
      char* ve = v + strlen(v);
      while (ve > v && (ve[-1] == ' ' || ve[-1] == '\t')) *--ve = 0;
      r->headers[r->n_headers].name = line;
      r->headers[r->n_headers].value = v;
      r->n_headers++;
    }
    line = le + 2;
  }
  r->hdr_block = h;   /* the header strings point into it; freed in free_request */
  const char* conn = http_header(r, "Connection");
  if (conn && header_has_token(conn, "close")) r->keep_alive = 0;
  if (conn && header_has_token(conn, "keep-alive")) r->keep_alive = 1;
  size_t maxb = cfg->max_body ? cfg->max_body : (size_t)100 << 20;
  const char* te = http_header(r, "Transfer-Encoding");
  const char* cl = http_header(r, "Content-Length");
  const char* ex = http_header(r, "Expect");
  int chunked = te && header_has_token(te, "chunked");
  long long clen = cl ? atoll(cl) : 0;
  if (!chunked && clen > 0 && (size_t)clen > maxb) {
    r->keep_alive = 0;
    http_respond_error(r, 413, "invalid_request_error", "body larger than %zu bytes", maxb);
    return -1;
  }
  if (ex && !strcasecmp(ex, "100-continue") && (chunked || clen > 0)) send_all(c->fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
  if (chunked) {
    size_t cap = 1 << 16, bl = 0;
    char* body = malloc(cap + 1);
    for (;;) {
      char* le;
      while (!(le = c->len ? (char*)memfind(c->buf, c->len, "\r\n", 2) : NULL)) {
        if (c->len > 1024 || conn_fill(c, c->cap > (1 << 16) ? c->cap : (1 << 16))) { free(body); return -1; }
      }
      size_t sz = (size_t)strtoull(c->buf, NULL, 16);
      conn_consume(c, (size_t)(le - c->buf) + 2);
      if (sz == 0) {   /* trailers until the blank line */
        for (;;) {
          while (!(le = c->len ? (char*)memfind(c->buf, c->len, "\r\n", 2) : NULL))
            if (conn_fill(c, c->cap > (1 << 16) ? c->cap : (1 << 16))) { free(body); return -1; }
          size_t l = (size_t)(le - c->buf);
          conn_consume(c, l + 2);
          if (l == 0) break;
        }
        break;
      }
      if (bl + sz > maxb) {
        free(body);
        r->keep_alive = 0;
        http_respond_error(r, 413, "invalid_request_error", "body larger than %zu bytes", maxb);
        return -1;
      }
      while (cap < bl + sz) cap *= 2;
      body = realloc(body, cap + 1);
      while (c->len < sz + 2)
        if (conn_fill(c, sz + 2 + 4096 > c->cap ? sz + 2 + 4096 : c->cap)) { free(body); return -1; }
      memcpy(body + bl, c->buf, sz);
      bl += sz;
      conn_consume(c, sz + 2);
    }
    body[bl] = 0;
    r->body = body;
    r->body_len = bl;
  } else {
    size_t need = clen > 0 ? (size_t)clen : 0;
    char* body = malloc(need + 1);
    size_t got = c->len < need ? c->len : need;
    memcpy(body, c->buf, got);
    conn_consume(c, got);
    while (got < need) {
      ssize_t k;
      do k = recv(c->fd, body + got, need - got, 0);
      while (k < 0 && errno == EINTR);
      if (k <= 0) { free(body); return -1; }
      got += (size_t)k;
    }
    body[need] = 0;
    r->body = body;
    r->body_len = need;
  }
  return 0;
}

static void free_request(http_req* r) {
  free(r->body);
  free(r->hdr_block);
  r->hdr_block = NULL;
  r->body = NULL;
}

typedef struct {
  int fd;
  const http_server_config* cfg;
  char peer[64];
} conn_arg;

static void* conn_thread(void* a) {
  conn_arg* ca = a;
  conn_t c = {ca->fd, NULL, 0, 0};
  const http_server_config* cfg = ca->cfg;
  struct timeval tv = {cfg->idle_timeout_s ? cfg->idle_timeout_s : 60, 0};
  setsockopt(c.fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  int one = 1;
  setsockopt(c.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
  setsockopt(c.fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  for (;;) {
    http_req r;
    if (read_request(&c, &r, cfg)) break;
    cfg->handler(&r, cfg->user);
    if (cfg->access_log)
      fprintf(stderr, "%s \"%s %s\" %d %.1fms\n", ca->peer, r.method, r.path, r.status, http_now_ms() - r.t_start_ms);
    int ka = r.keep_alive && !r.upgraded;
    free_request(&r);
    if (!ka || atomic_load(&g_stop)) break;
  }
  close(c.fd);
  free(c.buf);
  free(ca);
  atomic_fetch_sub(&g_conns, 1);
  return NULL;
}

int http_healthcheck(int port, const char* path) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 1;
  struct sockaddr_in a;
  memset(&a, 0, sizeof a);
  a.sin_family = AF_INET;
  a.sin_port = htons((unsigned short)port);
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  struct timeval tv = {5, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  if (connect(fd, (struct sockaddr*)&a, sizeof a)) { close(fd); return 1; }
  char req[512], resp[64] = {0};
  int n = snprintf(req, sizeof req, "GET %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", path);
  if (send_all(fd, req, (size_t)n)) { close(fd); return 1; }
  ssize_t k = recv(fd, resp, sizeof resp - 1, 0);
  close(fd);
  return k > 12 && !strncmp(resp + 9, "200", 3) ? 0 : 1;
}

void http_server_stop(void) {
  atomic_store(&g_stop, 1);
  if (g_listen_fd >= 0) shutdown(g_listen_fd, SHUT_RDWR);
}

int http_serve(const http_server_config* cfg) {
  signal(SIGPIPE, SIG_IGN);
  int fd = socket(AF_INET6, SOCK_STREAM, 0);
  int v6 = fd >= 0;
  if (!v6) fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { perror("socket"); return 1; }
  int one = 1, zero = 0;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  const char* host = cfg->host && *cfg->host ? cfg->host : "0.0.0.0";
  int rc;
  if (v6) {
    struct sockaddr_in6 a6;
    memset(&a6, 0, sizeof a6);
    a6.sin6_family = AF_INET6;
    a6.sin6_port = htons((unsigned short)cfg->port);
    if (!strcmp(host, "0.0.0.0") || !strcmp(host, "::")) {
      setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);   /* dual stack */
      a6.sin6_addr = in6addr_any;
    } else if (strchr(host, ':')) inet_pton(AF_INET6, host, &a6.sin6_addr);
    else {   /* an IPv4 address: v4-mapped */
      char m[64];
      snprintf(m, sizeof m, "::ffff:%s", host);
      setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);
      inet_pton(AF_INET6, m, &a6.sin6_addr);
    }
    rc = bind(fd, (struct sockaddr*)&a6, sizeof a6);
  } else {
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)cfg->port);
    inet_pton(AF_INET, host, &a.sin_addr);
    rc = bind(fd, (struct sockaddr*)&a, sizeof a);
  }
  if (rc || listen(fd, 128)) { perror("bind/listen"); close(fd); return 1; }
  g_listen_fd = fd;
  int maxc = cfg->max_connections > 0 ? cfg->max_connections : 64;
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
  pthread_attr_setstacksize(&at, (size_t)8 << 20);   /* the request runs inference on this thread */
  while (!atomic_load(&g_stop)) {
    struct sockaddr_storage ss;
    socklen_t sl = sizeof ss;
    int c = accept(fd, (struct sockaddr*)&ss, &sl);
    if (c < 0) {
      if (errno == EINTR) continue;
      if (atomic_load(&g_stop)) break;
      continue;
    }
    if (atomic_fetch_add(&g_conns, 1) >= maxc) {
      atomic_fetch_sub(&g_conns, 1);
      const char* bb = "{\"error\":{\"message\":\"too many connections\",\"type\":\"server_busy\",\"code\":503}}";
      char busy[512];
      int bl = snprintf(busy, sizeof busy, "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                        "Content-Length: %zu\r\nConnection: close\r\nRetry-After: 1\r\n\r\n%s", strlen(bb), bb);
      send_all(c, busy, (size_t)bl);
      close(c);
      continue;
    }
    conn_arg* ca = calloc(1, sizeof *ca);
    ca->fd = c;
    ca->cfg = cfg;
    if (ss.ss_family == AF_INET6) inet_ntop(AF_INET6, &((struct sockaddr_in6*)&ss)->sin6_addr, ca->peer, sizeof ca->peer);
    else inet_ntop(AF_INET, &((struct sockaddr_in*)&ss)->sin_addr, ca->peer, sizeof ca->peer);
    pthread_t th;
    if (pthread_create(&th, &at, conn_thread, ca)) {
      close(c);
      free(ca);
      atomic_fetch_sub(&g_conns, 1);
    }
  }
  pthread_attr_destroy(&at);
  close(fd);
  return 0;
}
