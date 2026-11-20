/* http.h: a small HTTP/1.1 + WebSocket (RFC 6455) server for the Arc deploy images. POSIX sockets and pthreads only:
 * no third-party code. One thread per connection (capped), keep-alive, Content-Length and chunked request bodies,
 * Expect: 100-continue, multipart/form-data, query strings, base64, and WebSocket upgrade / framing.
 *
 * The same file ships in WhissleAI/arc_transcriber and WhissleAI/arc-tagger-c (deploy/server/); keep the copies equal.
 * Part of the Arc deploy service (Whissle Inc.); licensed as the repository it is in (see LICENSE). */
#ifndef ARC_HTTP_H
#define ARC_HTTP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HTTP_MAX_HEADERS 64

typedef struct {
  char* name;
  char* value;
} http_header_kv;

typedef struct http_req {
  int fd;
  char method[16];
  char path[1024];      /* without the query string, percent-decoded */
  char query[4096];     /* raw query string (no '?') */
  int n_headers;
  http_header_kv headers[HTTP_MAX_HEADERS];
  char* body;           /* NUL-terminated (body_len bytes + 0) */
  size_t body_len;
  int keep_alive;       /* the response keeps the connection open */
  int upgraded;         /* set by ws_accept: the connection belongs to the WebSocket session now */
  double t_start_ms;
  char* hdr_block;      /* internal */
  int status;           /* set by http_respond (for the access log) */
} http_req;

typedef void (*http_handler)(http_req* r, void* user);

typedef struct {
  const char* host;          /* "0.0.0.0" */
  int port;
  int max_connections;       /* concurrent connections (threads); 0 = 64 */
  size_t max_body;           /* bytes; 0 = 100 MB */
  int idle_timeout_s;        /* keep-alive idle timeout; 0 = 60 */
  int access_log;            /* 1: one line per request on stderr */
  http_handler handler;
  void* user;
} http_server_config;

/* Listen and serve forever (returns only on a bind / listen error, or after http_server_stop). */
int http_serve(const http_server_config* c);
/* Ask http_serve to return (from a signal handler: only sets a flag and closes the listening socket). */
void http_server_stop(void);

/* Case-insensitive header lookup; NULL if absent. */
const char* http_header(const http_req* r, const char* name);
/* Query parameter (percent-decoded, '+' = space) into out; 1 if present, 0 if not. */
int http_query(const http_req* r, const char* key, char* out, size_t out_len);
/* 1 if the media type of Content-Type (before ';', case-insensitive) starts with prefix ("audio/", "application/json") */
int http_ctype_is(const http_req* r, const char* prefix);

/* Responses (each sends a full response with Content-Length). */
void http_respond(http_req* r, int status, const char* content_type, const char* body, size_t len);
void http_respond_json(http_req* r, int status, const char* json);
/* {"error": {"message": msg, "type": type, "code": status}} (OpenAI-style error envelope) */
void http_respond_error(http_req* r, int status, const char* type, const char* fmt, ...);
const char* http_reason(int status);

/* multipart/form-data. Parts point into the (modified) body; data is not NUL-terminated (len gives its size). */
typedef struct {
  const char* name;
  const char* filename;      /* NULL for a plain field */
  const char* content_type;  /* NULL if not given */
  const char* data;
  size_t len;
} http_part;
/* Returns the number of parts (<= max), or -1 when the body is not valid multipart. */
int http_multipart(http_req* r, http_part* parts, int max);
/* A field's value as a malloc'd NUL-terminated string (NULL if absent). */
char* http_part_str(const http_part* parts, int n, const char* name);
const http_part* http_part_get(const http_part* parts, int n, const char* name);

/* base64 (standard alphabet, padding optional on decode). Return malloc'd buffers; NULL on bad input. */
char* b64_encode(const unsigned char* data, size_t len);
unsigned char* b64_decode(const char* s, size_t len, size_t* out_len);

/* WebSocket. ws_accept answers the upgrade (101); then ws_recv / ws_send on r->fd. */
enum { WS_TEXT = 1, WS_BINARY = 2, WS_CLOSE = 8, WS_PING = 9, WS_PONG = 10 };
int http_is_ws_upgrade(const http_req* r);
int ws_accept(http_req* r);
/* Next complete message (fragments joined; pings answered; pongs skipped). *data is malloc'd (NUL-terminated).
 * Returns the opcode (WS_TEXT / WS_BINARY), WS_CLOSE when the peer closed, or -1 on error / timeout. */
int ws_recv(int fd, unsigned char** data, size_t* len, size_t max_len);
int ws_send(int fd, int opcode, const void* data, size_t len);   /* 0 / -1; thread-safe per fd is the caller's job */
int ws_send_text(int fd, const char* s);
void ws_close(int fd, int code, const char* reason);

/* Docker HEALTHCHECK without curl in the image: GET path on 127.0.0.1:port; 0 if it answers 200. */
int http_healthcheck(int port, const char* path);

double http_now_ms(void);

#ifdef __cplusplus
}
#endif
#endif
