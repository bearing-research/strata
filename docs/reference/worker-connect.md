# Worker Relay Protocol

`strata-worker --connect <url>` runs a worker that binds no port. It opens one outbound WebSocket to a relay, and the relay presents the worker at an HTTP URL the notebook server dispatches to like any other worker. Each HTTP request the relay receives for that URL travels over the WebSocket as a stream of frames, and the worker's in-process app answers it the same way. A worker behind NAT, or a firewall that allows only outbound connections, can then serve `# @worker` cells.

Strata ships the worker side. The relay is not part of Strata; this page is what a relay implements. Everything the server and worker say to each other is the [executor protocol](executor-protocol.md), unchanged: this page only says how those HTTP requests ride over one WebSocket.

```
notebook server ──HTTP──► relay ◄══WebSocket══ strata-worker --connect
                 POST https://relay/w/abc/v1/execute      (worker dialled out)
```

## What goes over the socket, and what does not

Only requests **to** the worker go through the relay: the dispatch (`/v1/execute`, `/v1/notebook-execute`, `/v1/execute-manifest`, `/execute`), the cancel (`/v1/executions/{build_id}/cancel`) and `GET /health`.

Requests the worker **makes** go out directly, as from any worker: a signed worker downloads inputs, uploads its bundle, finalizes and posts console chunks to the `log_url` over ordinary outbound HTTPS. A `direct` worker makes none; its console arrives in the bundle when the cell ends. So a `signed` worker needs outbound reach to the server's signed URLs (or the object store, with presigned URLs), and a `direct` worker needs nothing but the relay.

## Connecting

The worker opens a WebSocket to the `--connect` URL (`ws://` or `wss://`; use `wss://` anywhere but a test) with:

- `Authorization: Bearer <token>`, where the token is the value of `STRATA_WORKER_CONNECT_TOKEN` on the worker, if set. It is the relay's credential for this worker and nothing else: the relay checks it to decide which worker this socket is. The worker never puts it on the command line, where the cells it runs could read it, and removes it from its environment at startup like its other secrets.
- `Sec-WebSocket-Protocol: strata-worker-connect.v1`. The relay must select this subprotocol in its handshake response. A relay that selects none, or another one, is refused by the worker as one that does not speak this version.

The relay refuses a worker by answering the handshake with an HTTP status instead of `101`. A `4xx` other than `408` and `429` is final: `strata-worker` logs it and exits with status 1, since the same token would be refused again. A `408`, `429`, `5xx`, a network error, or a connection that closes later, is retried.

**Reconnecting.** After a failure or a close, the worker waits a random time between half and all of the current delay, then reconnects. The delay starts at 1 s, doubles on each consecutive failure up to 60 s, and goes back to 1 s once a handshake succeeds. A request in flight when the connection drops is lost to its caller (the relay should fail it, for example with `502`); the cell it started keeps running on the worker until it finishes or the server cancels it, as when a plain HTTP client hangs up.

**Keepalive.** The worker sends a WebSocket ping every 20 s and closes the connection if no pong arrives within 20 s. A relay answers pings as any WebSocket implementation does. That traffic also keeps a NAT mapping open.

## Mapping URLs

The relay chooses the URL it presents the worker at, typically with a per-worker prefix such as `https://relay.example.com/w/<worker-id>`. It strips that prefix before forwarding, so the worker sees the paths of the executor protocol. The server is registered with the relay URL, for example `config.url = "https://relay.example.com/w/abc/v1/execute"`, and derives `/health`, `/v1/execute-manifest` and the cancel URL from it keeping the prefix, so all of them reach the relay under the same prefix.

## Frames

Every WebSocket message is one frame. A **text** message is a JSON object, a **control frame**. A **binary** message is a **body chunk**: 4 bytes holding the stream id as an unsigned big-endian integer, then up to 1 MiB (1,048,576 bytes) of body. The worker refuses a message larger than 1 MiB plus 4 bytes by closing the connection (close code 1009), so a relay splits larger bodies. An empty chunk carries nothing and need not be sent.

Each HTTP request is a **stream**, named by an `id` the relay picks: an integer from 1 to 2³²−1, unique among the streams in flight on this connection. A relay may reuse an id once its stream has ended. Frames of different streams interleave freely; frames of one stream arrive in the order sent, since one WebSocket preserves order.

### Relay to worker

**`request`** opens a stream:

```json
{
  "type": "request",
  "id": 7,
  "method": "POST",
  "path": "/v1/execute",
  "query": "",
  "headers": [["authorization", "Bearer <STRATA_WORKER_TOKEN>"], ["content-type", "multipart/form-data; boundary=..."], ["content-length", "52133"]]
}
```

| Field | Description |
| --- | --- |
| `method` | The HTTP method. |
| `path` | The request path after the relay's own prefix is stripped, starting with `/`, percent-encoded as it was on the wire. |
| `query` | The raw query string without the `?`, or `""`. |
| `headers` | `[name, value]` pairs in order, repeats allowed, as Latin-1 strings. Forward the caller's headers, `Authorization` included, minus hop-by-hop ones (`Connection`, `Keep-Alive`, `Transfer-Encoding`, `Upgrade`, `TE`, `Trailer`, `Proxy-*`). |

Then the request body as body chunks with this `id`, then:

**`end`**, `{"type": "end", "id": 7}`, when the body is complete. A request without a body (`GET /health`) is a `request` frame followed by `end`. A `Content-Length` header, if forwarded, must match the bytes sent.

**`abort`**, `{"type": "abort", "id": 7}`, when the relay gives up on the stream, for example because its caller disconnected. The worker drops what is left of the body and tells the app its client has gone; it sends nothing more for the stream. A running cell is not stopped by this, as with a plain HTTP disconnect: the server stops a cell through the cancel route, which arrives as a stream of its own.

### Worker to relay

**`response`** starts the answer:

```json
{"type": "response", "id": 7, "status": 200, "headers": [["content-type", "application/x-tar"], ["x-strata-executor-protocol", "v1"]]}
```

Then the response body as body chunks with this `id`, then **`end`**, `{"type": "end", "id": 7}`. The worker sends the response only after the app produces it, which for a cell is when the cell finishes; the relay holds its caller's request open meanwhile, so it should allow at least as long as the longest cell timeout. If `headers` carries `Content-Length`, the body has exactly that many bytes, and a relay may forward it or re-frame the body (chunked) and drop it.

If the app fails before it has a response, the worker answers `500` with a short text body. If it fails after the response has started, the worker sends **`abort`**, `{"type": "abort", "id": 7}`, and the relay ends its caller's response as failed (closing the connection rather than completing it).

Both sides ignore a frame with an unknown `type`, and frames for an id that is not open (a stream already finished or aborted).

## Concurrency and flow control

Every stream is served concurrently: the server's cancel for a cell arrives as a second stream while the dispatch stream is still open, and is answered at once. There is no per-stream flow control. The worker holds at most 16 chunks of a request body that the app has not read yet; beyond that it stops reading the WebSocket until the app catches up, which pauses every stream on the connection the way a slow reader pauses a TCP connection. The app reads request bodies as they arrive (a multipart body is spooled to disk), so this only bites on a slow disk. A request the app answers without reading its body, such as a `401`, has the rest of its body discarded rather than held. In the other direction the worker writes response chunks as the WebSocket accepts them; a relay that stops reading likewise pauses the worker's responses.

## Authentication, end to end

Two credentials, for two hops:

| Credential | Presented by | To | Checked by |
| --- | --- | --- | --- |
| `STRATA_WORKER_CONNECT_TOKEN` | the worker | the relay, on the WebSocket handshake | the relay |
| `STRATA_WORKER_TOKEN` | the server (`config.token` / `config.token_env`) | the worker, in each request's `Authorization` header, forwarded by the relay | the worker, per request, as on a bound worker |

The relay never needs the worker token, and the worker token is not the relay's business beyond forwarding the header. Set `STRATA_WORKER_TOKEN` on a connected worker as on any reachable one: without it, anyone who can send a request to the relay URL can run code on the worker.

## Why this framing

The executor protocol's requests are few, long, and large: one dispatch per cell, a body of inputs, a bundle back, and an occasional cancel alongside. A JSON control frame plus raw binary chunks covers that with nothing a relay author has to pull in beyond a WebSocket library, and it can be implemented and debugged by reading the messages. Multiplexing through yamux or HTTP/2 over the socket would add a dependency on both ends, plus window-based flow control that this traffic does not need, for a relay that is not Strata's to write.
