# Example: A Short RFC-Style Document

This is a sample document that ships with airton_f so the scholar
character is testable out of the box. Replace or augment it with
your own markdown corpus — anything under `seed_documents/` is
ingested into the doc tree at session start.

## 1. Scope

This document describes a tiny imaginary protocol for sending
greetings between two processes. It exists to exercise the doc-tree
retrieval path: section-shaped chunks, hierarchical anchors, and
grounded citations.

## 2. Terminology

### 2.1 Greeter

The party that initiates a greeting. The greeter sends a `HELLO`
frame and waits for a `HELLO-ACK` response.

### 2.2 Greetee

The party that receives a greeting. The greetee replies with
`HELLO-ACK` within the timeout window or the greeter retries.

## 3. Frame Format

Every frame is a single line of UTF-8 text, terminated by `\n`. The
greeter and greetee each emit at most one frame per round trip.

### 3.1 HELLO

`HELLO <name>` — sent by the greeter. `<name>` is the greeter's
display name. Length must not exceed 64 bytes.

### 3.2 HELLO-ACK

`HELLO-ACK <name>` — sent by the greetee in response. `<name>` is
the greetee's display name. Same 64-byte limit.

## 4. Timeouts

The greeter retries the `HELLO` frame after 1 second without an
ack. After three retries the greeter SHOULD log an error and
abandon the exchange. The greetee MUST NOT reply to a duplicate
`HELLO` more than once per round trip.

## 5. Security Considerations

This protocol is illustrative; it has none. Do not use it in
production. Names are unauthenticated; replays are possible; the
transport is unencrypted. The example exists to demonstrate
section-grounded retrieval, not to defend any traffic.
