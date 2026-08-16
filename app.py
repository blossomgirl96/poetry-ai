#!/usr/bin/env python3
"""Mr. Meter — the chat app.

    ./run.sh            (this)
    ./run.sh --cli      (v1's four questions)

The conversation and the poem are two different model calls with two different
system prompts. See persona.py for why they stay apart.
"""

import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import poem
from persona import BOOTSTRAP, CHAT_MAX_TOKENS, MAX_TURNS, PERSONA, WRITE_POEM_TOOL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE_DIR, "static", "index.html")

app = FastAPI()
client = anthropic.Anthropic()


@dataclass
class Session:
    id: str
    created_at: datetime
    messages: list = field(default_factory=list)   # Anthropic-shaped
    transcript: list = field(default_factory=list)  # display / sidecar-shaped
    turns: int = 0
    chat_usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


# Single local user, so in-memory is enough. Restarting the server drops
# in-progress conversations; saved poems are on disk and unaffected.
SESSIONS: dict[str, Session] = {}


class TurnIn(BaseModel):
    session_id: str
    message: str = ""


class WriteIn(BaseModel):
    session_id: str


def sse(**payload):
    return f"data: {json.dumps(payload)}\n\n"


def get_session(session_id):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session expired")
    return session


def iter_chat(messages, force=False):
    """Stream one Mr. Meter turn. Yields ("text", chunk) then one ("final", message).

    Thinking stays on (Opus 5's default — we pass no `thinking` field). Disabling
    it would be faster, but on Opus 5 a tool call can then arrive as plain
    visible text: the turn succeeds, write_poem never fires, and no error is
    raised. That would silently break the handoff. Use effort to control cost.
    """
    kwargs = {
        "model": poem.MODEL,
        "max_tokens": CHAT_MAX_TOKENS,
        "system": PERSONA,
        "messages": messages,
        "tools": [WRITE_POEM_TOOL],
        "output_config": {"effort": "low"},
    }
    if force:
        kwargs["tool_choice"] = {"type": "tool", "name": WRITE_POEM_TOOL["name"]}

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield "text", text
        yield "final", stream.get_final_message()


def run_turn(session, user_message, force=False):
    """One turn: Mr. Meter replies, and if he called the tool, the poem follows.

    Both halves ride the same connection — the handoff happens inside a single
    turn, so splitting them would cost a round trip and open a race.
    """
    # Build the pending message list without committing it: if the stream dies
    # we don't want a half-turn poisoning the next request's context.
    pending = list(session.messages)
    if not pending:
        pending.append({"role": "user", "content": BOOTSTRAP})
        session.transcript.append({"role": "bootstrap", "content": BOOTSTRAP})
    elif user_message:
        pending.append({"role": "user", "content": user_message})
        session.transcript.append({"role": "user", "content": user_message})

    try:
        # --- Mr. Meter ---
        yield sse(type="chat_start")
        reply_parts, final = [], None
        for kind, payload in iter_chat(pending, force=force):
            if kind == "text":
                reply_parts.append(payload)
                yield sse(type="chat_delta", text=payload)
            else:
                final = payload
        yield sse(type="chat_end")

        reply = "".join(reply_parts).strip()
        pending.append({"role": "assistant", "content": final.content})

        # Only commit once the stream completed cleanly.
        session.messages = pending
        if reply:
            session.transcript.append({"role": "assistant", "content": reply})
        session.turns += 1
        session.chat_usage["input_tokens"] += final.usage.input_tokens
        session.chat_usage["output_tokens"] += final.usage.output_tokens

        tool_use = next(
            (b for b in final.content if b.type == "tool_use"), None
        )
        if tool_use is None:
            return

        # --- the poet ---
        yield sse(type="poem_start")
        answers = {k: (tool_use.input.get(k) or "") for k in poem.QUESTION_TEXT}
        chat_only = [t for t in session.transcript if t["role"] in ("user", "assistant")]
        user_prompt = poem.build_prompt(answers, transcript=chat_only)

        poem_parts, poem_final = [], None
        for kind, payload in poem.iter_poem(client, user_prompt):
            if kind == "text":
                poem_parts.append(payload)
                yield sse(type="poem_delta", text=payload)
            else:
                poem_final = payload

        text = "".join(poem_parts).strip()
        if poem_final.stop_reason == "refusal":
            yield sse(type="error", message="Claude declined to write this one.")
            return
        if not text:
            yield sse(type="error", message="Nothing came back — no poem text.")
            return

        poem_path, meta_path = poem.save_run(
            answers,
            user_prompt,
            text,
            poem_final,
            mode="chat",
            transcript=session.transcript,
            persona_prompt=PERSONA,
            chat_model=poem.MODEL,
            chat_usage=session.chat_usage,
        )
        yield sse(
            type="poem_end",
            poem_path=os.path.relpath(poem_path, BASE_DIR),
            meta_path=os.path.relpath(meta_path, BASE_DIR),
            stop_reason=poem_final.stop_reason,
            usage={
                "input_tokens": poem_final.usage.input_tokens,
                "output_tokens": poem_final.usage.output_tokens,
            },
        )

    except anthropic.APIError as e:
        # Past the first byte, an exception can no longer become a status code.
        yield sse(type="error", message=str(e))
    finally:
        yield sse(type="done")


STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.post("/session")
def new_session():
    sid = secrets.token_urlsafe(8)
    SESSIONS[sid] = Session(id=sid, created_at=datetime.now())
    return {"session_id": sid}


@app.post("/turn")
def turn(body: TurnIn):
    session = get_session(body.session_id)
    # The persona's "up to 5 turns" is a soft ceiling it judges itself; this is
    # only the safety net for when it talks past it.
    force = session.turns >= MAX_TURNS
    return StreamingResponse(
        run_turn(session, body.message, force=force),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@app.post("/write")
def write_now(body: WriteIn):
    session = get_session(body.session_id)
    return StreamingResponse(
        run_turn(session, "", force=True),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
