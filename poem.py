#!/usr/bin/env python3
"""Ask four questions, write one poem.

Usage:
    ./run.sh            (reads your key from .env)
"""

import hashlib
import json
import os
import sys
from datetime import datetime

import anthropic
from dotenv import load_dotenv

# Resolve everything against this script's directory, so it works from any cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = "claude-opus-5"
# Opus 5 thinks by default, and thinking counts against max_tokens alongside the
# poem itself — a short poem can still spend a lot of budget getting there. We
# stream, so a large ceiling costs nothing when it goes unused.
MAX_TOKENS = 64000

QUESTIONS = [
    {
        "key": "subject",
        "prompt": "Who's this poem about?",
        "hint": "a friend, family member, partner — anyone",
        "required": True,
    },
    {
        "key": "favourites",
        "prompt": "What are your favourite things about them?",
        "hint": "the small specific ones beat the big general ones",
        "required": True,
    },
    {
        "key": "memory",
        "prompt": "Any shared memories or stories that resonate with you in this moment?",
        "hint": "whichever one is on your mind right now",
        "required": True,
    },
    {
        "key": "feeling",
        "prompt": "How do you want the reader to feel reading this poem?",
        "hint": "e.g. ached-out relief, quiet dread, wanting to call someone",
        "required": True,
    },
]

FORM = "free verse"

# Canonical question text, so a chat-mode sidecar records the same wording the
# CLI would have asked even though nobody asked it out loud.
QUESTION_TEXT = {q["key"]: q["prompt"] for q in QUESTIONS}

SYSTEM = """You are a poet. You write poems that earn their images instead of \
decorating with them.

Rules:
- Use the details they gave you — the things they love, and any memory they told.
  Don't swap them for prettier ones of your own.
- Concrete over abstract. Show the thing, don't name the emotion.
- No greeting-card lines, no "in the end", no forced profundity.
- Vary line length. Let the poem breathe where it needs to.
- Give the poem a title on the first line, then a blank line, then the poem.
- Output only the title and the poem. No preamble, no explanation, no notes."""


def ask(question):
    """Prompt until we get an answer (or accept blank if optional)."""
    print(f"\n\033[1m{question['prompt']}\033[0m")
    print(f"\033[2m{question['hint']}\033[0m")
    while True:
        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOk — nothing written.")
            sys.exit(0)
        if answer or not question["required"]:
            return answer
        print("\033[2m(this one's needed)\033[0m")


def build_prompt(answers, transcript=None):
    """Fold the answers into one user turn.

    `transcript` is an optional list of {"role", "content"} from a chat session,
    appended as labeled source material so the poet gets the person's own
    phrasing alongside the distilled slots.
    """
    lines = [
        f"Write a poem about: {answers['subject']}",
        f"Their favourite things about the subject: {answers['favourites']}",
        f"Form: {FORM}",
    ]
    if answers.get("memory"):
        lines.append(f"A shared memory that matters to them right now: {answers['memory']}")
    if answers.get("feeling"):
        lines.append(f"The reader should end up feeling: {answers['feeling']}")

    if transcript:
        lines.append("")
        lines.append("--- The conversation that produced this, verbatim ---")
        lines.append(
            "Only the person's words are source material. The interviewer's lines "
            "are context, and anything the interviewer said about themselves is "
            "not about the subject."
        )
        lines.append("")
        for turn in transcript:
            # The persona is labeled generically so no character leaks into the
            # poet's context.
            if turn["role"] == "user":
                lines.append(f"Them: {turn['content']}")
            elif turn["role"] == "assistant":
                lines.append(f"Interviewer: {turn['content']}")
    return "\n".join(lines)


def iter_poem(client, user_prompt):
    """Stream one poem. Yields ("text", chunk) then exactly one ("final", message).

    Shared by the CLI (which prints) and the web app (which frames as SSE).
    """
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield "text", text
        yield "final", stream.get_final_message()


def save_run(
    answers,
    user_prompt,
    poem,
    message,
    *,
    mode="cli",
    transcript=None,
    persona_prompt=None,
    chat_model=None,
    chat_usage=None,
):
    """Write the poem, plus a JSON sidecar holding everything that produced it.

    Chat-mode runs add keys rather than changing existing ones, so every jq
    recipe in the README keeps working across a mixed v1/v2 corpus.
    """
    outdir = os.path.join(BASE_DIR, "poems")
    os.makedirs(outdir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Two saves in the same second would otherwise overwrite each other.
    if os.path.exists(os.path.join(outdir, stamp + ".txt")):
        n = 2
        while os.path.exists(os.path.join(outdir, f"{stamp}-{n}.txt")):
            n += 1
        stamp = f"{stamp}-{n}"

    poem_path = os.path.join(outdir, stamp + ".txt")
    with open(poem_path, "w") as f:
        f.write(poem + "\n")

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": MODEL,
        # Short hash lets you group runs by prompt version without diffing text.
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()[:12],
        "system_prompt": SYSTEM,
        "questions": [
            {
                "key": key,
                "question": QUESTION_TEXT[key],
                "answer": answers.get(key, ""),
            }
            for key in QUESTION_TEXT
        ],
        "resolved_form": FORM,
        "user_prompt": user_prompt,
        "poem": poem,
        "stop_reason": message.stop_reason,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
        "mode": mode,
        # None until the person says; "up" or "down" after.
        "rating": None,
    }

    if mode == "chat":
        # Kept out of `usage` so the token-spend recipe still means "poem cost".
        record["persona_prompt"] = persona_prompt
        record["persona_prompt_sha256"] = hashlib.sha256(
            (persona_prompt or "").encode()
        ).hexdigest()[:12]
        record["transcript"] = transcript or []
        record["turns"] = sum(
            1 for t in (transcript or []) if t["role"] == "assistant"
        )
        record["chat_model"] = chat_model
        record["chat_usage"] = chat_usage or {}

    meta_path = os.path.join(outdir, stamp + ".json")
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return poem_path, meta_path


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "No API key found.\n"
            "  1. cp .env.example .env      (if you haven't already)\n"
            "  2. open .env and paste your key after ANTHROPIC_API_KEY=\n"
            "     get one at https://console.anthropic.com/settings/keys"
        )

    print("\n\033[1mFour questions, then a poem.\033[0m")
    print("\033[2mCtrl-C to bail at any point.\033[0m")

    answers = {q["key"]: ask(q) for q in QUESTIONS}

    print("\n\033[2mWriting...\033[0m\n")
    print("\033[2m" + "─" * 50 + "\033[0m")

    client = anthropic.Anthropic()
    user_prompt = build_prompt(answers)
    poem_parts = []
    final = None
    try:
        for kind, payload in iter_poem(client, user_prompt):
            if kind == "text":
                poem_parts.append(payload)
                print(payload, end="", flush=True)
            else:
                final = payload
    except anthropic.APIError as e:
        sys.exit(f"\nAPI error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nOk — nothing written.")

    if final.stop_reason == "refusal":
        sys.exit("\nClaude declined to write this one.")

    poem = "".join(poem_parts).strip()
    print("\n\033[2m" + "─" * 50 + "\033[0m")

    if final.stop_reason == "max_tokens":
        print("\n\033[1mThat one hit the token ceiling before it finished.\033[0m")
        print(f"\033[2mRaise MAX_TOKENS (currently {MAX_TOKENS:,}) or try again.\033[0m")

    if not poem:
        sys.exit("\nNothing to save — no poem text came back.")

    try:
        save = input("\nSave it? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        save = "n"
    if save == "y":
        poem_path, meta_path = save_run(answers, user_prompt, poem, final)
        print(f"Saved  {os.path.relpath(poem_path, BASE_DIR)}")
        print(f"       {os.path.relpath(meta_path, BASE_DIR)}  (inputs + prompts)")


if __name__ == "__main__":
    main()


def set_rating(meta_path, rating):
    """Record a thumbs up or down against a saved poem.

    Rewrites the sidecar in place rather than keeping ratings in a second file,
    so a poem and what someone thought of it stay together.
    """
    with open(meta_path) as f:
        record = json.load(f)
    record["rating"] = rating
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record["rating"]
