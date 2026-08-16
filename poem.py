#!/usr/bin/env python3
"""Ask five questions, write one poem.

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

QUESTIONS = [
    {
        "key": "subject",
        "prompt": "What's the poem about?",
        "hint": "a person, a place, a moment, a feeling — anything",
        "required": True,
    },
    {
        "key": "image",
        "prompt": "One specific image you can't shake?",
        "hint": "something you can see, hear, smell, or touch",
        "required": True,
    },
    {
        "key": "feeling",
        "prompt": "What should the reader feel at the end?",
        "hint": "e.g. ached-out relief, quiet dread, wanting to call someone",
        "required": False,
    },
    {
        "key": "audience",
        "prompt": "Who is it for?",
        "hint": "press Enter to skip",
        "required": False,
    },
    {
        "key": "form",
        "prompt": "Form? [1] free verse  [2] rhyming  [3] haiku  [4] sonnet",
        "hint": "Enter for free verse",
        "required": False,
    },
]

FORMS = {
    "1": "free verse",
    "2": "a rhyming poem with a consistent scheme",
    "3": "a haiku (5-7-5)",
    "4": "a sonnet (14 lines, iambic pentameter)",
    "": "free verse",
}

SYSTEM = """You are a poet. You write poems that earn their images instead of \
decorating with them.

Rules:
- Use the person's specific image. Don't swap it for a prettier one.
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


def build_prompt(answers):
    form = FORMS.get(answers["form"], answers["form"] or "free verse")
    lines = [
        f"Write a poem about: {answers['subject']}",
        f"It must include this image: {answers['image']}",
        f"Form: {form}",
    ]
    if answers["feeling"]:
        lines.append(f"The reader should end up feeling: {answers['feeling']}")
    if answers["audience"]:
        lines.append(f"It is for: {answers['audience']}")
    return "\n".join(lines)


def save_run(answers, user_prompt, poem, message):
    """Write the poem, plus a JSON sidecar holding everything that produced it."""
    outdir = os.path.join(BASE_DIR, "poems")
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

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
            {"key": q["key"], "question": q["prompt"], "answer": answers[q["key"]]}
            for q in QUESTIONS
        ],
        "resolved_form": FORMS.get(answers["form"], answers["form"] or "free verse"),
        "user_prompt": user_prompt,
        "poem": poem,
        "stop_reason": message.stop_reason,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    }

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

    print("\n\033[1mFive questions, then a poem.\033[0m")
    print("\033[2mCtrl-C to bail at any point.\033[0m")

    answers = {q["key"]: ask(q) for q in QUESTIONS}

    print("\n\033[2mWriting...\033[0m\n")
    print("\033[2m" + "─" * 50 + "\033[0m")

    client = anthropic.Anthropic()
    user_prompt = build_prompt(answers)
    poem_parts = []
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                poem_parts.append(text)
                print(text, end="", flush=True)
            final = stream.get_final_message()
    except anthropic.APIError as e:
        sys.exit(f"\nAPI error: {e}")

    if final.stop_reason == "refusal":
        sys.exit("\nClaude declined to write this one.")

    poem = "".join(poem_parts).strip()
    print("\n\033[2m" + "─" * 50 + "\033[0m")

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
