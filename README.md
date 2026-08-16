# poetry-ai

A CLI that asks you five questions and writes you a poem.

## Setup (once)

```bash
cp .env.example .env     # then paste your key into .env
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
It should look like `sk-ant-api03-…` — one opaque string, no dots.

## Run (every time)

```bash
./run.sh
```

That's it. `run.sh` creates the venv and installs dependencies on first run, then
reads your key from `.env` and starts the questions.

## How it works

`poem.py` is one file, ~145 lines:

1. **`.env` load** — `python-dotenv` reads the key, resolved relative to the script
   so `./run.sh` works from any directory.
2. **Fixed question set** (`QUESTIONS`) — subject, a concrete image, target feeling,
   audience, form. Subject and image are required; the rest are skippable.
3. **Prompt assembly** (`build_prompt`) — folds the answers into a single user turn.
   The craft rules live in `SYSTEM`, separate from the answers.
4. **One streaming call** to `claude-opus-5`, printed token by token.
5. Optional save (`save_run`) — writes two files per poem.

## What gets saved

Answering `y` writes a pair, sharing a timestamp:

- `poems/<ts>.txt` — the poem, for reading
- `poems/<ts>.json` — everything that produced it, for comparing

The JSON holds the full `SYSTEM` text plus a 12-char SHA-256 of it, every question
with its answer (blanks preserved, so a skip is distinguishable from a non-question),
the resolved form, the assembled user prompt, the poem, `stop_reason`, and token
usage.

Group runs by prompt version:

```bash
# how many poems per SYSTEM version?
jq -r .system_prompt_sha256 poems/*.json | sort | uniq -c

# every poem written under one version
jq -r 'select(.system_prompt_sha256=="109e6fe4050c") | .poem' poems/*.json

# token spend so far
jq -s 'map(.usage.output_tokens) | add' poems/*.json
```

Edit `SYSTEM`, run the same inputs again, and the hash changes — that's your A/B.

## Where to tune it

| Want to change | Edit |
|---|---|
| The questions | `QUESTIONS` list in `poem.py` |
| Poem quality / voice | `SYSTEM` string — this is the main lever |
| Form options | `FORMS` dict |
| Model or length | `MODEL`, `max_tokens` |

## Files

| File | Purpose |
|---|---|
| `run.sh` | The entry point. Handles venv + deps + launch. |
| `poem.py` | The tool. |
| `.env` | Your API key. Gitignored — never commit it. |
| `.env.example` | Template to copy. Safe to commit. |
| `poems/` | Saved poems. Gitignored. |

## Next steps worth considering

- **A/B the system prompt** — save each poem alongside the prompt version that made
  it, so you can tell whether a prompt edit actually helped. Nothing tracks this yet.
- **AI-generated follow-ups** — keep Q1 fixed, have Claude generate Q2–Q5 from the
  answers. More magical, more latency, harder to debug.
- **Web wrapper** — a small FastAPI/Flask endpoint around `build_prompt` + the
  streaming call, so the whole thing is shareable.
