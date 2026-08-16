# poetry-ai

A CLI that asks you four questions and writes you a free-verse poem.

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

`poem.py` is one file, ~195 lines:

1. **`.env` load** — `python-dotenv` reads the key, resolved relative to the script
   so `./run.sh` works from any directory.
2. **Fixed question set** (`QUESTIONS`) — who it's about, favourite things about them,
   a shared memory, target feeling. All four are required; blank answers are
   re-prompted. Form isn't asked: every poem is free verse (`FORM`).
3. **Prompt assembly** (`build_prompt`) — folds the answers into a single user turn.
   The craft rules live in `SYSTEM`, separate from the answers.
4. **One streaming call** to `claude-opus-5`, printed token by token. Ctrl-C bails
   cleanly at any point, including mid-poem.
5. Optional save (`save_run`) — writes two files per poem.

### On `MAX_TOKENS`

Opus 5 thinks by default, and thinking spends the same `max_tokens` budget as the
poem — a short poem can still spend a lot of budget getting there. `MAX_TOKENS` is
set to 64,000, and the call streams, so a high ceiling costs nothing when it goes
unused. If a poem ever does hit the ceiling, the CLI says so instead of silently
saving a truncated one, and a run that produces no text at all is never offered for
saving.

## What gets saved

Answering `y` writes a pair, sharing a timestamp:

- `poems/<ts>.txt` — the poem, for reading
- `poems/<ts>.json` — everything that produced it, for comparing

The JSON holds the full `SYSTEM` text plus a 12-char SHA-256 of it, every question
with its answer, the resolved form, the assembled user prompt, the poem,
`stop_reason`, and token usage. Because the questions are recorded by key, sidecars
written before a question changed still say which question was actually asked.

Group runs by prompt version:

```bash
# how many poems per SYSTEM version?
jq -r .system_prompt_sha256 poems/*.json | sort | uniq -c

# every poem written under one version
jq -r 'select(.system_prompt_sha256=="ec7e69afcba5") | .poem' poems/*.json

# token spend so far
jq -s 'map(.usage.output_tokens) | add' poems/*.json
```

Edit `SYSTEM`, run the same inputs again, and the hash changes — that's your A/B.

## Where to tune it

| Want to change | Edit |
|---|---|
| The questions | `QUESTIONS` list in `poem.py` |
| Poem quality / voice | `SYSTEM` string — this is the main lever |
| The form every poem takes | `FORM` string |
| Model or length | `MODEL`, `MAX_TOKENS` |

## Files

| File | Purpose |
|---|---|
| `run.sh` | The entry point. Handles venv + deps + launch. |
| `poem.py` | The tool. |
| `.env` | Your API key. Gitignored — never commit it. |
| `.env.example` | Template to copy. Safe to commit. |
| `poems/` | Saved poems. Gitignored. |

## Next steps worth considering

- **Compare prompt versions properly** — the sidecars already record which `SYSTEM`
  produced each poem, but nothing scores them. A rating prompt would turn that record
  into an actual A/B.
- **AI-generated follow-ups** — keep Q1 fixed, have Claude generate Q2–Q4 from the
  answers. More magical, more latency, harder to debug.
- **Web wrapper** — a small FastAPI/Flask endpoint around `build_prompt` + the
  streaming call, so the whole thing is shareable.
