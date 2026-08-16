# poetry-ai

Chat with Mr. Meter about someone you love. He writes them a free-verse poem.

## Setup (once)

```bash
cp .env.example .env     # then paste your key into .env
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
It should look like `sk-ant-api03-…` — one opaque string, no dots.

## Run (every time)

```bash
./run.sh            # chat with Mr. Meter in the browser
./run.sh --cli      # the original four-question terminal flow
```

`run.sh` creates the venv, installs dependencies whenever `requirements.txt`
changes, reads your key from `.env`, and opens the page at `localhost:8000`.

## How it works

Two model calls with two different system prompts, and keeping them apart is the
whole design:

| | Prompt | What it does |
|---|---|---|
| **Mr. Meter** | `PERSONA` in `persona.py` | Talks to you. Introduces himself, asks who the poem is for, one question at a time, shares little asides of his own. Gathers material. |
| **The poet** | `SYSTEM` in `poem.py` | Writes the poem. Never speaks to you directly. Unchanged since v1. |

Mr. Meter carries one tool, `write_poem`, whose four slots are the same four
questions v1 asked out loud — `subject`, `favourites`, `memory`, `feeling`. When
he has enough, he says his handoff line and calls it in the same message; the
server streams that line and then continues straight into the poem on the same
connection. The poet receives both the distilled slots **and** the verbatim
conversation, so it gets structure without losing your own phrasing.

Files:

| File | Role |
|---|---|
| `persona.py` | Mr. Meter's prompt, the `write_poem` tool, `MAX_TURNS` |
| `app.py` | FastAPI — sessions, SSE streaming, the poem handoff |
| `static/index.html` | The whole UI. No build step, no npm |
| `poem.py` | The poet + the v1 CLI. `iter_poem`, `build_prompt`, `save_run` |

### Two things that will bite you

**Don't disable thinking on the chat call.** On Opus 5 with `thinking` disabled,
a tool call can arrive as *plain visible text* — the turn succeeds, `write_poem`
never fires, no error is raised, and the app appears to work while never writing
a poem. Thinking stays on (Opus 5's default); cost is controlled with
`output_config={"effort": "low"}` instead.

**Don't run uvicorn with `--reload`.** Sessions live in a dict in memory, so
every file save would wipe conversations in progress. A stale session id returns
404 and the page says so rather than silently starting over.

### On `MAX_TOKENS`

Opus 5 thinks by default, and thinking spends the same `max_tokens` budget as the
poem — a short poem can still spend a lot of budget getting there. `MAX_TOKENS` is
set to 64,000, and the call streams, so a high ceiling costs nothing when it goes
unused. If a poem ever does hit the ceiling, the CLI says so instead of silently
saving a truncated one, and a run that produces no text at all is never offered for
saving.

## What gets saved

Every poem writes a pair sharing a timestamp — automatically in chat mode, on `y`
in the CLI:

- `poems/<ts>.txt` — the poem, for reading
- `poems/<ts>.json` — everything that produced it, for comparing

The JSON holds the full `SYSTEM` text plus a 12-char SHA-256 of it, every question
with its answer, the resolved form, the assembled user prompt, the poem,
`stop_reason`, and token usage. Because the questions are recorded by key, sidecars
written before a question changed still say which question was actually asked.

Chat runs **add** keys rather than changing any, so every recipe below works across
a mixed v1/v2 corpus: `mode` (`cli` | `chat`), `persona_prompt` + its own
`persona_prompt_sha256`, the full `transcript`, `turns`, `chat_model`, and
`chat_usage`. Chat tokens are kept out of `usage` so the token-spend recipe still
means "what the poem cost".

In chat mode the `questions` list is synthesized from the tool's slots — nobody
asked them out loud, but the record is shaped identically so nothing downstream
has to care.

Group runs by prompt version:

```bash
# how many poems per SYSTEM version?
jq -r .system_prompt_sha256 poems/*.json | sort | uniq -c

# every poem written under one version
jq -r 'select(.system_prompt_sha256=="ec7e69afcba5") | .poem' poems/*.json

# token spend so far
jq -s 'map(.usage.output_tokens) | add' poems/*.json

# chat vs CLI, and which persona
jq -r '[.system_prompt_sha256, (.persona_prompt_sha256 // "cli")] | @tsv' poems/*.json | sort | uniq -c

# read a conversation back
jq -r '.transcript[] | "\(.role): \(.content)"' poems/<ts>.json
```

Edit `SYSTEM`, run the same inputs again, and the hash changes — that's your A/B.
One caveat now that there are two prompts: changing `PERSONA` changes what the
poet *receives*, so it confounds a `SYSTEM` comparison. Both hashes are recorded
so the confound is at least visible; group by the pair when comparing.

## Where to tune it

| Want to change | Edit |
|---|---|
| How Mr. Meter talks | `PERSONA` in `persona.py` |
| When he decides he has enough | `WRITE_POEM_TOOL` description, and `MAX_TURNS` for the backstop |
| Poem quality / voice | `SYSTEM` in `poem.py` — still the main lever |
| The CLI's questions | `QUESTIONS` list in `poem.py` |
| The form every poem takes | `FORM` string |
| Model or length | `MODEL`, `MAX_TOKENS`, `CHAT_MAX_TOKENS` |

## Files

| File | Purpose |
|---|---|
| `run.sh` | The entry point. venv + deps + launch, `--cli` for v1. |
| `app.py` | The server. Sessions, SSE, the handoff. |
| `persona.py` | Mr. Meter. |
| `poem.py` | The poet, and the v1 CLI. |
| `static/index.html` | The UI, entire. |
| `.env` | Your API key. Gitignored — never commit it. |
| `poems/` | Saved poems. Gitignored. |

## Next steps worth considering

- **Compare prompt versions properly** — the sidecars record which `SYSTEM` and
  which `PERSONA` produced each poem, but nothing scores them. A rating prompt
  would turn that record into an actual A/B.
- **Watch for Mr. Meter leaking into the poems.** His persona *requires* him to
  share anecdotes about himself, and the poet is told to use "the details they gave
  you". The transcript labels him `Interviewer:` and warns the poet off his lines,
  but that's a prompt-level mitigation, not a guarantee. If one of his forts or
  bicycles turns up in a poem, switch the transcript appendix to user turns only.
- **Nothing persists across a restart** except saved poems. Conversations live in
  a dict; reloading the page starts a new one.
