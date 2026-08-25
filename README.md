# poetry-ai

Chat with Mr. Meter about someone you love. He writes them a free-verse poem,
and reads it to you.

## What's new in v3

**Mr. Meter talks, and listens.** ElevenLabs voices him over a WebSocket the
browser holds itself, so his replies are spoken as they stream and the poem is
read stanza by stanza *while it's still being written* — the poem's own line
breaks become the pauses. The microphone is Chrome's, which keeps live interim
words appearing as you speak. Voice is optional: with no key configured the app
is exactly the silent text app it was.

**Dictation is punctuated before it's sent.** Browser speech recognition returns
no punctuation and every "um", which is ugly to read and worse as the verbatim
source material the poet sees. A one-second pass adds punctuation, drops fillers
and repairs stumbles without changing your words.

**The chat has faces.** A quill beside Mr. Meter, an anonymous face beside you —
both inline SVG drawn in `currentColor`, so there's still nothing to fetch and
they follow the light and dark palettes on their own.

**Mr. Meter got quieter.** The instruction that made him volunteer an anecdote in
every reply is gone, and the seven-angle question list collapsed to the two rules
that were doing the work — open a different door each time, never twice from the
same angle in a row.

Things that had to be learned the hard way, all now handled: `eleven_v3` and
`eleven_v3_conversational` are refused by the streaming socket (silently, with an
abnormal close); free-tier accounts cannot use library voices via the API;
"Write it now" mid-conversation ended the message list on an assistant turn,
which Opus 5 rejects; and the microphone had to be closed by *every* exit from a
turn, not just the mic button, or it transcribed Mr. Meter's own voice back into
the box.

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
| `static/index.html` | The whole UI — markup, styles, voice, avatars. No build step, no npm |
| `poem.py` | The poet + the v1 CLI. `iter_poem`, `build_prompt`, `save_run` |

### Voice

Optional, and off unless configured. Without the ElevenLabs keys the app behaves
exactly as before — silent text — so voice can never be the reason it breaks.

| Direction | Who does it | Cost |
|---|---|---|
| Mr. Meter speaks | ElevenLabs | ~$0.05 per 1,000 characters |
| You speak | The browser's own `SpeechRecognition` | free |


ElevenLabs only voices him; your microphone is handled by Chrome. That's deliberate
— browser recognition gives you live interim words as you talk, and you can see and
fix the transcript before sending. ElevenLabs' batch transcription would take that
away for accuracy you don't need on your own side of the conversation.

**Dictation is punctuated before it's sent.** Browser speech recognition
returns no punctuation and every "um", which is ugly to look at and worse as
source material for the poet. Tapping the mic to finish routes the text
through `POST /tidy` first — a one-second pass on Haiku that adds punctuation,
drops fillers and repairs stumbles without changing their words. It is
best-effort: a failure, a timeout, or an answer that no longer resembles what
they said all fall back to the raw transcript.

**The mic stays open until you tap it again.** Chrome ends recognition at the
first pause, which meant thinking mid-sentence cut you off and sent the
fragment. It now listens continuously and restarts itself through silences;
tapping the mic a second time is what ends the turn and sends.

**The browser holds its own socket to ElevenLabs.** `POST /voice-token` mints a
short-lived, single-use token server-side, and the page opens
`wss://.../stream-input` with it — so Claude's text goes straight from this page
into speech and no audio ever transits FastAPI. The API key stays on the server.

**The poem is spoken stanza by stanza while it's still being written.** Deltas are
buffered and flushed at blank lines, so the poem's own line breaks become the
pauses, and speech starts long before the last stanza exists.

Two knobs worth knowing, both in `static/index.html`:

- `chunk_length_schedule` — how much text ElevenLabs buffers before generating.
  The default first threshold (120 chars) is longer than most of Mr. Meter's
  replies, so audio would sit waiting; it's lowered to 50. Too low and prosody
  suffers, because it starts synthesising half-sentences. Tune by ear.
- `VOICE_MODES` — chat uses `eleven_flash_v2_5`, the poem uses
  `eleven_multilingual_v2` (better prosody for verse, double the per-character
  price at ~$0.10/1k). **Neither `eleven_v3` nor `eleven_v3_conversational`
  works here** — the streaming-input socket refuses both, and does it with an
  abnormal close carrying no error message. Verified: `flash_v2_5`,
  `multilingual_v2` and `turbo_v2_5` all work.

At ~1,650 spoken characters per session, voice roughly doubles the cost of a poem:
about $0.08 on top of $0.07 of Claude.

Worth knowing before you use it on family stories: **Chrome's speech recognition
uploads your audio to Google** to transcribe. It is not local. The mute button
silences Mr. Meter; there's no equivalent for the microphone except not using it.

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
| Dictation clean-up | `TIDY_MODEL` and `TIDY_PROMPT` in `app.py` |
| His voice | `ELEVENLABS_VOICE_ID` in `.env`; `VOICE_MODES` in `static/index.html` |

## Files

| File | Purpose |
|---|---|
| `run.sh` | The entry point. venv + deps + launch, `--cli` for v1. |
| `app.py` | The server. Sessions, SSE, the handoff. |
| `persona.py` | Mr. Meter. |
| `poem.py` | The poet, and the v1 CLI. |
| `static/index.html` | The UI, entire. |
| `.env` | Your API keys. Gitignored — never commit it. |
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
