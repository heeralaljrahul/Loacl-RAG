# The story engine

A role-playing game that remembers. Every entry follows a fixed format, and
the campaign holds its own continuity — the clock, who is in the room, how
they feel about each other, what happened on day one — rather than hoping a
model keeps track across three hundred turns.

```
python play.py new reina --seed seeds/reina.json
python play.py serve reina                 # play in the browser
python play.py play reina                  # or in the terminal
```

---

## The problem this solves

A local 9B model has an 8,000-token context. An 850-word entry is about
1,200 tokens. So the model can see roughly the last four entries — and after
that, everything you have played is gone. Ask about the hairpin she lost on
day one and it will invent a new one, confidently.

Stuffing the transcript into a vector index does not fix it. Index every
entry and within a week the index is mostly "Reina nodded", "the rain kept
falling", "Min laughed" — a thousand chunks of atmosphere that match every
query weakly, so the one chunk that matters never wins. Retrieval gets
*worse* the more you play, which is the opposite of what you want.

Four mechanisms, in order of how much they matter.

### 1. State is injected. Memory is retrieved. They are different things.

Reina's height is not a memory. It is true on every turn, including turns
about something else entirely — which is exactly when a vector search will
fail to return it, because the conversation is about lunch.

So character sheets, the clock, the present cast, relationship standings and
active flags are **injected verbatim into every single prompt**, and the
narrator is told they override anything recalled. Retrieval spends whatever
prompt budget is left over, never the other way round.

A model that has forgotten Reina's height is broken. A model that missed one
callback to turn 12 has merely missed a callback.

### 2. Raw narration is never embedded.

After each entry, a second model pass — the archivist, with its own cold,
literal prompt — reads the new turn and returns structured JSON: one
summary, any durable facts, any events, who is now present, relationship
movements, time elapsed.

Only that distillate is indexed. A fact is written to stand alone —

> Min asked Reina to come to the Thursday showcase.

not "she asked her about it" — because a memory retrieved on turn 300
arrives with no surrounding context and has to be intelligible cold.

### 3. Summaries fold into arcs.

Every ten turns the summaries are compressed into one arc summary. At turn
400 recalling day one does not mean out-ranking four hundred sibling chunks;
it means matching one of forty arc summaries — and the recent arcs are
injected every turn regardless, so the spine of the campaign is always
present.

### 4. The engine owns the clock.

Ask a model to track dates across hundreds of entries and it will drift: a
repeated Tuesday, a skipped week, a quiet reset to the opening scene.

So the date and time are state. The engine advances them by the minutes the
archivist reports, formats the header, and rebuilds whatever the model wrote
to match. The model chooses only the mini-title.

---

## The format

Every entry is checked, not trusted:

```
📅 Monday, April 14th, 2025 — 🕛 3:35 PM — 🧹 The Podium

<800-900 words of narration>

**What will you do?**

1️⃣ 🔔 The final bell — end of the school day
2️⃣ 🧹 Cleaning duty — Group Four this week
3️⃣ 🍱 Someone suggests going somewhere after school
4️⃣ 💫 Min corners Reina about Thursday
5️⃣ 🚪 Genkan and home — Monday closes
```

A 9B model hits an exact 800-900 word window maybe two thirds of the time
and occasionally returns four options. Accepting that silently makes every
later turn worse — the transcript teaches the model that four options is
fine, and drift compounds. So the entry is measured and **one targeted
revision pass** is spent on precisely what was wrong: too short gets "open
it out, invent no new events"; four options gets the closing block rewritten
alone, not the whole entry.

If repair still fails the entry is kept, the shortfall is reported rather
than hidden, and five options reach you regardless. You are never handed a
dead end.

---

## Playing

Type `1`-`5` to take an option, or write anything you like. Custom input is
treated as fully canon — the narrator is instructed never to soften,
redirect or refuse a bold action, and to let the world react honestly.

The chosen option is never echoed back at you. It is interpreted and arrives
as the character's own decision, already in motion.

| command | |
|---|---|
| `new <slug> --seed seeds/reina.json` | create a campaign |
| `new <slug> --seed … --opening story.txt` | start from prose you already wrote |
| `serve <slug>` | play in the browser (recommended) |
| `play <slug>` | play in the terminal |
| `turn <slug> "<action>"` | one turn, non-interactive |
| `status <slug>` | clock, cast, standings, memory counts |
| `history <slug>` | recent turns and their summaries |
| `recall <slug> "<query>"` | search the campaign's memory directly |
| `lore <slug> <paths…>` | index a character bible into this campaign |
| `list` | all campaigns |

`recall` is the debugging tool. When an entry contradicts something you
established long ago, ask the archive first — if the memory is there, the
model ignored it (raise `GAME_RECALL_K`, or the fact is phrased too vaguely
to retrieve). If it is not there, the archivist never wrote it down, and
that is a different fix.

---

## Starting from your own writing

You already have story. Don't retype it:

```
python play.py new reina --seed seeds/reina.json --opening my_chapter.txt
```

That prose is stored as turn 1 and distilled into memory exactly as if it
had been played. Then add your character bible as searchable background:

```
python play.py lore reina "C:\path\to\character bible"
```

Lore is indexed rather than injected — it is background that matters when
the scene touches it, unlike a character sheet, which matters every turn.

---

## The seed file

`seeds/reina.json` is filled in from the scene you shared, and it is
deliberately thin: it records what that excerpt actually established and
invents nothing. Expand it — the sheets are the highest-leverage thing in
the whole system.

```json
{
  "clock": {"date": "2025-04-14", "time": "15:35", "location": "Class 1-A"},
  "characters": [
    {"slug": "reina", "name": "Reina", "protagonist": true, "present": true,
     "sheet": "195 cm tall. National champion. Heir to the largest company on Earth."}
  ],
  "relationships": [{"other": "mirajane", "label": "closest", "closeness": 9,
                     "note": "Will not let Reina lift a finger on cleaning duty."}],
  "flags": {"cleaning_duty": "Group Four this week"},
  "lore": ["Background that only matters when the scene touches it."]
}
```

**Keep sheets short.** They are injected on every turn, so every line costs
prompt budget forever. Five sharp lines about how someone talks and what
they want beats forty lines of biography — and a long sheet gets skimmed by
a small model anyway. Put the biography in `lore`, where it is retrieved
only when relevant.

Anything the archivist learns later — new characters, shifting standings,
new flags — is added automatically as you play.

---

## Tuning

Settings live in `bat\_env.bat`, or as `GAME_*` environment variables.

| variable | default | |
|---|---|---|
| `GAME_MIN_WORDS` / `GAME_MAX_WORDS` | 800 / 900 | the length window |
| `GAME_MAX_REPAIRS` | 1 | revision passes per turn; 0 is faster, looser |
| `GAME_RECALL_K` | 6 | memories injected per turn |
| `GAME_ARC_EVERY` | 10 | how often summaries fold into an arc |
| `GAME_VERBATIM_TURNS` | 1 | previous entries kept word-for-word |
| `RAG_NUM_CTX` | 8192 | raise `VERBATIM_TURNS` only if you raise this |

On a 10 GB RTX 3080 expect roughly 25-40 seconds per turn: the entry itself,
plus the archivist pass, plus a revision when the length missed.
`GAME_MAX_REPAIRS=0` cuts that materially and lets more short entries
through — your call which annoys you more.

---

## Known limitations

- **Prose quality is the model's, not the engine's.** The engine guarantees
  format, continuity and memory. A 9B model writes like a 9B model; the
  memory system is what makes it feel like it knows Reina, not the parameter
  count.
- **The archivist is the weak link.** If it fails to record a fact, that
  fact is not in memory. Check with `recall` after an important turn. A
  broken archivist response degrades to "this turn produced no structured
  memory" and never loses the entry itself.
- **Relationship closeness is a blunt 0-10 number.** It steers tone; it does
  not model anything real.
- **Long timeskips need a nudge.** The archivist reports minutes elapsed and
  is honest about hours, but "three months later" is better done by saying
  so in your action.
- **One protagonist per campaign.** Two AUs mean two campaigns, which is
  also what keeps their memories from bleeding into each other.
