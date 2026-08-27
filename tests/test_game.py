"""The engine end to end, with a scripted model — no Ollama, no network."""

from game.testing import configure


# -- seeding ---------------------------------------------------------------


def test_seeding_populates_state(campaign):
    status = campaign.status()
    assert status["characters"] == 4
    assert set(status["present"]) == {"Reina", "Mirajane", "Louis"}
    assert status["present"][0] == "Reina", "the protagonist is listed first"
    assert "Class 1-A" in status["location"]
    assert campaign.state.protagonist.name == "Reina"


def test_lore_is_indexed_as_memory_not_injected(campaign):
    assert campaign.archive.count().get("lore") == 1
    hits = campaign.archive.recall("classroom rituals", 3)
    assert any("rituals" in h.text for h in hits)


# -- one turn --------------------------------------------------------------


def test_a_turn_produces_a_conforming_entry(campaign):
    result = campaign.engine.play("Reina reads at the podium while they clean")
    assert result.n == 1
    assert 800 <= result.entry.words <= 900
    assert len(result.entry.choices) == 5
    assert result.text.startswith("📅 Monday, April 14th, 2025")
    assert "**What will you do?**" in result.text
    assert result.text.rstrip().splitlines()[-1].startswith("5️⃣")


def test_the_engine_owns_the_clock_not_the_model(campaign):
    """The scripted model always writes the same header. The engine must
    still advance time, because a model asked to track dates across hundreds
    of entries will drift."""
    campaign.engine.play("first")
    assert campaign.state.clock.time == "16:00"     # 15:35 + 25 minutes
    campaign.engine.play("second")
    assert campaign.state.clock.time == "16:25"
    assert campaign.state.turn(2).in_time == "16:00"


def test_header_is_rebuilt_from_state(campaign):
    campaign.state.clock = campaign.state.clock.advance(60 * 24)  # next day
    result = campaign.engine.play("go")
    assert "Tuesday, April 15th, 2025" in result.text
    assert "🧹 A Quiet Room" in result.text, "the model still picks the title"


def test_choosing_by_number_resolves_to_the_option_text(campaign):
    campaign.engine.play("open the scene")
    pending = campaign.state.pending_choices()
    result = campaign.engine.play("4")
    assert result.action == pending[3]
    assert campaign.state.turn(2).player_input == pending[3]


def test_custom_input_is_passed_through_untouched(campaign):
    campaign.engine.play("open")
    wild = "Reina flips the entire desk and walks out into the rain"
    result = campaign.engine.play(wild)
    assert result.action == wild


# -- format repair ---------------------------------------------------------


def test_a_short_entry_is_revised_not_accepted(campaign):
    configure(words=300)
    result = campaign.engine.play("go")
    assert result.repairs, "a 300-word entry must trigger a revision"
    assert 800 <= result.entry.words <= 900


def test_a_long_entry_is_revised_down(campaign):
    configure(words=1500)
    result = campaign.engine.play("go")
    assert 800 <= result.entry.words <= 900


def test_four_options_are_repaired_to_five(campaign):
    configure(choices=4)
    result = campaign.engine.play("go")
    assert len(result.entry.choices) == 5


def test_the_player_is_never_handed_a_dead_end(campaign):
    """Even when every repair fails, five options reach the player."""
    configure(choices=0)
    campaign.engine.cfg.max_repairs = 0
    result = campaign.engine.play("go")
    assert len(result.entry.choices) == 5
    assert result.notes


# -- memory ----------------------------------------------------------------


def test_a_turn_is_distilled_into_memory(campaign):
    result = campaign.engine.play("clean the room")
    assert result.distillate.summary
    assert campaign.state.turn(1).summary
    counts = campaign.archive.count()
    assert counts.get("summary") == 1
    assert counts.get("fact") == 1
    assert counts.get("event") == 1


def test_raw_narration_is_never_indexed(campaign):
    """The rule the whole design rests on. Index the transcript and within a
    week retrieval returns nothing but atmosphere."""
    campaign.engine.play("go")
    narration = campaign.state.turn(1).narration
    indexed = [r["text"] for r in
               campaign.app.store.db.execute("SELECT text FROM chunks")]
    assert narration not in indexed
    assert all(len(text) < 600 for text in indexed), "memories are distillate, not prose"


def test_archivist_updates_relationships_and_flags(campaign):
    configure(archivist={
        "summary": "Min asked Reina to come on Thursday.",
        "facts": ["Min asked Reina to come to the Thursday showcase."],
        "events": [], "present": ["reina", "min"],
        "relationships": [{"other": "min", "closeness": 8, "label": "close friend",
                           "note": "Asked Reina to Thursday."}],
        "flags": {"thursday_showcase": "invited"},
        "minutes_elapsed": 15, "location": "Corridor", "title": "The Ask",
    })
    campaign.engine.play("talk to Min")
    standing = {r["other"]: r for r in campaign.state.relationships()}
    assert standing["min"]["closeness"] == 8
    assert campaign.state.flags()["thursday_showcase"] == "invited"
    assert campaign.state.location == "Corridor"
    assert {c.name for c in campaign.state.characters(present_only=True)} == {"Reina", "Min"}


def test_unknown_slugs_from_the_archivist_are_ignored(campaign):
    configure(archivist={
        "summary": "Someone new appeared.", "facts": [], "events": [],
        "present": ["reina", "a_character_that_does_not_exist"],
        "relationships": [{"other": "nobody", "closeness": 9}],
        "flags": {}, "minutes_elapsed": 5, "location": "", "title": "x",
    })
    campaign.engine.play("go")
    assert [c.slug for c in campaign.state.characters(present_only=True)] == ["reina"]
    assert not [r for r in campaign.state.relationships() if r["other"] == "nobody"]


def test_a_broken_archivist_response_never_loses_the_turn(campaign):
    configure(archivist="I'm sorry, I can't help with that.")
    result = campaign.engine.play("go")
    assert campaign.state.turn_count == 1, "the entry is still committed"
    assert result.distillate.error
    assert campaign.state.turn(1).summary, "a fallback summary is still recorded"


# -- the long game ---------------------------------------------------------


def test_arcs_fold_on_schedule(campaign):
    for i in range(3):
        campaign.engine.play(f"turn {i}")
    arcs = campaign.state.arcs()
    assert len(arcs) == 1
    assert arcs[0]["from_turn"] == 1 and arcs[0]["to_turn"] == 3
    assert campaign.archive.count().get("arc") == 1


def test_day_one_is_still_recallable_after_many_turns(campaign):
    """The whole point. A detail established on turn 1 must still be
    retrievable once it is far outside any context window."""
    configure(archivist={
        "summary": "Reina lost a silver hairpin behind the piano in the music room.",
        "facts": ["Reina lost a silver hairpin behind the piano in the music room."],
        "events": [], "flags": {}, "minutes_elapsed": 10,
        "location": "Music room", "title": "The Hairpin",
    })
    campaign.engine.play("look behind the piano")

    configure(archivist=None)          # back to unrelated turns
    for i in range(25):
        campaign.engine.play(f"an ordinary turn {i}")
    assert campaign.state.turn_count == 26

    hits = campaign.archive.recall("silver hairpin piano music room", 5)
    assert any("hairpin" in h.text for h in hits), \
        "a turn-1 fact must survive 25 turns of unrelated play"


def test_the_prompt_always_carries_the_character_sheet(campaign):
    """Sheets are injected, never retrieved. If Reina's height depended on a
    vector search, every turn about something else would invent one."""
    from game.context import build_user_prompt

    for i in range(12):
        campaign.engine.play(f"turn {i}")
    prompt = build_user_prompt(campaign.state, campaign.game_cfg,
                               "something entirely unrelated", [])
    assert "195 cm tall" in prompt
    assert "WORLD STATE (authoritative)" in prompt
    assert "THE STORY SO FAR" in prompt, "arc summaries carry the early game"
    assert "cleaning_duty" in prompt


def test_context_stays_bounded_as_the_campaign_grows(campaign):
    from game.context import build_user_prompt

    sizes = []
    for i in range(20):
        campaign.engine.play(f"turn {i}")
        sizes.append(len(build_user_prompt(campaign.state, campaign.game_cfg, "x", [])))
    assert sizes[-1] < sizes[4] * 2.5, \
        "prompt size must not grow linearly with the transcript"


def test_opening_prose_becomes_memory(campaign):
    n = campaign.open_with("Reina sat by the window watching the rain come down.",
                           title="Opening")
    assert n == 1
    assert campaign.state.turn(1).summary
    assert campaign.archive.count().get("summary") == 1


# -- durability ------------------------------------------------------------


def test_a_turn_interrupted_before_archiving_is_repaired_next_session(campaign):
    """The entry is committed before the archivist runs. A crash in between
    used to leave a turn that existed in the story but not in memory —
    invisible to every later recall, forever."""
    campaign.engine.play("the turn that got interrupted")
    # Simulate the crash: undo the archiving, leaving the transcript intact.
    campaign.state.db.execute("UPDATE turns SET archived=0, summary=''")
    campaign.state.db.execute(
        "DELETE FROM chunks WHERE doc_id IN "
        "(SELECT id FROM documents WHERE path LIKE 'memory://summary%')")
    campaign.state.db.execute("DELETE FROM documents WHERE path LIKE 'memory://summary%'")
    campaign.state.db.commit()
    campaign.app.store.bump_generation()
    assert len(campaign.state.unarchived()) == 1

    repaired = campaign.engine.catch_up()
    assert repaired == [1]
    assert campaign.state.turn(1).summary
    assert campaign.archive.count().get("summary") == 1
    assert not campaign.state.unarchived()


def test_catch_up_runs_automatically_on_the_next_turn(campaign):
    campaign.engine.play("first")
    campaign.state.db.execute("UPDATE turns SET archived=0")
    campaign.state.db.commit()
    events = [e["type"] for e in campaign.engine.play_events("second")]
    assert "repaired" in events


def test_catching_up_an_old_turn_does_not_rewind_the_campaign(campaign):
    """Replaying a stale turn's state patch would drag the clock and the
    location back to where the story used to be."""
    campaign.engine.play("turn one")
    configure(archivist={
        "summary": "They moved to the roof.", "facts": [], "events": [],
        "present": ["reina"], "relationships": [], "flags": {},
        "minutes_elapsed": 30, "location": "Rooftop", "title": "Up High",
    })
    campaign.engine.play("go to the roof")
    assert campaign.state.location == "Rooftop"
    later_clock = campaign.state.clock.time

    campaign.state.db.execute("UPDATE turns SET archived=0 WHERE n=1")
    campaign.state.db.commit()
    configure(archivist={
        "summary": "Back in the classroom.", "facts": [], "events": [],
        "present": ["reina"], "relationships": [], "flags": {},
        "minutes_elapsed": 25, "location": "Class 1-A", "title": "Down Again",
    })
    campaign.engine.catch_up()
    assert campaign.state.location == "Rooftop", "an old turn must not move the story"
    assert campaign.state.clock.time == later_clock


def test_repaired_memory_is_dated_from_the_turn_not_from_now(campaign):
    """A memory recovered days of in-story time later must still be filed
    under the date it happened, or the archive's chronology quietly rots."""
    campaign.engine.play("first")
    campaign.state.clock = campaign.state.clock.advance(60 * 24 * 3)  # three days on
    for _ in range(3):
        campaign.engine.play("later")

    campaign.state.db.execute("UPDATE turns SET archived=0, summary='' WHERE n=1")
    campaign.state.db.execute(
        "DELETE FROM chunks WHERE doc_id IN (SELECT id FROM documents "
        "WHERE path LIKE 'memory://%/00001/%')")
    campaign.state.db.execute("DELETE FROM documents WHERE path LIKE 'memory://%/00001/%'")
    campaign.state.db.commit()
    campaign.app.store.bump_generation()

    campaign.engine.catch_up()
    heading = campaign.state.db.execute(
        "SELECT c.heading FROM chunks c JOIN documents d ON d.id=c.doc_id "
        "WHERE d.path LIKE 'memory://summary/00001/%'").fetchone()["heading"]
    assert heading.startswith("Monday, April 14th, 2025"), heading


def test_the_scripted_model_can_actually_be_reset(campaign):
    """Guards the test suite itself. `archivist=None` used to mean "leave it
    alone", so a test that thought it had gone back to ordinary turns was
    still replaying the previous script — and passed for the wrong reason."""
    from game.testing import configure, reset

    configure(archivist={"summary": "A very specific thing happened.",
                         "facts": [], "events": [], "flags": {},
                         "minutes_elapsed": 5, "location": "", "title": "x"})
    campaign.engine.play("one")
    assert campaign.state.turn(1).summary == "A very specific thing happened."

    configure(archivist=None)
    campaign.engine.play("two")
    assert campaign.state.turn(2).summary != "A very specific thing happened."
    reset()
