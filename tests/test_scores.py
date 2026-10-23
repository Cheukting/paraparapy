"""The local score board (parapara.scores)."""

from datetime import datetime, timedelta

from parapara import scores


def _summary(score: int, routine: str = "rave-easy", early: bool = False) -> dict:
    return {"song_id": "rave", "routine_id": routine, "difficulty": "easy", "score": score,
            "accuracy_pct": score / 500, "max_combo": 10, "ended_early": early}  # fmt: skip


def test_ranks_best_first_and_keeps_levels_apart(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAPARA_DATA_DIR", str(tmp_path))
    t0 = datetime(2026, 10, 23, 18, 0)
    first = scores.record(_summary(30000), t0)
    assert first.rank == 1 and first.new_best and first.previous_best is None
    second = scores.record(_summary(20000), t0 + timedelta(minutes=5))
    assert second.rank == 2 and not second.new_best and second.previous_best == 30000
    third = scores.record(_summary(40000), t0 + timedelta(minutes=10))
    assert third.new_best and third.previous_best == 30000 and third.entries == 3
    assert [e["score"] for e in scores.top("rave", "rave-easy")] == [40000, 30000, 20000]
    scores.record(_summary(99999, routine="rave-hard"), t0)  # another level: its own board
    assert scores.best("rave", "rave-easy")["score"] == 40000
    assert scores.best("rave", "rave-hard")["score"] == 99999
    assert scores.best("other", "x") is None


def test_ended_early_is_not_ranked_and_board_is_capped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAPARA_DATA_DIR", str(tmp_path))
    assert scores.record(_summary(10**6, early=True)) is None
    assert scores.top("rave", "rave-easy") == []
    t0 = datetime(2026, 10, 23, 18, 0)
    for i in range(scores.KEEP + 5):
        scores.record(_summary(1000 + i), t0 + timedelta(seconds=i))
    board = scores.top("rave", "rave-easy", n=100)
    assert len(board) == scores.KEEP and board[-1]["score"] == 1005  # the lowest five dropped
    (tmp_path / "scores.json").write_text("not json")
    assert scores.top("rave", "rave-easy") == []  # a broken file starts a new board
