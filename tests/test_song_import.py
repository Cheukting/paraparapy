"""Tests for the suitability check, song import, the user library and the
Jamendo candidate search (faked API, no network)."""

import json
from pathlib import Path

import numpy as np
import pytest

from parapara.audio import find_songs, importer
from parapara.audio.song import Song, SongError
from parapara.audio.suitability import (
    TrackAnalysis,
    beat_clarity,
    check_track,
    judge_suitability,
    license_warnings,
)
from parapara.audio.synth import render_demo_song, write_wav

ROOT = Path(__file__).resolve().parents[1]
GOOD = TrackAnalysis(bpm=150.0, first_beat_s=0.5, drift_ms=4.0, beats_used=200, beats_detected=200,
                     beat_clarity=2.0, duration_s=90.0)  # fmt: skip


@pytest.fixture(scope="module")
def tracks(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("tracks")
    out = {}
    for name, kwargs in {"good": {}, "slow": {"bpm": 120.0}, "short": {"phrases": 0, "outro_counts": 0}}.items():
        song = render_demo_song(**kwargs)
        out[name] = d / f"{name}.wav"
        write_wav(out[name], song.samples, song.sample_rate)
    rng = np.random.default_rng(0)
    out["noise"] = d / "noise.wav"
    write_wav(out["noise"], rng.normal(0, 0.2, 44100 * 20), 44100)
    return out


# --- suitability rules -----------------------------------------------------------


def test_good_track_passes() -> None:
    report = judge_suitability(GOOD, "CC-BY-4.0")
    assert report.ok and report.warnings == ()
    assert report.metrics["duration_beats"] == int((90 - 0.5) * 150 / 60)


@pytest.mark.parametrize(
    ("changes", "text"),
    [
        ({"bpm": 128.0}, "needs about 140-160 BPM"),
        ({"bpm": 172.0}, "needs about 140-160 BPM"),
        ({"drift_ms": 22.0}, "tempo not steady"),
        ({"beats_used": 100}, "follow one constant tempo"),
        ({"beat_clarity": 1.05}, "beat not clear enough"),
        ({"duration_s": 12.0}, "too short"),
    ],
)
def test_rejections(changes, text) -> None:
    from dataclasses import replace

    report = judge_suitability(replace(GOOD, **changes), "CC-BY-4.0")
    assert not report.ok and any(text in r for r in report.rejections), report.rejections


@pytest.mark.parametrize(
    ("changes", "text"),
    [
        ({"bpm": 137.0}, "outside the ideal"),
        ({"drift_ms": 12.0}, "check by ear"),
        ({"beat_clarity": 1.4}, "hard to follow"),
        ({"duration_s": 25.0}, "short track"),
    ],
)
def test_warnings(changes, text) -> None:
    from dataclasses import replace

    report = judge_suitability(replace(GOOD, **changes), "CC-BY-4.0")
    assert report.ok and any(text in w for w in report.warnings), report.warnings


def test_licence_rules() -> None:
    assert judge_suitability(GOOD, "").ok  # no licence needed for personal use
    assert judge_suitability(GOOD, "").warnings == ()
    assert license_warnings("CC-BY-4.0") == []
    assert len(license_warnings("CC BY-NC-ND 4.0")) == 2
    assert "share-alike" in license_warnings("CC-BY-SA")[0]
    assert judge_suitability(None, "CC-BY").rejections[-1] == "no steady beat could be detected"


def test_beat_clarity_on_synthetic_envelopes() -> None:
    fps, bpm = 100.0, 150.0
    env = np.full(3000, 0.1)
    on_beat = env.copy()
    on_beat[(np.arange(0, 30, 0.4) * fps).astype(int) + 2] = 1.0  # spikes on the beats (+ detection lag)
    assert beat_clarity(on_beat, fps, 0.0, bpm) > 5
    assert beat_clarity(env, fps, 0.0, bpm) == pytest.approx(1.0)


def test_check_track_on_real_audio(tracks) -> None:
    good = check_track(tracks["good"], "own work")
    assert good.ok, good.rejections
    assert good.analysis.bpm == pytest.approx(150, abs=0.1) and good.analysis.beat_clarity > 1.5
    assert any("BPM" in r for r in check_track(tracks["slow"], "own work").rejections)
    assert any("too short" in r for r in check_track(tracks["short"], "own work").rejections)
    assert not check_track(tracks["noise"], "own work").ok


# --- import ------------------------------------------------------------------


INFO = importer.SongInfo("Demo", "Me")  # no licence or source: personal use


def test_import_writes_an_unverified_song(tracks, tmp_path) -> None:
    song, report = importer.import_song(tracks["good"], INFO, library=tmp_path)
    assert song.id == "me-demo" and song.directory == tmp_path / "me-demo"
    assert (tmp_path / "me-demo" / "me-demo.wav").exists()
    assert song.bpm == pytest.approx(150, abs=0.1)
    assert song.license.name == "unspecified (personal use)"
    assert not song.beat_map.verified and song.beat_map.method == "librosa"
    assert any("not verified" in w for w in song.warnings)
    with pytest.raises(SongError, match="already exists"):
        importer.import_song(tracks["good"], INFO, library=tmp_path)
    importer.import_song(tracks["good"], INFO, library=tmp_path, replace=True)


def test_rejected_import_writes_nothing(tracks, tmp_path) -> None:
    with pytest.raises(importer.ImportRejected) as err:
        importer.import_song(tracks["slow"], INFO, library=tmp_path)
    assert err.value.report.rejections and not any(tmp_path.iterdir())


def test_import_input_errors(tracks, tmp_path) -> None:
    with pytest.raises(SongError, match="format"):
        importer.import_song(tmp_path / "x.txt", INFO, library=tmp_path)
    with pytest.raises(SongError, match="no such file"):
        importer.import_song(tmp_path / "missing.wav", INFO, library=tmp_path)


def test_cli_exit_codes_and_catalog(tracks, tmp_path, capsys) -> None:
    lib = str(tmp_path / "lib")
    with pytest.raises(SystemExit) as rejected:
        importer.main([str(tracks["slow"]), "--library", lib])
    assert rejected.value.code == 2 and "REJECTED" in capsys.readouterr().out
    importer.main([str(tracks["good"]), "--catalog", "soulbringer-goa-trolls", "--library", lib])
    song = Song.load(tmp_path / "lib" / "soulbringer-goa-trolls")
    assert song.license.name == "CC-BY-4.0" and "Soulbringer" in song.attribution
    assert song.source.startswith("https://freemusicarchive.org/")
    importer.main([str(tracks["good"]), "--library", lib])  # nothing but the file
    assert Song.load(tmp_path / "lib" / "good").title == "good"  # title from the file name


def test_user_library_location(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PARAPARA_SONGS_DIR", str(tmp_path))
    assert importer.default_user_library() == tmp_path
    monkeypatch.delenv("PARAPARA_SONGS_DIR")
    assert "parapara" in str(importer.default_user_library())


def test_load_all_songs_merges_libraries(tracks, monkeypatch, tmp_path) -> None:
    builtin, user = tmp_path / "builtin", tmp_path / "user"
    monkeypatch.setenv("PARAPARA_SONGS_DIR", str(user))
    importer.import_song(tracks["good"], INFO, song_id="demo", library=builtin)
    importer.import_song(tracks["good"], INFO, library=user)
    assert set(importer.load_all_songs(builtin=builtin)) == {"demo", "me-demo"}
    importer.import_song(tracks["good"], INFO, song_id="demo", library=user)  # clashes with built-in
    with pytest.raises(SongError, match="exists in both"):
        importer.load_all_songs(builtin=builtin)


def test_fetch_demo_songs_imports_the_catalog(tracks, tmp_path) -> None:
    from parapara.audio import fetch_demo_songs

    payload = tracks["good"].read_bytes()
    downloads = []

    def fake_download(url, path):
        downloads.append(url)
        path.write_bytes(payload)

    out = fetch_demo_songs.fetch_demo_songs(tmp_path, ids=["soulbringer-goa-trolls"], download=fake_download)
    assert out["soulbringer-goa-trolls"].startswith("imported") and len(downloads) == 1
    song = Song.load(tmp_path / "soulbringer-goa-trolls")
    assert song.license.name == "CC-BY-4.0" and "Soulbringer" in song.attribution
    again = fetch_demo_songs.fetch_demo_songs(tmp_path, ids=["soulbringer-goa-trolls"], download=fake_download)
    assert again["soulbringer-goa-trolls"] == "already present" and len(downloads) == 1


def test_catalog_entries_are_complete() -> None:
    catalog = json.loads((ROOT / "assets/songs/catalog.json").read_text())
    assert 3 <= len(catalog["candidates"]) <= 4
    for e in catalog["candidates"]:
        assert e["license"]["name"] == "CC-BY-4.0" and e["license"]["url"].startswith("https://creativecommons.org/")
        assert e["title"] in e["attribution"] and e["artist"] in e["attribution"]
        assert 135 <= e["measured"]["bpm"] <= 165 and e["measured"]["drift_ms"] <= 15
    with pytest.raises(KeyError):
        importer.catalog_entry("nope")


def test_slugify() -> None:
    assert importer.slugify("Shane Ivers", "80s Motivational Chiptune!") == "shane-ivers-80s-motivational-chiptune"
    assert importer.slugify("!!!") == "song"


# --- Jamendo search (faked) ---------------------------------------------------------


def _result(i, license_url, allowed=True):
    return {"id": str(i), "name": f"Track {i}", "artist_name": "Artist", "shareurl": f"https://jamen.do/t/{i}",
            "license_ccurl": license_url, "audiodownload": f"https://dl/{i}.mp3",
            "audiodownload_allowed": allowed}  # fmt: skip


def test_jamendo_keeps_only_downloadable_plain_cc_by() -> None:
    results = [_result(1, "http://creativecommons.org/licenses/by/3.0/"),
               _result(2, "http://creativecommons.org/licenses/by-nc/3.0/"),
               _result(3, "http://creativecommons.org/licenses/by-sa/3.0/"),
               _result(4, "http://creativecommons.org/licenses/by/3.0/", allowed=False)]  # fmt: skip
    entries = find_songs.candidates(results)
    assert [e["id"] for e in entries] == ["jamendo-1"]
    assert entries[0]["license"]["name"] == "CC-BY" and "Track 1" in entries[0]["attribution"]


def test_jamendo_search_url() -> None:
    url = find_songs.search_url("abc", ("trance", "techno"), 20)
    assert url.startswith("https://api.jamendo.com/v3.0/tracks/?client_id=abc")
    assert "speed=high+veryhigh" in url and "fuzzytags=trance+techno" in url and "include=licenses+musicinfo" in url


def test_jamendo_find_downloads_and_checks(tracks, tmp_path) -> None:
    payload = tracks["good"].read_bytes()
    fake = {"results": [_result(7, "https://creativecommons.org/licenses/by/4.0/")]}
    entries = find_songs.find("abc", tmp_path, fetch=lambda url: fake,
                              download=lambda url, path: path.write_bytes(payload))  # fmt: skip
    (e,) = entries
    assert e["check"]["ok"] and e["check"]["measured"]["bpm"] == pytest.approx(150, abs=0.1)
