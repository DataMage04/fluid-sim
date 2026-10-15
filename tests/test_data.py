from pathlib import Path

from data import write_to_output


def test_write_to_output_replaces_previous_file_and_adds_newlines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = Path("results")
    results.mkdir()

    output = results / "output.txt"
    output.write_text("old data\n", encoding="utf-8")

    write_to_output(["line one", "line two"])

    assert output.read_text(encoding="utf-8") == "line one\nline two\n"
