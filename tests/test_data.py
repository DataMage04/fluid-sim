from pathlib import Path

from fluid_sim.io.output import TextReport, write_to_output


def test_write_to_output_replaces_previous_file_and_adds_newlines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = Path("results")
    results.mkdir()

    output = results / "output.txt"
    output.write_text("old data\n", encoding="utf-8")

    write_to_output(["line one", "line two"])

    assert output.read_text(encoding="utf-8") == "line one\nline two\n"


def test_text_report_formats_sections_fields_and_tables():
    report = TextReport("TEST REPORT", subtitle="Case: unit-test")
    report.add_section("Grid")
    report.add_field("Cells", "8 x 4")
    report.add_table(
        ("Step", "Before", "After"),
        [(1, "1.0e-2", "1.0e-15")],
    )

    text = report.render()

    assert "TEST REPORT" in text
    assert "Case: unit-test" in text
    assert "GRID" in text
    assert "Cells" in text
    assert "Before" in text
    assert "1.0e-15" in text


def test_write_to_output_accepts_rendered_report(tmp_path):
    output = tmp_path / "case" / "output.txt"
    report = TextReport("REPORT")
    report.add_field("Status", "COMPLETED")

    write_to_output(report.render(), output)

    text = output.read_text(encoding="utf-8")
    assert "REPORT" in text
    assert "COMPLETED" in text
