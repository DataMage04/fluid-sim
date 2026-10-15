from pathlib import Path

from fluid_sim.io.output import TextReport, write_table_csv, write_to_output


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


def test_write_table_csv_writes_multiple_rows(tmp_path):
    output = tmp_path / "pressure_profile.csv"

    write_table_csv(
        [
            {"x_m": 0.25, "mean_pressure_Pa": 2.0, "mean_Cp": 4.0},
            {"x_m": 0.75, "mean_pressure_Pa": 1.0, "mean_Cp": 2.0},
        ],
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert "x_m,mean_pressure_Pa,mean_Cp" in text
    assert "0.25,2.0,4.0" in text
    assert "0.75,1.0,2.0" in text
