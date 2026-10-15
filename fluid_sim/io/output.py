from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


REPORT_WIDTH = 78


class TextReport:
    """Small helper for producing readable plain-text engineering reports."""

    def __init__(self, title: str, subtitle: str | None = None):
        self.lines: list[str] = [
            "=" * REPORT_WIDTH,
            title.center(REPORT_WIDTH),
            "=" * REPORT_WIDTH,
        ]
        if subtitle:
            self.lines.append(subtitle)

    def add_section(self, title: str) -> None:
        self.lines.extend(
            [
                "",
                title.upper(),
                "-" * REPORT_WIDTH,
            ]
        )

    def add_field(self, label: str, value: object, unit: str = "") -> None:
        suffix = f" {unit}" if unit else ""
        self.lines.append(f"{label:<32} {value}{suffix}")

    def add_fields(
        self,
        fields: Iterable[tuple[str, object] | tuple[str, object, str]],
    ) -> None:
        for field in fields:
            if len(field) == 2:
                label, value = field
                self.add_field(label, value)
            else:
                label, value, unit = field
                self.add_field(label, value, unit)

    def add_table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[object]],
    ) -> None:
        text_rows = [[str(value) for value in row] for row in rows]
        if not text_rows:
            self.lines.append("No data recorded.")
            return

        header_values = [str(header) for header in headers]
        widths = [
            max(
                len(header_values[column]),
                max(len(row[column]) for row in text_rows),
            )
            for column in range(len(header_values))
        ]

        def format_row(row: Sequence[str]) -> str:
            cells = []
            for column, value in enumerate(row):
                if column == 0:
                    cells.append(value.ljust(widths[column]))
                else:
                    cells.append(value.rjust(widths[column]))
            return "  ".join(cells)

        self.lines.append(format_row(header_values))
        self.lines.append(
            "  ".join("-" * width for width in widths)
        )
        for row in text_rows:
            self.lines.append(format_row(row))

    def add_lines(self, lines: Iterable[str]) -> None:
        self.lines.extend(lines)

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


def write_to_output(
    data: str | Iterable[str],
    file_path: str | Path = "results/output.txt",
) -> None:
    """Write a report, replacing any previous output file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, str):
        text = data if data.endswith("\n") else data + "\n"
    else:
        text = "\n".join(str(line) for line in data) + "\n"

    file_path.write_text(text, encoding="utf-8")
    print(f"Output written to {file_path}")
