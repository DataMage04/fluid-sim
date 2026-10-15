from pathlib import Path


def write_to_output(data: list[str], file_path="results/output.txt"):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.unlink(missing_ok=True)

    with file_path.open("a", encoding="utf-8") as file:
        file.writelines(f"{line}\n" for line in data)

    print(f"Output written to {file_path}")
