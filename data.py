from pathlib import Path

def write_to_output(data:list):
    file_path = Path("results/output.txt")
    file_path.unlink(missing_ok=True)
    
    with open("results/output.txt", "a") as file:
        file.writelines(data)

