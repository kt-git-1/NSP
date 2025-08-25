import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run optimization pipeline and produce shift_summary.csv."""
    root = Path(__file__).resolve().parent
    (root / "output").mkdir(exist_ok=True)
    scripts = ["optimize_1.py", "optimize_2.py", "optimize_3.py"]
    for script in scripts:
        try:
            subprocess.run([sys.executable, str(root / script)], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error while executing {script}: {exc}")
            raise


if __name__ == "__main__":
    main()
