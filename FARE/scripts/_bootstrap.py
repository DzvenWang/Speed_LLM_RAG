from pathlib import Path
import sys
folder=str(Path(__file__).resolve().parents[1])
if folder not in sys.path:sys.path.insert(0,folder)
