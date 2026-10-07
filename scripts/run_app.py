from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"app"))
from app import app
app.run(host="127.0.0.1",port=5000,debug=False)
