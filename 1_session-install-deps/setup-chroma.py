import subprocess
import os

if os.getenv("VECTOR_DB") == "CHROMA":
    print(subprocess.run(["sh 0_install_prerequisites/setup-chroma.sh"], shell=True))