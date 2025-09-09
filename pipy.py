import sys
import numpy as np

print("Python:", sys.version)
print("NumPy:", np.__version__)

try:
    import mordred
    print("Mordred import: OK")
except Exception as e:
    print("Mordred import error:", e)
