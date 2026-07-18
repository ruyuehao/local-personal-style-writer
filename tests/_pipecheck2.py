import os
names = os.listdir(r"\\.\pipe\\")
print("PIPE_HELD:", "local-style-writer" in names)
