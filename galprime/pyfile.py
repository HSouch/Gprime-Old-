import sys
import time

name = sys.argv[1]
if name == 'file1': # or name == 'file2':
    time.sleep(4)
print(name)