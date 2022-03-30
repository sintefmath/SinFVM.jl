import re
import sys

with open(sys.argv[1]) as inputfile:
    for line in inputfile:
        line = re.sub(r'(\d+)f0', r'\1', line)
        line = re.sub(r'Float32\((\d+.\d+)\)', r'\1', line)
        line = re.sub(r'Int32\((\d+)\)', r'\1', line)
        
        sys.stdout.write(line)
        