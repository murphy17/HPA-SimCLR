import os

bash = lambda s: os.popen(s).read().rstrip().split('\n')