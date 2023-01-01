import os
import sys
#filerun = sys.argv[1]

os.system("python generate_data.py")

for i in range(10):
    os.system("python train1.py " +str(i))
for i in range(10):
    os.system("python train1.py " +str(i))



