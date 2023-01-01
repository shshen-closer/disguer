import os
import sys


list_file = os.listdir('runs/')
os.system("python generate_test.py")

for i in range(len(list_file)):
    print(list_file[i], i)
    os.system("python test.py " + list_file[i] )
