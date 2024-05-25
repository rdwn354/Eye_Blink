import os
import sys

print("E-Blink".center(50,"-"))

print()
print("command")
print("1 ==> Open Camera")
print("2 ==> Start Program")
print("3 ==> Data Visualize")
print("4 ==> Review")
print("5 ==> exit")
command = input(str("Please insert command : "))
print()

if command == "1":
    os.system("python Camera.py")
elif command == "2":
    os.system("python Recording.py")
elif command == "3":
    os.system("python Calculate.py")
elif command == "4":
    os.system("python Graph .py")
elif command == "5":
    sys.exit()
else :
    print("Command not found :v")
    os.system("python3 run.py")

