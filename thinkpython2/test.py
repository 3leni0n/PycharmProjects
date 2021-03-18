def countdown(n):
    if n <= 0:
        print('Blastoff!')
    else:
        print(n)
        countdown(n-1)

def print_n(s, n):
    if n <= 0:
        return
    print(s)
    print_n(s, n-1)

def do_n(print_n, n):
    print_n(s, n)

def recurse():
    recurse()


import time

