"""Excercise 5.1."""

import time

def GMT():
    secs_from_epoch = time.time()  # number of seconds from the epoch (1 January 1970)
    secs_per_day = 60 * 60 * 24  # number of seconds in a day
    days_from_epoch = secs_from_epoch / secs_per_day  # number of days since the epoch
    remainder = days_from_epoch % 1  # Remainder of the previous division. Today expressed as a floating point 0-1
    secs_today = remainder * secs_per_day

    seconds = secs_today % 60  # remainder in seconds of the division seconds of today / 60 (minutes)
    minutes = secs_today / 60 % 60  # remainder in minutes of the division seconds of today / 60 / 60
    hours = secs_today / 60 // 60  #

    print('The current time is', int(hours), 'hours', int(minutes), 'minutes', int(seconds), 'seconds', '(GMT)')
    print('The number of days since the epoch is', int(days_from_epoch))


"""Exercise 5.2."""

def check_fermat(a, b, c, n):
    # Fermat's Last Theorem: a**n + b**n = c**n
    a = a**n
    b = b**n
    c = c**n

    if n > 2 and c == a + b:
        print('Holy smokes, Fermat was wrong!')
    else:
        print("No, that doesn't work")

def fermat_last_theorem():
    print('a b and c must be positive integers; n must be a positive integer > 2')

    a = input('a = ')
    a = int(a)
    b = input('b = ')
    b = int(b)
    c = input('c = ')
    c = int(c)
    n = input('n = ')
    n = int(n)

    check_fermat(a, b, c, n)


"""Exercise 5.3."""

def is_triangle(a, b, c):

    # In mathematics, the triangle inequality states that for any triangle, the sum of the lengths of any two sides must
    # be greater than or equal to the length of the remaining side.
    if a > b + c:
        print('No')
    elif b > a + c:
        print('No')
    elif c > a + b:
        print('No')
    else:
        print('Yes')


def check_triangle():
    print('a b and c must be positive integers')

    a = input('a = ')
    a = int(a)
    b = input('b = ')
    b = int(b)
    c = input('c = ')
    c = int(c)

    is_triangle(a, b, c)


"""Exercise 5.4."""

def recurse(n, s):
    """This function call itself until reach the base case and returns the value of s. n must be a positive integer.
    """
    if n == 0:
        print(s)
    else:
        recurse(n-1, n+s)

# If I'd call recurse(-1, 0) I'd get a 'RecursionError:maximum recursion depth exceeded while calling a Python object'.
# The reason is that n would never reach the base case (n=0).


"""Exercise 5.5."""

def draw(t, length, n):
    if n == 0:
        return
    angle = 50
    t.fd(length*n)
    t.lt(angle)
    draw(t, length, n-1)
    t.rt(2*angle)
    draw(t, length, n-1)
    t.lt(angle)
    t.bk(length*n)