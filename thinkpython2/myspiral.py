import turtle

import math

def draw_spiral(t, n):

    # Parameters
    theta = 0
    a = 0.1
    b = 0.001

    for i in range(n):
        t.fd(1)
        r = a + b * theta
        r = 1 / r
        t.lt(r)
        theta = theta + r

# create the world and bob
bob = turtle.Turtle()
draw_spiral(bob, 1000)

turtle.mainloop()