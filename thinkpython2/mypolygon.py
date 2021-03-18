import turtle

# print(bob)

# bob.fd(100)
# bob.lt(90)
#
# bob.fd(100)
# bob.lt(90)
#
# bob.fd(100)
# bob.lt(90)
#
# bob.fd(100)

# for i in range(4):
#     print('Hello!')
    
# for i in range(4):
#     bob.fd(100)
#     bob.lt(90)
    
def square(t, length):
    for i in range(4):
        t.fd(length)
        t.lt(90)

# square(bob, 100)

def polygon(t, length, n):
    angle = 360 / n
    for i in range(n):
        t.fd(length)
        t.lt(angle)
        
# polygon(bob, length=70, n=7)

import math

def circle(t, r):
    circumference = 2 * math.pi * r
#    n = 50
    n = int(circumference / 3) + 3
    length = circumference / n
    polygon(t, length, n)
    
# circle(bob, 100)

def arc(t, r, angle):
    arc_length = 2 * math.pi * r * angle / 360
    n = int(arc_length / 3) + 1
    step_length = arc_length / n
    step_angle = angle / n
    
    for i in range(n):
        t.fd(step_length)
        t.lt(step_angle)
    
# arc(bob, 100, 360)

def polyline(t, length, n, angle):
    """Draws n line segments with the given length and
    angle (in degrees) between them. t is a turtle.
    """
    for i in range(n):
        t.fd(length)
        t.lt(angle)
#    print(t, length, n, angle)
        
def polygon_v2(t, length, n):
    angle = 360 / n
    polyline(t, length, n, angle)
    
# polygon_v2(bob, length=70, n=7)

def arc_v2(t, r, angle):
    arc_length = 2 * math.pi * r * angle / 360
    n = int(arc_length / 4) + 1
    step_length = arc_length / n
    step_angle = angle / n
    
    # making a slight left turn before starting reduces
    # the error caused by the linear approximation of the arc
    t.lt(step_angle/2)
    polyline(t, step_length, n, step_angle)
    t.rt(step_angle/2)
#    print(t, r, angle, arc_length, n, step_length, step_angle)
    
# arc_v2(bob, 100, 360)

def circle_v2(t, r):
    arc_v2(t, r, 360)
#    print(t, r)
    
# circle_v2(bob, 100)

# the following condition checks whether we are
# running as a script, in which case run the test code,
# or being imported, in which case don't.

if __name__ == '__main__':
    bob = turtle.Turtle()

    # draw a circle centered on the origin
    radius = 100
    bob.pu()
    bob.fd(radius)
    bob.lt(90)
    bob.pd()
    circle_v2(bob, radius)

    # wait for the user to close the window
    turtle.mainloop()

# stack diagram

"""

__main__:
    bob     --> turtle.Turtle
    radius  --> 100
    
polyline:
    t       --> bob
    length  --> 3.9766995615060674
    n       --> 158
    angle   --> 2.278481012658228
    
arc:
    t           --> bob
    r           --> 100
    angle       --> 360
    arc_length  --> 628.3185307179587
    n           --> 158
    step_length --> 3.9766995615060674
    step_angle  --> 2.278481012658228
    
circle:
    t   --> bob
    r   --> 100
    
"""