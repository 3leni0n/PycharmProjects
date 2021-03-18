import turtle

from mypolygon import arc_v2

bob = turtle.Turtle()

def myflower(t, r, n, angle):
    """Draw a flower by repeating arc shaped petals

    t = turtle
    r = radius
    n = n petals
    angle = angle of the arc (in degrees); width of the petal
    """
    for i in range(n):
        arc_v2(t, r, angle)
        t.lt(180 - angle)
        arc_v2(t, r, angle)
        t.lt(180 - angle)
        t.lt(360 / n)

turtle.mainloop()
turtle.clear()
turtle.bye()
bob.reset()