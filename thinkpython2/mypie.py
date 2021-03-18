import turtle
import math

from mypolygon import polygon_v2

bob = turtle.Turtle()

def mypie(t, length, n):
    """This piece of shit code finds fucking finally the fucking radius of the imaginary circle in which the fucking
     polygon is fucking inscribed. It does so by fucking splitting the target isosceles fucking triangle into 2 fucking
     right fucking triangles. Then uses sine to find c. Note that needs to input length/2 and internal angle/2 as well,
     BITCH. Using polygon_v2.
    """
    angle = 360 / n
    radians = angle / 180 * math.pi
    external_angle = 180 - angle
    c = (length / 2) / math.sin(radians / 2)    # from sine formula

    polygon_v2(t, length, n)
    t.lt(external_angle / 2)
    t.fd(c)

    for i in range(n - 2):
        t.rt(external_angle)
        t.fd(c)
        t.lt(180)
        t.fd(c)

    t.rt(external_angle)
    t.fd(c)

angle = 360 / n
radians = angle / 180 * math.pi

def isosceles(t, r, angle):
    """Draws an isosceles triangle.

    t = turtle
    r = radius
    angle = angle of the unique one
    """
    a = r * math.sin(radians / 2)
    t.rt(angle / 2)
    t.fd(r)
    t.lt((90 + angle / 2))
    t.fd(2 * a)     # a is only the length of half side
    t.lt((90 + angle / 2))
    t.fd(r)

def mypie_v2(t, r, n):
    """Draws a pie within a regular polygon using triangles. It does not require polygon_v2.
    """

    for i in range(n):
        isosceles(t, r, angle)
        t.lt(180 + angle / 2)