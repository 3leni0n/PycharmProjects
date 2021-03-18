import math

def mykoch(t, length, iteration):

    # angle = 60

    if iteration == 0:
        t.fd(length)
        return

    for angle in [60, -120, 60, 0]:
        mykoch(t, length / 3, iteration - 1)
        t.lt(angle)

    # length = length/3
    # koch(t, length, iteration - 1)
    # t.lt(angle)
    # koch(t, length, iteration - 1)
    # t.rt(2 * angle)
    # koch(t, length, iteration - 1)
    # t.lt(angle)
    # koch(t, length, iteration - 1)

# Test
t.pu()
t.bk(length/2)
t.pd()
mykoch(t, 500, 4)

def mykoch_snowflake(t, length, iteration):
    for i in range(3):
        mykoch(t, length, iteration)
        t.rt(120)


t.pu()
t.bk(length/2)
# Use Pythagoras Theorem to center the snowflake vertically
# b = math.sqrt((length/3)**2 - (length/3/2)**2)
t.lt(90)
t.fd(b)
t.rt(90)
t.pd()
mykoch_snowflake(bob, 500, 4)
turtle.mainloop()


def mycesaro(t, length, iteration):

    # angle = 60

    if iteration == 0:
        t.fd(length)
        return

    for angle in [85, -170, 85, 0]:
        mycesaro(t, length / 3, iteration - 1)
        # mycesaro(t, length, iteration - 1)
        t.lt(angle)

    # length = length/3
    # koch(t, length, iteration - 1)
    # t.lt(angle)
    # koch(t, length, iteration - 1)
    # t.rt(2 * angle)
    # koch(t, length, iteration - 1)
    # t.lt(angle)
    # koch(t, length, iteration - 1)

a = math.sin(math.radians(5)) * length/3

t.pu()
t.bk(length/3 + a)
t.pd()
mycesaro(t, 300, 4)

def mycesaro_antisnowflake(t, length, iteration):
    for i in range(4):
        mycesaro(t, length, iteration)
        t.lt(90)

mycesaro_antisnowflake(bob, 500, 4)
turtle.mainloop()