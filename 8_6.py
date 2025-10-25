#encoding: utf-8
from __future__ import division
from nodebox.graphics import *
import pymunk
import pymunk.pyglet_util
import random, math
import numpy as np

space = pymunk.Space()

def createBody(x,y,shape,*shapeArgs):
    body = pymunk.Body()
    body.position = x, y
    s = shape(body, *shapeArgs)
    s.mass = 1
    s.friction = 1
    space.add(body, s)
    return s

s0=createBody(300, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s0.score=0
s3=createBody(200, 300, pymunk.Poly, ((-20,-5),(-20,5),(20,15),(20,-15)))
s3.color = (0, 255, 0, 255)
s3.score=0
s3.body.Q=[[0, 0], [0, 0], [0, 0]]
s3.body.last_state = 0
s3.body.last_action = 0

s1=createBody(300, 200, pymunk.Circle, 10, (0,0))
S1 = [s1]
S2=[]
for i in range(1):
    s2=createBody(350, 250, pymunk.Circle, 10, (0,0))
    s2.color = (255, 0, 0, 255)
    S2.append(s2)

def getAngle(x,y,x1,y1):
    return math.atan2(y1-y, x1-x)

def getDist(x,y,x1,y1):
    return ((x-x1)**2+(y-y1)**2)**0.5

def inCircle(x,y,cx,cy,R):
    if (x-cx)**2+(y-cy)**2 < R**2:
        return True
    return False

def getNearest(s, S):
    x,y=s.position
    dist, nearest_s = 1000, S[0]
    for s_i in S:
        x1,y1=s_i.body.position
        d=getDist(x,y,x1,y1)
        if d<dist:
            dist=d
            nearest_s=s_i
    return nearest_s.body, dist

def state_action(b,s,a):
    v=100
    if s==0:
        if a==0: b.velocity=v*cos(b.angle), v*sin(b.angle)
    if s==1:
        b.angle=2*math.pi*random.random()
    if s==2:
        if random.random()>0.5: b.angle+=math.pi

def updateQ(body, reward):
    global S1, S2
    alpha = 0.5
    gamma = 0.9

    s = body
    x,y = s.position
    s1_body, dist1 = getNearest(s, S1)
    s2_body, dist2 = getNearest(s, S2)

    if dist1 < dist2:
        new_state = 0
    else:
        new_state = 1

    old_state = body.last_state
    action = body.last_action

    old_value = body.Q[old_state][action]
    future_optimal_value = np.max(body.Q[new_state])

    new_value = old_value + alpha * (reward + gamma * future_optimal_value - old_value)
    body.Q[old_state][action] = new_value
    # print body.Q # Розкоментуйте, щоб бачити процес навчання

def score():
    global s0, s3, S1, S2
    for s in S1:
        bx,by = s.body.position
        s0x,s0y=s0.body.position
        s3x,s3y=s3.body.position
        if not inCircle(bx,by,350,250,180):
            if getDist(bx,by,s0x,s0y)<getDist(bx,by,s3x,s3y):
                s0.score=s0.score+1
            else:
                s3.score=s3.score+1
                updateQ(s3.body, 1)
            s.body.position=random.randint(200,400),random.randint(200,300)

    for s in S2:
        bx,by = s.body.position
        s0x,s0y=s0.body.position
        s3x,s3y=s3.body.position
        if not inCircle(bx,by,350,250,180):
            if getDist(bx,by,s0x,s0y)<getDist(bx,by,s3x,s3y):
                s0.score=s0.score-1
            else:
                s3.score=s3.score-1
                updateQ(s3.body, -1)
            s.body.position=random.randint(200,400),random.randint(200,300)

def autoControl():
    global s3, S1, S2
    epsilon = 0.3

    s = s3.body
    
    if canvas.frame % 10 == 0:
        x, y = s.position
        s1_body, dist1 = getNearest(s, S1)
        s2_body, dist2 = getNearest(s, S2)
        
        if dist1 < dist2 and dist1 < 150:
            state = 0
            target_angle = getAngle(x, y, *s1_body.position)
        elif dist2 < dist1 and dist2 < 150:
            state = 1
            target_angle = getAngle(x, y, *s2_body.position)
        else:
            state = 2
            state_action(s, 2, 0)
            return

        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(s.Q[state])

        if action == 0:
            s.angle = target_angle
        elif action == 1:
            s.angle = target_angle + math.pi
        
        s.last_state = state
        s.last_action = action
    
    state_action(s, 0, 0)

def manualControl():
    v=10
    b=s0.body
    a=b.angle
    x,y=b.position
    vx,vy=b.velocity
    if canvas.keys.char=="a":
        b.angle-=0.1
    if canvas.keys.char=="d":
        b.angle+=0.1
    if canvas.keys.char=="w":
        b.velocity=vx+v*cos(a), vy+v*sin(a)
    if canvas.mouse.button==LEFT:
        b.angle=getAngle(x,y,*canvas.mouse.xy)
        b.velocity=vx+v*cos(a), vy+v*sin(a)

def simFriction():
    for s in [s0,s1,s3]+S2:
        s.body.velocity=s.body.velocity[0]*0.9, s.body.velocity[1]*0.9
        s.body.angular_velocity=s.body.angular_velocity*0.9

draw_options = pymunk.pyglet_util.DrawOptions()

def draw(canvas):
    canvas.clear()
    fill(0,0,0,1)
    text("%i %i"%(s0.score,s3.score),20,20)
    nofill()
    ellipse(350, 250, 350, 350, stroke=Color(0))
    manualControl()
    autoControl()
    score()
    simFriction()
    space.step(0.02)
    space.debug_draw(draw_options)

canvas.size = 700, 500
canvas.run(draw)