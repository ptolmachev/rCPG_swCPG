import turtle
import numpy as np
import json


class Neural_node():
    def __init__(self, type, name, pos, r):
        self.type = type
        self.name = name
        self.r = r
        self.pos = pos
        if (type == 'excitatory'):
            self.x_ex = pos[0]
            self.y_ex = pos[1]
            self.color = '#FFCCCC'
        elif (type == 'inhibitory'):
            self.x_inh = pos[0]
            self.y_inh = pos[1]
            self.color = '#CCCCFF'
        elif type == 'both':
            self.x_inh = pos[0] + self.r / 4.0
            self.y_inh = pos[1] + self.r / 4.0
            self.x_ex = pos[0] - self.r / 4.0
            self.y_ex = pos[1] - self.r / 4.0
            self.color_ex = '#FFCCCC'
            self.color_inh = '#CCCCFF'
        elif type == 'motor':
            self.x = pos[0]
            self.y = pos[1]
            self.color = '#AAAA44'
        else:
            raise NameError

    def draw(self, t):
        if self.type != 'both':
            t.setposition(self.pos)
            t.fillcolor(self.color)
            t.begin_fill()
            t.pendown()
            t.circle(self.r)
            t.end_fill()
            t.penup()
            if self.type == 'excitatory':
                x = self.x_ex
                y = self.y_ex
            elif self.type =='inhibitory':
                x = self.x_inh
                y = self.y_inh
            else:
                x = self.x
                y = self.y
            t.setposition((x, y + self.r - int(self.r/4)))
            t.write(self.name, align="center", font=("Arial", int(self.r/2), "normal"))
            t.setposition((x, y))
        else:
            #draw excitatory part
            t.setposition((self.x_ex, self.y_ex))
            t.fillcolor(self.color_ex)
            t.begin_fill()
            t.pendown()
            t.circle(self.r)
            t.end_fill()
            t.penup()

            #the inhibitory one
            t.setposition((self.x_inh, self.y_inh))
            t.fillcolor(self.color_inh)
            t.begin_fill()
            t.pendown()
            t.circle(self.r)
            t.end_fill()
            t.penup()

            t.setposition((self.x_inh, self.y_inh + self.r - int(self.r/4)))
            t.write(self.name, align="center", font=("Arial", int(self.r/2), "normal"))
            t.setposition((self.x_inh, self.y_inh))


def draw_line(t, pos1, pos2, style_params):
    size = style_params['size']
    color = style_params['color']
    cap = style_params['cap']

    t.pen(pencolor=color, pensize=np.exp(1.5 * (np.abs(size) - np.median(np.abs(b)))))
    t.penup()
    t.setposition(pos1)
    a = t.towards(pos2)
    t.setheading(a)
    t.pendown()
    t.forward(t.distance(pos2))
    t.color(color)
    t.shape(cap)
    t.shapesize(size)
    t.stamp()
    t.penup()

def draw_connection(node_A, node_B, strength):
    if strength > 0:
        color = '#FF5555'
        con = 'ex'
        style_params = {'size':  np.abs(strength), 'color': color, 'cap': 'arrow'}
        if node_A.type == 'motor':
            x_start = node_A.x
            y_start = node_A.y
        else:
            x_start = node_A.x_ex
            y_start = node_A.y_ex
    else:
        color = '#5555FF'
        con = 'inh'
        style_params = {'size' : np.abs(strength), 'color': color, 'cap' : 'circle'}
        x_start = node_A.x_inh
        y_start = node_A.y_inh

    if node_B.type == 'excitatory':
        x = node_B.x_ex
        y = node_B.y_ex + node_B.r
    if node_B.type == 'inhibitory':
        x = node_B.x_inh
        y = node_B.y_inh + node_B.r
    if node_B.type == 'motor':
        x = node_B.x
        y = node_B.y + node_B.r
    if node_B.type == 'both':
        x = node_B.x_inh
        y = node_B.y_inh + node_B.r

    t.setposition((x_start, y_start + node_A.r))
    angle = t.towards(x,y)
    t.setheading(angle)
    t.penup()
    t.forward(node_A.r)
    x_d = x - 1.05 * node_B.r * np.cos(np.pi * angle / 180.0)
    y_d = y - 1.05 * node_B.r * np.sin(np.pi * angle / 180.0)
    draw_line(t, t.position(), (x_d, y_d), style_params)

if __name__ == '__main__':
    turtle.setup(1500, 1500)
    r = 50
    t = turtle.Turtle()

    t.speed(0.0)
    t.penup()

    SensoryInp = Neural_node(type='motor', name='SensoryInp', pos=(-6*r, 4 * r), r=r)
    Relay = Neural_node(type='both', name='Relay', pos=(-2 * r, 4 * r), r=r)
    Sw1 = Neural_node(type='both', name='Sw1', pos=(3.5*r, 2*r), r=r)
    Sw2 = Neural_node(type='inhibitory', name='Sw2', pos=(0.5*r, 2*r), r=r)
    Sw3 = Neural_node(type='excitatory', name='Sw3', pos=(1.75*r, -0.25*r), r=r)
    PreI = Neural_node(type='excitatory', name='PreI', pos=(2*r, -3*r), r=r)
    EarlyI = Neural_node(type='both', name='EarlyI', pos=(2*r, -7*r), r=r)
    PostI = Neural_node(type='both', name='PostI', pos=(-2*r, -3*r), r=r)
    AugE = Neural_node(type='inhibitory', name='AugE', pos=(-2*r, -7*r), r=r)
    RampI  = Neural_node(type='excitatory', name='RampI', pos=(5*r, -5*r), r=r)
    KF_t = Neural_node(type='excitatory', name='KF_t', pos=(-6 * r, -2 * r), r=r)
    KF_p = Neural_node(type='excitatory', name='KF_p', pos=(-6 * r, -4.5 * r), r=r)
    KF_r = Neural_node(type='inhibitory', name='KF_r', pos=(-3.5 * r, 0.5 * r), r=r)
    HN  = Neural_node(type='motor', name='HN', pos=(6 * r, -1*r), r=r)
    PN = Neural_node(type='motor', name='PN', pos=(8 * r, -5 * r), r=r)
    VN = Neural_node(type='motor', name='VN', pos=(6 * r, -9 * r), r=r)

    SensoryInp.draw(t)
    Relay.draw(t)
    Sw1.draw(t)
    Sw2.draw(t)
    Sw3.draw(t)
    PreI.draw(t)
    EarlyI.draw(t)
    PostI.draw(t)
    AugE.draw(t)
    RampI.draw(t)
    KF_t.draw(t)
    KF_p.draw(t)
    KF_r.draw(t)
    HN.draw(t)
    PN.draw(t)
    VN.draw(t)

    # #draw connection
    # # load gson
    file = open("rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])

    # 0- PreI   # 1 - EarlyI  # 2 - PostI
    # 3 - AugE  # 4 - RampI   # 5 - Relay
    # 6 - NTS1  # 7 - NTS2    # 8 - NTS3
    # 9 - KF_t   # 10 - KF_p    # 11 - M_HN
    # 12- M_PN  # 13 - M_VN   # 14 - KF_inh
    # 15 - NTS_inh
    # write table of correpondance of nodes to numbers
    table = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p", "KF_r", "HN", "PN", "VN"]
    for i in range(len(table)):
        for j in range(len(table)):
            if b[i, j] != 0:
                strength = b[i, j]
                name_A = table[i]
                name_B = table[j]
                draw_connection(eval(name_A), eval(name_B), strength)
    draw_connection(SensoryInp, Relay, 0.4)

    ts = turtle.getscreen()
    ts.getcanvas().postscript(file="../img/Model_11_02_2020/connections.eps")