import turtle
import numpy as np
import json

from utils.gen_utils import get_project_root


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
    max_size = style_params['max_size']
    color = style_params['color']
    cap = style_params['cap']

    t.pen(pencolor=color, pensize=5 * size / max_size) #np.exp(0.01 * (np.abs(size) - np.median(np.abs(b))))
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

def draw_connection(node_A, node_B, strength, max_strength):
    if strength > 0:
        color = '#FF5555'
        con = 'ex'
        style_params = {'size':  np.abs(strength), 'color': color, 'cap': 'arrow', 'max_size' : max_strength}
        if node_A.type == 'motor':
            x_start = node_A.x
            y_start = node_A.y
        else:
            x_start = node_A.x_ex
            y_start = node_A.y_ex
    else:
        color = '#5555FF'
        con = 'inh'
        style_params = {'size' : np.abs(strength), 'color': color, 'cap' : 'circle', 'max_size' : max_strength}
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
    file_name = "../../img/connections.eps"
    turtle.setup(1500, 1500)
    r = 50
    t = turtle.Turtle()

    t.speed(0.0)
    t.penup()

    x = "r * 0.75 * np.random.rand() - r"
    SensoryInp = Neural_node(type='motor', name='SensoryInp', pos=(-6*(r) + eval(x), 4 *(r)+ eval(x)), r=r)
    Relay = Neural_node(type='both', name='Relay', pos=(-2 *(r)+ eval(x), 4 * (r) + eval(x)), r=r)
    Sw1 = Neural_node(type='both', name='Sw1', pos=(3.5*(r)+ eval(x), 2*(r)+ eval(x)), r=r)
    Sw2 = Neural_node(type='inhibitory', name='Sw2', pos=(0.5*(r)+ eval(x), 2*(r)+ eval(x)), r=r)
    NTS_drive = Neural_node(type='excitatory', name='NTS_drive', pos=(1.75*(r)+ eval(x), -0.25*(r)+ eval(x)), r=r)
    PreI = Neural_node(type='excitatory', name='PreI', pos=(2*(r)+ eval(x), -3*(r)+ eval(x)), r=r)
    EarlyI = Neural_node(type='both', name='EarlyI', pos=(2*(r)+ eval(x), -7*(r)+ eval(x)), r=r)
    PostI = Neural_node(type='both', name='PostI', pos=(-2*(r)+ eval(x), -3*(r)+ eval(x)), r=r)
    AugE = Neural_node(type='inhibitory', name='AugE', pos=(-2*(r)+ eval(x), -7*(r)+ eval(x)), r=r)
    RampI  = Neural_node(type='excitatory', name='RampI', pos=(5*(r)+ eval(x), -5*(r)+ eval(x)), r=r)
    KF_t = Neural_node(type='excitatory', name='KF_t', pos=(-6 *(r)+ eval(x), -2 *(r)+ eval(x)), r=r)
    KF_p = Neural_node(type='excitatory', name='KF_p', pos=(-6*(r)+ eval(x), -4.5*(r)+ eval(x)), r=r)
    KF_relay = Neural_node(type='inhibitory', name='KF_relay', pos=(-3.5*(r)+ eval(x), 0.5*(r)+ eval(x)), r=r)
    HN  = Neural_node(type='motor', name='HN', pos=(6 * r + eval(x), -1*r + eval(x)), r=r)
    PN = Neural_node(type='motor', name='PN', pos=(8 * r + eval(x) , -5 * r + eval(x)), r=r)
    VN = Neural_node(type='motor', name='VN', pos=(6 * r + eval(x), -9 * r + eval(x)), r=r)

    SensoryInp.draw(t)
    Relay.draw(t)
    Sw1.draw(t)
    Sw2.draw(t)
    NTS_drive.draw(t)
    PreI.draw(t)
    EarlyI.draw(t)
    PostI.draw(t)
    AugE.draw(t)
    RampI.draw(t)
    KF_t.draw(t)
    KF_p.draw(t)
    KF_relay.draw(t)
    HN.draw(t)
    PN.draw(t)
    VN.draw(t)

    # #draw connection
    x = 1
    y = 1
    population_names = ['PreI', 'EarlyI', "PostI", "AugE", "KF_t", "KF_p", "KF_relay", "NTS_drive",
                        "Sw1", "Sw2", "Relay", "RampI"]
    N = len(population_names)
    p = population_names
    W = np.zeros((N,N))
    W[p.index("PreI"), p.index("EarlyI")] = 0.40 # PreI -> EarlyI
    W[p.index("PreI"), p.index("PostI")] = 0.00 # PreI -> PostI
    W[p.index("PreI"), p.index("AugE")] = 0.00 # PreI -> AugE
    W[p.index("PreI"), p.index("RampI")] = 0.90 # PreI -> RampI

    W[p.index("EarlyI"), p.index("PreI")] = -0.08 # EarlyI -> PreI
    W[p.index("EarlyI"), p.index("PostI")] = -0.25 # EarlyI -> PostI
    W[p.index("EarlyI"), p.index("AugE")] = -0.63 # EarlyI -> AugE
    W[p.index("EarlyI"), p.index("KF_p")] = -0.10 # EarlyI -> KF_p
    W[p.index("EarlyI"), p.index("Sw1")] = -0.003 # EarlyI -> Sw1
    W[p.index("EarlyI"), p.index("RampI")] = -0.15 # EarlyI -> RampI

    W[p.index("PostI"), p.index("PreI")] = -0.35 # PostI -> PreI
    W[p.index("PostI"), p.index("EarlyI")] = -0.22 # PostI -> EarlyI
    W[p.index("PostI"), p.index("AugE")] = -0.36 # PostI -> AugE
    W[p.index("PostI"), p.index("RampI")] = -0.50 # PostI -> RampI

    W[p.index("AugE"), p.index("PreI")] = -0.30 # AugE -> PreI
    W[p.index("AugE"), p.index("EarlyI")] = -0.43 # AugE -> EarlyI
    W[p.index("AugE"), p.index("PostI")] = -0.06 # AugE -> PostI
    W[p.index("AugE"), p.index("RampI")] = -0.50 # AugE -> RampI

    W[p.index("KF_t"), p.index("PreI")] = +0.16 * x # KF_t -> PreI
    W[p.index("KF_t"), p.index("EarlyI")] = +0.66 * x # KF_t -> EarlyI
    W[p.index("KF_t"), p.index("PostI")] = +1.10 * x # KF_t -> PostI
    W[p.index("KF_t"), p.index("AugE")] = +0.72 * x # KF_t -> AugE
    W[p.index("KF_t"), p.index("KF_relay")] = +0.7 * x  # KF_t -> KF_relay

    W[p.index("KF_p"), p.index("PreI")] = +0.00 * x # KF_p -> PreI
    W[p.index("KF_p"), p.index("EarlyI")] = +0.00 * x# KF_p -> EarlyI
    W[p.index("KF_p"), p.index("PostI")] = +0.60 * x# KF_p -> PostI
    W[p.index("KF_p"), p.index("AugE")] = +0.00 * x# KF_p -> AugE

    W[p.index("KF_relay"), p.index("Sw1")] = -0.09  # KF_relay -> Sw1
    W[p.index("KF_relay"), p.index("Sw2")] = -0.05  # KF_relay -> Sw2

    W[p.index("NTS_drive"), p.index("PostI")] = 0.42  # NTS_drive -> PostI


    W[p.index("Sw1"), p.index("PreI")] = -0.30   # Sw1 -> PreI
    W[p.index("Sw1"), p.index("EarlyI")] = -0.17   # Sw1 -> EarlyI
    W[p.index("Sw1"), p.index("AugE")] = -0.15   # Sw1 -> AugE
    W[p.index("Sw1"), p.index("Sw2")] = -0.56  # Sw1 -> Sw2
    W[p.index("Sw2"), p.index("Sw1")] = -0.39  # Sw2 -> Sw1

    W[p.index("Relay"), p.index("PreI")] = -0.30 * y # Relay -> PreI
    W[p.index("Relay"), p.index("EarlyI")] = -0.30 * y  # Relay -> EarlyI
    W[p.index("Relay"), p.index("AugE")] = -0.30 * y # Relay -> AugE
    W[p.index("Relay"), p.index("RampI")] = -0.30 * y # Relay -> RampI
    W[p.index("Relay"), p.index("KF_t")] = 0.15 * x * y  # Relay -> KF_t
    W[p.index("Relay"), p.index("KF_p")] = 0.15 * x * y # Relay -> KF_p
    W[p.index("Relay"), p.index("Sw1")] = 0.74 * y  # Relay -> Sw1
    W[p.index("Relay"), p.index("Sw2")] = 0.71* y  # Relay -> Sw2
    W[p.index("Relay"), p.index("NTS_drive")] = 0.15 * y  # Relay -> NTS_drive

    drives = np.zeros((3, N))
    # other
    drives[0, p.index("KF_t")] = 0.81 * x  # -> KF_t
    drives[0, p.index("KF_p")] = 0.50 * x  # -> KF_p
    drives[0, p.index("NTS_drive")] = 0.68 * y  # -> NTS_drive
    drives[0, p.index("Sw1")] = 0.33 * y  # -> Sw1
    drives[0, p.index("Sw2")] = 0.45 * y  # -> Sw2

    # BotC
    drives[1, p.index("PreI")] = 0.09  # -> PreI
    drives[1, p.index("EarlyI")] = 0.27  # -> EarlyI
    drives[1, p.index("PostI")] = 0.00  # -> PostI
    drives[1, p.index("AugE")] = 0.42  # -> AugE
    drives[1, p.index("RampI")] = 0.50  # -> RampI

    c = drives
    b = W

    # write table of correpondance of nodes to numbers
    table = population_names
    for i in range(len(table)):
        for j in range(len(table)):
            if b[i, j] != 0:
                strength = b[i, j]
                name_A = table[i]
                name_B = table[j]
                draw_connection(eval(name_A), eval(name_B), strength, max_strength=np.max(np.abs(b)))
    draw_connection((SensoryInp), (Relay), 1, max_strength=np.max(np.abs(b)))

    draw_connection((PreI), (PN), 0.1, max_strength=np.max(np.abs(b)))
    draw_connection((RampI), (PN), 0.9, max_strength=np.max(np.abs(b)))
    draw_connection((PreI), (HN), 0.7, max_strength=np.max(np.abs(b)))
    draw_connection((RampI), (HN), 0.15, max_strength=np.max(np.abs(b)))
    draw_connection((Sw1), (HN), 0.35, max_strength=np.max(np.abs(b)))
    draw_connection((RampI), (VN), 0.75, max_strength=np.max(np.abs(b)))
    draw_connection((Sw1), (VN), 0.8, max_strength=np.max(np.abs(b)))
    draw_connection((PostI), (VN), 0.6, max_strength=np.max(np.abs(b)))
    draw_connection((KF_p), (VN), 0.4, max_strength=np.max(np.abs(b)))


    turtle.hideturtle()
    ts = turtle.getscreen()
    ts.getcanvas().postscript(file=file_name)

    from PIL import Image
    TARGET_BOUNDS = (1024, 1024)

    # Load the EPS at 10 times whatever size Pillow thinks it should be
    # (Experimentaton suggests that scale=1 means 72 DPI but that would
    #  make 600 DPI scale=8⅓ and Pillow requires an integer)
    pic = Image.open(file_name)
    pic.load(scale=10)

    # Ensure scaling can anti-alias by converting 1-bit or paletted images
    if pic.mode in ('P', '1'):
        pic = pic.convert("RGB")

    # Calculate the new size, preserving the aspect ratio
    ratio = min(TARGET_BOUNDS[0] / pic.size[0],
                TARGET_BOUNDS[1] / pic.size[1])
    new_size = (int(pic.size[0] * ratio), int(pic.size[1] * ratio))

    # Resize to fit the target size
    pic = pic.resize(new_size, Image.ANTIALIAS)

    # Save to PNG
    pic.save(file_name.split(".eps")[0] + ".png")