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
    color = style_params['color']
    cap = style_params['cap']

    t.pen(pencolor=color, pensize=1) #np.exp(0.01 * (np.abs(size) - np.median(np.abs(b))))
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
    file_name = "../../img/connections.eps"
    turtle.setup(1500, 1500)
    r = 50
    t = turtle.Turtle()

    t.speed(0.0)
    t.penup()


    x = "r * 0.5 * np.random.rand() - r"
    SensoryInp = Neural_node(type='motor', name='SensoryInp', pos=(-6*(r) + eval(x), 4 *(r)+ eval(x)), r=r)
    Relay = Neural_node(type='both', name='Relay', pos=(-2 *(r)+ eval(x), 4 * (r) + eval(x)), r=r)
    Sw1 = Neural_node(type='both', name='Sw1', pos=(3.5*(r)+ eval(x), 2*(r)+ eval(x)), r=r)
    Sw2 = Neural_node(type='inhibitory', name='Sw2', pos=(0.5*(r)+ eval(x), 2*(r)+ eval(x)), r=r)
    Sw3 = Neural_node(type='excitatory', name='Sw3', pos=(1.75*(r)+ eval(x), -0.25*(r)+ eval(x)), r=r)
    PreI = Neural_node(type='excitatory', name='PreI', pos=(2*(r)+ eval(x), -3*(r)+ eval(x)), r=r)
    EarlyI = Neural_node(type='both', name='EarlyI', pos=(2*(r)+ eval(x), -7*(r)+ eval(x)), r=r)
    PostI = Neural_node(type='both', name='PostI', pos=(-2*(r)+ eval(x), -3*(r)+ eval(x)), r=r)
    AugE = Neural_node(type='inhibitory', name='AugE', pos=(-2*(r)+ eval(x), -7*(r)+ eval(x)), r=r)
    RampI  = Neural_node(type='excitatory', name='RampI', pos=(5*(r)+ eval(x), -5*(r)+ eval(x)), r=r)
    KF_t = Neural_node(type='excitatory', name='KF_t', pos=(-6 *(r)+ eval(x), -2 *(r)+ eval(x)), r=r)
    KF_p = Neural_node(type='excitatory', name='KF_p', pos=(-6*(r)+ eval(x), -4.5*(r)+ eval(x)), r=r)
    KF_r = Neural_node(type='inhibitory', name='KF_r', pos=(-3.5*(r)+ eval(x), 0.5*(r)+ eval(x)), r=r)
    HN  = Neural_node(type='motor', name='HN', pos=(6 * r + eval(x), -1*r + eval(x)), r=r)
    PN = Neural_node(type='motor', name='PN', pos=(8 * r + eval(x) , -5 * r + eval(x)), r=r)
    VN = Neural_node(type='motor', name='VN', pos=(6 * r + eval(x), -9 * r + eval(x)), r=r)

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
    data_path = str(get_project_root()) + "/data"
    file = open(f"{data_path}/rCPG_swCPG.json", "rb+")
    params = json.load(file)
    b = np.array(params["b"])
    c = np.array(params["c"])

    # 0- PreI   # 1 - EarlyI  # 2 - PostI
    # 3 - AugE  # 4 - RampI   # 5 - Relay
    # 6 - Sw 1  # 7 - Sw2     # 8 - Sw3
    # 9 - KF_t   # 10 - KF_p   # 11 - KF_r
    # 12 - M_HN  # 13- M_PN  # 14 - M_VN
    # 15 - KF_inh # 16 - NTS_inh # 17 - SI

    # write table of correpondance of nodes to numbers
    table = ["PreI", "EarlyI", "PostI", "AugE", "RampI", "Relay", "Sw1", "Sw2", "Sw3", "KF_t", "KF_p", "KF_r",
             "HN", "PN", "VN"] #, "KF_inh", "NTS_inh", "SI"]
    for i in range(len(table)):
        for j in range(len(table)):
            if b[i, j] != 0:
                strength = b[i, j]
                name_A = table[i]
                name_B = table[j]
                draw_connection(eval(name_A), eval(name_B), strength)
    draw_connection((SensoryInp), (Relay), 1)
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