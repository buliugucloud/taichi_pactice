import taichi as ti

# import numpy as np

ti.init(arch=ti.cpu)
# ti.init(arch=ti.cpu, debug=True)

res_x, res_y = 640, 640  # 屏幕分辨率
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
gui = ti.GUI('N-body Game_v2', res=(res_x, res_y), fast_gui=False)
# gui.fps_limit = 24
obj_num = 20
my_pos = ti.Vector.field(2, ti.f32, ())  # 自己的位置
obj_pos = ti.Vector.field(2, ti.f32, obj_num)  # 障碍物的位置
obj_vel = ti.Vector.field(2, ti.f32, obj_num)  # 障碍物的速度
obj_force = ti.Vector.field(2, ti.f32, obj_num)  # 障碍物所受的力

# constants
G = 1
PI = 3.1415926
my_mass = 2.0
obj_mass = 0.2

dt_show = 1e-3
dt_subnum = 5
epsilon= 1e-5  # 偏置项ε
move_dpos_x = 0.005 / dt_subnum
move_dpos_y = move_dpos_x * res_x / res_y

score = ti.field(ti.i32, ())  # 得分
is_alive = ti.field(ti.i32, ())  # 当前是否存活


@ti.kernel
def initialize():
    safe_distance = 0.2  # 两障碍物初始间距在safe_distance以下时为安全
    obj_cnt = 0
    my_pos[None] = ti.Vector([0.5, 0.5])  # 自己被初始化在屏幕中间
    while obj_cnt < obj_num:
        x = ti.random()
        y = ti.random()
        now_pos = ti.Vector([x, y])
        if (now_pos - my_pos[None]).norm(epsilon) < safe_distance:  # 两障碍物初始太近则重新初始
            continue
        obj_pos[obj_cnt] = now_pos
        obj_vel[obj_cnt] = ti.Vector([ti.random(), ti.random()]) * 1
        obj_cnt += 1

    is_alive[None] = True
    score[None] = -50  # 反应速度会有一定延迟


@ti.kernel
def compute_force():
    ti.block_local(obj_pos)
    # 计算物体对障碍物们的引力
    for i in range(obj_num):
        # n_force[i] = ti.Vector([0.0, 0.0])
        diff = obj_pos[i] - my_pos[None]
        r = diff.norm(epsilon)  # 直线距离

        if score[None] > 0 and r < 8 / max(res_x, res_y):  # 判断是否死亡
            is_alive[None] = False
            print(r, my_pos[None], obj_pos[i])
        f = -(1.0 / r) ** 3 * diff * G * my_mass * obj_mass  # 万有引力
        obj_force[i] = f

    # 计算障碍物间的引力
    for i in range(obj_num):
        p = obj_pos[i]
        for j in range(obj_num):
            if i != j:
                diff = p - obj_pos[j]
                r = diff.norm(epsilon)
                f = -(1.0 / r) ** 3 * diff * G * obj_mass ** 2
                obj_force[i] += f


@ti.kernel
def update():
    dt = dt_show / dt_subnum
    for i in range(obj_num):
        obj_vel[i] += dt * obj_force[i]
        obj_pos[i] += dt * obj_vel[i]

        bnd_gain = 1.5  # 每次撞墙速度都加50%
        vel_max = 5   # 最大速度
        x, y = obj_pos[i]
        if x > 1 or x < 0:  # 障碍物撞到左右边界
            obj_vel[i][0] = -obj_vel[i][0]  # 速度x分量反向
            if obj_vel[i].norm() * bnd_gain < vel_max:
                obj_vel[i] *= bnd_gain
        if y > 1 or y < 0:
            obj_vel[i][1] = -obj_vel[i][1]
            if obj_vel[i].norm() * bnd_gain < vel_max:
                obj_vel[i] *= bnd_gain


def move():
    x, y = my_pos[None][0], my_pos[None][1]
    # if gui.get_event(ti.GUI.PRESS):
    # if gui.event.key == 'r': initialize()
    # elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: return
    if gui.is_pressed('Up'): y += move_dpos_y
    if gui.is_pressed('Down'): y -= move_dpos_y
    if gui.is_pressed('Left'): x -= move_dpos_x
    if gui.is_pressed('Right'): x += move_dpos_x

    x = min(1, max(0, x))
    y = min(1, max(0, y))

    my_pos[None][0], my_pos[None][1] = x, y


def draw():
    if is_alive[None]:
        gui.text(str(score[None]), (0.5, 0.95), font_size=20, color=0x777777)
        score[None] += 1
    else:
        gui.text('Die ' + str(score[None]), (0.5, 0.5), font_size=80)

    gui.circle(my_pos[None].to_numpy(), color=0xff0000, radius=10)
    gui.circles(obj_pos.to_numpy(), color=0x0055ff, radius=6)
    gui.show()
    # gui.show(f'img/{51+score[None]:0>3d}.png')


initialize()
while gui.running:
    for i in range(dt_subnum):
        if is_alive[None]:
            move()
            compute_force()
            update()
    draw()
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r': initialize()
