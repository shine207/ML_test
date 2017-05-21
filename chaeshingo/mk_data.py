#-*- coding: utf-8 -*- 

import glob
import re

import numpy as np
import matplotlib.pyplot as plt

black_pt = b'(.)*\xc8\xe6(.)*'
white_pt = b'(.)*\xb9\xe9(.)*'

black = 1
white = 2

gibo_size = 19

# for gibo_data
info_data_size = 2  # [none,black,white(0,1,2)] , [cnt of live route(0,1,2,3,4)]
stone_info_idx = 0
live_route_info_idx = 1

# for capture_data
cap_info_data_size = 2 # [ flag of reenterance ( true/ false ) ] , [ plot variable ]
reenterance_info_idx = 0
plot_var_info_idx = 1

# for direction
left = 0
right = 1
up = 2
down = 3

# [19 x 19 x 2]

#files = glob.glob("./data/*.sgf")
#files = glob.glob("./data/150530*.sgf")
files = glob.glob("./data/19901010kuk34-d03.sgf")
p1 = re.compile('^RE')
p2 = re.compile('^;')
pb = re.compile(black_pt)
pw = re.compile(white_pt)

black_file = open("./black.csv", "w")
white_file = open("./white.csv", "w")

def write_data( dfile, data, solution_pos ):
    dfile.write(','.join(str(x) for x in data.flatten()))
    dfile.write(','+str(solution_pos)+'\n')


def trans_xy(gibo_xy):
    col = ord(gibo_xy[0])-ord('a')
    row = ord(gibo_xy[1])-ord('a')
    return row * gibo_size + col, row, col

def calc_live_route(data, row, col):
    alive_left = 1
    alive_right = 1
    alive_up = 1
    alive_down = 1

    #left
    if col <= 0:
        alive_left = 0
    elif data[row][col-1][stone_info_idx] != 0:
        alive_left = 0
    #right
    if col >= (gibo_size-1):
        alive_right = 0
    elif data[row][col+1][stone_info_idx] != 0:
        alive_right = 0
    #up
    if row >= (gibo_size-1):
        alive_up = 0
    elif data[row+1][col][stone_info_idx] != 0:
        alive_up = 0
    #down
    if row <= 0:
        alive_down = 0
    elif data[row-1][col][stone_info_idx] != 0:
        alive_down = 0
 
    return alive_left+alive_right+alive_up+alive_down

def calc_live_route_all_stone(data) :
    for row in range(gibo_size) :
        for col in range(gibo_size) :
            if data[row][col][stone_info_idx] != 0 :
                calc_live_route(data, row, col)

def iscapture(data, row, col, except_direction, stone) :
    global capture_data
    global left, right, up, down
    if ( except_direction != left ) :
        # check reenterance
        # check stone
    if ( except_direction != right ) :
        # check reenterance
        # check stone
    if ( except_direction != up ) :
        # check reenterance
        # check stone
    if ( except_direction != down ) :
        # check reenterance
        # check stone
    

def capture_stone() :
    global capture_data
    global left, right, up, down
    
def init_reenterance() :
    global capture_data
    for row in range(gibo_size) :
        for col in range(gibo_size) :
            capture_data[row][col][reenterance_info_idx] = False


def put_stone(data, row, col, stone):
    # calc my live route
    live_route = calc_live_route(data,row,col)
    # modify info data
    data[row][col] = [stone, live_route]
    # change 4way live route
    if col > 0 and data[row][col-1][stone_info_idx] != 0 :
        data[row][col-1][live_route_info_idx] = max([data[row][col-1][live_route_info_idx]-1, 0])
    if col < (gibo_size-1) and data[row][col+1][stone_info_idx] != 0 :
        data[row][col+1][live_route_info_idx] = max([data[row][col+1][live_route_info_idx]-1, 0])
    if row < (gibo_size-1) and data[row+1][col][stone_info_idx] != 0 :
        data[row+1][col][live_route_info_idx] = max([data[row+1][col][live_route_info_idx]-1, 0])
    if row > 0 and data[row-1][col][stone_info_idx] != 0 :
        data[row-1][col][live_route_info_idx] = max([data[row-1][col][live_route_info_idx]-1, 0])

    global capture_data
    global left, right, up, down
    # check capture opponent stones
    opp_stone = stone == black ? white : black
    # check left
    if (col > 0 and data[row][col-1] == [ opp_stone, 0 ] ) :
        capture_data[row][col-1][reenterance_info_idx] = true
        if iscapture(data, row, col-1, right, opp_stone) :
            capture_stone()
            # if capture, recalcurate live route of all stones
            calc_live_route_all_stone(data)
        else :
            init_reenterance()
    # check right
    if (col < (gibo_size-1) and data[row][col+1] == [ opp_stone, 0 ] ) :
        capture_data[row][col+1][reenterance_info_idx] = true
        if iscapture(data, row, col+1, left, opp_stone) :
            capture_stone()
            # if capture, recalcurate live route of all stones
            calc_live_route_all_stone(data)
        else :
            init_reenterance()
    # check up
    if (row < (gibo_size-1) and data[row+1][col] == [ opp_stone, 0 ] ) :
        capture_data[row+1][col][reenterance_info_idx] = true
        if iscapture(data, row+1, col, down, opp_stone) :
            capture_stone()
            # if capture, recalcurate live route of all stones
            calc_live_route_all_stone(data)
        else :
            init_reenterance()
    # check down
    if (row > 0 and data[row-1][col] == [ opp_stone, 0 ] ) :
        capture_data[row-1][col][reenterance_info_idx] = true
        if iscapture(data, row-1, col, up, opp_stone) :
            capture_stone()
            # if capture, recalcurate live route of all stones
            calc_live_route_all_stone(data)
        else :
            init_reenterance()


    
    # draw stone
    global ax, plt
    global black, white
    if stone == white:
        capture_data[row][col][plot_var_info_idx], = ax.plot(col, (gibo_size-1)-row, 'o' ,markersize=28 , markeredgecolor=(0,0,0), markerfacecolor='w', markeredgewidth=1)
    elif stone == black:
        capture_data[row][col][plot_var_info_idx], = ax.plot(col, (gibo_size-1)-row, 'o' ,markersize=28 , markeredgecolor=(.5,.5,.5), markerfacecolor='k', markeredgewidth=1)
    plt.draw()
    plt.pause(1)
    

for f in files:

    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    fig.patch.set_facecolor((1,1,.8))

    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(gibo_size):
        ax.plot([x, x], [0,gibo_size-1], 'k')
    for y in range(gibo_size):
        ax.plot([0, gibo_size-1], [y,y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1,19)
    ax.set_ylim(-1,19)

    # draw Go stones at (10,10) and (13,16)
    #s1, = ax.plot(10,10,'o',markersize=28, markeredgecolor=(0,0,0), markerfacecolor='w', markeredgewidth=2)
    #s2, = ax.plot(13,16,'o',markersize=28, markeredgecolor=(.5,.5,.5), markerfacecolor='k', markeredgewidth=2)

    plt.draw()


    print('make data : '+f)
    ef = open(f, "r", encoding='euc-kr')
    text = ef.readlines()
    is_bw = 'none'
    for el in text:
        if p1.match(el):
            if pb.match(el.encode('euc-kr')):
                print('black win')
                is_bw = 'black'
            elif pw.match(el.encode('euc-kr')):
                print('white win')
                is_bw = 'white'
            else:
                print('I don\'t know who is winner')
                break
        elif p2.match(el):
            #gibo_data = [[[0 for data in range(info_data_size)] for col in range(gibo_size)] for row in range(gibo_size)] 
            gibo_data = np.zeros((gibo_size, gibo_size, info_data_size))
            capture_data = [[[False, None] for col in range(gibo_size)] for row in range(gibo_size)]
            if is_bw == 'black':
                # make data for black alphago
                gibo = el[1:len(el)-2]
                s_gibo = gibo.split(';')
                for e_point in s_gibo:
                    trans_point, row, col = trans_xy(e_point[2:4])
                    if( e_point[0] == 'B'):
                        write_data(black_file, gibo_data, trans_point)
                        put_stone(gibo_data, row, col, black)
                    else:
                        put_stone(gibo_data, row, col, white)

            elif is_bw == 'white':
                # make data for black alphago
                gibo = el[1:len(el)-2]
                s_gibo = gibo.split(';')
                for e_point in s_gibo:
                    trans_point, row, col = trans_xy(e_point[2:4])
                    if( e_point[0] == 'W'):
                        write_data(white_file, gibo_data, trans_point)
                        put_stone(gibo_data, row, col, white)
                    else:
                        put_stone(gibo_data, row, col, black)

            else:
                print('I don\'t know who is winner')
                break

    ef.close()

black_file.close();
white_file.close();
    



'''
def onclick(event):
    global ix, iy
    ix, iy = round(event.xdata), round(event.ydata)
    print( 'x = %d, y = %d'%(ix, iy))
    ax.plot(ix,iy,'o',markersize=28, markeredgecolor=(0,0,0), markerfacecolor='w', markeredgewidth=2)
    plt.draw()

    fig.canvas.mpl_disconnect(cid)

    return coords

cid = fig.canvas.mpl_connect('button_press_event', onclick)
'''
plt.draw()


