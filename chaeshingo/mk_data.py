 #-*- coding: utf-8 -*- 

import glob
import re

black = b'(.)*\xc8\xe6(.)*'
white = b'(.)*\xb9\xe9(.)*'

gibo_size = 19


#files = glob.glob("./data/*.sgf")
files = glob.glob("./data/150530*.sgf")
p1 = re.compile('^RE')
p2 = re.compile('^;')
pb = re.compile(black)
pw = re.compile(white)

black_file = open("./black.csv", "w")
white_file = open("./white.csv", "w")

def write_data( dfile, data, solution_pos ):
    data[len(data)-1] = solution_pos
    dfile.write(','.join(str(x) for x in data)+'\n')


def trans_xy(gibo_xy):
    return ((ord(gibo_xy[1])-ord('a')) * gibo_size )+ ord(gibo_xy[0])-ord('a')

for f in files:
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
            gibo_data = [0]*(gibo_size*gibo_size+1)
            if is_bw == 'black':
                # make data for black alphago
                gibo = el[1:len(el)-2]
                s_gibo = gibo.split(';')
                for e_point in s_gibo:
                    trans_point = trans_xy(e_point[2:4])
                    if( e_point[0] == 'B'):
                        write_data(black_file, gibo_data, trans_point)
                        gibo_data[trans_point] = 1
                    else:
                        gibo_data[trans_point] = 2

            elif is_bw == 'white':
                # make data for black alphago
                gibo = el[1:len(el)-2]
                s_gibo = gibo.split(';')
                for e_point in s_gibo:
                    if( e_point[0] == 'W'):
                        trans_point = trans_xy(e_point[2:4])
                        write_data(white_file, gibo_data, trans_point)
                        gibo_data[trans_point] = 2
                    else:
                        gibo_data[trans_point] = 1

            else:
                print('I don\'t know who is winner')
                break

    ef.close()

black_file.close();
white_file.close();
    

