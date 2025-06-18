import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos
from math import pi
from math import sqrt







def dct(block):
    dct_block = []
    for i in range(8):
        dct_block.append([])
        for j in range(8):
            new_pixel = 0
            for x in range(8):
                for y in range(8):
                    cx = 1
                    cy = 1
                    if i == 0:
                        cx = (1/sqrt(2))
                    if j == 0:
                        cy = (1/sqrt(2))
                    new_pixel += cx * cy * block[x][y] * cos((2*y+1)*j*pi/16) * cos((2*x+1)*i*pi/16)
                    #print(new_pixel)
            
            dct_block[i].append(new_pixel*0.25)
    #print('dct_block', dct_block, '\n')  

    return dct_block

def idct(dctblock):
    idct_block = []
    for i in range(8):
        idct_block.append([])
        for j in range(8):
            old_pixel = 0.0
            for x in range(8):
                for y in range(8):
                    cx = 1
                    cy = 1
                    if x == 0:
                        cx = (1/sqrt(2))
                    if y == 0:
                        cy = (1/sqrt(2))
                    old_pixel += cx * cy * dctblock[x][y] * cos((2*i+1)*x*pi/16) * cos((2*j+1)*y*pi/16)

                    
            #print(old_pixel, dctblock[i][j])
            idct_block[i].append(int(old_pixel*0.25))
            
    #print('idct_block', idct_block, '\n')  
    return idct_block


block = [[126,130,119,123,124,121,115,117],
    [126,131,119,124,125,122,116,118],
    [124,122,119,119,118,117,112,111],
    [118,113,119,110,117,114,102,107],
    [105,111,113,112,116,113,108,108],
    [107,110,105,108,109,108,108,106],
    [110,111,108,109,109,105,104,107],
    [110,108,108,107,106,104,110,105]]

dctblock = dct(block)
print(dctblock, '\n')
print(block, '\n')
print(idct(dctblock))




'''
def make_blocks(layer):
    M = len(layer)
    list_of_blocks = []
    block = []
    for i in range(0,M,2):
        for j in range(0,M,2):
            k = layer[i:i+2]            
            for l in range(2):
                block.append((k[l][j:j+2]))            
            #print('block', block, '\n')
            list_of_blocks.append(block)
            block = []
    return list_of_blocks


def image_from_blocks(blocks_list,M):
    image_list = np.full((M,M),0)
    for k in range(len(blocks_list)):
        for i in range(len(blocks_list[k])):
            for j in range(len(blocks_list[k])):
                ost = k % (8//2)
                print(k)
                new_i = (k - ost ) // 2 + i
                new_j = ost * 2 + j
                print(k, i, j, new_i, new_j)
                image_list[new_i][new_j] = blocks_list[k][i][j]
    print(image_list)
    return image_list


    
lll = [[126,130,119,123,124,121,115,117],
    [126,131,119,124,125,122,116,118],
    [124,122,119,119,118,117,112,111],
    [118,113,119,110,117,114,102,107],
    [105,111,113,112,116,113,108,108],
    [107,110,105,108,109,108,108,106],
    [110,111,108,109,109,105,104,107],
    [110,108,108,107,106,104,110,105]]

blocks_list = make_blocks(lll)
print(blocks_list)
image_from_blocks(blocks_list,8)


print(11//4, 8//4)
'''