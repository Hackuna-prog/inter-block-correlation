import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos
from math import pi
from math import sqrt




def inverse_Arnold(list_of_wm_bites):
    M = len(list_of_wm_bites)
    temp_matrix = np.array([[2,-1],[-1,1]])
    transformed_watermark = np.full((M,M),0)
    for x in range(M):
        for y in range(M) :
            coordinats = np.array([x,y])
            new_coordinats = (temp_matrix @ coordinats) % (M)
            new_x = int(new_coordinats[0])
            new_y = int(new_coordinats[1])
            transformed_watermark[new_x][new_y] = list_of_wm_bites[x][y]
    #print('watermark', watermark, '\ntransformed', transformed_watermark)
    
    return transformed_watermark
                


def make_blocks(layer):
    M = len(layer)
    list_of_blocks = []
    block = []
    for i in range(0,M,8):
        for j in range(0,M,8):
            k = layer[i:i+8]            
            for l in range(8):
                block.append((k[l][j:j+8]))            
            #print('block', block, '\n')
            list_of_blocks.append(block)
            block = []
            
    return list_of_blocks


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



    



def extracting(list_of_dct_blocks):
    x = int(len(list_of_dct_blocks[0])/2)
    y = int(len(list_of_dct_blocks[0])/2)
    
    wm_bits = []
    
    for i in range(len(list_of_dct_blocks)):
        #print('before emb ', list_of_dct_blocks[i],'\n')
        b_pq = list_of_dct_blocks[i][x][y]
        b_pq_plus = list_of_dct_blocks[(i + 1) % len(list_of_dct_blocks)][x][y]
        delta = b_pq - b_pq_plus
        print(b_pq, b_pq_plus,delta)
        T = 80
        #print(delta)
        if (delta < -T) or ((delta > 0) and (delta < T)):
            wm_bits.append(255)
        else:
            wm_bits.append(0)
            
    return wm_bits



'''
wm_nums =[2,4,8,20,40]
img_nums = [16,32,64,160,320]
'''
wm_nums =[8]
img_nums = [64]

for num in range(len(img_nums)):       

    img_name = f'new_image{img_nums[num]}.png'

    image_cv2 = cv2.imread(img_name,cv2.IMREAD_COLOR)
    b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(image_cv2)
    image_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]
    image_layers = []
    #wm_layers = []
    #only blue
    for layer in range(1):
        image_layers.append([])
        for i in range(len(b_layer_cv2)):
            image_layers[layer].append([])
            for j in range(len(b_layer_cv2)):
                image_layers[layer][i].append(int(image_layers_cv2[layer][i][j]) - 128)
        #print('image layer ', image_layers[layer], '\n')

        
        
        list_of_blocks = make_blocks(image_layers[layer])
        #print('list_of_blocks ', list_of_blocks, '\n')
        list_of_dct_blocks = []
        #list_of_mod_power = []
        for block in list_of_blocks:
            #print('block', block)
            dct_block = dct(block)
            list_of_dct_blocks.append(dct_block)
            #list_of_mod_power.append(block_modification_power)

        wm_bits = extracting(list_of_dct_blocks)
        print(wm_bits)
        #print(wm_bits, '\n')

        wm_bits_layer = []
        wm_length = img_nums[num]//8
        for i in range(wm_length):
            wm_bits_layer.append([])
            for j in range(wm_length):
                wm_bits_layer[i].append(wm_bits[wm_length*i + j])
        #print(wm_bits_layer)

        new_wm = inverse_Arnold(wm_bits_layer)
        

        cv2.imwrite(f'restored_wm{wm_length}.png', new_wm)

