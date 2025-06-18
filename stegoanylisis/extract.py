import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos
from math import pi
from math import sqrt



alpha = 4

empty_data_name = 'img640.png'
stego_data_name = f'fast_alpha_{alpha}_level_06.jpg'



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
            idct_block[i].append(old_pixel*0.25)
            
    for i in range(8):
        for j in range(8):
            idct_block[i][j] = int(idct_block[i][j])
    #print(i ,'idct_block', idct_block, '\n')  
    return idct_block





def extracting(empty_list_of_dct_blocks, stego_list_of_dct_blocks, alpha, length):
    watermark_bits = []
    
    x = int(len(empty_list_of_dct_blocks[0])/2)
    y = int(len(empty_list_of_dct_blocks[0])/2)
    
    
    for i in range(len(empty_list_of_dct_blocks)):
        #print('before emb ', list_of_dct_blocks[i],'\n')
        c_i_empty = int(empty_list_of_dct_blocks[i][x][y])
        c_i_stego = int(stego_list_of_dct_blocks[i][x][y])

        #print('empty', empty_list_of_dct_blocks[i], '\n', c_i_empty)
        #print('stego' , stego_list_of_dct_blocks[i], '\n', c_i_stego)

        w_i = (c_i_stego - c_i_empty)# - alpha
        #print(w_i,'\n\n\n')
        watermark_bits.append(w_i)

    
    watermark = np.full((length,length),0)
    
    for i in range(length):
        for j in range(length):
            wm_bit = watermark_bits[length*i + j]
            if wm_bit < 3:
                watermark[i][j] = 255 
            else:
                watermark[i][j] = 0

    #print(watermark)

    return watermark






    










empty_data = cv2.imread(empty_data_name,cv2.IMREAD_COLOR)
b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(empty_data)
empty_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]
empty_layers = []

stego_data = cv2.imread(stego_data_name,cv2.IMREAD_COLOR)
b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(stego_data)
stego_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]
stego_layers = []

for layer in range(3):

    empty_layers.append([])
    for i in range(len(b_layer_cv2)):
        empty_layers[layer].append([])
        for j in range(len(b_layer_cv2)):
            empty_layers[layer][i].append(int(empty_layers_cv2[layer][i][j]) - 128)
    #print('image layer ', image_layers[layer], '\n')

    empty_list_of_blocks = make_blocks(empty_layers[layer])
    #print('list_of_blocks ', list_of_blocks, '\n')
    empty_list_of_dct_blocks = []
    
    for block in empty_list_of_blocks:
        dct_block = idct(block)
        #print('block', block, '\nidct block', dct_block,'\n\n')
        empty_list_of_dct_blocks.append(dct_block)



    stego_layers.append([])
    for i in range(len(b_layer_cv2)):
        stego_layers[layer].append([])
        for j in range(len(b_layer_cv2)):
            stego_layers[layer][i].append(int(stego_layers_cv2[layer][i][j]) - 128)
    
    stego_list_of_blocks = make_blocks(stego_layers[layer])
    #print('list_of_blocks ', list_of_blocks, '\n')
    stego_list_of_dct_blocks = []
    
    for block in stego_list_of_blocks:
        #print('block', block)
        dct_block = idct(block)
        stego_list_of_dct_blocks.append(dct_block)
    
    #print(empty_list_of_dct_blocks[0][0], '\n\n', stego_list_of_dct_blocks[0][0])

    length = len(b_layer_cv2) // 8
    wm_bites = extracting(empty_list_of_dct_blocks, stego_list_of_dct_blocks, alpha, length)
    #print(wm_bites)
    #cv2.imwrite(f'befarnold_wm{layer}.png', wm_bites)
    
    watermark = inverse_Arnold(wm_bites)
    cv2.imwrite(f'restored_wm{layer}_alpha_{alpha}_level_06.png', watermark)

