import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos
from math import pi
from math import sqrt




img_name = 'img640_blue.png'
dwm_name = 'wm80.png'
alpha = 9



def Arnold_transformation(watermark):
    M = len(watermark)
    temp_matrix = np.array([[1,1],[1,2]])
    transformed_watermark = np.full((M,M),0)
    for x in range(M):
        for y in range(M) :
            coordinats = np.array([x,y])
            new_coordinats = (temp_matrix @ coordinats) % (M)
            new_x = int(new_coordinats[0])
            new_y = int(new_coordinats[1])
            transformed_watermark[new_x][new_y] = watermark[x][y]
    #print('watermark', watermark, '\ntransformed', transformed_watermark)
    cv2.imwrite('transformed_wm.png',transformed_watermark)
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

   
    #print(block, '\n')
    return(dct_block)


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

    



def embedding(list_of_dct_blocks, list_of_transformed_wm_bits, alpha):
    x = int(len(list_of_dct_blocks[0])/2)
    y = int(len(list_of_dct_blocks[0])/2)
    
    for i in range(len(list_of_dct_blocks)):
        #print('before emb ', list_of_dct_blocks[i],'\n', list_of_dct_blocks[i][x][y],'\n')
        
        if list_of_transformed_wm_bits[i] == 255:
            w_i = 0
        else:
            w_i = 1

        c_i = list_of_dct_blocks[i][x][y]      
        c_i_new = c_i + alpha*w_i
        
        #if c_i != c_i_new: print(c_i, c_i_new)
            
        list_of_dct_blocks[i][x][y] = c_i_new
        #print('after emb ', list_of_dct_blocks[i],'\n', list_of_dct_blocks[i][x][y],'\n\n\n')

    return list_of_dct_blocks


def image_from_blocks(blocks_list,M):
    #print(len(blocks_list),'\n\n\n\n\n\n\n')
    image_list = np.full((M,M),0)
    for k in range(len(blocks_list)):
        for i in range(len(blocks_list[k])):
            for j in range(len(blocks_list[k])):
                ost = k % (M//8)
                #print(k)
                new_i = (k // (M//8)) * 8 + i
                new_j = ost * 8 + j
                #print(k, i, j, new_i, new_j)
                image_list[new_i][new_j] = blocks_list[k][i][j] + 128
    #print(image_list)
    return image_list



    




watermark_cv2 = cv2.imread(dwm_name,cv2.IMREAD_GRAYSCALE)
watermark = []
for i in range(len(watermark_cv2)):
    watermark.append([])
    for j in range(len(watermark_cv2)):
        #watermark[i].append( int(watermark_cv2[i][j]) - 128 )
        watermark[i].append( int(watermark_cv2[i][j]))
#print(watermark, '\n\n')
transformed_watermark = Arnold_transformation(watermark)
list_of_transformed_wm_bits = []
for i in transformed_watermark:
    for j in i:
        list_of_transformed_wm_bits.append(int(j))

#print('list_of_transformed_wm_bits\n', list_of_transformed_wm_bits)





image_cv2 = cv2.imread(img_name,cv2.IMREAD_COLOR)
b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(image_cv2)
image_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]
image_layers = []
new_image_layers = []
for layer in range(3):
    image_layers.append([])
    for i in range(len(b_layer_cv2)):
        image_layers[layer].append([])
        for j in range(len(b_layer_cv2)):
            image_layers[layer][i].append(int(image_layers_cv2[layer][i][j]) - 128)
    #print('image layer ', image_layers[layer], '\n')

    
    
    list_of_blocks = make_blocks(image_layers[layer])
    #print('list_of_blocks ', list_of_blocks, '\n')
    list_of_dct_blocks = []
    
    for block in list_of_blocks:
        #print('block', block)
        dct_block = dct(block)
        list_of_dct_blocks.append(dct_block)
       
    list_of_dct_blocks = embedding(list_of_dct_blocks,  list_of_transformed_wm_bits, alpha)
    #print('list_of_dct_blocks ', list_of_dct_blocks, '\n')
    
    list_of_idct_blocks = []
    for block in list_of_dct_blocks:
        list_of_idct_blocks.append(idct(block))

    #print('list_of_idct_blocks', list_of_idct_blocks)
    #print('length list_of_idct_blocks', len(list_of_idct_blocks))

    new_image_layer = image_from_blocks(list_of_idct_blocks, len(image_layers[layer]))
    #print('new image ', new_image_layer)
    new_image_layers.append(new_image_layer)
    
    #print(image_layers_cv2[layer],'\n\n', new_image_layer,'\n\n\n')

    '''
    for i in range(len(list_of_dct_blocks)):
        print(list_of_blocks[i],'\n',\
            list_of_idct_blocks[i], '\n\n')
    '''

rgb_image_with_watermark = cv2.merge([new_image_layers[0], new_image_layers[1], new_image_layers[2]])
cv2.imwrite(f'fast_alpha_{alpha}.png', rgb_image_with_watermark)





#dct checker https://asecuritysite.com/comms/dct2