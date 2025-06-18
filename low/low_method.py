import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos
from math import pi
from math import sqrt
"""
img_name = 'img16.png'
dwm_name = 'wm2.png'
"""




def Arnold_transformation(watermark,num):
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
    wm_length = num //8
    cv2.imwrite(f'transformed_wm{wm_length}.png',transformed_watermark)
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
    #print("\n\n\n\n", list_of_blocks, "\n\n\n\n")
            
    return list_of_blocks


def mediana(block):
    l = [block[0][1], block[0][2], block[0][3],\
        block[1][0], block[1][1], block[1][2], \
        block[2][0], block[2][1], block[3][0], \
        ]
    l = sorted(l)
    mediana_value = l[5]
    dc = block[0][0]
    return (dc, mediana_value)



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

    block_dc, block_mediana = mediana(dct_block)        
    block_modification_power = 0
    if abs(block_dc) > 1000 or abs(block_dc) < 1:
        block_modification_power = abs(2*block_mediana)
    else:
        block_modification_power = abs(2*((block_dc - block_mediana)/block_dc))
    
    return(dct_block, block_dc, block_modification_power)


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

    



def embedding(list_of_dct_blocks, list_of_mod_power, list_of_transformed_wm_bits):
    x = int(len(list_of_dct_blocks[0])/2)
    y = int(len(list_of_dct_blocks[0])/2)
    T = 80
    K = 12
    for i in range(len(list_of_dct_blocks)):
        
        b_pq = list_of_dct_blocks[i][x][y]
        b_pq_plus = list_of_dct_blocks[(i + 1) % len(list_of_dct_blocks)][x][y]
        delta = b_pq - b_pq_plus
        mod_pow = list_of_mod_power[i]
        print('before emb ', b_pq, b_pq_plus, delta)
        print('modpow',mod_pow)
        #print(b_pq, b_pq_plus)
        #break
        print('lll',list_of_transformed_wm_bits[i])
        if list_of_transformed_wm_bits[i] == -128:

            if delta > (T-K):
                while delta > (T-K):
                    #print(1, delta)
                    b_pq -= mod_pow
                    delta = b_pq - b_pq_plus

            elif (K > delta) and (delta > -T/2):
                while delta < K:
                    #print(2, delta)
                    #exit
                    b_pq += mod_pow
                    delta = b_pq - b_pq_plus

            elif delta < -T/2:
                while delta > -T -K:
                    #print(3, delta)
                    b_pq -= mod_pow
                    delta = b_pq - b_pq_plus

        elif list_of_transformed_wm_bits[i] == 127:

            if delta > T/2:
                while delta <= T+K:
                    #print(4, delta)
                    b_pq += mod_pow
                    delta = b_pq - b_pq_plus

            elif -K < delta < T/2:
                while delta >= -K:
                    #print(5, delta)
                    b_pq -= mod_pow
                    delta = b_pq - b_pq_plus

            elif delta < K-T:
                while delta <= K-T:
                    #print(6,delta)
                    b_pq += mod_pow
                    delta = b_pq - b_pq_plus
                    
        list_of_dct_blocks[i][x][y] = b_pq
        print('after emb ', b_pq, '\n')

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


    image_list = np.full((M,M),0)
    for k in range(len(blocks_list)):
        for i in range(len(blocks_list[k])):
            for j in range(len(blocks_list[k])):
                ost = k % (8//2)
                #print(k)
                new_i = (k - ost ) // 2 + i
                new_j = ost * 2 + j
                #print(k, i, j, new_i, new_j)
                image_list[new_i][new_j] = blocks_list[k][i][j]
    #print(image_list)
    return image_list
    











'''
wm_nums =[2,4,8,20,40]
img_nums = [16,32,64,160,320]
'''
wm_nums =[2]
img_nums = [16]


for num in range(len(img_nums)):       

    img_name = f'img{img_nums[num]}.png'
    dwm_name = f'wm{wm_nums[num]}.png'

    watermark_cv2 = cv2.imread(dwm_name,cv2.IMREAD_GRAYSCALE)
    watermark = []
    for i in range(len(watermark_cv2)):
        watermark.append([])
        for j in range(len(watermark_cv2)):
            watermark[i].append( int(watermark_cv2[i][j]) - 128 )
    #print(watermark, '\n\n')
    transformed_watermark = Arnold_transformation(watermark, img_nums[num])
    #print('trans ',transformed_watermark)
    list_of_transformed_wm_bits = []
    for i in transformed_watermark:
        for j in i:
            list_of_transformed_wm_bits.append(int(j))

    #print('list_of_transformed_wm_bits\n', list_of_transformed_wm_bits)





    image_cv2 = cv2.imread(img_name,cv2.IMREAD_COLOR)
    #print(image_cv2)
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
        list_of_mod_power = []
        for block in list_of_blocks:
            #print('block', block)
            dct_block, block_dc, block_modification_power = dct(block)
            list_of_dct_blocks.append(dct_block)
            list_of_mod_power.append(block_modification_power)

        list_of_dct_blocks = embedding(list_of_dct_blocks, list_of_mod_power, list_of_transformed_wm_bits)
        #print('list_of_dct_blocks ', list_of_dct_blocks, '\n')
        
        list_of_idct_blocks = []
        for block in list_of_dct_blocks:
            list_of_idct_blocks.append(idct(block))

        #print('list_of_idct_blocks', list_of_idct_blocks)
        #print('length list_of_idct_blocks', len(list_of_idct_blocks))

        new_image_layer = image_from_blocks(list_of_idct_blocks, len(image_layers[layer]))
        #print('new image ', new_image_layer)
        new_image_layers.append(new_image_layer)


    rgb_image_with_watermark = cv2.merge([new_image_layers[0], new_image_layers[1], new_image_layers[2]])
    cv2.imwrite(f'new_image{img_nums[num]}.png', rgb_image_with_watermark)





    #dct checker https://asecuritysite.com/comms/dct2
    #https://d1wqtxts1xzle7.cloudfront.net/102624166/j.aeue.2013.08.01820230525-1-co7ejz-libre.pdf?1684994920=&response-content-disposition=inline%3B+filename%3DA_novel_blind_robust_image_watermarking.pdf&Expires=1750274315&Signature=e8Pwi4sf4MM-HRjalMl2haTezKT6N3Rft0FVpTUT~n5V6-zAmPS54eGuLmyvpYiriqzfIxa8im9le-1~gpitc4tWHZUXsdC5CCiu0VWQkdl81AwWB5sDAuGGULO1FGWJhaSNqL-xIcummkd7KLQvubulDOSqgu3theeR~H4am4hYnVMl8f2cqAZRahK3cfOS-R1gZHupveTBcy6Pr1pt0YR7vZMpfsoedQmyQMQQv~99G3UkrRqMUxXUgJVRPfqN85A7XMamYgpzlSnKmEyD9ovuddOPKpCMsaTIeq-FBVTcHrZTcZBfFKBqV7rZtKfOXM3HfqSEAT-TVnJ31gPylA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA