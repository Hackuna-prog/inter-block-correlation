import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import log10
from math import sqrt




def eq_of_PSNR(alpha):    
    empty_data_name = 'img640.png'
    stego_data_name = f'fast_alpha_{alpha}_second.jpg'

    empty_data = cv2.imread(empty_data_name,cv2.IMREAD_COLOR)
    b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(empty_data)
    empty_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]


    stego_data = cv2.imread(stego_data_name,cv2.IMREAD_COLOR)
    b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(stego_data)
    stego_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]

    delta_f_x_y = 0
    N = len(b_layer_cv2)

    for layer in range(3):
        for x in range(N):
            for y in range(N):
                f_x_y = int(empty_layers_cv2[layer][x][y])
                f_new_x_y = int(stego_layers_cv2[layer][x][y])
                delta_f_x_y += (f_x_y - f_new_x_y)**2

    PSNR = 1/3 * 10 * log10(255**2 * N**2 / delta_f_x_y)
    print('alpha = ', alpha, 'PSNR = ', PSNR)


def eq_of_NCC(alpha):
    empty_data_name = 'img640.png'
    stego_data_name = f'fast_alpha_{alpha}.png'

    empty_data = cv2.imread(empty_data_name,cv2.IMREAD_COLOR)
    b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(empty_data)
    empty_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]


    stego_data = cv2.imread(stego_data_name,cv2.IMREAD_COLOR)
    b_layer_cv2, g_layer_cv2, r_layer_cv2 = cv2.split(stego_data)
    stego_layers_cv2 = [b_layer_cv2, g_layer_cv2, r_layer_cv2]

    N = len(b_layer_cv2)
    f_avg = 0
    f_new_avg = 0

    for layer in range(3):
        for x in range(N):
            for y in range(N):
                f_avg += int(empty_layers_cv2[layer][x][y])
                f_new_avg += int(stego_layers_cv2[layer][x][y])
    
    f_avg = 1/3 * (f_avg /(N**2))
    f_new_avg = 1/3 * (f_new_avg/ N**2)
    #print('avg ', f_avg, f_new_avg)

    
    NCC = 0

    for layer in range(3):
        temp1 = 0
        temp2 = 0
        temp3 = 0
        for x in range(N):
            for y in range(N):
                f_x_y = int(empty_layers_cv2[layer][x][y])
                f_new_x_y = int(stego_layers_cv2[layer][x][y])
                temp1 += (f_x_y - f_avg) * (f_new_x_y - f_new_avg)
                temp2 += (f_x_y - f_avg)**2
                temp3 += (f_new_x_y - f_new_avg)**2
        temp2 = sqrt(temp2)
        temp3 = sqrt(temp3)
        NCC +=  (temp1 / (temp2 * temp3))
    
    NCC /= 3

    print('alpha = ', alpha, 'NCC = ', NCC)

                





alpha = 4
eq_of_PSNR(alpha)
eq_of_NCC(alpha)
print('\n')
