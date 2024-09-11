
import os
import colour
from colour.plotting import *

import matplotlib.pyplot as plt

#來源圖片處理
import cv2
import numpy as np
import rawpy
import math
from PIL import Image

plot_chromaticity_diagram_CIE1931(standalone=False)

plt_2020_CIE=1
plt_re_709_CIE=1
plt_re_2020_CIE=1

CIE_dwn_sp_ratio=50
CIE_s=0.05
CIE_alpha=0.1
#'C:/Users/eric/Desktop/run1/test1/test_img/test_imgout/1.bmp'
#A = np.array([])
#'C:/Users/eric/Desktop/mmmm/1.bmp'
#讀檔
img_path = './709_2020_img/9.bmp'

if img_path.split( ".")[-1].lower() == "cr2":
    print("cr2 file")
    rawImg = rawpy.imread(img_path)
    #print(rawImg.shape)
    #img = rawImg.postprocess()

    #r,g,b = cv2.split(img)#拆分通道
    #img = cv2.merge([b,g,r])#合并通道
    #以下两行可能解决偏色问题，output_bps=16表示输出是 16 bit (2^16=65536)需要转换一次
    im = rawImg.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    rgb = np.float32(im / 65535*255.0)
    img = np.asarray(rgb,np.uint8)
    b,g,r = cv2.split(img)#拆分通道
    img = cv2.merge([r,g,b])#合并通道

else:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
# 讓視窗可以自由縮放大小
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if img_path.split( ".")[-1].lower() == "cr2":
    n = 8  #2020
    n1 = 8  #709
    x = 2
    B, G, R = cv2.split(img)
elif img_path.split( ".")[-1].lower() == "bmp":
    n = 8  #2020
    n1 = 8  #709
    x = 2
    B, G, R = cv2.split(img)
    
    
width, height, color = img.shape
print(width, height)

#'C:/Users/eric/Desktop/run1/test2/picture/8.bmp'
#po-cnn2020-10.bmp

#'C:/Users/eric/Desktop/mmmm/1.bmp'
#讀檔2
#'C:/Users/eric/Desktop/mmmm/1.bmp'
img_path1 = './2020/3.bmp'

if img_path1.split( ".")[-1].lower() == "cr2":
    print("cr2-1 file")
    rawImg1 = rawpy.imread(img_path1)
    #print(rawImg.shape)
    #img = rawImg.postprocess()

    #r,g,b = cv2.split(img)#拆分通道
    #img = cv2.merge([b,g,r])#合并通道
    #以下两行可能解决偏色问题，output_bps=16表示输出是 16 bit (2^16=65536)需要转换一次
    im1 = rawImg1.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    rgb1 = np.float32(im1 / 65535*255.0)
    img1 = np.asarray(rgb1,np.uint8)
    b1,g1,r1 = cv2.split(img1)#拆分通道
    img1 = cv2.merge([r1,g1,b1])#合并通道

else:
    img1 = cv2.imread(img_path1, cv2.IMREAD_UNCHANGED)
    
# 讓視窗可以自由縮放大小
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if img_path1.split( ".")[-1].lower() == "cr2":
    n = 8  #2020
    n1 = 8  #709
    x = 2
    B1, G1, R1 = cv2.split(img1)
elif img_path1.split( ".")[-1].lower() == "bmp":
    n = 8  #2020
    n1 = 8  #709
    x = 2
    B1, G1, R1 = cv2.split(img1)
    
    
width1, height1, color = img1.shape
print(width1, height1)    


    
R_M = np.zeros((width, height), dtype=np.single)
G_M = np.zeros((width, height), dtype=np.single)
B_M = np.zeros((width, height), dtype=np.single)

R_M1 = np.zeros((width1, height1), dtype=np.single)
G_M1 = np.zeros((width1, height1), dtype=np.single)
B_M1 = np.zeros((width1, height1), dtype=np.single)

#BT2020 to 709    
    
 #Q RGB   
R_Q = (R / (2 ** (n - 8)) - 16) / 219
G_Q = (G / (2 ** (n - 8)) - 16) / 219
B_Q = (B / (2 ** (n - 8)) - 16) / 219

R_Q1 = (R1 / (2 ** (n - 8)) - 16) / 219
G_Q1 = (G1 / (2 ** (n - 8)) - 16) / 219
B_Q1 = (B1 / (2 ** (n - 8)) - 16) / 219
    
 #上升曲線
count = 0
i = 0


max_rgb =1.092
    
min_rgb =-0.074

R_Q = (R_Q - min_rgb) / (max_rgb - min_rgb)
G_Q = (G_Q - min_rgb) / (max_rgb - min_rgb)
B_Q = (B_Q - min_rgb) / (max_rgb - min_rgb)

R_Q1 = (R_Q1 - min_rgb) / (max_rgb - min_rgb)
G_Q1 = (G_Q1 - min_rgb) / (max_rgb - min_rgb)
B_Q1 = (B_Q1 - min_rgb) / (max_rgb - min_rgb)        
                
R_up = R_Q
G_up = G_Q
B_up = B_Q

R_up1 = R_Q1
G_up1 = G_Q1
B_up1 = B_Q1
        
print("Gamma up")

R_up = np.where((R_up>=0) & (R_up<=1), R_up ** x, R_up)
G_up = np.where((G_up>=0) & (G_up<=1), G_up ** x, G_up)
B_up = np.where((B_up>=0) & (B_up<=1), B_up ** x, B_up)

R_up1 = np.where((R_up1>=0) & (R_up1<=1), R_up1 ** x, R_up1)
G_up1 = np.where((G_up1>=0) & (G_up1<=1), G_up1 ** x, G_up1)
B_up1 = np.where((B_up1>=0) & (B_up1<=1), B_up1 ** x, B_up1)
    
print("BT2020 to CIE")
    
#BT2020 to CIExy

R_cie = 0.637 * R_up + 0.1446 * G_up + 0.1689 * B_up
G_cie = 0.2627 * R_up + 0.678 * G_up + 0.0593 * B_up
B_cie = 0 * R_up + 0.0281 * G_up + 1.061 * B_up

R_cie1 = 0.6370 * R_up1 + 0.1446 * G_up1 + 0.1689 * B_up1
G_cie1 = 0.2627 * R_up1 + 0.6780 * G_up1 + 0.0593 * B_up1
B_cie1 = 0.0000 * R_up1 + 0.0281 * G_up1 + 1.0610 * B_up1
    
RGB_sum = R_cie + G_cie + B_cie

RGB_sum1 = R_cie1 + G_cie1 + B_cie1
    
max_sum = np.max(RGB_sum)

max_sum1 = np.max(RGB_sum1)
    
CIE_x = R_cie/RGB_sum
CIE_y = G_cie/RGB_sum
CIE_z = 1 - CIE_x - CIE_y
    
CIE_x1 = R_cie1/RGB_sum1
CIE_y1 = G_cie1/RGB_sum1
CIE_z1 = 1 - CIE_x1 - CIE_y1
   
#clipping the saturation area
    
    #define the saturation area for BT2020 to 709
    
R_709_x = 0.640074
R_709_y = 0.329971
R_709_z = 1 - R_709_x - R_709_y    
    
G_709_x = 0.3 
G_709_y = 0.6
G_709_z = 1 - G_709_x - G_709_y
    
B_709_x = 0.150017
B_709_y = 0.060007
B_709_z = 1 - B_709_x - B_709_y
    
O_709_x = 0.312716 
O_709_y = 0.329001
O_709_z = 1 - O_709_x - O_709_y
    
    #find the line, L_any,  that crossed  (CIE_x, CIE_y) and O_709
    #then find the point of intersection of  L_any and L1, L2, L3
    
m1 = (R_709_y - G_709_y)/(R_709_x - G_709_x)
m2 = (G_709_y - B_709_y)/(G_709_x - B_709_x)
m3 = (R_709_y - B_709_y)/(R_709_x - B_709_x)
    
    #L1: a1*x+b1*y = p1
a1 = m1
b1 = -1
p1 = m1*G_709_x - G_709_y

    #L2: a2*x+b2*y = p2    
a2 = m2
b2 = -1
p2 = m2*B_709_x - B_709_y

    #L3: a3*x+b3*y = p2    
a3 = m3
b3 = -1
p3 = m3*B_709_x - B_709_y
    
C = (CIE_y - O_709_y)/(CIE_x - O_709_x)
d = -1
Q = C*O_709_x - O_709_y
    
print("find line functions")
    
F1 = CIE_y - G_709_y - m1*(CIE_x - G_709_x)
F2 = CIE_y - B_709_y - m2*(CIE_x - B_709_x)
F3 = CIE_y - B_709_y - m3*(CIE_x - B_709_x)

    
print("find ponits of intersecton")
    
Den1 =  a1*d-b1*C
Den1 =  np.where(Den1==0, 0.0001, Den1)
X1 = (p1*d-b1*Q)/Den1
Y1 = (a1*Q-p1*C)/Den1
    
Den2 =  a2*d-b2*C
Den2 =  np.where(Den2==0, 0.0001, Den2)
X2 = (p2*d-b2*Q)/Den2
Y2 = (a2*Q-p2*C)/Den2 
    
Den3 =  a3*d-b3*C
Den3 =  np.where(Den3==0, 0.0001, Den3)
X3 = (p3*d-b3*Q)/Den3
Y3 = (a3*Q-p3*C)/Den3

print("find region")     
## 有問題   
##    region 1
CIE_x = np.where((F1 > 0) & (F2 > 0) & (F3 > 0), CIE_x1, CIE_x)
CIE_y = np.where((F1 > 0) & (F2 > 0) & (F3 > 0), CIE_y1, CIE_y)
##    
#    #region 2
CIE_x = np.where((F1 < 0) & (F2 < 0) & (F3 > 0), CIE_x1, CIE_x)
CIE_y = np.where((F1 < 0) & (F2 < 0) & (F3 > 0), CIE_y1, CIE_y)
#    
    #region 3
CIE_x = np.where((F1 > 0) & (F2 < 0) & (F3 > 0), CIE_x1, CIE_x)
CIE_y = np.where((F1 > 0) & (F2 < 0) & (F3 > 0), CIE_y1, CIE_y)
###
###    #region 4
CIE_x = np.where((F1 < 0) & (F2 > 0) & (F3 > 0), CIE_x1, CIE_x)
CIE_y = np.where((F1 < 0) & (F2 > 0) & (F3 > 0), CIE_y1, CIE_y)
###
###    #region 5
CIE_x = np.where((F1 > 0) & (F2 < 0) & (F3 > 0), CIE_x1, CIE_x)
CIE_y = np.where((F1 > 0) & (F2 < 0) & (F3 > 0), CIE_y1, CIE_y)
####
####    #region 6
#CIE_x = np.where((F1 > 0) & (F2 > 0) & (F3 > 0), CIE_x1, CIE_x)
#CIE_y = np.where((F1 > 0) & (F2 > 0) & (F3 > 0), CIE_y1, CIE_y)

#CIE_x = np.where((F1 < 0) & (F2 < 0) & (F3 < 0), CIE_x1, CIE_x)
#CIE_y = np.where((F1 < 0) & (F2 < 0) & (F3 < 0), CIE_y1, CIE_y)  


          
print("find new cie points")
    
R_cie = CIE_x*RGB_sum
G_cie = CIE_y*RGB_sum
B_cie = RGB_sum - R_cie - G_cie
    
print("CIE to BT2020")
    
R_M = 1.7167 * R_cie - 0.3557 * G_cie - 0.2534 * B_cie
G_M = -0.6667 * R_cie + 1.6165 * G_cie + 0.0158 * B_cie
B_M = 0.0176 * R_cie - 0.0428 * G_cie + 0.9421 * B_cie

    
print("scaling down")
    
max_rgb_M = np.max([R_M, G_M, B_M])
    
R_M = np.where(max_rgb_M>1.0, R_M/max_rgb_M, R_M)
G_M = np.where(max_rgb_M>1.0, G_M/max_rgb_M, G_M)
B_M = np.where(max_rgb_M>1.0, B_M/max_rgb_M, B_M)
    
R_M = np.where(R_M<0.0, 0.0, R_M)
G_M = np.where(G_M<0.0, 0.0, G_M)
B_M = np.where(B_M<0.0, 0.0, B_M)
    
#下降曲線
print("Gamma down")
    
R_cie_re = 0.412 * R_M + 0.358 * G_M + 0.181 * B_M
G_cie_re = 0.213 * R_M + 0.715 * G_M + 0.072 * B_M
B_cie_re = 0.019 * R_M + 0.119 * G_M + 0.951 * B_M 
    
RGB_sum_re = R_M + G_M + B_M
      
CIE_x_re = R_M/RGB_sum_re
CIE_y_re = G_M/RGB_sum_re
CIE_z_re = 1 - CIE_x_re - CIE_y_re

RGB_sum_re = R_cie_re + G_cie_re + B_cie_re
      
CIE_x_re = R_cie_re/RGB_sum_re
CIE_y_re = G_cie_re/RGB_sum_re
CIE_z_re = 1 - CIE_x_re - CIE_y_re
    

    
R_sat = np.where((R_M>=0) & (R_M<=1), R_M ** (1/x), R_M)
G_sat = np.where((G_M>=0) & (G_M<=1), G_M ** (1/x), G_M)
B_sat = np.where((B_M>=0) & (B_M<=1), B_M ** (1/x), B_M)
    
max_R = np.max(R_sat)
max_G = np.max(G_sat)
max_B = np.max(B_sat)

R_sat =  (max_rgb - min_rgb)*R_sat + min_rgb
G_sat =  (max_rgb - min_rgb)*G_sat + min_rgb
B_sat =  (max_rgb - min_rgb)*B_sat + min_rgb
       
#Q RGB2
R_Q1 = (219 * R_sat + 16) * 2 ** (n1 - 8)
G_Q1 = (219 * G_sat + 16) * 2 ** (n1 - 8)
B_Q1 = (219 * B_sat + 16) * 2 ** (n1 - 8)
    
R_Q1 = np.where((R_Q1<0), 0, R_Q1)
R_Q1 = np.where((R_Q1>255), 255, R_Q1)
G_Q1 = np.where((G_Q1<0), 0, G_Q1)
G_Q1 = np.where((G_Q1>255), 255, G_Q1)
B_Q1 = np.where((B_Q1<0), 0, B_Q1)
B_Q1 = np.where((B_Q1>255), 255, B_Q1)

    
#R、G、B的合併
img = cv2.merge([R_Q1,G_Q1,B_Q1])
#存圖 

arr = np.array(img , dtype = 'uint8')
arr=Image.fromarray(arr)
arr.save('./integrate/' + '1.bmp' )

    
#downsampling (CIE_x, CIE_y) by 2-D 1/10
CIE_x_re_dwn_sp = CIE_x[::CIE_dwn_sp_ratio,:]
CIE_y_re_dwn_sp = CIE_y[::CIE_dwn_sp_ratio,:]       
    
plt.scatter(CIE_x_re_dwn_sp, CIE_y_re_dwn_sp,  s=CIE_s,  c='black', alpha=CIE_alpha)

t1 = plt.Polygon([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], facecolor="none", edgecolor='black')
plt.gca().add_patch(t1)

t2 = plt.Polygon([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], facecolor="none", edgecolor='black')
plt.gca().add_patch(t2)
       
print("End")
