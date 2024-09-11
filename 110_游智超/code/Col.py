#pip install colour-science
#pip install colour
#pip install matplotlib

import colour
from colour.plotting import *
import matplotlib.pyplot as plt

# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get
# displayed and can be used as a basis for other plots.
plot_chromaticity_diagram_CIE1931(standalone=False)


#來源圖片處理
import cv2
import numpy as np
import rawpy
import math
img_path = 'C:/Users/User/Desktop/gmcode/code/overlap/seaview/img_out/1.bmp'

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

#to 709
#Q RGB
R_Q = (R / (2 ** (n - 8)) - 16) / 219
G_Q = (G / (2 ** (n - 8)) - 16) / 219
B_Q = (B / (2 ** (n - 8)) - 16) / 219
#上升曲線

count = 0
i = 0

max_r =np.max(R_Q)
max_g =np.max(G_Q)
max_b =np.max(B_Q)

min_r =np.min(R_Q)
min_g =np.min(G_Q)
min_b =np.min(B_Q)

print ("Max_r value element : " , max_r);
print ("Max_g value element : " , max_g);
print ("Max_b value element : " , max_b);

print ("Min_r value element : " , min_r);
print ("Min_g value element : " , min_g);
print ("Min_b value element : " , min_b);

R_Q = (R_Q - min_r) / (max_r - min_r)
G_Q = (G_Q - min_g) / (max_g - min_g)
B_Q = (B_Q - min_b) / (max_b - min_b)

        
R_up = R_Q
G_up = G_Q
B_up = B_Q

R_up = np.where((R_up>=0) & (R_up<=1), R_up ** x, R_up)
G_up = np.where((G_up>=0) & (G_up<=1), G_up ** x, G_up)
B_up = np.where((B_up>=0) & (B_up<=1), B_up ** x, B_up)
            
#M  709 to cie
#R_M = 0.412 * R_up + 0.358 * G_up + 0.181 * B_up
#G_M = 0.213 * R_up + 0.715 * G_up + 0.072 * B_up
#B_M = 0.019 * R_up + 0.119 * G_up + 0.951 * B_up 
        
        
#M  2020 to cie
R_M = 0.637 * R_up + 0.145 * G_up + 0.169 * B_up
G_M = 0.263 * R_up + 0.678 * G_up + 0.059 * B_up
B_M = 0.000 * R_up + 0.028 * G_up + 1.061 * B_up


RGB_s = R_M + G_M + B_M
x_M = R_M / RGB_s
y_M = G_M / RGB_s

print("Process OK")

#畫三角形
##plt.plot(x_M, y_M, "o" ,color='black')
plt.scatter(x_M,y_M,  s=0.000005,  c='black', alpha=0.5)

t1 = plt.Polygon([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], facecolor="none", edgecolor='black')
plt.gca().add_patch(t1)

t2 = plt.Polygon([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], facecolor="none", edgecolor='black')
plt.gca().add_patch(t2)


# Displaying the plot.
render(
    standalone=True,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True)
