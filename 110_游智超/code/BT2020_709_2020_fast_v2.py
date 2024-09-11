#引入函示庫(代改)
#pip install colour-science
#pip install colour
#pip install matplotlib

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

plt_2020_CIE=1
plt_re_709_CIE=1
plt_re_2020_CIE=1

CIE_dwn_sp_ratio=50
CIE_s=0.05
CIE_alpha=0.1

#讀檔
for a in range (1,89):
  
    img_path = "C:/Users/eric/Desktop/cr2_2/" + str(a).zfill(1) + ".CR2"
    #rawImg = imageio.imread(train)
    #rawImg = rawImg*255
    print("CR2 file:"+ str(a) +".CR2")
    if img_path.split( ".")[-1].lower() == "cr2":
        rawImg = rawpy.imread(img_path)
   #print(rawImg.shape)
   #img = rawImg.postprocess()

   #r,g,b = cv2.split(img)#拆分通道
   #img = cv2.merge([b,g,r])#合并通道
   #以下两行可能解决偏色问题，output_bps=16表示输出是 16 bit (2^16=65536)需要转换一次
        im = rawImg.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        rgb = np.float32(im/65535.0*255.0)        
        img = np.asarray(rgb,np.uint8)
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        img = cv2.merge([b,g,r])#合并通道
        path = "./test_img_bmp/"
        if not os.path.isdir(path):
               os.mkdir(path)
        cv2.imwrite('./test_img_bmp/' + str(a) + '.bmp', img)
        
    else:
       img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img_path.split( ".")[-1].lower() == "cr2":
        n = 8  #2020
        n1 = 8  #709
        x = 2
        B, G, R = cv2.split(img)

    width, height, color = img.shape
    print(width, height)
    
    
    R_M = np.zeros((width, height), dtype=np.single)
    G_M = np.zeros((width, height), dtype=np.single)
    B_M = np.zeros((width, height), dtype=np.single)

#BT2020 to 709    
    
 #Q RGB   
    R_Q = (R / (2 ** (n - 8)) - 16) / 219
    G_Q = (G / (2 ** (n - 8)) - 16) / 219
    B_Q = (B / (2 ** (n - 8)) - 16) / 219
    
 #上升曲線
    count = 0
    i = 0


    max_rgb =1.092
    
    min_rgb =-0.074

    R_Q = (R_Q - min_rgb) / (max_rgb - min_rgb)
    G_Q = (G_Q - min_rgb) / (max_rgb - min_rgb)
    B_Q = (B_Q - min_rgb) / (max_rgb - min_rgb)        
                
    R_up = R_Q
    G_up = G_Q
    B_up = B_Q
        
    print("Gamma up")

    R_up = np.where((R_up>=0) & (R_up<=1), R_up ** x, R_up)
    G_up = np.where((G_up>=0) & (G_up<=1), G_up ** x, G_up)
    B_up = np.where((B_up>=0) & (B_up<=1), B_up ** x, B_up)
    
    print("BT2020 to CIE")
    
#BT2020 to CIExy

    R_cie = 0.637 * R_up + 0.1446 * G_up + 0.1689 * B_up
    G_cie = 0.2627 * R_up + 0.678 * G_up + 0.0593 * B_up
    B_cie = 0 * R_up + 0.0281 * G_up + 1.061 * B_up
    
    RGB_sum = R_cie + G_cie + B_cie
    
    max_sum = np.max(RGB_sum)
    
    CIE_x = R_cie/RGB_sum
    CIE_y = G_cie/RGB_sum
    CIE_z = 1 - CIE_x - CIE_y
    
    if plt_2020_CIE==1:
        
       path = "./2020_CIE/"
       if not os.path.isdir(path):
               os.mkdir(path)
               
       print("plot 2020 CIE_xy plane")
       #畫三角形
       ##plt.plot(CIE_x, CIE_y, "o" ,color='black')
       plot_chromaticity_diagram_CIE1931(standalone=False)
       
       #downsampling (CIE_x, CIE_y) by 2-D 1/10
       CIE_x_dwn_sp = CIE_x[::CIE_dwn_sp_ratio,:]
       CIE_y_dwn_sp = CIE_y[::CIE_dwn_sp_ratio,:]       
    
       plt.scatter(CIE_x_dwn_sp, CIE_y_dwn_sp,  s=CIE_s,  c='black', alpha=CIE_alpha)

       t1 = plt.Polygon([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], facecolor="none", edgecolor='black')
       plt.gca().add_patch(t1)

       t2 = plt.Polygon([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], facecolor="none", edgecolor='black')
       plt.gca().add_patch(t2)
       
       plt.savefig('./2020_CIE/'+ str(a) + '_2020_to_CIE_plot.png')

       # Displaying the plot.
       render(
       standalone=True,
       limits=(-0.1, 0.9, -0.1, 0.9),
       x_tighten=True,
       y_tighten=True)       
   
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
    
    #region 1
    CIE_x = np.where((F1 > 0) & (F2 < 0) & (F3 < 0), R_709_x, CIE_x)
    CIE_y = np.where((F1 > 0) & (F2 < 0) & (F3 < 0), R_709_y, CIE_y)
    
    #region 2
    CIE_x = np.where((F1 > 0) & (F2 < 0) & (F3 > 0), X1, CIE_x)
    CIE_y = np.where((F1 > 0) & (F2 < 0) & (F3 > 0), Y1, CIE_y)
    
    #region 3
    CIE_x = np.where((F1 > 0) & (F2 > 0) & (F3 > 0), G_709_x, CIE_x)
    CIE_y = np.where((F1 > 0) & (F2 > 0) & (F3 > 0), G_709_y, CIE_y)

    #region 4
    CIE_x = np.where((F1 < 0) & (F2 > 0) & (F3 > 0), X2, CIE_x)
    CIE_y = np.where((F1 < 0) & (F2 > 0) & (F3 > 0), Y2, CIE_y)

    #region 5
    CIE_x = np.where((F1 < 0) & (F2 > 0) & (F3 < 0), B_709_x, CIE_x)
    CIE_y = np.where((F1 < 0) & (F2 > 0) & (F3 < 0), B_709_y, CIE_y)

    #region 6
    CIE_x = np.where((F1 < 0) & (F2 < 0) & (F3 < 0), X3, CIE_x)
    CIE_y = np.where((F1 < 0) & (F2 < 0) & (F3 < 0), Y3, CIE_y) 
    
    print("find new cie points")
    
    R_cie = CIE_x*RGB_sum
    G_cie = CIE_y*RGB_sum
    B_cie = RGB_sum - R_cie - G_cie
    
    print("CIE to BT709")
    
    R_M = 3.2410 * R_cie - 1.5374 * G_cie - 0.4986 * B_cie
    G_M = -0.9692 * R_cie + 1.8760 * G_cie + 0.0416 * B_cie
    B_M = 0.0556 * R_cie - 0.2040 * G_cie + 1.0570 * B_cie
    
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
    
    RGB_sum_re = R_cie_re + G_cie_re + B_cie_re
      
    CIE_x_re = R_cie_re/RGB_sum_re
    CIE_y_re = G_cie_re/RGB_sum_re
    CIE_z_re = 1 - CIE_x_re - CIE_y_re
    
    if plt_re_709_CIE==1:
       
       path = "./to_709_CIE/"
       if not os.path.isdir(path):
               os.mkdir(path)
        
       print("plot 2020_to_709 CIE_xy plane")
       #畫三角形
       ##plt.plot(CIE_x, CIE_y, "o" ,color='black')
       plot_chromaticity_diagram_CIE1931(standalone=False)
       
       #downsampling (CIE_x, CIE_y) by 2-D 1/10
       CIE_x_re_dwn_sp = CIE_x_re[::CIE_dwn_sp_ratio,:]
       CIE_y_re_dwn_sp = CIE_y_re[::CIE_dwn_sp_ratio,:]       
    
       plt.scatter(CIE_x_re_dwn_sp, CIE_y_re_dwn_sp,  s=CIE_s,  c='black', alpha=CIE_alpha)

       t1 = plt.Polygon([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], facecolor="none", edgecolor='black')
       plt.gca().add_patch(t1)

       t2 = plt.Polygon([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], facecolor="none", edgecolor='black')
       plt.gca().add_patch(t2)

       plt.savefig('./to_709_CIE/' + str(a) + '_2020_to_709_CIE_plot.png')

       # Displaying the plot.
       render(
       standalone=True,
       limits=(-0.1, 0.9, -0.1, 0.9),
       x_tighten=True,
       y_tighten=True)
    
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

    print("2020 to 709 ok")
    
    #R、G、B的合併
    img = cv2.merge([B_Q1,G_Q1,R_Q1])
#存圖 
    #arr = np.array(img , dtype = 'uint8')
    #arr=Image.fromarray(arr)
    #arr.save('D:/huh/python/SW_code/' + str(a) + '_2020_to_709.bmp' )
    path = "./out_img_709/"
    if not os.path.isdir(path):
           os.mkdir(path)
    
    cv2.imwrite('./out_img_709/' + str(a) + '_2020_to_709.bmp', img)
    
#709 TO 2020
#Q RGB

    R_up1 = (R_Q1 / ( 2 ** ( n - 8 )) - 16) / 219
    G_up1 = (G_Q1 / ( 2 ** ( n - 8 )) - 16) / 219
    B_up1 = (B_Q1 / ( 2 ** ( n - 8 )) - 16) / 219

#上升曲線

    R_up1 = (R_up1 - min_rgb) / (max_rgb - min_rgb)
    G_up1 = (G_up1 - min_rgb) / (max_rgb - min_rgb)
    B_up1 = (B_up1 - min_rgb) / (max_rgb - min_rgb)        
             
        

    R_up1 = np.where((R_up1 >= 0) & (R_up1 <= 1), R_up1 ** x, R_up1)
    G_up1 = np.where((G_up1 >= 0) & (G_up1 <= 1), G_up1 ** x, G_up1)
    B_up1 = np.where((B_up1 >= 0) & (B_up1 <= 1), B_up1 ** x, B_up1)
            
            
            
#M2
    print("BT709 to BT2020")    
    
    R_M1 = 0.6274 * R_up1 + 0.3293 * G_up1 + 0.0443 * B_up1
    G_M1 = 0.0691 * R_up1 + 0.9195 * G_up1 + 0.0114 * B_up1
    B_M1 = 0.0164 * R_up1 + 0.0880 * G_up1 + 0.8956 * B_up1
    
#下降曲線

    R_M1 = np.where((R_M1<0.0), 0.0, R_M1)
    R_M1 = np.where((R_M1>1.0), 1.0, R_M1)
    
    G_M1 = np.where((G_M1<0.0), 0.0, G_M1)
    G_M1 = np.where((G_M1>1.0), 1.0, G_M1)
    
    B_M1 = np.where((B_M1<0.0), 0.0, B_M1)
    B_M1 = np.where((B_M1>1.0), 1.0, B_M1)

    R_M1 = np.where((R_M1 >= 0) & (R_M1 <= 1), R_M1 ** (1/x), R_M1)
    G_M1 = np.where((G_M1 >= 0) & (G_M1 <= 1), G_M1 ** (1/x), G_M1)
    B_M1 = np.where((B_M1 >= 0) & (B_M1 <= 1), B_M1 ** (1/x), B_M1)
    
    R_M1 =  (max_rgb - min_rgb)*R_M1 + min_rgb
    G_M1 =  (max_rgb - min_rgb)*G_M1 + min_rgb
    B_M1 =  (max_rgb - min_rgb)*B_M1 + min_rgb
                
#Q RGB2
    R_E = (219 * R_M1 + 16) * 2 ** (n1 - 8)
    G_E = (219 * G_M1 + 16) * 2 ** (n1 - 8)
    B_E = (219 * B_M1 + 16) * 2 ** (n1 - 8)
    
    R_E = np.where((R_E<0), 0, R_E)
    R_E = np.where((R_E>255), 255, R_E)
    G_E = np.where((G_E<0), 0, G_E)
    G_E = np.where((G_E>255), 255, G_E)
    B_E = np.where((B_E<0), 0, B_E)
    B_E = np.where((B_E>255), 255, B_E)

#R、G、B的合併
    img = cv2.merge([B_E,G_E,R_E])
#存圖 
    #arr = np.array(img , dtype = 'uint8')
    print("OK")
    #arr=Image.fromarray(arr)
    #arr.save('D:/huh/python/SW_code/' + str(a) + '_709_to_2020.bmp' )
    path = "./out_img_2020/"
    if not os.path.isdir(path):
           os.mkdir(path)
           
    cv2.imwrite('./out_img_2020/' + str(a) + '.bmp', img)    
    
##    cv2.imshow("Merged",merged)
##    cv2.waitKey(0)
    
    print("BT2020 to CIE")
    
    R_Q2 = (R_E / (2 ** (n - 8)) - 16) / 219
    G_Q2 = (G_E / (2 ** (n - 8)) - 16) / 219
    B_Q2 = (B_E / (2 ** (n - 8)) - 16) / 219
    
    R_Q2 = (R_Q2 - min_rgb) / (max_rgb - min_rgb)
    G_Q2 = (G_Q2 - min_rgb) / (max_rgb - min_rgb)
    B_Q2 = (B_Q2 - min_rgb) / (max_rgb - min_rgb)

        
    R_up2 = R_Q2
    G_up2 = G_Q2
    B_up2 = B_Q2

    R_up2 = np.where((R_up2>=0) & (R_up2<=1), R_up2 ** x, R_up2)
    G_up2 = np.where((G_up2>=0) & (G_up2<=1), G_up2 ** x, G_up2)
    B_up2 = np.where((B_up2>=0) & (B_up2<=1), B_up2 ** x, B_up2)
            
       
    #M  2020 to cie
    R_M2 = 0.637 * R_up2 + 0.1446 * G_up2 + 0.1689 * B_up2
    G_M2 = 0.2627 * R_up2 + 0.678 * G_up2 + 0.0593 * B_up2
    B_M2 = 0.000 * R_up2 + 0.0281 * G_up2 + 1.061 * B_up2


    RGB_s2 = R_M2 + G_M2 + B_M2
    x_M2 = R_M2 / RGB_s2
    y_M2 = G_M2 / RGB_s2
    
    if plt_re_2020_CIE==1:
        
       path = "./to_709_to_2020_CIE/"
       if not os.path.isdir(path):
               os.mkdir(path)
               
       print("plot 709_to_2020 CIE_xy plane")
       #畫三角形
       ##plt.plot(CIE_x, CIE_y, "o" ,color='black')
       plot_chromaticity_diagram_CIE1931(standalone=False)
       
       #downsampling (CIE_x, CIE_y) by 2-D 1/10
       x_M2_dwn_sp = x_M2[::CIE_dwn_sp_ratio,:]
       y_M2_dwn_sp = y_M2[::CIE_dwn_sp_ratio,:]
    
       plt.scatter(x_M2_dwn_sp, y_M2_dwn_sp,  s=CIE_s,  c='black', alpha=CIE_alpha)

       t1 = plt.Polygon([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], facecolor="none", edgecolor='black')
       plt.gca().add_patch(t1)

       t2 = plt.Polygon([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], facecolor="none", edgecolor='black')
       plt.gca().add_patch(t2)
       
       plt.savefig('./to_709_to_2020_CIE/' + str(a) + '_2020_to_709_to_2020_CIE_plot.png')
       
       # Displaying the plot.
       render(
       standalone=True,
       limits=(-0.1, 0.9, -0.1, 0.9),
       x_tighten=True,
       y_tighten=True)
       
    print("End")
