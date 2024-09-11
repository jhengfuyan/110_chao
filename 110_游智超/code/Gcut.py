# -*- coding: utf-8 -*-
"""9
Created on Mon Jan 28 22:26:45 2019

@author: Richard
"""
import imageio
import numpy as np
import os

#目標圖跟訓練圖要一樣多

#64*64
n=1
#m = imageio.imread('C:/Users/USER/Desktop/test_graph/test_graph_orginal/7.bmp')
#print(m)
#cut = np.zeros((64,64,3))
#cut = np.uint8(cut)
##for file in os.listdir('C:/Users/USER/Desktop/test_graph/test_graph_orginal/7.bmp'):
#for i in range(1,2):#裁的張數(range的1是從圖片的檔名1開始，如果是7那就是從7開始)
#    for a in range(1,2):#一次丟幾張照片進來裁
#        m = imageio.imread('C:/Users/USER/Desktop/test_graph/test_graph_orginal/'+str(i)+'.bmp')
#        print(m.shape)  
#        #print(m)
#        #print(a)
#        if m.shape[0]%64==0:
#            x=64
#        else:
#            x=0
#        if m.shape[1]%64==0:
#            y=64
#        else:
#            y=0
#        for i in range(64,m.shape[0]+x,64):
#            for j in range(64,m.shape[1]+y,64):
#                cut=m[i-64:i,j-64:j]#最後一個冒號代表通道數
#                cut = np.uint8(cut)
#                #cut = np.float32(cut)
#                imageio.imwrite('C:/Users/USER/Desktop/test_graph/test_graph_cut/'+str(n)+'.bmp',cut,format='bmp' ) #輸入 nparrray float32
#                print(cut)
#                n=n+1
#    i=i+1;

#32*32
#n=1
#m = imageio.imread('C:/Users/USER/Desktop/test_graph/test_graph_orginal/1.bmp')
#print(m.shape)
#cut = np.zeros((32,32,3))
#cut = np.uint8(cut)
##for file in os.listdir('D:/Downloads/Raise/dataset1/HDR/'):
#for i in range(5,47):
#    for a in range(1,2):
#        m = imageio.imread('D:/tensorflow_c/tensorflow/HDREye/LDR/'+str(i)+'.hdr')
#        print(m.shape)  
##        print(m)
#        print(a)
#        if m.shape[0]%32==0:
#            x=32
#        else:
#            x=0
#        if m.shape[1]%32==0:
#            y=32
#        else:
#            y=0
#        for i in range(32,m.shape[0]+x,32):
#            for j in range(32,m.shape[1]+y,32):
#                cut=m[i-32:i,j-32:j,:]
#                cut = np.float32(cut)
#                imageio.imwrite('D:/tensorflow_c/tensorflow/HDREye/LDR_CUT_32/'+str(n)+'.hdr',cut,format='hdr' ) #輸入 nparrray float32
##                print(cut)
#                n=n+1
#    i=i+1;

#for a in range(5,47):
#    m = imageio.imread('C:/tensorflow/trainimage/NEW/HDR/'+str(a)+'.hdr')
#    print(m.shape)  
#    print(a) 
#    for i in range(32,m.shape[0],32):
#        for j in range(32,m.shape[1]+32,32):
#            cut=m[i-32:i,j-32:j,:]
#            cut = np.float32(cut)
#            imageio.imwrite('C:/tensorflow/HDR_CNN/data/train/HDR/'+str(n)+'.hdr',cut,format='hdr' ) #輸入 nparrray float32
#            n=n+1
#        num=num+1
#    else:
#        for i in range(256,m.shape[0],256):
#            for j in range(256,m.shape[1],256):
#                cut=m[i-256:i,j-256:j,:]
#                cut = np.float32(cut)
#                imageio.imwrite('C:/tensorflow/HDR_CNN/data/train/HDR/'+str(n)+'.hdr',cut,format='hdr' ) #輸入 nparrray float32
#                n=n+1
#        num=num+1

    
#overlapping是用來避免出圖之後圖中呈現一條一條的分隔線
#看照片大小
#m = imageio.imread('C:/Users/User/Desktop/picture_best/1/Arches_E_PineTree.hdr')
#print(m.shape)#印出m的矩陣或維數
m = imageio.imread('C:/Users/User/Desktop/TC/1.bmp')
print(m.shape)
m_resize = np.zeros((m.shape[0]+10,m.shape[1]+8,m.shape[2]))
print(m_resize.shape)
m_resize[5:m_resize.shape[0]-5,4:m_resize.shape[1]-4,:]=m
print(m_resize.shape)
#m_resize = np.zeros((m.shape[0]+42,m.shape[1]+42,m.shape[2]))
#m_resize[21:m_resize.shape[0]-21,21:m_resize.shape[1]-21,:]=m
#print(m_resize.shape)
#np.pad(m_resize,((0,8),(8,0),(0,0)),'constant',constant_values = (0,0))
n=1
cut = np.zeros((64,64,3))
cut = np.uint8(cut)
#print(m_resize.shape)    
for i in range(64,m_resize.shape[0]+32,32):
    for j in range(64,m_resize.shape[1]+32,32):
        cut=m_resize[i-64:i,j-64:j,:]#把圖片切成64*64
        cut = np.uint8(cut)
        imageio.imwrite('C:/Users/User/Desktop/TC/o/'+str(n)+'.bmp',cut,format='bmp' ) #輸入 nparrray float32
        if (cut.shape[0]!=64 or cut.shape[1]!=64):
            print(str(n)+'.',cut.shape)
        n=n+1