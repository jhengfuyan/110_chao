# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:08:06 2019

@author: Richard
"""

#import HDRReader
import imageio
import numpy as np
#n=1
#merged = np.zeros((6720,4480,3))
#for i in range(64,merged.shape[0]+64,64):
#    for j in range(64,merged.shape[1]+64,64):
#        temp=imageio.imread('C:/Users/USER/Desktop/test_graph/test_graph_cut/'+str(n)+'.bmp') 
#        merged[i-64:i,j-64:j,:]=temp
#        n=n+1
#        #print(temp.shape)
#        
#merged = np.uint8(merged)
#imageio.imwrite('C:/Users/USER/Desktop/test_graph/test_graph_merged/12.bmp',merged,format='.bmp' )  #輸入 nparrray float32


#m = imageio.imread('D:/tensorflow/BasketballCourt_3k12bit.hdr')
#m = imageio.imread('C:/Users/User/Desktop/picture_best/s.hdr')
#print(m.shape)

#overlapping
n=1
for x in range(5,6): 

    merged = np.zeros((4480, 6720,3))#返回來一個給定形狀和類型的用0填充的數組；
    for i in range(32,merged.shape[0]+16,32):
        for j in range(32,merged.shape[1]+16,32):
            temp=imageio.imread('C:/Users/User/Desktop/TC/o/'+str(n)+'.bmp')
            print(temp.shape)
            merged[i-32:i,j-32:j,:]=temp[16:temp.shape[0]-16,16:temp.shape[1]-16,:3]
            print(merged.shape)
            n=n+1
merged = np.uint8(merged)
imageio.imwrite('C:/Users/User/Desktop/TC/2.bmp',merged,format='.bmp' ) #輸入 nparrray float32for x in range(5,6): 
#merged[i-64:i,j-64:j,:]=temp[16:temp.shape[0]-16,16:temp.shape[1]-16,:]

