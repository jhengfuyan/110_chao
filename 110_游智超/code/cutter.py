import imageio
import numpy as np
import os

#for a in range (1 ,92):
#    m = imageio.imread('D:/training91-2020/' + str(a) + '.bmp')
#    print(m.shape)
#    m_resize = np.zeros((m.shape[0] + 32 , m.shape[1] + 32 , m.shape[2]))
#    m_resize[16 : m_resize.shape[0] - 16 , 16 : m_resize.shape[1] - 16,:] = m

#    n = 1

#    cut = np.zeros((32 , 32 , 3))
#    cut = np.float32(cut)
#    print(m_resize.shape)

#    for i in range(32 , m_resize.shape[0] + 16 , 16):
#        for j in range(32 , m_resize.shape[1] + 16 , 16):
#            cut = m_resize[ i - 32 : i , j - 32 : j , : ]
#            cut = np.float32(cut)
#            imageio.imwrite('D:/training91-cutter/' + str(n) +'.bmp' , cut , format = 'bmp')
#            n=n+1

#%%
#m = imageio.imread('C:/Users/eric/Desktop/Own_code/Original _image/1.bmp')
#print(m.shape)
n=1

cut = np.zeros((64,64,3))
cut = np.uint8(cut)
##for file in os.listdir('D:/Downloads/Raise/dataset1/HDR/'):
for i in range(1,88):
    for a in range(1,2):
        m = imageio.imread('C:/Users/User/Desktop/run2/2020img/1' + str(i) + '.bmp')
        print(m.shape)  
        print(a)
        if m.shape[0]%64==0:
            x=64
        else:
            x=0
        if m.shape[1]%64==0:
            y=64
        else:
            y=0
        for i in range(64,m.shape[0]+x,64):
            for j in range(64,m.shape[1]+y,64):
                cut=m[i-64:i,j-64:j,:]
                cut = np.uint8(cut)
                imageio.imwrite('C:/Users/User/Desktop/run2/test1/' + str(n) +'.bmp' , cut , format = 'bmp' ) #輸入 nparrray float32
                n=n+1
    i=i+1;

