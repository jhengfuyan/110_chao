import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import rawpy
import cv2
import imageio

def rmse(predictions, targets):
    
    predictions = predictions.astype(np.float64)
    
    targets = targets.astype(np.float64)

    differences = predictions - targets     #the DIFFERENCEs

    differences_squared = differences ** 2  #the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()    #the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)     #ROOT of ^

    return rmse_val

def rmsepsnr():
#    for i in range(5, 6):
        
#   CR2檔案
#        REF = rawpy.imread("7.CR2")
#        REF = REF.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#        REF = np.float32(REF/65535.0*255.0)
#        REF = np.asarray(REF,np.uint8)
#        r,g,b = cv2.split(REF)#拆分通道
#        REF = cv2.merge([b,g,r])#合并通道
#        cv2.namedWindow('REF', cv2.WINDOW_NORMAL)
#        cv2.imshow("REF", REF)
 
#  D:/project_use/classify4/img_out/reconstruct1.bmp  
#D:/for_tech/img_out/reconstruct1dev4.hdr    
#   bmp檔案        
        REF = cv2.imread("./overlap/bluebird/img_out/1.bmp")
#        print(REF.shape)
        cv2.namedWindow('REF', cv2.WINDOW_NORMAL)
        cv2.imshow("REF", REF) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        TAR = cv2.imread("./overlap/bluebird/1.bmp")
        cv2.namedWindow('TAR', cv2.WINDOW_NORMAL)
        cv2.imshow("TAR", TAR) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print('RMSE:', rmse(REF, TAR))

        REF = tf.cast(REF, tf.float64)
        TAR = tf.cast(TAR, tf.float64)
        psnr = tf.image.psnr(REF, TAR, max_val=255)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("PSNR:", psnr.eval(session=sess))

rmsepsnr()


#SSIM
#def cal_ssim(im1,im2):
#      assert len(im1.shape) == 2 and len(im2.shape) == 2
#      assert im1.shape == im2.shape
#      mu1 = im1.mean()
#      mu2 = im2.mean()
#      sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
#      sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
#      sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
#      k1, k2, L = 0.01, 0.03, 255
#      C1 = (k1*L) ** 2
#      C2 = (k2*L) ** 2
#      C3 = C2/2
#      l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
#      c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
#      s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
#      ssim = l12 * c12 * s12
#      return ssim
