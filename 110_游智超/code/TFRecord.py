import tensorflow as tf
import numpy as np
import os
import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

train_bt2020_dir = 'C:/Users/USER/Desktop/jerry_data/chao_code0701/test_test1/'
train_cnn2020_dir = 'C:/Users/USER/Desktop/jerry_data/chao_code0701/out_2020_test1/'

def get_files(filebt2020_dir,filecnn2020_dir):

    picturebt2020 = []

    picturecnn2020 = []


    for i in range(1,7351):
            picturebt2020.append(filebt2020_dir + str(i) +".bmp")

    for i in range(1,7351):
            picturecnn2020.append(filecnn2020_dir + str(i) +".bmp")

    print('There are %d picturebt2020\nThere are %d picturecnn2020' %(len(picturebt2020),len(picturecnn2020)))
    
    imagebt2020_list = picturebt2020
    imagecnn2020_list = picturecnn2020
    
    temp = np.array([imagebt2020_list, imagecnn2020_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    imagebt2020_list = list(temp[: , 0])
    imagecnn2020_list = list(temp[: , 1])
    
    return imagebt2020_list, imagecnn2020_list

    
def bytes_feature(value):
  return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to_tfrecord(bt2020s, cnn2020s, save_dir, name):
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(cnn2020s)
    print(n_samples)


    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in range(0, n_samples):
        try:
            if i%1000==0:
                print(i)
            bt2020_lab = np.float32(imageio.imread(bt2020s[i])) # type(image) must be array!
            cnn2020_lab = np.float32(imageio.imread(cnn2020s[i])) # type(image) must be array!

            bt2020_lab = bt2020_lab.flatten()
            cnn2020_lab = cnn2020_lab.flatten()
            example = tf.train.Example(features=tf.train.Features(feature={
                            'cnn2020': _floats_feature(cnn2020_lab),
                            'bt2020': _floats_feature(bt2020_lab)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', bt2020s[i])
            print('Could not read:', cnn2020s[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
    


save_dir = 'C:/Users/USER/Desktop/jerry_data/chao_code0701/TFRecord_11/'

name_test = '64x64-float32-training'
bt2020s, cnn2020s = get_files(train_bt2020_dir,train_cnn2020_dir)
convert_to_tfrecord(bt2020s, cnn2020s, save_dir, name_test)


##test
#def bytes_feature(value):
#  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#def _int64_feature(value):
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#
#def get_files_(DIR):
#    '''
#    Args:
#        file_dir: file directory
#    Returns:
#        list of images and labels
#    '''
#    picturebt2020 = []
#    for i in range(1,29401):
#            picturebt2020.append(DIR + str(i) +".bmp")
#           # label_pictureLDR.append(i)
#           # i=i+1
#    print('There are %d picturebt2020\n' %(len(picturebt2020)))
#    
#    imagebt2020_list = picturebt2020
#    
#    
#    return imagebt2020_list
#def convert_to_tfrecord_(bt2020s, save_dir, name):
##    '''convert all images and labels to one tfrecord file.
##    Args:
##        images: list of image directories, string type
##        labels: list of labels, int type
##        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
##        name: the name of tfrecord file, string type, e.g.: 'train'
##    Return:
##        no return
##    Note:
##        converting needs some time, be patient...
##    '''
#    
#    filename = os.path.join(save_dir, name + '.tfrecords')
#    n_samples = len(bt2020s)
#    print(n_samples)
#    
#    writer = tf.python_io.TFRecordWriter(filename)
#    print('\nTransform start......')
#    for i in range(0, n_samples):
#        try:
#            if i%1000==0:
#                print(i)
#            bt2020 = np.uint8(imageio.imread(bt2020s[i])*255) # type(image) must be array!
#            bt2020 = bt2020.flatten()
#            example = tf.train.Example(features=tf.train.Features(feature={
#                            'bt2020': _int64_feature(bt2020)}))
#            writer.write(example.SerializeToString())
#        except IOError as e:
#            print('Could not read:', bt2020s[i])
#            print('error: %s' %e)
#            print('Skip it!\n')
#    writer.close()
#    print('Transform done!')
#save_dir = 'C:/Users/User/Desktop/run5/TFRecord_out/'
#
#
#name_test = 'bluebird-overlapping-64x64'
#bt2020s= get_files_('C:/Users/User/Desktop/run5/picture_cut/')
#convert_to_tfrecord_(bt2020s, save_dir, name_test)