import tensorflow as tf
import numpy as np
import os
import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#test
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list =tf.train.FloatList(value=value))

def get_files_(DIR):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    picturebt2020 = []
    for i in range(1,29401):
            picturebt2020.append(DIR + str(i) +".bmp")
           # label_pictureLDR.append(i)
           # i=i+1
    print('There are %d picturebt2020\n' %(len(picturebt2020)))
    
    imagebt2020_list = picturebt2020
    
    
    return imagebt2020_list
def convert_to_tfrecord_(bt2020s, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(bt2020s)
    print(n_samples)
    
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in range(0, n_samples):
        try:
            if i%1000==0:
                print(i)
            bt2020 = np.float32(imageio.imread(bt2020s[i])) # type(image) must be array!
            bt2020 = bt2020.flatten()
            example = tf.train.Example(features=tf.train.Features(feature={
                            'bt2020': _floats_feature(bt2020)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', bt2020s[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
save_dir = './overlap/redbird/'


name_test = 'Seaview-overlapping-64x64'
bt2020s= get_files_('./overlap/redbird/o/')
convert_to_tfrecord_(bt2020s, save_dir, name_test)