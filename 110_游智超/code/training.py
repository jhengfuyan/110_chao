import  os
import numpy as np
import tensorflow as tf
import model
import cv2
#import modela
import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow
tensorflow.__version__

#%%

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
config.gpu_options.allow_growth = True
 
sess0 = tf.InteractiveSession(config = config)
#%%

BATCH_SIZE = 64 #一次炫戀的圖
learning_rate = 1e-2#啜物綠的降點
epochs = 50000 #加設我有100張炫練50000次

#%%
def run_training():
    #將目錄更改為你的目錄
    logs_train_dir = './log64/' #學長是train的目錄
    filenames = './TFRecord/64x64-float32-training.tfrecords' #學長是64x64x4095x78.tfrecords的目錄
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat(epochs)
    is_training = True
    
    def parser(record):
        img_features ={
                        'cnn2020': tf.VarLenFeature(tf.float32),
                        'bt2020': tf.VarLenFeature(tf.float32),
                            }
        parsed = tf.parse_single_example(record , img_features)
        bt2020 = tf.sparse_tensor_to_dense(parsed['bt2020'], default_value = 0)
        cnn2020 = tf.sparse_tensor_to_dense(parsed['cnn2020'], default_value = 0)
        bt2020 = tf.reshape(bt2020, [64,64,3])
        cnn2020 = tf.reshape(cnn2020, [64,64,3])
        print(cnn2020.shape)
        return bt2020,cnn2020
  

    print('here1')
    dataset = dataset.shuffle(buffer_size = 100000)
    dataset = dataset.map(parser, num_parallel_calls = 8)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size = 360)
    iterator = dataset.make_initializable_iterator()
    bt2020,cnn2020 =iterator.get_next()  
    print('here2')
#    bt2020 = RGB2Lab(bt2020)
#    cnn2020 = RGB2Lab(cnn2020)
    print(bt2020.shape)
    print(cnn2020.shape)
    
    train_logits = model.inference(bt2020 , is_training)
    print(train_logits.shape)
    train_loss = model.losses(train_logits , cnn2020)
    
    print(train_loss.shape)
    
    train_op = model.trainning(train_loss , learning_rate)
    print(train_op)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir , sess.graph)
    saver = tf.train.Saver(var_list = tf.global_variables())
    epoch_loss = 0
    epoch_min = 5000
    step = 0
    
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())

#讀檔
    with tf.Graph().as_default():
        print("Reading checkpoints...")
        
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess , ckpt.model_checkpoint_path)
        
            print('Loading success , global_step is %s' % global_step)
        
        else:
        
            print('No checkpoint file cound')
    try:
        while True:
            
            _, tra_loss = sess.run([train_op , train_loss])
            step += 1
            epoch_loss = epoch_loss + tra_loss
            
            if step % 50 == 0 :
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str , step)
                
            if step % 1776 == 0 :
                epoch_loss_mean_average = epoch_loss / 1776
                epoch_loss = 0
                
                if epoch_min > epoch_loss_mean_average :
                    print('Step %d, epoch_loss = %f, ' % (step , epoch_loss_mean_average))
                    epoch_min = epoch_loss_mean_average
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess , checkpoint_path , global_step =step)
    
    except tf.errors.OutOfRangeError:
      print("end!")
      print('Step %d  ' % (step ))
      
#%%     
def evaluate_image():
     
    #將目錄更改為你的目錄
    logs_train_dir = './log64/'#train的檔案
    filenames = './bluebird/bluebird-overlapping-64x64.tfrecords' #wasteland_clouds_4k12bitTM.tfrecords的檔案
    is_training = False
    dataset = tf.data.TFRecordDataset(filenames)
    def parser(record):
        img_features = {
                        'bt2020' : tf.VarLenFeature(tf.float32),
                       }
        parsed = tf.parse_single_example(record , img_features)
        bt2020 = tf.sparse_tensor_to_dense(parsed ['bt2020'] , default_value = 0)
        bt2020 = tf.reshape(bt2020 , [ 64 , 64 , 3])
        
        return bt2020
    
    dataset = dataset.map(parser)
    dataset = dataset.repeat(1).batch(1)
    iterator = dataset.make_initializable_iterator()
    bt2020 = iterator.get_next()
    train_logits = model.inference(bt2020 , is_training)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        with tf.Graph().as_default():
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess , ckpt.model_checkpoint_path)
                print('Loading success , global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
        n=1
        
        try:
            while True:

                out = sess.run(train_logits)
                m = out[0,:,:,:]
                np.clip( m, 0 , 255 ,out = m)
                img = np.asarray(m,np.uint8)
#                Lab2RGB(img)
                imageio.imwrite('./evaluate_image_out/' + str(n) + '.bmp' , img ,format = 'bmp')
            
                if n % 10000 == 0 :
                    print(n)
                n = n + 1
        except tf.errors.OutOfRangeError:
            print("end!")