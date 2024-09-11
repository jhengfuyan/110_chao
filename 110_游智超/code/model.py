
#%%

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

#%%
def inference(images,is_training):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    def convs_s(inputs,kernelw,kernelh,in_chanel,out_chanel,strides,is_training):
        weights = tf.get_variable('weights', 
                                  shape = [kernelw,kernelh,in_chanel, out_chanel],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[out_chanel],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(inputs, weights, strides=[1,strides,strides,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        y_bn1 = tf.layers.batch_normalization(pre_activation, training=is_training,name=scope)        
        return y_bn1

    def convs_v(inputs,kernelw,kernelh,in_chanel,out_chanel,strides,is_training):
        weights = tf.get_variable('weights', 
                                  shape = [kernelw,kernelh,in_chanel, out_chanel],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[out_chanel],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(inputs, weights, strides=[1,strides,strides,1], padding='VAILD')
        pre_activation = tf.nn.bias_add(conv, biases)
        y_bn1 = tf.layers.batch_normalization(pre_activation, training=is_training,name=scope)        
        return y_bn1

        
    with tf.variable_scope('res') as scope:
            res = convs_s(images,3,3,3,64,1,is_training)
            res = tf.nn.relu(res, name= 'res')
        
    with tf.variable_scope('res6_1') as scope:
            res6_1 = convs_s(res,3,3,64,64,1,is_training)        
            res6_1 = tf.nn.relu((res6_1), name= 'res6_1')   
         
    with tf.variable_scope('res6_2') as scope:
            res6_2 = convs_s(res6_1,3,3,64,64,1,is_training)        
            res6_2 = tf.nn.relu((res6_2), name= 'res6_2')  
    res_1o = res + res6_2
    with tf.variable_scope('res6_3') as scope:
            res6_3 = convs_s(res_1o,3,3,64,64,1,is_training)        
            res6_3 = tf.nn.relu((res6_3), name= 'res6_3')   
    
    with tf.variable_scope('res6_4') as scope:
            res6_4 = convs_s(res6_3,3,3,64,64,1,is_training)        
            res6_4 = tf.nn.relu((res6_4), name= 'res6_4')  
    res_2o = res_1o + res6_4 
    
    with tf.variable_scope('res_5') as scope:
            res_5 = convs_s(res_2o,3,3,64,64,1,is_training)        
            res_5 = tf.nn.relu((res_5), name= 'res_5')    
    
    with tf.variable_scope('res_6') as scope:
            res_6 = convs_s(res_5,3,3,64,64,1,is_training)        
            res_6 = tf.nn.relu((res_6), name= 'res_6')  
    res_3o = res_2o + res_6    
    

    with tf.variable_scope('res6_17') as scope:
            res6_17 = convs_s(res_3o,3,3,64,3,1,is_training)        
            res6_17 = tf.nn.relu((res6_17), name= 'res6_17') 
    with tf.variable_scope('res6_18') as scope:
            res6_18 = convs_s(res6_17,1,1,3,3,1,is_training)        
            res6_18 = tf.nn.relu((res6_18), name= 'res6_18')     
    with tf.variable_scope('res6_19') as scope:
            res6_19 = convs_s(res6_18,1,1,3,3,1,is_training)        
            res6_19 = tf.nn.relu((res6_19), name= 'res6_19')   
    res_9o = res6_17 + res6_19
    
    
    with tf.variable_scope('res6_20') as scope:
            res6_20 = convs_s(res_9o,1,1,3,3,1,is_training)        
            res6_20 = tf.nn.relu((res6_20), name= 'res6_20')
    with tf.variable_scope('res6_661') as scope:
            res6_661 = convs_s(res6_20,1,1,3,3,1,is_training)        
            res6_661 = tf.nn.relu((res6_661), name= 'res6_661')
    res_10o = res_9o + res6_661    
    
    
    
    
    









          
    
    
    
    
    
#    
#    with tf.variable_scope('res1') as scope:
#            res1 = convs_s(images,3,3,3,32,1,is_training)
#            res1 = tf.nn.relu(res1, name= 'res1')
#        
#    with tf.variable_scope('res6_11') as scope:
#            res6_11 = convs_s(res1,3,3,32,32,1,is_training)        
#            res6_11 = tf.nn.relu((res6_11), name= 'res6_11')   
#         
#    with tf.variable_scope('res6_21') as scope:
#            res6_21 = convs_s(res6_11,3,3,32,32,1,is_training)        
#            res6_21 = tf.nn.relu((res6_21), name= 'res6_21')  
#    res_11o = res1 + res6_21
#    with tf.variable_scope('res6_31') as scope:
#            res6_31 = convs_s(res_11o,3,3,32,32,1,is_training)        
#            res6_31 = tf.nn.relu((res6_31), name= 'res6_31')   
#    
#    with tf.variable_scope('res6_41') as scope:
#            res6_41 = convs_s(res6_31,3,3,32,32,1,is_training)        
#            res6_41 = tf.nn.relu((res6_41), name= 'res6_41')  
#    res_21o = res_11o + res6_41
#    
#    with tf.variable_scope('res_51') as scope:
#            res_51 = convs_s(res_21o,3,3,32,32,1,is_training)        
#            res_51 = tf.nn.relu((res_51), name= 'res_51')    
#    
#    with tf.variable_scope('res_61') as scope:
#            res_61 = convs_s(res_51,3,3,32,32,1,is_training)        
#            res_61 = tf.nn.relu((res_61), name= 'res_61')  
#    res_31o = res_21o + res_61   
##    
#    with tf.variable_scope('res6_171') as scope:
#            res6_171 = convs_s(res_31o,3,3,32,3,1,is_training)        
#            res6_171 = tf.nn.relu((res6_171), name= 'res6_171') 
#    with tf.variable_scope('res6_181') as scope:
#            res6_181 = convs_s(res6_171,1,1,3,3,1,is_training)        
#            res6_181 = tf.nn.relu((res6_181), name= 'res6_181')     
#    with tf.variable_scope('res6_191') as scope:
#            res6_191 = convs_s(res6_181,1,1,3,3,1,is_training)        
#            res6_191 = tf.nn.relu((res6_191), name= 'res6_191') 
#    res_91o = res6_171 + res6_191
#    
#    with tf.variable_scope('res6_201') as scope:
#            res6_201 = convs_s(res_91o,1,1,3,3,1,is_training)        
#            res6_201 = tf.nn.relu((res6_201), name= 'res6_201')
#    with tf.variable_scope('res6_6661') as scope:
#            res6_6661 = convs_s(res6_201,1,1,3,3,1,is_training)        
#            res6_6661 = tf.nn.relu((res6_6661), name= 'res6_6661')
#    res_101o = res_91o + res6_6661     
##    
##    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



#
##
#    with tf.variable_scope('res11') as scope:
#            res11 = convs_s(images,3,3,3,3,1,is_training)
#            res11 = tf.nn.relu(res11, name= 'res11')
#        
#    with tf.variable_scope('res6_111') as scope:
#            res6_111 = convs_s(res11,3,3,3,3,1,is_training)        
#            res6_111 = tf.nn.relu((res6_111), name= 'res6_111')   
#         
#    with tf.variable_scope('res6_211') as scope:
#            res6_211 = convs_s(res6_111,3,3,3,3,1,is_training)        
#            res6_211 = tf.nn.relu((res6_211), name= 'res6_211')  
#    res_111o = res11 + res6_211
#    
#    with tf.variable_scope('res6_311') as scope:
#            res6_311 = convs_s(res_111o,3,3,3,3,1,is_training)        
#            res6_311 = tf.nn.relu((res6_311), name= 'res6_311')   
#    
#    with tf.variable_scope('res6_411') as scope:
#            res6_411 = convs_s(res6_311,3,3,3,3,1,is_training)        
#            res6_411 = tf.nn.relu((res6_411), name= 'res6_411')  
#    res_211o = res_111o + res6_411 
#    
#    with tf.variable_scope('res_511') as scope:
#            res_511 = convs_s(res_211o,3,3,3,3,1,is_training)        
#            res_511 = tf.nn.relu((res_511), name= 'res_511')    
#    
#    with tf.variable_scope('res_611') as scope:
#            res_611 = convs_s(res_511,3,3,3,3,1,is_training)        
#            res_611 = tf.nn.relu((res_611), name= 'res_611')  
#    res_311o = res_211o + res_611 
#
#    with tf.variable_scope('res6_1711') as scope:
#            res6_1711 = convs_s(res_311o,3,3,3,3,1,is_training)        
#            res6_1711 = tf.nn.relu((res6_1711), name= 'res6_1711') 
#    with tf.variable_scope('res6_1811') as scope:
#            res6_1811 = convs_s(res6_1711,1,1,3,3,1,is_training)        
#            res6_1811 = tf.nn.relu((res6_1811), name= 'res6_1811')     
#    with tf.variable_scope('res6_1911') as scope:
#            res6_1911 = convs_s(res6_1811,1,1,3,3,1,is_training)        
#            res6_1911 = tf.nn.relu((res6_1911), name= 'res6_1911')   
#    res_911o = res6_1711 + res6_1911 
#    
#    with tf.variable_scope('res6_2011') as scope:
#            res6_2011 = convs_s(res_911o,1,1,3,3,1,is_training)        
#            res6_2011 = tf.nn.relu((res6_2011), name= 'res6_2011')
#    with tf.variable_scope('res6_2111') as scope:
#            res6_2111 = convs_s(res6_2011,1,1,3,3,1,is_training)        
#            res6_2111 = tf.nn.relu((res6_2111), name= 'res6_2111')
#    res_1011o = res_911o + res6_2111 
    
    
    
    
    

#    res_all= res_10o + res_101o+res_1011o
#    res_all= res_10o + res_101o
    return  res_10o
#    with tf.variable_scope('res6_2') as scope:
#        res6_2 = convs_v(res_1o,3,3,1,1,2,is_training)
#        res6_2 = tf.nn.relu((res6_2), name= 'res6_2')
#    with tf.variable_scope('res6_3') as scope:
#        res6_3 = convs_v(res_1o,3,3,1,1,2,is_training)        
#        res6_3 = tf.nn.relu((res6_3), name= 'res6_3')   
#    res_2o = res6_2 + res6_3
#
#    print("shape of res_2o=",res_2o.shape)
    
#    with tf.variable_scope('res6_4') as scope:
#        res6_4 = convs(res_2o,3,3,3,3,1,is_training)
#        res6_4 = tf.nn.relu((res6_4), name= 'res6_4')
#    with tf.variable_scope('res6_5') as scope:
#        res6_5 = convs(res6_4,3,3,3,3,1,is_training)        
#        res6_5 = tf.nn.relu((res6_5), name= 'res6_5')   
#    res_3o = res_2o + res6_5
#    with tf.variable_scope('res6_6') as scope:
#        res6_6 = convs(res_3o,3,3,3,3,1,is_training)
#        res6_6 = tf.nn.relu((res6_6), name= 'res6_6')
#    with tf.variable_scope('res6_7') as scope:
#        res6_7 = convs(res6_6,3,3,3,3,1,is_training)        
#        res6_7 = tf.nn.relu((res6_7), name= 'res6_3')   
#    res_4o = res_3o + res6_7
#    with tf.variable_scope('res6_8') as scope:
#        res6_8 = convs(res_4o,3,3,3,3,1,is_training)
#        res6_8 = tf.nn.relu((res6_8), name= 'res6_8')
#    with tf.variable_scope('res6_9') as scope:
#        res6_9 = convs(res6_2,3,3,3,3,1,is_training)        
#        res6_9 = tf.nn.relu((res6_9), name= 'res6_9')   
#    res_5o = res_4o + res6_9
#    with tf.variable_scope('res6_10') as scope:
#        res6_10 = convs(res_5o,3,3,3,3,1,is_training)
#        res6_10 = tf.nn.relu((res6_10), name= 'res6_10')
#    with tf.variable_scope('res6_11') as scope:
#        res6_11 = convs(res6_10,3,3,3,3,1,is_training)        
#        res6_11 = tf.nn.relu((res6_11), name= 'res6_11')   
#    res_6o = res_5o + res6_11
#    with tf.variable_scope('res6_12') as scope:
#        res6_12 = convs(res_6o,3,3,3,3,1,is_training)
#        res6_12 = tf.nn.relu((res6_12), name= 'res6_12')
#    with tf.variable_scope('res6_13') as scope:
#        res6_13 = convs(res6_12,3,3,3,3,1,is_training)         
#        res6_13 = tf.nn.relu((res6_13), name= 'res6_13')   
#    res_7o = res_6o + res6_13
#    with tf.variable_scope('res6_14') as scope:
#        res6_14 = convs(res_7o,3,3,3,3,1,is_training)
#        res6_14 = tf.nn.relu((res6_14), name= 'res6_14')
#    with tf.variable_scope('res6_15') as scope:
#        res6_15 = convs(res6_14,3,3,3,3,1,is_training)        
#        res6_15 = tf.nn.relu((res6_15), name= 'res6_15')   
#    res_8o = res_7o + res6_15
#    with tf.variable_scope('res6_16') as scope:
#        res6_16 = convs(res_8o,3,3,3,3,1,is_training)
#        res6_16 = tf.nn.relu((res6_16), name= 'res6_16')
#    with tf.variable_scope('res6_17') as scope:
#        res6_17 = convs(res6_16,3,3,3,3,1,is_training)        
#        res6_17 = tf.nn.relu((res6_17), name= 'res6_17')   
#    res_9o = res_8o + res6_17
#    with tf.variable_scope('res6_18') as scope:
#        res6_18 = convs(res_9o,3,3,3,3,1,is_training)
#        res6_18 = tf.nn.relu((res6_18), name= 'res6_18')
#    with tf.variable_scope('res6_19') as scope:
#        res6_19 = convs(res6_18,3,3,3,3,1,is_training)        
#        res6_19 = tf.nn.relu((res6_19), name= 'res6_19')   
#    res_10o = res_9o + res6_19
#%%

   


def losses(x_reconstruct, x_origin):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
   
    with tf.variable_scope('loss') as scope:
        loss = tf.reduce_mean(tf.pow(tf.subtract(x_reconstruct,x_origin), 2), name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)

    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-10,
    )
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
#def evaluation(logits, labels):
#  """Evaluate the quality of the logits at predicting the label.
#  Args:
#    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
#    labels: Labels tensor, int32 - [batch_size], with values in the
#      range [0, NUM_CLASSES).
#  Returns:
#    A scalar int32 tensor with the number of examples (out of batch_size)
#    that were predicted correctly.
#  """
#  with tf.variable_scope('accuracy') as scope:
#      correct = tf.nn.in_top_k(logits, labels, 1)
#      correct = tf.cast(correct, tf.float16)
#      accuracy = tf.reduce_mean(correct)
#      tf.summary.scalar(scope.name+'/accuracy', accuracy)
#  return accuracy

#%%





    
    
    
    
    
    
    
    
    
    
    
    
    
    
              
#    res_1o = res1+res3    
#    with tf.variable_scope('res4') as scope:
#        res4 = convs_s(res_1o,1,1,3,3,1,is_training)
#        res4 = tf.nn.relu(res4, name= 'res4')
##    res_2o = res_1o+res4
#  
#    with tf.variable_scope('res5') as scope:
#        res5 = convs_s(res4,1,1,3,3,1,is_training)
#        res5 = tf.nn.relu(res5, name= 'res5')
#     
#    with tf.variable_scope('res6') as scope:
#        res6 = convs_s(res5,1,1,3,3,1,is_training)
#        res6 = tf.nn.relu(res6, name= 'res6') 
#    res_2o = res4+res6
#    res_3o = res_2o+res6   
#    with tf.variable_scope('res7') as scope:
#        res7 = convs_s(res_3o,3,3,3,3,1,is_training)
#        res7 = tf.nn.relu(res7, name= 'res7')
#    with tf.variable_scope('res8') as scope:
#        res8 = convs_s(res7,3,3,3,3,1,is_training)
#        res8 = tf.nn.relu(res8, name= 'res8')
#    res_4o = res_3o+res8
#    with tf.variable_scope('res15') as scope:
#        res15 = convs_s(res_4o,3,3,3,3,1,is_training)
#        res15 = tf.nn.relu(res15, name= 'res15')
#        
#    with tf.variable_scope('res16') as scope:
#        res16 = convs_s(res15,3,3,3,3,1,is_training)
#        res16 = tf.nn.relu(res16, name= 'res16')
#    res_8o = res_4o+res16
    
#    with tf.variable_scope('res9') as scope:
#        res9 = convs_s(res_3o,3,3,3,3,1,is_training)
#        res9 = tf.nn.relu(res9, name= 'res9')
##        
#    with tf.variable_scope('res10') as scope:
#        res10 = convs_s(res9,3,3,3,3,1,is_training)
#        res10 = tf.nn.relu(res10, name= 'res10')
#    res_5o = res9+res10 
#    with tf.variable_scope('res11') as scope:
#        res11 = convs_s(res_5o,3,3,3,3,1,is_training)
#        res11 = tf.nn.relu(res11, name= 'res11')
#        
#    with tf.variable_scope('res12') as scope:
#        res12 = convs_s(res11,3,3,3,3,1,is_training)
#        res12 = tf.nn.relu(res12, name= 'res12')
#    res_6o = res_5o+res12
# 
#    with tf.variable_scope('res13') as scope:
#        res13 = convs_s(res_6o,3,3,3,3,1,is_training)
#        res13 = tf.nn.relu(res13, name= 'res13')
##        
#    with tf.variable_scope('res14') as scope:
#        res14 = convs_s(res13,3,3,3,3,1,is_training)
#        res14 = tf.nn.relu(res14, name= 'res14')
#    res_7o = res_6o+res14 
##    with tf.variable_scope('res15') as scope:
##        res15 = convs_s(res_7o,3,3,3,3,1,is_training)
##        res15 = tf.nn.relu(res15, name= 'res15')
##        
##    with tf.variable_scope('res16') as scope:
##        res16 = convs_s(res15,3,3,3,3,1,is_training)
##        res16 = tf.nn.relu(res16, name= 'res16')
##    res_8o = res_7o+res16
#    
#    
#    
#    with tf.variable_scope('res17') as scope:
#        res17 = convs_s(res_7o,3,3,3,3,1,is_training)
#        res17 = tf.nn.relu(res17, name= 'res17')
#        
#    with tf.variable_scope('res18') as scope:
#        res18 = convs_s(res17,3,3,3,3,1,is_training)
#        res18 = tf.nn.relu(res18, name= 'res18')
#    res_8o = res17+res18    
#
#    with tf.variable_scope('res19') as scope:
#        res19 = convs_s(res_8o,3,3,3,3,1,is_training)
#        res19 = tf.nn.relu(res19, name= 'res19')
#        
#    with tf.variable_scope('res20') as scope:
#        res20 = convs_s(res19,3,3,3,3,1,is_training)
#        res20 = tf.nn.relu(res20, name= 'res20')
#    res_9o = res_8o+res20  
#        
#    with tf.variable_scope('res21') as scope:
#        res21 = convs_s(res_9o,3,3,3,3,1,is_training)
#        res21 = tf.nn.relu(res21, name= 'res21')
#    with tf.variable_scope('res22') as scope:
#        res22 = convs_s(res21,3,3,3,3,1,is_training)
#        res22 = tf.nn.relu(res22, name= 'res22')
#    res_10o = res_9o+res22
#    
#    
#    
#    with tf.variable_scope('res23') as scope:
#        res23 = convs_s(res_10o,3,3,3,3,1,is_training)
#        res23 = tf.nn.relu(res23, name= 'res23')
#        
#    with tf.variable_scope('res24') as scope:
#        res24 = convs_s(res23,3,3,3,3,1,is_training)
#        res24 = tf.nn.relu(res24, name= 'res24')
#    res_11o = res23+res24    

#    with tf.variable_scope('res25') as scope:
#        res25 = convs_s(res_11o,3,3,3,3,1,is_training)
#        res25 = tf.nn.relu(res25, name= 'res25')
#        
#    with tf.variable_scope('res26') as scope:
#        res26 = convs_s(res25,3,3,3,3,1,is_training)
#        res26 = tf.nn.relu(res26, name= 'res26')
#    res_12o = res_11o+res26  
#        
#    with tf.variable_scope('res27') as scope:
#        res27 = convs_s(res_12o,3,3,3,3,1,is_training)
#        res27 = tf.nn.relu(res27, name= 'res27')
#    with tf.variable_scope('res28') as scope:
#        res28 = convs_s(res27,3,3,3,3,1,is_training)
#        res28 = tf.nn.relu(res28, name= 'res28')
#    res_13o = res_12o+res28
#    
#    
#    
#    with tf.variable_scope('res29') as scope:
#        res29 = convs_s(images,3,3,3,3,1,is_training)
#        res29 = tf.nn.relu(res29, name= 'res29')
#        
#    with tf.variable_scope('res30') as scope:
#        res30 = convs_s(res29,3,3,3,3,1,is_training)
#        res30 = tf.nn.relu(res30, name= 'res30')
#    res_14o = res29+res30    
#
#    with tf.variable_scope('res31') as scope:
#        res31 = convs_s(res_14o,3,3,3,3,1,is_training)
#        res31 = tf.nn.relu(res31, name= 'res31')
#        
#    with tf.variable_scope('res32') as scope:
#        res32 = convs_s(res31,3,3,3,3,1,is_training)
#        res32 = tf.nn.relu(res32, name= 'res32')
#    res_15o = res_14o+res32  
#        
#    with tf.variable_scope('res33') as scope:
#        res33 = convs_s(res_15o,3,3,3,3,1,is_training)
#        res33 = tf.nn.relu(res33, name= 'res33')
#    with tf.variable_scope('res34') as scope:
#        res34 = convs_s(res33,3,3,3,3,1,is_training)
#        res34 = tf.nn.relu(res34, name= 'res34')
#    res_16o = res_15o+res34
    
    
    
    
    
    
    
#    with tf.variable_scope('res23') as scope:
#        res23 = convs_s(images,3,3,3,3,1,is_training)
#        res23 = tf.nn.relu(res23, name= 'res23')
#        
#    with tf.variable_scope('res24') as scope:
#        res24 = convs_s(res23,3,3,3,3,1,is_training)
#        res24 = tf.nn.relu(res24, name= 'res24')
#    res_11o = res23+res24    
#
#    with tf.variable_scope('res25') as scope:
#        res25 = convs_s(res_11o,3,3,3,3,1,is_training)
#        res25 = tf.nn.relu(res25, name= 'res25')
#        
#    with tf.variable_scope('res26') as scope:
#        res26 = convs_s(res25,3,3,3,3,1,is_training)
#        res26 = tf.nn.relu(res26, name= 'res26')
#    res_12o = res_11o+res26  
#        
#    with tf.variable_scope('res27') as scope:
#        res27 = convs_s(res_12o,3,3,3,3,1,is_training)
#        res27 = tf.nn.relu(res27, name= 'res27')
#    with tf.variable_scope('res28') as scope:
#        res28 = convs_s(res27,3,3,3,3,1,is_training)
#        res28 = tf.nn.relu(res28, name= 'res28')
#    res_13o = res_12o+res28
#    
    
#        
#   
#    
##    with tf.variable_scope('res10') as scope:
##        res10 = convs_s(images,3,3,3,3,1,is_training)
##        res10 = tf.nn.relu(res10, name= 'res10')
#    res_5o = res_4o+res10  
    
#    res_1o = res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9# +res10 +res11 +res12 +res13 +res14 +res15
#    tf.reduce_mean(res_1o)
#    res_all = res_3o+res_7o+res_10o+res_13o+res_16o
#    res_all_1 = tf.divide(res_all,3)
    
#    print("shape of images=",images.shape)    
#    print("shape of res=",res_5o.shape)
    
#    ress_all=res_9o+res_91o+res_92o
#    return ress_all
#    with tf.variable_scope('res6_2') as scope:
#        res6_2 = convs_v(res_1o,3,3,1,1,2,is_training)
#        res6_2 = tf.nn.relu((res6_2), name= 'res6_2')
#    with tf.variable_scope('res6_3') as scope:
#        res6_3 = convs_v(res_1o,3,3,1,1,2,is_training)        
#        res6_3 = tf.nn.relu((res6_3), name= 'res6_3')   
#    res_2o = res6_2 + res6_3
#
#    print("shape of res_2o=",res_2o.shape)
    
#    with tf.variable_scope('res6_4') as scope:
#        res6_4 = convs(res_2o,3,3,3,3,1,is_training)
#        res6_4 = tf.nn.relu((res6_4), name= 'res6_4')
#    with tf.variable_scope('res6_5') as scope:
#        res6_5 = convs(res6_4,3,3,3,3,1,is_training)        
#        res6_5 = tf.nn.relu((res6_5), name= 'res6_5')   
#    res_3o = res_2o + res6_5
#    with tf.variable_scope('res6_6') as scope:
#        res6_6 = convs(res_3o,3,3,3,3,1,is_training)
#        res6_6 = tf.nn.relu((res6_6), name= 'res6_6')
#    with tf.variable_scope('res6_7') as scope:
#        res6_7 = convs(res6_6,3,3,3,3,1,is_training)        
#        res6_7 = tf.nn.relu((res6_7), name= 'res6_3')   
#    res_4o = res_3o + res6_7
#    with tf.variable_scope('res6_8') as scope:
#        res6_8 = convs(res_4o,3,3,3,3,1,is_training)
#        res6_8 = tf.nn.relu((res6_8), name= 'res6_8')
#    with tf.variable_scope('res6_9') as scope:
#        res6_9 = convs(res6_2,3,3,3,3,1,is_training)        
#        res6_9 = tf.nn.relu((res6_9), name= 'res6_9')   
#    res_5o = res_4o + res6_9
#    with tf.variable_scope('res6_10') as scope:
#        res6_10 = convs(res_5o,3,3,3,3,1,is_training)
#        res6_10 = tf.nn.relu((res6_10), name= 'res6_10')
#    with tf.variable_scope('res6_11') as scope:
#        res6_11 = convs(res6_10,3,3,3,3,1,is_training)        
#        res6_11 = tf.nn.relu((res6_11), name= 'res6_11')   
#    res_6o = res_5o + res6_11
#    with tf.variable_scope('res6_12') as scope:
#        res6_12 = convs(res_6o,3,3,3,3,1,is_training)
#        res6_12 = tf.nn.relu((res6_12), name= 'res6_12')
#    with tf.variable_scope('res6_13') as scope:
#        res6_13 = convs(res6_12,3,3,3,3,1,is_training)         
#        res6_13 = tf.nn.relu((res6_13), name= 'res6_13')   
#    res_7o = res_6o + res6_13
#    with tf.variable_scope('res6_14') as scope:
#        res6_14 = convs(res_7o,3,3,3,3,1,is_training)
#        res6_14 = tf.nn.relu((res6_14), name= 'res6_14')
#    with tf.variable_scope('res6_15') as scope:
#        res6_15 = convs(res6_14,3,3,3,3,1,is_training)        
#        res6_15 = tf.nn.relu((res6_15), name= 'res6_15')   
#    res_8o = res_7o + res6_15
#    with tf.variable_scope('res6_16') as scope:
#        res6_16 = convs(res_8o,3,3,3,3,1,is_training)
#        res6_16 = tf.nn.relu((res6_16), name= 'res6_16')
#    with tf.variable_scope('res6_17') as scope:
#        res6_17 = convs(res6_16,3,3,3,3,1,is_training)        
#        res6_17 = tf.nn.relu((res6_17), name= 'res6_17')   
#    res_9o = res_8o + res6_17
#    with tf.variable_scope('res6_18') as scope:
#        res6_18 = convs(res_9o,3,3,3,3,1,is_training)
#        res6_18 = tf.nn.relu((res6_18), name= 'res6_18')
#    with tf.variable_scope('res6_19') as scope:
#        res6_19 = convs(res6_18,3,3,3,3,1,is_training)        
#        res6_19 = tf.nn.relu((res6_19), name= 'res6_19')   
#    res_10o = res_9o + res6_19
#%%

   

#
#def losses(x_reconstruct, x_origin):
#    '''Compute loss from logits and labels
#    Args:
#        logits: logits tensor, float, [batch_size, n_classes]
#        labels: label tensor, tf.int32, [batch_size]
#        
#    Returns:
#        loss tensor of float type
#    '''
#   
#    with tf.variable_scope('loss') as scope:
#        loss = tf.reduce_mean(tf.pow(tf.subtract(x_reconstruct,x_origin), 2), name='loss')
#        tf.summary.scalar(scope.name+'/loss', loss)
#
#    return loss
#
##%%
#def trainning(loss, learning_rate):
#    '''Training ops, the Op returned by this function is what must be passed to 
#        'sess.run()' call to cause the model to train.
#        
#    Args:
#        loss: loss tensor, from losses()
#        
#    Returns:
#        train_op: The op for trainning
#    '''
#    with tf.name_scope('optimizer'):
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_ops):
#            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
#    beta1=0.9,
#    beta2=0.999,
#    epsilon=1e-10,
#    )
#            global_step = tf.Variable(0, name='global_step', trainable=False)
#            train_op = optimizer.minimize(loss, global_step= global_step)
#    return train_op

#%%
#def evaluation(logits, labels):
#  """Evaluate the quality of the logits at predicting the label.
#  Args:
#    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
#    labels: Labels tensor, int32 - [batch_size], with values in the
#      range [0, NUM_CLASSES).
#  Returns:
#    A scalar int32 tensor with the number of examples (out of batch_size)
#    that were predicted correctly.
#  """
#  with tf.variable_scope('accuracy') as scope:
#      correct = tf.nn.in_top_k(logits, labels, 1)
#      correct = tf.cast(correct, tf.float16)
#      accuracy = tf.reduce_mean(correct)
#      tf.summary.scalar(scope.name+'/accuracy', accuracy)
#  return accuracy

#%%




