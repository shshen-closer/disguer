# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf



def weight_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()(shape) , trainable=True, name=name)  
    return initial
def bias_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()(shape) , trainable=True, name=name) 
    return initial

def knowledge_representation_matrix(
        queries, 
        keys,
        num_heads, 
        num_units,
        dropout_prob,
        is_training,
        scope = 'krm'):
    
    Q = tf.compat.v1.layers.dense(queries, num_units, activation=tf.nn.relu) 
    K = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu) 
    V = tf.compat.v1.layers.dense(keys, num_units, activation=tf.nn.relu)#, activation=tf.nn.relu) 
    
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
    
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
    
    outputs = tf.nn.softmax(outputs) 

    outputs = tf.nn.dropout(outputs, dropout_prob)
    outputs = tf.matmul(outputs, V_) 
    
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    
    outputs += Q
    outputs =  tf.compat.v1.layers.batch_normalization(outputs, momentum=0.9, epsilon=1e-5, training=is_training)#
    
    return outputs



def multi_span(x, keys, drop_rate, is_training):
  #  x = tf.layers.dense(x, 512)
  #  with tf.name_scope("krm_{}".format(1)):

    x = knowledge_representation_matrix(queries = x, 
                                                keys = keys, 
                                                num_heads = 8,
                                                num_units = 256, 
                                                dropout_prob = drop_rate,
                                                is_training = is_training,
                                                ) 
    x1 = tf.compat.v1.layers.dense(x, 256, activation=tf.nn.swish) 
    x1 = tf.compat.v1.layers.dense(x1, 256, activation=tf.nn.swish) 
    x1 = tf.compat.v1.layers.dense(x1, 256, activation=tf.nn.swish) 
    x += x1
  
    print('x output: ', np.shape(x))
    
    return x

class ATT(object):

    def __init__(self, batch_size, num_steps, num_skills, hidden_size):
        
        self.batch_size = batch_size = batch_size
        self.hidden_size  = hidden_size
        self.num_steps = num_steps
        self.num_skills =  num_skills

        self.length = tf.compat.v1.placeholder(tf.float32, [batch_size, ], name="length")
        self.input_problem = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="input_problem")
        self.input_kc = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="input_kc")
        self.input_a = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="input_a")
        self.in_ty = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="in_ty")
        self.tar_id = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_id")
        self.tar_kc = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="tar_kc")
        self.tar_ty = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="tar_ty")
        self.tar_index = tf.compat.v1.placeholder(tf.int32, [None], name="tar_index")
        self.tar_corr = tf.compat.v1.placeholder(tf.float32, [None], name="tar_corr")
        self.ana_his = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="ana_his")
        self.ana_pre = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="ana_pre")
        self.int_k0 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_k0")
        self.tar_k0 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_k0")
        self.int_t = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t")
        self.tar_t = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t")
        self.int_t1 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t1")
        self.tar_t1 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t1")
        self.int_t2 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t2")
        self.tar_t2 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t2")
        self.int_t3 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t3")
        self.tar_t3 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t3")
        self.int_t4 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t4")
        self.tar_t4 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t4")
        self.int_t5 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t5")
        self.tar_t5 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t5")
        self.int_t6 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t6")
        self.tar_t6 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t6")
        self.int_t7 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_t7")
        self.tar_t7 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_t7")
        self.int_h = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h")
        self.tar_h = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h")
        self.int_h1 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h1")
        self.tar_h1 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h1")
        self.int_h2 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h2")
        self.tar_h2 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h2")
        self.int_h3 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h3")
        self.tar_h3 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h3")
        self.int_h4 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h4")
        self.tar_h4 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h4")
        self.int_h5 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h5")
        self.tar_h5 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h5")
        self.int_h6 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h6")
        self.tar_h6 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h6")
        self.int_h7 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_h7")
        self.tar_h7 = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_h7")
        self.int_cl = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_cl")
        self.tar_cl = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_cl")
        self.int_al = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="int_al")
        self.tar_al = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="tar_al")
        

        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer=  tf.compat.v1.keras.initializers.glorot_uniform()#tf.compat.v1.keras.initializers.VarianceScaling() # 
        
        

        #glorot, he, he
        #glorot_uniform
        # exercise
        zero_problem = tf.zeros((1, hidden_size))
        self.problem_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'problem_w')
        all_problem = tf.concat([zero_problem, self.problem_w ], axis = 0)
        in_id =  tf.nn.embedding_lookup(all_problem, self.input_problem)
        tar_id =  tf.nn.embedding_lookup(all_problem, self.tar_id)

        # exercise
        zero_problem = tf.zeros((1, hidden_size))
        self.kc_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'kc_w')
        all_problem = tf.concat([zero_problem, self.kc_w ], axis = 0)
        kc_embedding =  tf.nn.embedding_lookup(all_problem, self.input_kc)
        tar_kc =  tf.nn.embedding_lookup(all_problem, self.tar_kc)

        # type
        self.type_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'type_w')
        all_problem = tf.concat([zero_problem, self.type_w], axis = 0)
        input_type =  tf.nn.embedding_lookup(all_problem, self.in_ty)
        tar_type =  tf.nn.embedding_lookup(all_problem, self.tar_ty)

        self.analen_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'analen_w')
        all_problem = tf.concat([zero_problem, self.analen_w ], axis = 0)
        ana_his =  tf.nn.embedding_lookup(all_problem, self.ana_his)
        ana_pre =  tf.nn.embedding_lookup(all_problem, self.ana_pre)


        self.k0_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'k0_w')
        all_problem = tf.concat([zero_problem, self.k0_w ], axis = 0)
        int_k0 =  tf.nn.embedding_lookup(all_problem, self.int_k0)
        tar_k0 =  tf.nn.embedding_lookup(all_problem, self.tar_k0)

        self.al_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'al_w')
        all_problem = tf.concat([zero_problem, self.al_w ], axis = 0)
        int_al =  tf.nn.embedding_lookup(all_problem, self.int_al)
        tar_al =  tf.nn.embedding_lookup(all_problem, self.tar_al)

        self.cl_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'cl_w')
        all_problem = tf.concat([zero_problem, self.cl_w ], axis = 0)
        int_cl =  tf.nn.embedding_lookup(all_problem, self.int_cl)
        tar_cl =  tf.nn.embedding_lookup(all_problem, self.tar_cl)



        self.time_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time_w')
        all_problem = tf.concat([zero_problem, self.time_w ], axis = 0)
        int_t =  tf.nn.embedding_lookup(all_problem, self.int_t)
        tar_t =  tf.nn.embedding_lookup(all_problem, self.tar_t)
        self.time1_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time1_w')
        all_problem = tf.concat([zero_problem, self.time1_w ], axis = 0)
        int_t1 =  tf.nn.embedding_lookup(all_problem, self.int_t1)
        tar_t1 =  tf.nn.embedding_lookup(all_problem, self.tar_t1)
        self.time2_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time2_w')
        all_problem = tf.concat([zero_problem, self.time2_w ], axis = 0)
        int_t2 =  tf.nn.embedding_lookup(all_problem, self.int_t2)
        tar_t2 =  tf.nn.embedding_lookup(all_problem, self.tar_t2)
        self.time3_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time3_w')
        all_problem = tf.concat([zero_problem, self.time3_w ], axis = 0)
        int_t3 =  tf.nn.embedding_lookup(all_problem, self.int_t3)
        tar_t3 =  tf.nn.embedding_lookup(all_problem, self.tar_t3)
        self.time4_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time4_w')
        all_problem = tf.concat([zero_problem, self.time4_w ], axis = 0)
        int_t4 =  tf.nn.embedding_lookup(all_problem, self.int_t4)
        tar_t4 =  tf.nn.embedding_lookup(all_problem, self.tar_t4)
        self.time5_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time5_w')
        all_problem = tf.concat([zero_problem, self.time5_w ], axis = 0)
        int_t5 =  tf.nn.embedding_lookup(all_problem, self.int_t5)
        tar_t5 =  tf.nn.embedding_lookup(all_problem, self.tar_t5)
        self.time6_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time6_w')
        all_problem = tf.concat([zero_problem, self.time6_w ], axis = 0)
        int_t6 =  tf.nn.embedding_lookup(all_problem, self.int_t6)
        tar_t6 =  tf.nn.embedding_lookup(all_problem, self.tar_t6)
        self.time7_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'time7_w')
        all_problem = tf.concat([zero_problem, self.time7_w ], axis = 0)
        int_t7 =  tf.nn.embedding_lookup(all_problem, self.int_t7)
        tar_t7 =  tf.nn.embedding_lookup(all_problem, self.tar_t7)



        self.hope_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope_w')
        all_problem = tf.concat([zero_problem, self.hope_w ], axis = 0)
        int_h =  tf.nn.embedding_lookup(all_problem, self.int_h)
        tar_h =  tf.nn.embedding_lookup(all_problem, self.tar_h)
        self.hope1_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope1_w')
        all_problem = tf.concat([zero_problem, self.hope1_w ], axis = 0)
        int_h1 =  tf.nn.embedding_lookup(all_problem, self.int_h1)
        tar_h1 =  tf.nn.embedding_lookup(all_problem, self.tar_h1)
        self.hope2_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope2_w')
        all_problem = tf.concat([zero_problem, self.hope2_w ], axis = 0)
        int_h2 =  tf.nn.embedding_lookup(all_problem, self.int_h2)
        tar_h2 =  tf.nn.embedding_lookup(all_problem, self.tar_h2)
        self.hope3_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope3_w')
        all_problem = tf.concat([zero_problem, self.hope3_w ], axis = 0)
        int_h3 =  tf.nn.embedding_lookup(all_problem, self.int_h3)
        tar_h3 =  tf.nn.embedding_lookup(all_problem, self.tar_h3)
        self.hope4_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope4_w')
        all_problem = tf.concat([zero_problem, self.hope4_w ], axis = 0)
        int_h4 =  tf.nn.embedding_lookup(all_problem, self.int_h4)
        tar_h4 =  tf.nn.embedding_lookup(all_problem, self.tar_h4)
        self.hope5_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope5_w')
        all_problem = tf.concat([zero_problem, self.hope5_w ], axis = 0)
        int_h5 =  tf.nn.embedding_lookup(all_problem, self.int_h5)
        tar_h5 =  tf.nn.embedding_lookup(all_problem, self.tar_h5)
        self.hope6_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope6_w')
        all_problem = tf.concat([zero_problem, self.hope6_w ], axis = 0)
        int_h6 =  tf.nn.embedding_lookup(all_problem, self.int_h6)
        tar_h6 =  tf.nn.embedding_lookup(all_problem, self.tar_h6)
        self.hope7_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([7652, hidden_size]),dtype=tf.float32, trainable=True, name = 'hope7_w')
        all_problem = tf.concat([zero_problem, self.hope7_w ], axis = 0)
        int_h7 =  tf.nn.embedding_lookup(all_problem, self.int_h7)
        tar_h7 =  tf.nn.embedding_lookup(all_problem, self.tar_h7)
        



        in_ts = int_t+int_t1+int_t2+int_t3+int_t4+int_t5+int_t6+int_t7+int_h+int_h1+int_h2+int_h3+int_h4+ int_h5+int_h6+int_h7
        in_ts = tf.compat.v1.layers.dense(in_ts, units = hidden_size, activation=tf.nn.relu)
        input_p = 4*in_id+3*kc_embedding +2*input_type + int_k0 + ana_his+ int_al+ int_cl + in_ts  
        input_p = tf.compat.v1.layers.dense(input_p, units = hidden_size, activation=tf.nn.relu)
      #  input_p = tf.concat([input_p, input_a],axis = -1)
      #  input_p = tf.compat.v1.layers.dense(input_p, units = hidden_size)
        his_a_e = tf.expand_dims(self.input_a, -1)
        his_a_e = tf.cast(his_a_e, tf.float32)
        input_data = his_a_e*tf.compat.v1.layers.dense(input_p, units = hidden_size) + (1-his_a_e)*tf.compat.v1.layers.dense(input_p, units = hidden_size)

        tar_ts =  tar_t+tar_t1+tar_t2+tar_t3+tar_t4+tar_t5+tar_t6+tar_t7+tar_h+tar_h1+tar_h2+tar_h3+tar_h4+tar_h5+tar_h6+tar_h7
        tar_ts = tf.compat.v1.layers.dense(tar_ts, units = hidden_size, activation=tf.nn.relu) 
        tar_data = 4*tar_id+ 3*tar_kc+ 2*tar_type+tar_k0 + ana_pre+tar_al+tar_cl +tar_ts 
        tar_data = tf.compat.v1.layers.dense(tar_data, units = hidden_size, activation=tf.nn.relu)
 
        print(np.shape(tar_data))

        states =  multi_span(input_data , tar_data, self.dropout_keep_prob,  self.is_training) 
        states = tf.compat.v1.layers.dense(states, units = hidden_size, activation=tf.nn.relu)
     #  states = tf.nn.dropout(states, self.dropout_keep_prob)


        input_p  = tf.split(input_data, batch_size, axis = 0)
        states = tf.split(states, batch_size, axis = 0)
        length = tf.split(self.length, batch_size, axis = 0)
        tar_split  = tf.split(tar_data, batch_size, axis = 0)
        ooo = list()
        for iii in range(batch_size):
            keys = tf.squeeze(input_p[iii], 0) 
            item = tf.squeeze(states[iii], 0)  # steps, hidden
            value = tf.squeeze(tar_split[iii], 0)  # steps, hidden
            lll = tf.squeeze(length[iii], 0)
            lll = tf.range(0, lll)
            lll = tf.cast(lll, tf.int32)
            temp_state = tf.gather(item, lll, axis = 0)  
            temp_key = tf.gather(keys, lll, axis = 0)  
            pos = tf.range(num_steps, 0, -1)
            pos = tf.cast(pos, tf.int32)
            pos = tf.gather(pos, lll, axis = 0) 
            pos = pos/tf.reduce_sum(pos) / 1
            pos1 = tf.cast(pos, tf.float32)

            pos = tf.range(1, num_steps+1)
            pos = tf.cast(pos, tf.int32)
            pos = tf.gather(pos, lll, axis = 0) 
            pos = pos/tf.reduce_sum(pos) / 1
            pos2 = tf.cast(pos, tf.float32)
 

            temp_key = tf.transpose(temp_key, [1,0])# hidden  lll, 
            att = tf.matmul(value, temp_key) #steps, lll
            att = tf.nn.softmax(att, axis = 1)  - pos1 + pos2 #steps, lll
            temp = tf.matmul(att, temp_state) # steps, hidden
            temp = tf.expand_dims(temp, axis = 0)
            ooo.append(temp)
        states = tf.concat(ooo, axis = 0)
        print('states', np.shape(states))

        

        output = tf.concat([tar_data*states, tar_data, states], axis = -1)
        output = tf.compat.v1.layers.dense(output, units = hidden_size)
        
        logits = tf.reduce_mean(output, axis = -1, name="logits")
        print('logits',np.shape(logits))
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, self.tar_index)
        
        #make prediction
        self.pred = tf.sigmoid(selected_logits, name="pred")

        # loss function
        losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=self.tar_corr), name="losses")

        l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.compat.v1.trainable_variables()],
                                 name="l2_losses") * 0.000001
        self.loss = tf.add(losses, l2_losses, name="loss")
        
        self.cost = self.loss