# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from datetime import datetime 
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import checkmate as cm
from utils import data_helpers as dh

import json

# Parameters

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))
file_name = sys.argv[1]


MODEL = file_name
while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(90175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")



MODEL_DIR =  'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR =  'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 128, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 50 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("seq_len", 50, "Number of epochs to train for.")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


    

def test():

    # Load data
    logger.info("Loading data...")

    
    logger.info("test data processing...")
    test_students = np.load("data/test.npy", allow_pickle=True)
    test_students = np.concatenate([test_students, test_students[:FLAGS.batch_size]], axis = 0)
    
    max_num_steps = 550
    max_num_skills = 1175

    BEST_OR_LATEST = 'B'

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    if BEST_OR_LATEST == 'L':
        logger.info("latest")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            length = graph.get_operation_by_name("length").outputs[0]
            input_problem = graph.get_operation_by_name("input_problem").outputs[0]
            input_kc = graph.get_operation_by_name("input_kc").outputs[0]
            input_a = graph.get_operation_by_name("input_a").outputs[0]
            in_ty = graph.get_operation_by_name("in_ty").outputs[0]
            tar_id = graph.get_operation_by_name("tar_id").outputs[0]
            tar_kc = graph.get_operation_by_name("tar_kc").outputs[0]
            tar_ty = graph.get_operation_by_name("tar_ty").outputs[0]
            tar_index = graph.get_operation_by_name("tar_index").outputs[0]
            tar_corr = graph.get_operation_by_name("tar_corr").outputs[0]
            ana_his = graph.get_operation_by_name("ana_his").outputs[0]
            ana_pre = graph.get_operation_by_name("ana_pre").outputs[0]
            int_k0 = graph.get_operation_by_name("int_k0").outputs[0]
            tar_k0 = graph.get_operation_by_name("tar_k0").outputs[0]
            int_t = graph.get_operation_by_name("int_t").outputs[0]
            tar_t = graph.get_operation_by_name("tar_t").outputs[0]
            int_t1 = graph.get_operation_by_name("int_t1").outputs[0]
            tar_t1 = graph.get_operation_by_name("tar_t1").outputs[0]
            int_t2 = graph.get_operation_by_name("int_t2").outputs[0]
            tar_t2 = graph.get_operation_by_name("tar_t2").outputs[0]
            int_t3 = graph.get_operation_by_name("int_t3").outputs[0]
            tar_t3 = graph.get_operation_by_name("tar_t3").outputs[0]
            int_t4 = graph.get_operation_by_name("int_t4").outputs[0]
            tar_t4 = graph.get_operation_by_name("tar_t4").outputs[0]
            int_t5 = graph.get_operation_by_name("int_t5").outputs[0]
            tar_t5 = graph.get_operation_by_name("tar_t5").outputs[0]
            int_t6 = graph.get_operation_by_name("int_t6").outputs[0]
            tar_t6 = graph.get_operation_by_name("tar_t6").outputs[0]
            int_t7 = graph.get_operation_by_name("int_t7").outputs[0]
            tar_t7 = graph.get_operation_by_name("tar_t7").outputs[0]
            int_h = graph.get_operation_by_name("int_h").outputs[0]
            tar_h = graph.get_operation_by_name("tar_h").outputs[0]
            int_h1 = graph.get_operation_by_name("int_h1").outputs[0]
            tar_h1 = graph.get_operation_by_name("tar_h1").outputs[0]
            int_h2 = graph.get_operation_by_name("int_h2").outputs[0]
            tar_h2 = graph.get_operation_by_name("tar_h2").outputs[0]
            int_h3 = graph.get_operation_by_name("int_h3").outputs[0]
            tar_h3 = graph.get_operation_by_name("tar_h3").outputs[0]
            int_h4 = graph.get_operation_by_name("int_h4").outputs[0]
            tar_h4 = graph.get_operation_by_name("tar_h4").outputs[0]
            int_h5 = graph.get_operation_by_name("int_h5").outputs[0]
            tar_h5 = graph.get_operation_by_name("tar_h5").outputs[0]
            int_h6 = graph.get_operation_by_name("int_h6").outputs[0]
            tar_h6 = graph.get_operation_by_name("tar_h6").outputs[0]
            int_h7 = graph.get_operation_by_name("int_h7").outputs[0]
            tar_h7 = graph.get_operation_by_name("tar_h7").outputs[0]
            int_al = graph.get_operation_by_name("int_al").outputs[0]
            tar_al = graph.get_operation_by_name("tar_al").outputs[0]
            int_cl = graph.get_operation_by_name("int_cl").outputs[0]
            tar_cl = graph.get_operation_by_name("tar_cl").outputs[0]
           
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            pred = graph.get_operation_by_name("pred").outputs[0]
            
        
            a=datetime.now()
            data_size = len(test_students)
            index = 0
            actual_labels = []
            pred_labels = []
            while(index+FLAGS.batch_size <= data_size):
                input_problem_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_kc_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_a_b = np.zeros((FLAGS.batch_size, max_num_steps))
                in_ty_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_id_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_kc_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_ty_b = np.zeros((FLAGS.batch_size, max_num_steps))
                ana_his_b = np.zeros((FLAGS.batch_size, max_num_steps))
                ana_pre_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_k0_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_k0_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t1_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t1_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t2_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t2_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t3_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t3_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t4_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t4_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t5_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t5_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t6_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t6_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_t7_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_t7_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h1_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h1_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h2_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h2_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h3_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h3_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h4_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h4_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h5_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h5_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h6_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h6_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_h7_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_h7_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_al_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_al_b = np.zeros((FLAGS.batch_size, max_num_steps))
                int_cl_b = np.zeros((FLAGS.batch_size, max_num_steps))
                tar_cl_b = np.zeros((FLAGS.batch_size, max_num_steps))
                length_b = np.zeros((FLAGS.batch_size, ))
                tar_corr_b = []
                tar_index_b = []
                for i in range(FLAGS.batch_size):
                    item = test_students[index+i]
                    input_problem_b[i][:len(item[0])] = item[0]
                    input_kc_b[i][:len(item[1])] = item[1]
                    input_a_b[i][:len(item[2])] = item[2]
                    tar_id_b[i][:len(item[3])] = item[3]
                    tar_kc_b[i][:len(item[4])] = item[4]
                    length_b[i] = len(item[0])

                    in_ty_b[i][:len(item[6])] = item[6]
                    tar_ty_b[i][:len(item[7])] = item[7]
                    ana_his_b[i][:len(item[8])] = item[8]
                    ana_pre_b[i][:len(item[9])] = item[9]
                    int_k0_b[i][:len(item[8])] = item[10]
                    tar_k0_b[i][:len(item[9])] = item[11]
                    int_t_b[i][:len(item[8])] = item[12]
                    tar_t_b[i][:len(item[9])] = item[13]
                    int_t1_b[i][:len(item[8])] = item[14]
                    tar_t1_b[i][:len(item[9])] = item[15]
                    int_t2_b[i][:len(item[8])] = item[16]
                    tar_t2_b[i][:len(item[9])] = item[17]
                    int_t3_b[i][:len(item[8])] = item[18]
                    tar_t3_b[i][:len(item[9])] = item[19]
                    int_t4_b[i][:len(item[8])] = item[20]
                    tar_t4_b[i][:len(item[9])] = item[21]
                    int_t5_b[i][:len(item[8])] = item[22]
                    tar_t5_b[i][:len(item[9])] = item[23]
                    int_t6_b[i][:len(item[8])] = item[24]
                    tar_t6_b[i][:len(item[9])] = item[25]
                    int_t7_b[i][:len(item[8])] = item[26]
                    tar_t7_b[i][:len(item[9])] = item[27]
                    int_h_b[i][:len(item[8])] = item[28]
                    tar_h_b[i][:len(item[9])] = item[29]
                    int_h1_b[i][:len(item[8])] = item[30]
                    tar_h1_b[i][:len(item[9])] = item[31]
                    int_h2_b[i][:len(item[8])] = item[32]
                    tar_h2_b[i][:len(item[9])] = item[33]
                    int_h3_b[i][:len(item[8])] = item[34]
                    tar_h3_b[i][:len(item[9])] = item[35]
                    int_h4_b[i][:len(item[8])] = item[36]
                    tar_h4_b[i][:len(item[9])] = item[37]
                    int_h5_b[i][:len(item[8])] = item[38]
                    tar_h5_b[i][:len(item[9])] = item[39]
                    int_h6_b[i][:len(item[8])] = item[40]
                    tar_h6_b[i][:len(item[9])] = item[41]
                    int_h7_b[i][:len(item[8])] = item[42]
                    tar_h7_b[i][:len(item[9])] = item[43]
                    int_al_b[i][:len(item[8])] = item[44]
                    tar_al_b[i][:len(item[9])] = item[45]
                    int_cl_b[i][:len(item[8])] = item[46]
                    tar_cl_b[i][:len(item[9])] = item[47]

                    

                    for j in range(len(item[3])):
                        tar_index_b.append(i*max_num_steps+j)
                        tar_corr_b.append(0)

                index += FLAGS.batch_size

                feed_dict = {
                    length: length_b,
                    input_problem:input_problem_b,
                    input_kc: input_kc_b,
                    input_a:input_a_b,
                    in_ty: in_ty_b,
                    tar_id: tar_id_b,
                    tar_kc: tar_kc_b,
                    tar_ty:tar_ty_b,
                    tar_index: tar_index_b,
                    tar_corr: tar_corr_b,
                    ana_his: ana_his_b,
                    ana_pre: ana_pre_b,
                    int_k0: int_k0_b,
                    tar_k0: tar_k0_b,
                    int_t: int_t_b,
                    tar_t: tar_t_b,
                    int_t1: int_t1_b,
                    tar_t1: tar_t1_b,
                    int_t2: int_t2_b,
                    tar_t2: tar_t2_b,
                    int_t3: int_t3_b,
                    tar_t3: tar_t3_b,
                    int_t4: int_t4_b,
                    tar_t4: tar_t4_b,
                    int_t5: int_t5_b,
                    tar_t5: tar_t5_b,
                    int_t6: int_t6_b,
                    tar_t6: tar_t6_b,
                    int_t7: int_t7_b,
                    tar_t7: tar_t7_b,
                    int_h: int_h_b,
                    tar_h: tar_h_b,
                    int_h1: int_h1_b,
                    tar_h1: tar_h1_b,
                    int_h2: int_h2_b,
                    tar_h2: tar_h2_b,
                    int_h3: int_h3_b,
                    tar_h3: tar_h3_b,
                    int_h4: int_h4_b,
                    tar_h4: tar_h4_b,
                    int_h5: int_h5_b,
                    tar_h5: tar_h5_b,
                    int_h6: int_h6_b,
                    tar_h6: tar_h6_b,
                    int_h7: int_h7_b,
                    tar_h7: tar_h7_b,
                    int_al: int_al_b,
                    tar_al: tar_al_b,
                    int_cl: int_cl_b,
                    tar_cl: tar_cl_b,
                    dropout_keep_prob: 0.0,
                    is_training: False
                }
                pred_b = sess.run(pred, feed_dict)
                pred_labels.extend(pred_b.tolist())
            print(len(pred_labels))

        np.save('results/' + file_name, np.array(pred_labels))
       
            

                
                
            

    logger.info("Done.")


if __name__ == '__main__':
    test()
