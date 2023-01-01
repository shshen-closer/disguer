# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import tensorflow as tf
from datetime import datetime 
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from model import ATT
from utils import checkmate as cm
from utils import data_helpers as dh


TRAIN_OR_RESTORE = 'T' #input("Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()).replace(':', '_'))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()).replace(':', '_'))

tf.compat.v1.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.compat.v1.flags.DEFINE_float("norm_ratio", 10, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 256, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 50 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 4, "Number of epochs to train for.")


tf.compat.v1.flags.DEFINE_integer("decay_steps", 2, "how many steps before decay learning rate. (default: 500)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.2, "Rate of decay for learning rate. (default: 0.95)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


def train():
    """Training model."""

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    all_data = np.load("data/all_data.npy", allow_pickle=True)
    
    np.random.shuffle(all_data)
    aaa = int(0.1*len(all_data))
    folds = int(sys.argv[1])


    train_students = np.concatenate([all_data[:aaa*(folds)], all_data[aaa*(folds+1):]], axis = 0)
    valid_students = all_data[aaa*(folds):aaa*(folds+1)]
   

    
    print(np.shape(train_students))
    max_num_steps = 550
    max_num_skills = 1175

    gama = 0.01

    print((len(train_students)//FLAGS.batch_size + 1) * FLAGS.decay_steps)
    # Build a graph and lstm_3 object
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            att = ATT(
                batch_size = FLAGS.batch_size,
                num_steps = max_num_steps,
                num_skills = max_num_skills,
                hidden_size = FLAGS.hidden_size, 
                )
            

            # Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=att.global_step, decay_steps=(len(train_students)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
               # learning_rate = tf.train.piecewise_constant(FLAGS.epochs, boundaries=[7,10], values=[0.005, 0.0005, 0.0001])
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
               # grads, vars = zip(*optimizer.compute_gradients(att.loss))
                #grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                #train_op = optimizer.apply_gradients(zip(grads, vars), global_step=att.global_step, name="train_op")
                train_op = optimizer.minimize(att.loss, global_step=att.global_step, name="train_op")

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.compat.v1.summary.scalar("loss", att.loss)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=1, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load att model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())



            current_step = sess.run(att.global_step)

            def train_step(length, input_problem, input_kc, input_a, in_ty, tar_id, tar_kc, tar_ty, tar_index, tar_corr, 
            ana_his,ana_pre, int_k0, tar_k0, int_t, tar_t,int_t1, tar_t1, int_t2, tar_t2, int_t3, tar_t3, int_t4, tar_t4, 
            int_t5, tar_t5, int_t6, tar_t6, int_t7, tar_t7,int_h, tar_h,int_h1, tar_h1, int_h2, tar_h2, int_h3, tar_h3, int_h4, 
            tar_h4, int_h5, tar_h5, int_h6, tar_h6, int_h7, tar_h7, int_al, tar_al, int_cl, tar_cl):
                """A single training step"""
                
                feed_dict = {
                    att.length: length,
                    att.input_problem: input_problem,
                    att.input_kc: input_kc,
                    att.input_a: input_a,
                    att.in_ty: in_ty,
                    att.tar_id: tar_id,
                    att.tar_kc: tar_kc,
                    att.tar_ty: tar_ty,
                    att.tar_index: tar_index,
                    att.tar_corr: tar_corr,
                    att.ana_his: ana_his,
                    att.ana_pre: ana_pre,
                    att.int_k0: int_k0,
                    att.tar_k0: tar_k0,
                    att.int_t: int_t,
                    att.tar_t: tar_t,
                    att.int_t1: int_t1,
                    att.tar_t1: tar_t1,
                    att.int_t2: int_t2,
                    att.tar_t2: tar_t2,
                    att.int_t3: int_t3,
                    att.tar_t3: tar_t3,
                    att.int_t4: int_t4,
                    att.tar_t4: tar_t4,
                    att.int_t5: int_t5,
                    att.tar_t5: tar_t5,
                    att.int_t6: int_t6,
                    att.tar_t6: tar_t6,
                    att.int_t7: int_t7,
                    att.tar_t7: tar_t7,
                    att.int_h: int_h,
                    att.tar_h: tar_h,
                    att.int_h1: int_h1,
                    att.tar_h1: tar_h1,
                    att.int_h2: int_h2,
                    att.tar_h2: tar_h2,
                    att.int_h3: int_h3,
                    att.tar_h3: tar_h3,
                    att.int_h4: int_h4,
                    att.tar_h4: tar_h4,
                    att.int_h5: int_h5,
                    att.tar_h5: tar_h5,
                    att.int_h6: int_h6,
                    att.tar_h6: tar_h6,
                    att.int_h7: int_h7,
                    att.tar_h7: tar_h7,
                    att.int_al: int_al,
                    att.tar_al: tar_al,
                    att.int_cl: int_cl,
                    att.tar_cl: tar_cl,
                    
                    att.dropout_keep_prob: FLAGS.keep_prob,
                    att.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, att.global_step, train_summary_op, att.pred, att.loss], feed_dict)

                if step%10==0:
                    logger.info("step {0}: loss {1:g} ".format(step,loss))
                train_summary_writer.add_summary(summaries, step)
                return pred

            def validation_step(length, input_problem, input_kc, input_a, in_ty, tar_id, tar_kc, tar_ty, tar_index, tar_corr, 
            ana_his,ana_pre, int_k0, tar_k0, int_t, tar_t,int_t1, tar_t1, int_t2, tar_t2, int_t3, tar_t3, int_t4, tar_t4, 
            int_t5, tar_t5, int_t6, tar_t6, int_t7, tar_t7,int_h, tar_h,int_h1, tar_h1, int_h2, tar_h2, int_h3, tar_h3, int_h4, 
            tar_h4, int_h5, tar_h5, int_h6, tar_h6, int_h7, tar_h7, int_al, tar_al, int_cl, tar_cl):
                """Evaluates model on a validation set"""

                feed_dict = {
                    att.length: length,
                    att.input_problem: input_problem,
                    att.input_kc: input_kc,
                    att.input_a: input_a,
                    att.in_ty: in_ty,
                    att.tar_id: tar_id,
                    att.tar_kc: tar_kc,
                    att.tar_ty: tar_ty,
                    att.tar_index: tar_index,
                    att.tar_corr: tar_corr,
                    att.ana_his: ana_his,
                    att.ana_pre: ana_pre,
                    att.int_k0: int_k0,
                    att.tar_k0: tar_k0,
                    att.int_t: int_t,
                    att.tar_t: tar_t,
                    att.int_t1: int_t1,
                    att.tar_t1: tar_t1,
                    att.int_t2: int_t2,
                    att.tar_t2: tar_t2,
                    att.int_t3: int_t3,
                    att.tar_t3: tar_t3,
                    att.int_t4: int_t4,
                    att.tar_t4: tar_t4,
                    att.int_t5: int_t5,
                    att.tar_t5: tar_t5,
                    att.int_t6: int_t6,
                    att.tar_t6: tar_t6,
                    att.int_t7: int_t7,
                    att.tar_t7: tar_t7,
                    att.int_h: int_h,
                    att.tar_h: tar_h,
                    att.int_h1: int_h1,
                    att.tar_h1: tar_h1,
                    att.int_h2: int_h2,
                    att.tar_h2: tar_h2,
                    att.int_h3: int_h3,
                    att.tar_h3: tar_h3,
                    att.int_h4: int_h4,
                    att.tar_h4: tar_h4,
                    att.int_h5: int_h5,
                    att.tar_h5: tar_h5,
                    att.int_h6: int_h6,
                    att.tar_h6: tar_h6,
                    att.int_h7: int_h7,
                    att.tar_h7: tar_h7,
                    att.int_k0: int_k0,
                    att.tar_k0: tar_k0,
                    att.int_al: int_al,
                    att.tar_al: tar_al,
                    att.int_cl: int_cl,
                    att.tar_cl: tar_cl,

                    att.dropout_keep_prob: 0.0,
                    att.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [att.global_step, validation_summary_op, att.pred, att.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                
                return pred
            # Training loop. For each batch...
            
            run_time = []

            for iii in range(FLAGS.epochs):
                np.random.shuffle(train_students)
                a=datetime.now()
                data_size = len(train_students)
                index = 0
                actual_labels = []
                pred_labels = []
                while(index+FLAGS.batch_size <= data_size):
                    input_problem = np.zeros((FLAGS.batch_size, max_num_steps))
                    input_kc = np.zeros((FLAGS.batch_size, max_num_steps))
                    input_a = np.zeros((FLAGS.batch_size, max_num_steps))
                    in_ty = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_id = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_kc = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_ty = np.zeros((FLAGS.batch_size, max_num_steps))
                    ana_his = np.zeros((FLAGS.batch_size, max_num_steps))
                    ana_pre = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_k0 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_k0 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t1 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t1 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t2 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t2 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t3 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t3 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t4 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t4 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t5 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t5 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t6 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t6 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_t7 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_t7 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h1 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h1 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h2 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h2 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h3 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h3 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h4 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h4 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h5 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h5 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h6 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h6 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_h7 = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_h7 = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_al = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_al = np.zeros((FLAGS.batch_size, max_num_steps))
                    int_cl = np.zeros((FLAGS.batch_size, max_num_steps))
                    tar_cl = np.zeros((FLAGS.batch_size, max_num_steps))
                    length = np.zeros((FLAGS.batch_size, ))
                    tar_corr = []
                    tar_index = []
                    for i in range(FLAGS.batch_size):
                        student = train_students[index+i]
                        fens = np.random.randint(40,53) #50 #
                        splits = int(fens/100*len(student[0]))
                        input_problem[i][:splits] = student[0][:splits]
                        input_kc[i][:splits] = student[1][:splits]
                        input_a[i][:splits] = student[2][:splits]
                        in_ty[i][:splits] = student[3][:splits]
                        ana_his[i][:splits] = student[4][:splits]
                        int_k0[i][:splits] = student[5][:splits]
                        int_t[i][:splits] = student[6][:splits]
                        int_t1[i][:splits] = student[7][:splits]
                        int_t2[i][:splits] = student[8][:splits]
                        int_t3[i][:splits] = student[9][:splits]
                        int_t4[i][:splits] = student[10][:splits]
                        int_t5[i][:splits] = student[11][:splits]
                        int_t6[i][:splits] = student[12][:splits]
                        int_t7[i][:splits] = student[13][:splits]
                        int_h[i][:splits] = student[14][:splits]
                        int_h1[i][:splits] = student[15][:splits]
                        int_h2[i][:splits] = student[16][:splits]
                        int_h3[i][:splits] = student[17][:splits]
                        int_h4[i][:splits] = student[18][:splits]
                        int_h5[i][:splits] = student[19][:splits]
                        int_h6[i][:splits] = student[20][:splits]
                        int_h7[i][:splits] = student[21][:splits]
                        int_al[i][:splits] = student[22][:splits]
                        int_cl[i][:splits] = student[23][:splits]
                        length[i] = splits

                        tar_id[i][:len(student[0][splits:])] = student[0][splits:]
                        tar_kc[i][:len(student[1][splits:])] = student[1][splits:]
                        tar_ty[i][:len(student[3][splits:])] = student[3][splits:]
                        ana_pre[i][:len(student[4][splits:])] = student[4][splits:]
                        tar_k0[i][:len(student[5][splits:])] = student[5][splits:]
                        tar_t[i][:len(student[6][splits:])] = student[6][splits:]
                        tar_t1[i][:len(student[7][splits:])] = student[7][splits:]
                        tar_t2[i][:len(student[8][splits:])] = student[8][splits:]
                        tar_t3[i][:len(student[9][splits:])] = student[9][splits:]
                        tar_t4[i][:len(student[10][splits:])] = student[10][splits:]
                        tar_t5[i][:len(student[11][splits:])] = student[11][splits:]
                        tar_t6[i][:len(student[12][splits:])] = student[12][splits:]
                        tar_t7[i][:len(student[13][splits:])] = student[13][splits:]
                        tar_h[i][:len(student[14][splits:])] = student[14][splits:]
                        tar_h1[i][:len(student[15][splits:])] = student[15][splits:]
                        tar_h2[i][:len(student[16][splits:])] = student[16][splits:]
                        tar_h3[i][:len(student[17][splits:])] = student[17][splits:]
                        tar_h4[i][:len(student[18][splits:])] = student[18][splits:]
                        tar_h5[i][:len(student[19][splits:])] = student[19][splits:]
                        tar_h6[i][:len(student[20][splits:])] = student[20][splits:]
                        tar_h7[i][:len(student[21][splits:])] = student[21][splits:]
                        tar_al[i][:len(student[22][splits:])] = student[22][splits:]
                        tar_cl[i][:len(student[23][splits:])] = student[23][splits:]
                        for j in range(len(student[0][splits:])):
                            tar_index.append(i*max_num_steps+j)
                            tar_corr.append(student[2][j + splits])
                            actual_labels.append(student[2][j + splits])

                    index += FLAGS.batch_size
                    
                    pred = train_step(length, input_problem, input_kc, input_a, in_ty, tar_id, tar_kc, tar_ty, tar_index, tar_corr, 
                    ana_his,ana_pre, int_k0, tar_k0, int_t, tar_t,int_t1, tar_t1, int_t2, tar_t2, int_t3, tar_t3, int_t4, tar_t4, 
                    int_t5, tar_t5, int_t6, tar_t6, int_t7, tar_t7,int_h, tar_h,int_h1, tar_h1, int_h2, tar_h2, int_h3, tar_h3, int_h4, 
                    tar_h4, int_h5, tar_h5, int_h6, tar_h6, int_h7, tar_h7, int_al, tar_al, int_cl, tar_cl)
                    for p in pred:
                        pred_labels.append(p)
                    current_step = tf.compat.v1.train.global_step(sess, att.global_step)
                
                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)
                rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                auc = metrics.roc_auc_score(actual_labels, pred_labels)
                #calculate r^2
                r2 = r2_score(actual_labels, pred_labels)
                
                pred_score = np.greater_equal(pred_labels,0.5) 
                pred_score = pred_score.astype(int)
                pred_score = np.equal(actual_labels, pred_score)
                acc = np.mean(pred_score.astype(int))

                logger.info("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g}".format((iii +1),rmse, auc, r2, acc))

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")
                    
                    data_size = len(valid_students)
                    index = 0
                    actual_labels = []
                    pred_labels = []
                    while(index+FLAGS.batch_size <= data_size):
                        input_problem = np.zeros((FLAGS.batch_size, max_num_steps))
                        input_kc = np.zeros((FLAGS.batch_size, max_num_steps))
                        input_a = np.zeros((FLAGS.batch_size, max_num_steps))
                        in_ty = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_id = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_kc = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_ty = np.zeros((FLAGS.batch_size, max_num_steps))
                        ana_his = np.zeros((FLAGS.batch_size, max_num_steps))
                        ana_pre = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_k0 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_k0 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t1 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t1 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t2 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t2 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t3 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t3 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t4 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t4 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t5 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t5 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t6 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t6 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_t7 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_t7 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h1 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h1 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h2 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h2 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h3 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h3 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h4 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h4 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h5 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h5 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h6 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h6 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_h7 = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_h7 = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_al = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_al = np.zeros((FLAGS.batch_size, max_num_steps))
                        int_cl = np.zeros((FLAGS.batch_size, max_num_steps))
                        tar_cl = np.zeros((FLAGS.batch_size, max_num_steps))
                        length = np.zeros((FLAGS.batch_size, ))
                        tar_corr = []
                        tar_index = []
                        for i in range(FLAGS.batch_size):
                            student = valid_students[index+i]
                            splits = int(0.5*len(student[0]))
                            input_problem[i][:splits] = student[0][:splits]
                            input_kc[i][:splits] = student[1][:splits]
                            input_a[i][:splits] = student[2][:splits]
                            in_ty[i][:splits] = student[3][:splits]
                            ana_his[i][:splits] = student[4][:splits]
                            int_k0[i][:splits] = student[5][:splits]
                            int_t[i][:splits] = student[6][:splits]
                            int_t1[i][:splits] = student[7][:splits]
                            int_t2[i][:splits] = student[8][:splits]
                            int_t3[i][:splits] = student[9][:splits]
                            int_t4[i][:splits] = student[10][:splits]
                            int_t5[i][:splits] = student[11][:splits]
                            int_t6[i][:splits] = student[12][:splits]
                            int_t7[i][:splits] = student[13][:splits]
                            int_h[i][:splits] = student[14][:splits]
                            int_h1[i][:splits] = student[15][:splits]
                            int_h2[i][:splits] = student[16][:splits]
                            int_h3[i][:splits] = student[17][:splits]
                            int_h4[i][:splits] = student[18][:splits]
                            int_h5[i][:splits] = student[19][:splits]
                            int_h6[i][:splits] = student[20][:splits]
                            int_h7[i][:splits] = student[21][:splits]
                            int_al[i][:splits] = student[22][:splits]
                            int_cl[i][:splits] = student[23][:splits]
                            length[i] = splits

                            tar_id[i][:len(student[0][splits:])] = student[0][splits:]
                            tar_kc[i][:len(student[1][splits:])] = student[1][splits:]
                            tar_ty[i][:len(student[3][splits:])] = student[3][splits:]
                            ana_pre[i][:len(student[4][splits:])] = student[4][splits:]
                            tar_k0[i][:len(student[5][splits:])] = student[5][splits:]
                            tar_t[i][:len(student[6][splits:])] = student[6][splits:]
                            tar_t1[i][:len(student[7][splits:])] = student[7][splits:]
                            tar_t2[i][:len(student[8][splits:])] = student[8][splits:]
                            tar_t3[i][:len(student[9][splits:])] = student[9][splits:]
                            tar_t4[i][:len(student[10][splits:])] = student[10][splits:]
                            tar_t5[i][:len(student[11][splits:])] = student[11][splits:]
                            tar_t6[i][:len(student[12][splits:])] = student[12][splits:]
                            tar_t7[i][:len(student[13][splits:])] = student[13][splits:]
                            tar_h[i][:len(student[14][splits:])] = student[14][splits:]
                            tar_h1[i][:len(student[15][splits:])] = student[15][splits:]
                            tar_h2[i][:len(student[16][splits:])] = student[16][splits:]
                            tar_h3[i][:len(student[17][splits:])] = student[17][splits:]
                            tar_h4[i][:len(student[18][splits:])] = student[18][splits:]
                            tar_h5[i][:len(student[19][splits:])] = student[19][splits:]
                            tar_h6[i][:len(student[20][splits:])] = student[20][splits:]
                            tar_h7[i][:len(student[21][splits:])] = student[21][splits:]
                            tar_al[i][:len(student[22][splits:])] = student[22][splits:]
                            tar_cl[i][:len(student[23][splits:])] = student[23][splits:]

                            for j in range(len(student[0][splits:])):
                                tar_index.append(i*max_num_steps+j)
                                tar_corr.append(student[2][j + splits])
                                actual_labels.append(student[2][j + splits])

                        index += FLAGS.batch_size
                        pred  = validation_step(length, input_problem, input_kc, input_a, in_ty, tar_id, tar_kc, tar_ty, tar_index, tar_corr, 
                        ana_his,ana_pre, int_k0, tar_k0, int_t, tar_t,int_t1, tar_t1, int_t2, tar_t2, int_t3, tar_t3, int_t4, tar_t4, 
                        int_t5, tar_t5, int_t6, tar_t6, int_t7, tar_t7,int_h, tar_h,int_h1, tar_h1, int_h2, tar_h2, int_h3, tar_h3, int_h4, 
                        tar_h4, int_h5, tar_h5, int_h6, tar_h6, int_h7, tar_h7, int_al, tar_al, int_cl, tar_cl)
                        for p in pred:
                            pred_labels.append(p)
                    

                    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                    auc = metrics.roc_auc_score(actual_labels, pred_labels)
                    #calculate r^2
                    r2 = r2_score(actual_labels, pred_labels)
                    
                    pred_score = np.greater_equal(pred_labels,0.5) 
                    pred_score = pred_score.astype(int)
                    pred_score = np.equal(actual_labels, pred_score)
                    acc = np.mean(pred_score.astype(int))

                    logger.info("VALIDATION {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g} ".format((iii +1)/FLAGS.evaluation_interval,rmse, auc, r2, acc))

                    best_saver.handle(auc, sess, current_step)

              

                logger.info("Epoch {0} has finished!".format(iii + 1))
            
            logger.info("running time analysis: epoch{0}, avg_time{1}".format(len(run_time), np.mean(run_time)))

    logger.info("Done.")


if __name__ == '__main__':
    train()
