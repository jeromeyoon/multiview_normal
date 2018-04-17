import os,time,pdb,argparse,threading,ang_loss,pickle
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
from ops import *
from utils import *
from random import shuffle
from network import networks
from scipy import ndimage
import scipy.io
class DCGAN(object):
    def __init__(self, sess, image_size=108, is_train=True,is_crop=True,\
                 batch_size=12,num_block=1,ir_image_shape=[256, 256,1], normal_image_shape=[256, 256, 3],\
	         light_shape=[64,64,3],df_dim=64,dataset_name='default',checkpoint_dir=None):

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
	self.num_block = num_block
        self.checkpoint_dir = checkpoint_dir
	self.ir_image_shape=[64,64,4]
	self.normal_image_shape=[64,64,3]
	self.use_queue = True
	self.mean_nir = -0.3313 #-1~1
	self.dropout =0.7
	self.loss ='L2'
	self.lambda_d = 100
	self.input_type = 'single' #multi frequency
        self.list_val = [11,16,21,22,33,36,38,53,59,92]
	self.pair = False
	self.build_model()
	
    def build_model(self):
       
        self.images = tf.placeholder(tf.float32,shape=[self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],3])
	self.normal_images = tf.placeholder(tf.float32,shape=[self.batch_size,self.normal_image_shape[0],self.normal_image_shape[1],3])
	self.keep_prob = tf.placeholder(tf.float32)
	net  = networks(64,self.df_dim)
     	self.G = net.generator(self.images) 
        self.G = self.G[-1]
	################ Discriminator Loss ######################
        self.D = net.discriminator(self.normal_images,self.keep_prob)
	self.D_  = net.discriminator(self.G,self.keep_prob,reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D[-1]), self.D[-1])
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_[-1]), self.D_[-1])
        self.d_loss = self.d_loss_real + self.d_loss_fake 
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real",self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake",self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss",self.d_loss)

	########################## Generative loss ################################
	self.ang_loss = ang_loss.ang_error(self.G,self.normal_images)
        self.ang_loss_sum = tf.summary.scalar("ang_loss",self.ang_loss)
	
	if self.loss == 'L1':
            self.L_loss = tf.reduce_mean(tf.abs(tf.subtract(self.G,self.normal_images)))
	    self.L_loss_sum = tf.summary.scalar("L1_loss",self.L_loss)
	else:
            self.L_loss = tf.reduce_mean(tf.square(self.G-self.normal_images))
            self.L_loss_sum = tf.summary.scalar("L2_loss",self.L_loss)
            
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_[-1]), self.D_[-1])
	self.g_loss_sum = tf.summary.scalar("g_loss",self.g_loss)
        self.gen_loss = self.g_loss + (self.L_loss+self.ang_loss)*self.lambda_d
	self.gen_loss_sum = tf.summary.scalar("gen_loss",self.g_loss)
	t_vars = tf.trainable_variables()
	self.g_vars =[var for var in t_vars if 'g' in var.name]
	self.d_vars =[var for var in t_vars if 'dis' in var.name]

	self.saver = tf.train.Saver(max_to_keep=20)
    def train(self, config):
        #####Train DCGAN####

        global_step1 = tf.Variable(0,name='global_step1',trainable=False)
        global_step2 = tf.Variable(0,name='global_step2',trainable=False)
        d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.d_loss, global_step=global_step1,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.gen_loss, global_step=global_step2,var_list=self.g_vars)

	self.g_sum = tf.summary.merge([self.g_loss_sum,self.L_loss_sum,self.ang_loss_sum,self.gen_loss_sum])
	self.d_sum = tf.summary.merge([self.d_loss_sum,self.d_loss_real_sum,self.d_loss_fake_sum])

	self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        try:
	    tf.global_variables_initializer().run()
	except:
	    tf.initialize_all_variables().run()
	
        start_time = time.time()

        load,counter = self.load(self.checkpoint_dir)
	pdb.set_trace()
        if load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
	# loda training and validation dataset path
	with open("./traininput_list_468.txt",'rb') as f:
            train_input = pickle.load(f)
	train_input.sort()
	with open("./traingt_list_468.txt",'rb') as f:
            train_gt = pickle.load(f)
        train_gt.sort()

	shuf = range(len(train_input))
	random.shuffle(shuf)
        for epoch in xrange(config.epoch):
	    shuffle = np.random.permutation(range(len(train_input)))
	    batch_idxs = min(len(train_input), config.train_size)/config.batch_size

            for idx in xrange(0,batch_idxs):
                batch_files = shuffle[idx*config.batch_size:(idx+1)*config.batch_size]
                batches = [get_image(train_input[batch_file], train_gt[batch_file],self.image_size,is_crop=self.is_crop) for batch_file in batch_files]
	        batches = np.array(batches)
                batch_images = np.array(batches[:,:,:,0:3]).astype(np.float32)
                batchlabel_images = np.array(batches[:,:,:,3:6]).astype(np.float32)
                #batch_images = np.array(batches[:,:,:,0:4]).astype(np.float32)
                #batchlabel_images = np.array(batches[:,:,:,4:7]).astype(np.float32)
	        batch_images = (batch_images)/127.5 -1.0		
	        batchlabel_images = (batchlabel_images)/127.5 -1.0		
                start_time = time.time()
	        _,summary,d_err =self.sess.run([d_optim,self.d_sum,self.d_loss],feed_dict={self.images:batch_images,self.normal_images:batchlabel_images,self.keep_prob:self.dropout})
		self.writer.add_summary(summary, counter)
	        _,summary,g_err,L_err,ang_err,output =self.sess.run([g_optim,self.g_sum,self.g_loss,self.L_loss,self.ang_loss,self.G],feed_dict={self.images:batch_images,self.normal_images:batchlabel_images})
		self.writer.add_summary(summary, counter)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L: %.6f d_loss:%.4f ang_loss:%.6f" \
		         % (epoch, idx, batch_idxs,time.time() - start_time,g_err,L_err,d_err,ang_err))
		'''
	        input1 = batch_images[0,:,:,0]
		input1 = (input1+1.)/2.
                scipy.misc.imsave('./input1.png', input1)
	        input2 = batch_images[0,:,:,1]
		input2 = (input2+1.)/2.
                scipy.misc.imsave('./input2.png', input2)
	        input3 = batch_images[0,:,:,2]
		input3 = (input3+1.)/2.
                scipy.misc.imsave('./input3.png', input3)
	        input4 = batch_images[0,:,:,3]
		input4 = (input4+1.)/2.
                scipy.misc.imsave('./input4.png', input4)
	        sample = output[0,:,:,:]
		sample = np.squeeze(sample).astype(np.float32)
                output = np.sqrt(np.sum(np.power(sample,2),axis=2))
		output = np.expand_dims(output,axis=-1)
		output = sample/output
		output = (output+1.)/2.
                scipy.misc.imsave('./sample.png', output)
		'''
                if np.mod(global_step1.eval(),100) ==0 and global_step1 != 0:
	            self.save(config.checkpoint_dir,global_step1)

    		counter = counter+1
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),

                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        import re
        model_dir = "%s_%s" % (self.dataset_name,self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	pdb.set_trace()
	if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,ckpt.all_model_checkpoint_paths[-1])
	    counter = ckpt.all_model_checkpoint_paths[-1]
            counter = counter[37:].encode('utf-8')
	    #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print("[*] Success to read ")
            #print("[*] Success to read {}".format(ckpt_name))
            return True,int(counter)
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

