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

	pdb.set_trace()
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
	self.loss ='L1'
	self.input_type = 'single' #multi frequency
        self.list_val = [11,16,21,22,33,36,38,53,59,92]
	self.pair = False
	self.build_model()
	
    def build_model(self):
       
        self.images = tf.placeholder(tf.float32,shape=self.ir_image_shape)
	self.normal_images = tf.placeholder(tf.float32,shape=self.normal_image_shape)
	if not self.use_queue:

        	self.low_ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_ir_image_shape,
                                    name='low_ir_images')
        	self.low_normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_normal_image_shape,
                                    name='low_normal_images')
	else:
		print ' using queue loading'
	        self.image_single = tf.placeholder(tf.float32,shape=self.ir_image_shape)
	        self.normal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
                q = tf.RandomShuffleQueue(1000,100,[tf.float32,tf.float32],[[self.ir_image_shape[0],self.ir_image_shape[1],4],[self.normal_image_shape[0],self.normal_image_shape[1],3]])
	        self.enqueue_op = q.enqueue([self.image_single,self.normal_image_single])
	        self.images,self.normal_images = q.dequeue_many(self.batch_size)

	self.keep_prob = tf.placeholder(tf.float32)
	net  = networks(64,self.df_dim)
     	self.G = net.generator(self.images) 
	self.G = self.G[-1]

	######## evaluation #######
	'''
	if self.input_type == 'single':
	    self.sample_G = tf.placeholder(tf.float32,shape=[1,600,800,1],name='sampler') 

	else:
	    self.sample_low= tf.placeholder(tf.float32,shape=[1,600,800,1],name='sampler_low')
	    self.sample_high= tf.placeholder(tf.float32,shape=[1,600,800,1],name='sampler_high')
	    self.sample_low_G,self.sample_high_G =net.multi_freq_sampler(self.sample_low,self.sample_high)
	    self.sample_G = self.sample_low_G[-1] + self.sample_high_G[-1]	
	'''
	################ Discriminator Loss ######################
        self.D = net.discriminator(self.normal_images,self.keep_prob)
        self.D_  = net.discriminator(self.G,self.keep_prob,reuse=True)

	#### entire resolution ####
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D[-1]), self.D[-1])
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_[-1]), self.D_[-1])
        self.d_loss = self.d_loss_real + self.d_loss_fake 
	########################## Generative loss ################################
	self.ang_loss = ang_loss.ang_error(self.G,self.normal_images)
	
	if self.loss == 'L1':
            self.L_loss = tf.reduce_mean(tf.abs(tf.subtract(self.G,self.normal_images)))

	else:
            self.L_loss = tf.reduce_mean(tf.square(self.G-self.normal_images))

        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_[-1]), self.D_[-1])
        self.gen_loss = self.g_loss + (self.L_loss+self.ang_loss)*100
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


        try:
	    tf.global_variables_initializer().run()
	except:
	    tf.initialize_all_variables().run()
	
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
	with open("./traininput_list_3579.txt",'rb') as f:
            train_input = pickle.load(f)
	train_input.sort()
	with open("./traingt_list_3579.txt",'rb') as f:
            train_gt = pickle.load(f)
        train_gt.sort()
	#train_input =[data[idx] for idx in xrange(0,len(data))]
	#train_gt =[data_label[idx] for idx in xrange(0,len(data))]

	shuf = range(len(train_input))
	random.shuffle(shuf)
	
	if self.use_queue:
	    # creat thread
	    coord = tf.train.Coordinator()
            num_thread =1
            for i in range(num_thread):
 	        t = threading.Thread(target=self.load_and_enqueue_single,args=(coord,train_input,train_gt,shuf,i,num_thread))
	        t.start()

        for epoch in xrange(config.epoch):
	    shuffle = np.random.permutation(range(len(train_input)))
	    batch_idxs = min(len(train_input), config.train_size)/config.batch_size

            for idx in xrange(0,batch_idxs):
                #batches = [get_image(train_input[batch_file], train_gt[batch_file],self.image_size,is_crop=self.is_crop) for batch_file in batch_files]

                #batch_images = np.array(ir_batch).astype(np.float32)
                #batchlabel_images = np.array(normal_batchlabel).astype(np.float32)

                start_time = time.time()
	        _,d_err =self.sess.run([d_optim,self.d_loss],feed_dict={self.keep_prob:self.dropout})
	        _,g_err,L_err,ang_err =self.sess.run([g_optim,self.g_loss,self.L_loss,self.ang_loss])
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L: %.6f d_loss:%.4f ang_loss:%.6f" \
		         % (epoch, idx, batch_idxs,time.time() - start_time,g_err,L_err,d_err,ang_err))

                if np.mod(global_step1.eval(),400) ==0 and global_step1 != 0:
	            self.save(config.checkpoint_dir,global_step1)


	'''
	for epoch in xrange(config.epoch):
	    #shuffle = np.random.permutation(range(len(data)))
	    batch_idxs = min(len(train_input), config.train_size)/config.batch_size

            for idx in xrange(0,batch_idxs):
                start_time = time.time()
		_,d_err =self.sess.run([d_optim,self.d_loss],feed_dict={self.keep_prob:self.dropout})
		_,g_err,L_err,ang_err =self.sess.run([g_optim,self.g_loss,self.L_loss,self.ang_loss])
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L: %.6f d_loss:%.4f ang_loss:%.6f" \
		        % (epoch, idx, batch_idxs,time.time() - start_time,g_err,L_err,d_err,ang_err))

                if np.mod(global_step1.eval(),4000) ==0 and global_step1 != 0:
	                     self.save(config.checkpoint_dir,global_step1)

        '''
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

        model_dir = "%s_%s" % (self.dataset_name,self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

	
    def load_and_enqueue_single(self,coord,file_list,label_list,shuf,idx=0,num_thread=1):
	count =0;
	length = len(file_list)
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
            input_img = scipy.io.loadmat(file_list[shuf[i]])
	    input_img = input_img['input_']
	    gt_img = scipy.misc.imread(label_list[shuf[i]])
	    input_img = input_img/127.5 -1.
	    gt_img = gt_img/127.5 -1.
	
	    rand_x = np.random.randint(64,224-64)
	    rand_y = np.random.randint(64,224-64)
	    ipt =  input_img[rand_y:rand_y+64,rand_x:rand_x+64,:]
	    label = gt_img[rand_y:rand_y+64,rand_x:rand_x+64,:]
	    self.sess.run(self.enqueue_op,feed_dict={self.image_single:ipt,self.normal_image_single:label})
	    count +=1
   	    #print('count:%d' %count) 
		
