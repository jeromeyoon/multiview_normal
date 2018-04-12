import os,time
from glob import glob
import tensorflow as tf
from ops import *
from utils import *
from network import networks
class EVAL(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=1, num_block=1,ir_image_shape=[64, 64, 1],
                 df_dim=64,dataset_name='default',checkpoint_dir=None):

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
	self.num_block = num_block
        self.build_model()

    def build_model(self):

        
	self.images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='images')
	net  = networks(64,64)
        self.G = net.generator(self.images)
        self.G = self.G[-1]	
        self.saver = tf.train.Saver()


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        import re
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	pdb.set_trace()	
	if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,ckpt.all_model_checkpoint_paths[-1])
	    #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print("[*] Success to read ")
            #print("[*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

