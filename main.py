import numpy as np
import os
import tensorflow as tf
import random,time,json,pdb,scipy.misc,glob
from model_queue import DCGAN
from test import EVAL
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image
from numpy import inf
from sorting import natsorted
import matplotlib.image as mpimg
from scipy import ndimage
flags = tf.app.flags
flags.DEFINE_integer("epoch", 3, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 12, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "0422", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "output", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("input_size", 64, "The size of image input size")
flags.DEFINE_integer("num_block", 3, "The number of block for generator model")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def main(_):
    width_size = 905
    height_size = 565
    #width_size = 1104
    #height_size = 764
    #width_size = 1123
    #height_size = 900
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    #with tf.Session() as sess:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
        if FLAGS.is_train:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,\
	    num_block = FLAGS.num_block,dataset_name=FLAGS.dataset,is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
	    dcgan = EVAL(sess, batch_size=1,num_block=FLAGS.num_block,ir_image_shape=[None,None,1],dataset_name=FLAGS.dataset,\
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)
	    print('deep model test \n')

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            list_val = [11,16,21,22,33,36,38,53,59,92]
	    print '1: Estimating Normal maps from arbitary obejcts \n'
            print '2: EStimating Normal maps of NIR dataset \n'
	    x = input('Selecting a Evaluation mode:')
            VAL_OPTION = int(x)

            if VAL_OPTION ==1: # arbitary dataset 
                print("Computing arbitary dataset ")
		trained_models = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		trained_models  = natsorted(trained_models)
		model = trained_models[6]
		model = model.split('/')
		model = model[-1]
		print('Load trained network: %s\n' %model)
		dcgan.load(FLAGS.checkpoint_dir,model)
		datapath = '/research3/datain/gmchoe_normal/0403/IR_0.25'
                savepath = datapath
		mean_nir = -0.3313
		img_files = glob.glob(os.path.join(datapath,'*.png'))
		img_files = natsorted(img_files)
		pdb.set_trace()
                for idx in xrange(0,len(img_files)):
		    print('Processing %d/%d \n' %(len(img_files),idx))
		    input_= scipy.misc.imread(img_files[idx],'F').astype(float)
		    height_size = input_.shape[0]
		    width_size = input_.shape[1]
	            input_ = np.reshape(input_,(height_size,width_size,1)) # LF size:383 x 552 
		    nondetail_input_ = ndimage.gaussian_filter(input_,sigma=(1,1,0),order=0)	   
		    input_ = input_/127.5 -1.0 
	            nondetail_input_  = nondetail_input_/127.5 -1.0 # normalize -1 ~1
	            detail_input_ = input_ - nondetail_input_
	            nondetail_input_ = np.reshape(nondetail_input_,(1,height_size,width_size,1)) # LF size:383 x 552 
	            detail_input_ = np.reshape(detail_input_,(1,height_size,width_size,1))
	            #detail_input_  = detail_input_/127.5 -1.0 # normalize -1 ~1
	            start_time = time.time() 
	            sample = sess.run(dcgan.G, feed_dict={dcgan.nondetail_images: nondetail_input_,dcgan.detail_images:detail_input_})
		    print('time: %.8f' %(time.time()-start_time))     
	            sample = np.squeeze(sample).astype(np.float32)

                    # normalization #
		    output = np.sqrt(np.sum(np.power(sample,2),axis=2))
		    output = np.expand_dims(output,axis=-1)
		    output = sample/output
		    output = (output+1.)/2.
		    """
		    if not os.path.exists(os.path.join(savepath,'%s/%s/%s' %(FLAGS.dataset,model,listdir[idx]))):
		        os.makedirs(os.path.join(savepath,'%s/%s/%s' %(FLAGS.dataset,model,listdir[idx])))
                    """
		    savename = os.path.join(savepath,'result/%s.bmp' % (img_files[idx][-10:]))
		    #savename = os.path.join(savepath,'single_normal_%02d.bmp' % (idx+1))
		    #savename = os.path.join(savepath,'%s/%s/%s/single_normal.bmp' % (FLAGS.dataset,model,listdir[idx]))
		    scipy.misc.imsave(savename, output)

	    elif VAL_OPTION ==2: # light source fixed
                list_val = [11,16,21,22,33,36,38,53,59,92]
		load,iteration = dcgan.load(FLAGS.checkpoint_dir)
	        pdb.set_trace()
		savepath ='./singleview_nir/L2ang/%d' %iteration
	        obj =1
		count =1
                if load:
            	    for idx in range(len(list_val)):
			if not os.path.exists(os.path.join(savepath,'%03d' %list_val[idx])):
		            os.makedirs(os.path.join(savepath,'%03d' %list_val[idx]))
		        for idx2 in range(1,10): #tilt angles 
			    print("Selected material %03d/%d" %(list_val[idx],idx2))
			    img = './dataset/multi-view/testdata_3579/%03d/%03d/patch_%06d.mat' %(obj,idx2,count)
			    input_ = scipy.io.loadmat(img)
			    input_ = input_['input_']
	                    input_ = input_.astype(np.float)
			    #input_ = input_[:,:,0:3]
			    input_ = np.reshape(input_,(1,600,800,4))
			    input_ = input_/127.5 -1.0 
			    start_time = time.time() 
	                    sample = sess.run([dcgan.G], feed_dict={dcgan.images:input_})
			    print('time: %.8f' %(time.time()-start_time))     
			    # normalization #
			    sample = np.squeeze(sample).astype(np.float32)
                            output = np.sqrt(np.sum(np.power(sample,2),axis=2))
			    output = np.expand_dims(output,axis=-1)
			    output = sample/output
			    output = (output+1.)/2.
			    if not os.path.exists(os.path.join(savepath,'%03d/%03d' %(list_val[idx],idx2))):
			        os.makedirs(os.path.join(savepath,'%03d/%03d' %(list_val[idx],idx2)))
			    savename = os.path.join(savepath, '%03d/%03d/multiview_normal.bmp' % (list_val[idx],idx2))
			    scipy.misc.imsave(savename, output)
                            count = count +1
                        obj = obj +1
	        else:
	            print("Failed to load network")

if __name__ == '__main__':
    tf.app.run()
