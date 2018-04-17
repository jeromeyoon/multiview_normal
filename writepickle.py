import pickle,os
import glob,pdb 

paths = '/research2/PAMI_LightField/dataset/multi-view/trainingdata_3579'
data=[]
gt=[]
folders = os.listdir(paths)
folders.sort()
pdb.set_trace()
for folder in folders:
    GTS = glob.glob(os.path.join(os.path.join(paths,folder),'GT_*.png'))
    Inputs = glob.glob(os.path.join(os.path.join(paths,folder),'patch_*.mat'))
    Inputs.sort()
    GTS.sort()
    data = data + Inputs
    gt = gt + GTS

with open('traininput_list_3579.txt','wb') as f:
    pickle.dump(data,f)

with open('traingt_list_3579.txt','wb') as f:
    pickle.dump(gt,f)


