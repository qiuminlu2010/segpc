import pickle
import os
import json
from pycocotools import mask
import cv2
import numpy as np

def main():
    with open('./pred/model1_nu_test.pkl','rb') as f:
        nu_test_pkl = pickle.load(f,encoding='utf-8')

    img_dir = "./datasets/test/x" 
    img_ids = sorted(os.listdir(img_dir))
    imgs_gt = list(set([i.split('.')[0] for i in img_ids]))
    imgs_gt= sorted(imgs_gt)

    assert len(imgs_gt)==277
    nu_save_path = './pred/test/nu'
    if not os.path.exists(nu_save_path):
        os.mkdir(nu_save_path)
    
    for i,ann in enumerate(nu_test_pkl):
        bbox,masks = ann
        scores = bbox[0][...,4]
        masks = masks[0]
        temp =[]
        for j,m in enumerate(masks):
            temp.append(mask.decode(m))
        print(i,img_ids[i])
        nu_pr = np.where(np.array(temp).sum(0)>0,1,0)
        print(os.path.join(nu_save_path, img_ids[i]))
        cv2.imwrite(os.path.join(nu_save_path, img_ids[i]), nu_pr)
    
    with open('./pred/ensemble_cell_test.pkl','rb') as f:
        cell_test_pkl = pickle.load(f,encoding='utf-8')

    inst_save_path = './pred/test/inst'
    if not os.path.exists(inst_save_path):
        os.mkdir(inst_save_path)

    for i,ann in enumerate(cell_test_pkl):
        bbox,masks = ann
        scores = bbox[0][...,4]
        masks = masks[0]
        temp =[]
        nu_pr =  cv2.imread(os.path.join(nu_save_path,img_ids[i]),0)
        for j,m in enumerate(masks):
            cell_pr = mask.decode(m)
            if cell_pr.sum() < 1000:
                continue
            inst_pr = (nu_pr + cell_pr)*cell_pr
            if inst_pr.max() == 1:
                inst_pr[0][0] = 2
            pred = np.zeros_like(inst_pr)
            pred[inst_pr==1] = 20
            pred[inst_pr==2] = 40   
            cv2.imwrite(os.path.join(inst_save_path, imgs_gt[i]+"_"+str(j+1)+".bmp"), pred)                          
        print(i,img_ids[i])     
    

if __name__ == '__main__':
    main()





