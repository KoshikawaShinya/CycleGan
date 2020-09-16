import cv2
import glob
import numpy as np
import pickle


datas = []
labels = []


# 画像をテンソル化
for fold_path in glob.glob('data/vangogh2photo/*'):
    print(fold_path)
    imgs = glob.glob(fold_path + '/*')
    datas = []

    if 'trainA' in fold_path  or 'trainB' in fold_path:
        for img_path in imgs:
            print(img_path)
            img = cv2.imread(img_path)
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            datas.append(img)

        if 'trainA' in fold_path:
            A_imgs = np.array(datas)
        else:
            B_imgs = np.array(datas)
    


print(A_imgs.shape[0])
print(B_imgs.shape[0])


print('dataset化中')
np.savez('datasets/vangogh2photo_256_t.npz', A_imgs=A_imgs, B_imgs=B_imgs)


