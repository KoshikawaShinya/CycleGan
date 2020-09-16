import keras as K
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import keras_contrib

images = []
model = K.models.load_model('saved_model/photo2vangogh/vangogh.h5',custom_objects={'InstanceNormalization':keras_contrib.layers.normalization.instancenormalization.InstanceNormalization})

for img_path in glob.glob('images/original/*'):

    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

images = np.array(images)
images = images / 127.5 - 1.0

for i, image in enumerate(images):
    print(i)
    image = np.expand_dims(image, axis=0)
    translated_image = model.predict(image)
    translated_image = np.reshape(translated_image, (256, 256, 3))
    translated_image = 0.5 * translated_image + 0.5
    plt.imsave('images/translated/vangogh_%d.jpg' %(i+1), translated_image)

print(translated_image)
plt.imshow(translated_image)
plt.show()
