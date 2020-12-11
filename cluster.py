from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os
fileDir = os.path.dirname(os.path.realpath('__file__'))

model = VGG16(weights='imagenet', include_top=False)
# model.summary()

img_path = 'cluster/Class101/train_134.jpg'
img = image.load_img(img_path, target_size=(299, 299))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model.predict(img_data)

# print(vgg16_feature.shape)

vgg16_feature_list = []
subdir = f'{fileDir}/cluster/'
for d in os.listdir(subdir):
    # get the directory names, i.e., 'dogs' or 'cats'
    # ...
    direct = subdir + f'{d}'
    # print(direct)

    # for i, fname in enumerate(filenames):
    for filename in os.listdir(direct):
        # process the files under the directory 'dogs' or 'cats'
        # ...
        # print(filename)
        img_path = direct + f'/{filename}'
        img = image.load_img(img_path, target_size=(299, 299))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        # print(img_data)
        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())

vgg16_feature_list_np = np.array(vgg16_feature_list)
kmeans = KMeans(n_clusters=2, random_state=0).fit(vgg16_feature_list_np)
print(kmeans)
