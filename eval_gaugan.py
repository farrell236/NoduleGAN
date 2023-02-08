import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

from gaugan import GauGAN


data_root = '/mnt/nas_houbb/users/Benjamin/LUNA16'

a=1


# 109198: 'images/1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031_99.png'
# 109199: 'images/1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273_100.png'

# images = sorted(glob(os.path.join(data_root, f'images/*.png')))
# labels_ohe = sorted(glob(os.path.join(data_root, f'labels_ohe/*.png')))
# labels_rgb = sorted(glob(os.path.join(data_root, f'labels_rgb/*.png')))
#
# train_images, val_images = images[:109199], images[109199:]
# train_labels_ohe, val_labels_ohe = labels_ohe[:109199], labels_ohe[109199:]
# train_labels_rgb, val_labels_rgb = labels_rgb[:109199], labels_rgb[109199:]


# 1001: 'nodule_images/1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031_88.png'
# 1002: 'nodule_images/1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273_65.png'

nodule_images = sorted(glob(os.path.join(data_root, 'nodule_images/*.png')))
nodule_labels_ohe = sorted(glob(os.path.join(data_root, 'nodule_labels_ohe/*.png')))
nodule_labels_rgb = sorted(glob(os.path.join(data_root, 'nodule_labels_rgb/*.png')))

train_nodule_images, val_nodule_images = nodule_images[:1002], nodule_images[1002:]
train_nodule_labels_ohe, val_nodule_labels_ohe = nodule_labels_ohe[:1002], nodule_labels_ohe[1002:]
train_nodule_labels_rgb, val_nodule_labels_rgb = nodule_labels_rgb[:1002], nodule_labels_rgb[1002:]

# train_idx = np.random.default_rng(seed=42).choice(len(train_images), size=len(train_images)//10, replace=False)
# val_idx = np.random.default_rng(seed=42).choice(len(val_images), size=len(val_images)//10, replace=False)
#
# train_images_subset = train_nodule_images + [train_images[i] for i in train_idx]
# train_labels_ohe_subset = train_nodule_labels_ohe + [train_labels_ohe[i] for i in train_idx]
# train_labels_rgb_subset = train_nodule_labels_rgb + [train_labels_rgb[i] for i in train_idx]
# val_images_subset = val_nodule_images + [val_images[i] for i in val_idx]
# val_labels_ohe_subset = val_nodule_labels_ohe + [val_labels_ohe[i] for i in val_idx]
# val_labels_rgb_subset = val_nodule_labels_rgb + [val_labels_rgb[i] for i in val_idx]


IMG_HEIGHT = 512
NUM_CLASSES = 6
LATENT_DIM = 512
BATCH_SIZE = 16
NUM_EPOCHS = 100


def parse_function(image_file, label_ohe_file, label_rgb_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    label_ohe = tf.io.read_file(label_ohe_file)
    label_ohe = tf.image.decode_png(label_ohe, 1)
    label_ohe = tf.one_hot(label_ohe[..., 0], 6)

    label_rgb = tf.io.read_file(label_rgb_file)
    label_rgb = tf.image.decode_png(label_rgb, 3)
    label_rgb = tf.image.convert_image_dtype(label_rgb, tf.float32)

    return label_rgb, image, label_ohe


gaugan = GauGAN(IMG_HEIGHT, NUM_CLASSES, BATCH_SIZE, LATENT_DIM)
gaugan.load_weights('checkpoints/gaugan/gaugan_512x512.ckpt')


a=1

val_dataset = tf.data.Dataset.from_tensor_slices((val_nodule_images, val_nodule_labels_ohe, val_nodule_labels_rgb))
val_dataset = val_dataset.map(parse_function)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)


iterator = iter(val_dataset)
batch = next(iterator)
batch = next(iterator)
batch = next(iterator)
batch = next(iterator)


a=1

latent_vector = tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM), mean=0.0, stddev=2.0)
fake_image = gaugan.predict([latent_vector, batch[-1]])

for idx in range(BATCH_SIZE):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(batch[0][idx])
    ax2.imshow(batch[1][idx])
    ax3.imshow(fake_image[idx])
    plt.show()

