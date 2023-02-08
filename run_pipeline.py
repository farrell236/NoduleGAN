import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from stylegan import StyleGAN
from gaugan import GauGAN


res = 512
NUM_CLASSES = 6
LATENT_DIM = 512
BATCH_SIZE = 16
NUM_EPOCHS = 100


##### Load Models

style_gan = StyleGAN(start_res=4, target_res=512)
style_gan.grow_model(res)
style_gan.load_weights(os.path.join('checkpoints/stylegan/stylegan_512x512.ckpt')).expect_partial()

gaugan = GauGAN(res, 6, BATCH_SIZE, LATENT_DIM)
gaugan.load_weights('checkpoints/gaugan/gaugan_512x512.ckpt')


##### Run StyleGAN

z = tf.random.normal((BATCH_SIZE, style_gan.z_dim))
w = style_gan.mapping([z])
noise = style_gan.generate_noise(batch_size=BATCH_SIZE)
labels_ohe = style_gan.call({"style_code": w, "noise": noise, "alpha": 1.0})


##### Process Labels

labels_ohe = tf.keras.backend.softmax(labels_ohe)
labels_ohe = tf.cast(labels_ohe > 0.5, dtype=tf.int32)
labels_ohe = tf.concat([labels_ohe, tf.zeros((BATCH_SIZE, res, res, 1), dtype=tf.int32)], axis=-1)


##### Run GauGAN Labels

latent_vector = tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM), mean=0.0, stddev=2.0)
fake_image = gaugan.predict([latent_vector, labels_ohe])


##### Format and display images

labels = tf.argmax(labels_ohe, axis=-1)

label_to_color = {
    0: [  0,   0,   0],
    1: [242,  80,  34],
    2: [127, 186,   0],
    3: [  0, 164, 239],
    4: [255, 185,   0],
    5: [115, 115, 115],
}

labels_rgb = np.zeros((BATCH_SIZE, res, res, 3), dtype=np.uint8)
for gray, rgb in label_to_color.items():
    labels_rgb[labels == gray, :] = rgb

for idx in range(BATCH_SIZE):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    ax1.imshow(labels_rgb[idx])
    ax2.imshow(fake_image[idx])
    plt.show()
