import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from stylegan import StyleGAN

a=1

BATCH_SIZE = 4

res = 512

style_gan = StyleGAN(start_res=4, target_res=512)
style_gan.grow_model(res)
style_gan.load_weights(os.path.join(f'checkpoints/stylegan/stylegan_{res}x{res}.ckpt')).expect_partial()

z = tf.random.normal((BATCH_SIZE, style_gan.z_dim))
w = style_gan.mapping([z])
noise = style_gan.generate_noise(batch_size=BATCH_SIZE)
labels = style_gan.call({"style_code": w, "noise": noise, "alpha": 1.0})

a=1

labels = tf.keras.backend.softmax(labels)
labels = tf.cast(labels > 0.5, dtype=tf.float32)
labels = tf.argmax(labels, axis=-1)

label_to_color = {
    0: [  0,   0,   0],
    1: [242,  80,  34],
    2: [127, 186,   0],
    3: [  0, 164, 239],
    4: [255, 185,   0],
    5: [115, 115, 115],
}

label_rgb = np.zeros((BATCH_SIZE, res, res, 3), dtype=np.uint8)
for gray, rgb in label_to_color.items():
    label_rgb[labels == gray, :] = rgb


for i in range(BATCH_SIZE):
    plt.imshow(label_rgb[i])
    plt.show()


