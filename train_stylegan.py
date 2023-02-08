import math
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from functools import partial

from stylegan import StyleGAN, log2, batch_sizes, train_step_ratio


# root_dir = '/mnt/nas_houbb/users/Benjamin/LUNA16/'
root_dir = './data/LUNA16'

def load(res, label_file):

    # Load label from disk
    label = tf.io.read_file(label_file)
    label = tf.image.decode_png(label, 3)

    # Resize to [res x res]
    label = tf.image.resize(label, [res, res], method='nearest')

    # Random rotate (augmentation)
    # degree = tf.random.normal([]) * 360
    # label = tfa.image.rotate(label, degree * math.pi / 180., interpolation='nearest')
    # label = tf.image.random_flip_up_down(label)
    # label = tf.image.random_flip_left_right(label)

    # Make inputs one-hot
    label = tf.one_hot(label[..., 0], depth=5)

    return label

a=1

def create_dataloader(res):

    batch_size = batch_sizes[log2(res)]

    # 109198: 'images/1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031_99.png'
    # 109199: 'images/1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273_100.png'
    labels = sorted(glob(os.path.join(root_dir, f'labels_ohe/*.png')))[:109199]

    dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.repeat()
    dataset = dataset.map(partial(load, res), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


a=1


start_res = 4
target_res = 512
steps_per_epoch = 10000

start_res_log2 = int(np.log2(start_res))
target_res_log2 = int(np.log2(target_res))

style_gan = StyleGAN(start_res=start_res, target_res=target_res)
opt_cfg = {"learning_rate": 2e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

for res_log2 in range(start_res_log2, target_res_log2 + 1):
    res = 2 ** res_log2
    for phase in ["TRANSITION", "STABLE"]:
        if res == start_res and phase == "TRANSITION":
            continue

        train_dl = create_dataloader(res)

        steps = int(train_step_ratio[res_log2] * steps_per_epoch)
        style_gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
            g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
            loss_weights={"gradient_penalty": 10, "drift": 0.001},
            steps_per_epoch=steps,
            res=res,
            phase=phase,
            run_eagerly=False,
        )

        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            f"/tmp/checkpoints/stylegan/stylegan_{res}x{res}.ckpt",
            save_weights_only=True,
            verbose=0,
        )
        print(phase)
        style_gan.fit(
            train_dl, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
        )

