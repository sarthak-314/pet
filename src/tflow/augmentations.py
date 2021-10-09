"""
augmentations:
    # Basic Augmentations
    flip_prob: 0.25
    random_rotate_range: 30
    shear_prob: 0.25
    pixel: # Pixel level transforms - 1/5 chance for each
        random_saturation: {'lower': 0, 'upper': 2}
        random_contrast: {'lower': 0.8, 'upper': 2}
        random_brightness: {'max_delta': 0.2}
        random_gamma: {'gamma': 0.6}
    random_rot90: True

    random_erasing:
        p: 0.5
        min_area: 0.01
        max_area: 0.4 # 0.4
        max_aspect_ratio: 0.1 # 0.3

    augmix:
        p: 0.25
        severity: 3
        width: 3
        depth: -1

    gridmask:
        p: 0.25
        ratio: 0.6 # grid vs spacing
        rotate: 30 # rotation range for grid
        gridmask_size_ratio: 0.5 # gridmask to image size ratio


"""
import tensorflow_addons as tfa
import tensorflow as tf
from .augmix import augmix
from .gridmask import GridMask

def occur(prob):
    'prob: chance that this returns True'
    return prob < tf.random.uniform([], 0, 1.0, dtype=tf.float32)

def rand():
    'Get random number between 0, 1'
    return tf.random.uniform([], 0, 1.0, dtype=tf.float32)

def radians(degree):
    pi_on_180 = 0.017453292519943295
    return degree * pi_on_180

@tf.function
def random_rotate(img, p=0.75, range=30):
    if not occur(p): return img
    degree = tf.random.uniform([], -range, range)
    return tfa.image.rotate(img, radians(degree), interpolation='BILINEAR')

@tf.function
def random_shear_x(img, p, lower, upper):
    if not occur(p): return img
    shearx = tf.random.uniform([], lower, upper)
    return tfa.image.shear_x(img, level=shearx, replace=0)

@tf.function
def random_shear_y(img, p, lower, upper):
    if not occur(p): return img
    sheary = tf.random.uniform([], lower, upper)
    return tfa.image.shear_y(img, level=sheary, replace=0)

def basic_augmentations(img, aug_hp):
    if occur(aug_hp.flip_prob):
        img = tf.image.random_flip_left_right(img)
    if occur(aug_hp.flip_prob):
        img = tf.image.random_flip_up_down(img)

    # Random 90 degree rotation
    if aug_hp.random_rot90:
        p_rot = rand()
        if p_rot < 0.25:  tf.image.rot90(img, k=1) # rotate 90º
        elif p_rot < 0.5:  tf.image.rot90(img, k=2) # rotate 180º
        elif p_rot < 0.75:  tf.image.rot90(img, k=3) # rotate 270º

    # Pixel Level Transformations
    p_pixel = rand()
    if p_pixel < 0.2:
        img = tf.image.random_brightness(img, **aug_hp.pixel.random_brightness)
    elif p_pixel < 0.4:
        img = tf.image.random_saturation(img, **aug_hp.pixel.random_saturation)
    elif p_pixel < 0.6:
        img = tf.image.random_contrast(img, **aug_hp.pixel.random_contrast)
    elif p_pixel < 0.8:
        img = tf.image.adjust_gamma(img, gamma=aug_hp.pixel.random_gamma['gamma'])
    else:
        pass

    return img

def random_erase(img, p=0.5, min_area=0.02, max_area=0.4, max_aspect_ratio=0.3):
    if not occur(p): return img

    ht, wd, ch = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
    area = tf.cast(ht*wd, tf.float32)

    erase_area_low_bound = tf.cast(tf.round(tf.sqrt(min_area * area * max_aspect_ratio)), tf.int32)
    erase_area_up_bound = tf.cast(tf.round(tf.sqrt((max_area * area) / max_aspect_ratio)), tf.int32)
    h_upper_bound = tf.minimum(erase_area_up_bound, ht)
    w_upper_bound = tf.minimum(erase_area_up_bound, wd)
    h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
    w = tf.random.uniform([], erase_area_low_bound, w_upper_bound, tf.int32)

    x1 = tf.random.uniform([], 0, ht+1 - h, tf.int32)
    y1 = tf.random.uniform([], 0, wd+1 - w, tf.int32)
    erased_area = tf.cast(tf.random.uniform([h, w, ch], 0, 255, tf.int32), tf.uint8)
    erased_img = img[x1:x1+h, y1:y1+w, :].assign(erased_area)
    return erased_img

def augmentations_factory(aug_hp, img_size):
    # TODO: Random Scale, Random Rotate, Random Blur
    def unbatched_augmentations(img, label):
        img = basic_augmentations(img, aug_hp)
        img = random_shear_x(img, aug_hp.shear_prob, lower=0, upper=1)
        img = random_shear_y(img, aug_hp.shear_prob, lower=0, upper=1)
        img = random_rotate(img, p=0.75, range=aug_hp.random_rotate_range)
        # img = random_erase(img, **aug_hp.random_erasing)

        # TODO: Make it better
        if occur(aug_hp.augmix.p):
            img = augmix(img, img_size, aug_hp.augmix.severity, aug_hp.augmix.width, aug_hp.augmix.depth)
        if occur(aug_hp.gridmask.p):
            gridmask = GridMask((img_size, img_size), aug_hp.gridmask.ratio, aug_hp.gridmask.rotate, aug_hp.gridmask.gridmask_size_ratio, fill=0)
            img = gridmask(img, label)

        return img, label

    def batched_augmentations(img, label):
        return img, label

    return unbatched_augmentations, batched_augmentations