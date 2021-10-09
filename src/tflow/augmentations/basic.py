import tensorflow as tf
import numpy as np

def zoom_out(x, img_size, scale_factor):
    img_dim = (img_size, img_size)
    resize_x = tf.random.uniform(shape=[], minval=tf.cast(img_dim[1]//(1/scale_factor), tf.int32), maxval=img_dim[1], dtype=tf.int32)
    resize_y = tf.random.uniform(shape=[], minval=tf.cast(img_dim[0]//(1/scale_factor), tf.int32), maxval=img_dim[0], dtype=tf.int32)
    top_pad = (img_dim[0] - resize_y) // 2
    bottom_pad = img_dim[0] - resize_y - top_pad
    left_pad = (img_dim[1] - resize_x ) // 2
    right_pad = img_dim[1] - resize_x - left_pad

    x = tf.image.resize(x, (resize_y, resize_x))
    x = tf.pad([x], [[0,0], [top_pad, bottom_pad], [left_pad, right_pad], [0,0]])
    x = tf.image.resize(x, img_dim)
    return tf.squeeze(x, axis=0)

def zoom_in(x, img_size, scale_factor):
    img_dim = (img_size, img_size)
    scales = list(np.arange(scale_factor, 1.0, 0.05))
    boxes = np.zeros((len(scales),4))
    for i, scale in enumerate(scales):
        x_min = y_min = 0.5 - (0.5*scale)
        x_max = y_max = 0.5 + (0.5*scale)
        boxes[i] = [x_min, y_min, x_max, y_max]

    def random_crop(x):
        crop = tf.image.crop_and_resize([x], boxes=boxes, box_indices=np.zeros(len(boxes)), crop_size=img_dim)
        return crop[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    return random_crop(x)

# SCALE
def get_random_scale(img_size, aug_params_scale):
    def random_scale(img, label):
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < aug_params_scale['prob']:
            if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.75:
                img = zoom_in(img, img_size, aug_params_scale['zoom_in'])
            else:
                img = zoom_out(img, img_size, aug_params_scale['zoom_out'])
        return img, label
    return random_scale


################################################################################################

def image_rotate(image, img_size, angle):
    if len(image.get_shape().as_list()) != 3:
        raise ValueError('`image_rotate` only support image with 3 dimension(h, w, c)`')
    angle = tf.cast(angle, tf.float32)
    img_dim = (img_size, img_size)
    h, w, c = img_dim[0], img_dim[1], 3
    cy, cx = h//2, w//2
    ys = tf.range(h)
    xs = tf.range(w)
    ys_vec = tf.tile(ys, [w])
    xs_vec = tf.reshape( tf.tile(xs, [h]), [h,w] )
    xs_vec = tf.reshape( tf.transpose(xs_vec, [1,0]), [-1])
    ys_vec_centered, xs_vec_centered = ys_vec - cy, xs_vec - cx
    new_coord_centered = tf.cast(tf.stack([ys_vec_centered, xs_vec_centered]), tf.float32)
    inv_rot_mat = tf.reshape( tf.dynamic_stitch([0,1,2,3], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)]), [2,2])
    old_coord_centered = tf.matmul(inv_rot_mat, new_coord_centered)
    old_ys_vec_centered, old_xs_vec_centered = old_coord_centered[0,:], old_coord_centered[1,:]
    old_ys_vec = tf.cast( tf.round(old_ys_vec_centered+cy), tf.int32)
    old_xs_vec = tf.cast( tf.round(old_xs_vec_centered+cx), tf.int32)
    outside_ind = tf.logical_or( tf.logical_or(old_ys_vec > h-1 , old_ys_vec < 0), tf.logical_or(old_xs_vec > w-1 , old_xs_vec<0))
    old_ys_vec = tf.boolean_mask(old_ys_vec, tf.logical_not(outside_ind))
    old_xs_vec = tf.boolean_mask(old_xs_vec, tf.logical_not(outside_ind))
    ys_vec = tf.boolean_mask(ys_vec, tf.logical_not(outside_ind))
    xs_vec = tf.boolean_mask(xs_vec, tf.logical_not(outside_ind))
    old_coord = tf.cast(tf.transpose(tf.stack([old_ys_vec, old_xs_vec]), [1,0]), tf.int32)
    new_coord = tf.cast(tf.transpose(tf.stack([ys_vec, xs_vec]), [1,0]), tf.int64)
    channel_vals = tf.split(image, c, axis=-1)
    rotated_channel_vals = list()
    for channel_val in channel_vals:
        rotated_channel_val = tf.gather_nd(channel_val, old_coord)
        sparse_rotated_channel_val = tf.SparseTensor(new_coord, tf.squeeze(rotated_channel_val,axis=-1), [h, w])
        rotated_channel_vals.append(tf.sparse.to_dense(sparse_rotated_channel_val, default_value=0, validate_indices=False))
    rotated_image = tf.transpose(tf.stack(rotated_channel_vals), [1, 2, 0])
    return rotated_image

def random_blockout(img, img_size, sl=0.1, sh=0.2, rl=0.4):
    img_dim = (img_size, img_size)
    h, w, c = img_dim[0], img_dim[1], 3
    origin_area = tf.cast(h*w, tf.float32)

    e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
    e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

    e_height_h = tf.minimum(e_size_h, h)
    e_width_h = tf.minimum(e_size_h, w)

    erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)
    erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)

    erase_area = tf.zeros(shape=[erase_height, erase_width, c])
    erase_area = tf.cast(erase_area, tf.uint8)

    pad_h = h - erase_height
    pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
    pad_bottom = pad_h - pad_top

    pad_w = w - erase_width
    pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
    pad_right = pad_w - pad_left

    erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
    erase_mask = tf.squeeze(erase_mask, axis=0)
    erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))

    return tf.cast(erased_img, img.dtype)


def gaussian_blur(img, ksize=5, sigma=1):
    def gaussian_kernel(size=3, sigma=1):
        x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
        y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)
        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
        return tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)
    kernel = gaussian_kernel(ksize, sigma)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    r, g, b = tf.split(img, [1,1,1], axis=-1)
    r_blur = tf.nn.conv2d([r], kernel, [1,1,1,1], 'SAME')
    g_blur = tf.nn.conv2d([g], kernel, [1,1,1,1], 'SAME')
    b_blur = tf.nn.conv2d([b], kernel, [1,1,1,1], 'SAME')
    blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
    return tf.squeeze(blur_image, axis=0)


def get_random_rotate(img_size, rot_prob, rot_range=90):
    def random_rotate(img, label):
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < rot_prob:
            angle = tf.random.uniform(shape=[], minval=-rot_range, maxval=rot_range, dtype=tf.int32)
            img = image_rotate(img, img_size, angle)
        return img, label
    return random_rotate

def get_random_cutout(img_size, aug_params_cutout):
    def random_cutout(img, label):
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < aug_params_cutout['prob']:
            sl, sh, rl = aug_params_cutout['sl'], aug_params_cutout['sh'], aug_params_cutout['rl']
            img = random_blockout(img, img_size, sl, sh, rl)
        return img, label
    return random_cutout

def get_random_blur(aug_params_blur):
    def gaussian_blur_fn(img, label):
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < aug_params_blur['prob']:
            img = gaussian_blur(img, aug_params_blur['ksize'], sigma=1)
        return img, label
    return gaussian_blur_fn


def basic_augmentations(img):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    if p_spatial >= .25:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    # Rotates
    if p_rotate > .75: img = tf.image.rot90(img, k=3) # rotate 270º
    elif p_rotate > .5: img = tf.image.rot90(img, k=2) # rotate 180º
    elif p_rotate > .25: img = tf.image.rot90(img, k=1) # rotate 90º
    # Pixel-level transforms
    if p_pixel >= .2:
        if p_pixel >= .8:
            img = tf.image.random_saturation(img, lower=0, upper=2)
        elif p_pixel >= .6:
            img = tf.image.random_contrast(img, lower=.8, upper=2)
        elif p_pixel >= .4:
            img = tf.image.random_brightness(img, max_delta=.4) # Changed from 0.2
        else:
            img = tf.image.adjust_gamma(img, gamma=.6)
    return img

def get_resize_fn(img_size):
    def resize_fn(img, label):
        img =  tf.image.resize(img, size=[img_size, img_size])
        return img, label
    return resize_fn


def get_basic_augs(img_size):
    def basic_augs_fn(img, label):
        img = basic_augmentations(img)
        return img, label
    return basic_augs_fn

def get_basic_augmentations():
    return get_basic_augs, get_random_scale, get_random_rotate, get_random_cutout, get_random_blur, get_resize_fn



"""
get_basic_augs, get_random_scale, get_random_rotate, get_random_cutout, get_random_blur, get_resize_fn = get_basic_augmentations()

AUG_PARAMS = {
    'scale': {
        'zoom_in': 0.1,
        's
    }
}
basic_augs = get_basic_augs()
random_scale = get_random_scale(IMG_SIZE, AUG_PARAMS['scale'])
random_rotate = get_random_rotate(IMG_SIZE, AUG_PARAMS['rot_prob'])
random_cutout = get_random_cutout(IMG_SIZE, AUG_PARAMS['cutout'])
random_blur = get_random_blur(AUG_PARAMS['blur'])
resize = get_resize_fn(IMG_SIZE)

cutmix, mixup = get_cutmix_mixup(IMG_SIZE, classes, cutmix_prob=AUG_PARAMS['cutmix_prob'], mixup_prob=AUG_PARAMS['mixup_prob'])
gridmask = get_gridmask()
"""


