import tensorflow_probability as tfp
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import math

def int_parameter(level, maxval):
    return tf.cast(level * maxval / 10, tf.int32)

def float_parameter(level, maxval):
    return tf.cast((level) * maxval / 10., tf.float32)

def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)
    
def affine_transform_(image, img_size, transform_matrix):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    DIM = img_size
    XDIM = DIM%2 #fix for size 331
    
    x = tf.repeat(tf.range(DIM//2,-DIM//2,-1), DIM)
    y = tf.tile(tf.range(-DIM//2,DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(transform_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM,DIM,3])

def blend(image1, image2, factor):
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

def rotate(affine_transform, image, level):
    degrees = float_parameter(sample_level(level), 30)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    degrees = tf.cond(rand_var > 0.5, lambda: degrees, lambda: -degrees)

    angle = math.pi*degrees/180 # convert degrees to radians
    angle = tf.cast(angle, tf.float32)
    # define rotation matrix
    c1 = tf.math.cos(angle)
    s1 = tf.math.sin(angle)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(image, rotation_matrix)
    return transformed

def translate_x(affine_transform, image, img_size, level):
    lvl = int_parameter(sample_level(level), img_size / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_x_matrix = tf.reshape(tf.concat([one,zero,zero, zero,one,lvl, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(image, translate_x_matrix)
    return transformed

def translate_y(affine_transform, image, img_size, level):
    lvl = int_parameter(sample_level(level), img_size / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_y_matrix = tf.reshape(tf.concat([one,zero,lvl, zero,one,zero, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(image, translate_y_matrix)
    return transformed

def shear_x(affine_transform, image, level):
    lvl = float_parameter(sample_level(level), 0.3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    s2 = tf.math.sin(lvl)
    shear_x_matrix = tf.reshape(tf.concat([one,s2,zero, zero,one,zero, zero,zero,one],axis=0), [3,3])   

    transformed = affine_transform(image, shear_x_matrix)
    return transformed

def shear_y(affine_transform, image, level):
    lvl = float_parameter(sample_level(level), 0.3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    c2 = tf.math.cos(lvl)
    shear_y_matrix = tf.reshape(tf.concat([one,zero,zero, zero,c2,zero, zero,zero,one],axis=0), [3,3])   
    
    transformed = affine_transform(image, shear_y_matrix)
    return transformed

def solarize(affine_transform, image, level):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 1 - image)

def solarize_add(affine_transform, image, level):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    threshold = float_parameter(sample_level(level), 1)
    addition = float_parameter(sample_level(level), 0.5)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    addition = tf.cond(rand_var > 0.5, lambda: addition, lambda: -addition)

    added_image = tf.cast(image, tf.float32) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 1), tf.float32)
    return tf.where(image < threshold, added_image, image)

def posterize(affine_transform, image, level):
    lvl = int_parameter(sample_level(level), 8)
    shift = 8 - lvl
    shift = tf.cast(shift, tf.uint8)
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)
    image = tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    return tf.cast(tf.clip_by_value(tf.math.divide(image, 255), 0, 1), tf.float32)

def autocontrast(affine_transform, image, _):
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)

    def scale_channel(image):
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return tf.cast(tf.clip_by_value(tf.math.divide(image, 255), 0, 1), tf.float32)

def equalize(affine_transform, image, _):
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)

    def scale_channel(im, c):
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                        lambda: im,
                        lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)

    return tf.cast(tf.clip_by_value(tf.math.divide(image, 255), 0, 1), tf.float32)

def color(affine_transform, image, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    image = tf.cast(tf.math.scalar_mul(255, image), tf.uint8)
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    blended = blend(degenerate, image, factor)
    return tf.cast(tf.clip_by_value(tf.math.divide(blended, 255), 0, 1), tf.float32)

def brightness(affine_transform, image, level):
    delta = float_parameter(sample_level(level), 0.5) + 0.1
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    delta = tf.cond(rand_var > 0.5, lambda: delta, lambda: -delta) 
    return tf.image.adjust_brightness(image, delta=delta)

def contrast(affine_transform, image, level):
    factor = float_parameter(sample_level(level), 1.8) + 0.1
    factor = tf.reshape(factor, [])
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    factor = tf.cond(rand_var > 0.5, lambda: factor, lambda: 1.9 - factor  )

    return tf.image.adjust_contrast(image, factor)
means = {'R': 0.44892993872313053, 'G': 0.4148519066242368, 'B': 0.301880284715257}
stds = {'R': 0.24393544875614917, 'G': 0.2108791383467354, 'B': 0.220427056859487}

def substract_means(image):
    image = image - np.array([means['R'], means['G'], means['B']])
    return image

def normalize(image):
    image = substract_means(image)
    image = image / np.array([stds['R'], stds['G'], stds['B']])
    return tf.clip_by_value(image, 0, 1)

def apply_op(image, level, which, img_size):
    def affine_transform(image, transform_matrix): 
        return affine_transform_(image, img_size, transform_matrix)
    augmented = image
    # augmented = tf.cond(which == tf.constant([0], dtype=tf.int32), lambda: rotate(affine_transform, image, level), lambda: augmented)
    # augmented = tf.cond(which == tf.constant([1], dtype=tf.int32), lambda: translate_x(affine_transform, image, img_size, level), lambda: augmented)
    # augmented = tf.cond(which == tf.constant([2], dtype=tf.int32), lambda: translate_y(affine_transform, image, img_size, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([3], dtype=tf.int32), lambda: shear_x(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([4], dtype=tf.int32), lambda: shear_y(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([5], dtype=tf.int32), lambda: solarize_add(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([6], dtype=tf.int32), lambda: solarize(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([7], dtype=tf.int32), lambda: posterize(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([8], dtype=tf.int32), lambda: autocontrast(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([9], dtype=tf.int32), lambda: equalize(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([10], dtype=tf.int32), lambda: color(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([11], dtype=tf.int32), lambda: contrast(affine_transform, image, level), lambda: augmented)
    augmented = tf.cond(which == tf.constant([12], dtype=tf.int32), lambda: brightness(affine_transform, image, level), lambda: augmented)
    return augmented

def augmix(image, img_size, severity, width, depth):
    alpha = 1.
    dir_dist = tfp.distributions.Dirichlet([alpha]*width)
    ws = tf.cast(dir_dist.sample(), tf.float32)
    beta_dist = tfp.distributions.Beta(alpha, alpha)
    m = tf.cast(beta_dist.sample(), tf.float32)

    mix = tf.zeros_like(image, dtype='float32')

    def outer_loop_cond(i, depth, mix):
        return tf.less(i, width)

    def outer_loop_body(i, depth, mix):
        image_aug = tf.identity(image)
        depth = tf.cond(tf.greater(depth, 0), lambda: depth, lambda: tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32))

        def inner_loop_cond(j, image_aug):
            return tf.less(j, depth)

        def inner_loop_body(j, image_aug):
            which = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
            image_aug = apply_op(image_aug, severity, which, img_size)
            j = tf.add(j, 1)
            return j, image_aug
        
        j = tf.constant([0], dtype=tf.int32)
        j, image_aug = tf.while_loop(inner_loop_cond, inner_loop_body, [j, image_aug])

        wsi = tf.gather(ws, i)
        mix = tf.add(mix, wsi*normalize(image_aug))
        i = tf.add(i, 1)
        return i, depth, mix

    i = tf.constant([0], dtype=tf.int32)
    i, depth, mix = tf.while_loop(outer_loop_cond, outer_loop_body, [i, depth, mix])
    
    mixed = tf.math.scalar_mul((1 - m), normalize(image)) + tf.math.scalar_mul(m, mix)
    return tf.clip_by_value(mixed, 0, 1)



        