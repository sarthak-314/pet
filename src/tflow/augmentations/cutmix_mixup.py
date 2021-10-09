import tensorflow as tf

def cutmix(image, label, batch_size, img_size, classes=4, prob = 1.0):
    label = tf.cast(label, tf.float32)
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    imgs = []; labs = []
    for j in range(batch_size):
        # DO CUTMIX WITH prob DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<= prob, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast( tf.random.uniform([],0,batch_size),tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,img_size),tf.int32)
        y = tf.cast( tf.random.uniform([],0,img_size),tf.int32)
        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
        WIDTH = tf.cast( img_size * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(img_size,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(img_size,x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:img_size,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:img_size,:,:]],axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/img_size/img_size,tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],classes)
            lab2 = tf.one_hot(label[k],classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(batch_size,img_size,img_size,3))
    label2 = tf.reshape(tf.stack(labs),(batch_size,classes))
    return image2,label2


def mixup(image, label, batch_size, img_size, classes, prob = 1.0):
    label = tf.cast(label, tf.float32)
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    imgs = []; labs = []
    for j in range(batch_size):
        # DO MIXUP WITH prob DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=prob, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,batch_size),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        # MAKE CUTMIX LABEL
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],classes)
            lab2 = tf.one_hot(label[k],classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(batch_size,img_size,img_size,3))
    label2 = tf.reshape(tf.stack(labs),(batch_size,classes))
    return image2,label2

def get_cutmix_mixup(img_size, classes, batch_size, cutmix_prob=0.5, mixup_prob=0.5):
    def cutmix_fn(img, label): 
        return cutmix(img, label, batch_size, img_size, classes, prob=cutmix_prob)
    def mixup_fn(img, label): 
        return mixup(img, label, batch_size, img_size, classes, prob=mixup_prob)
    return cutmix_fn, mixup_fn