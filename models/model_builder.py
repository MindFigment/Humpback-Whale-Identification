from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras import backend as K

def build_model(lr, l2, activation='sigmoid', img_shape=(384, 384, 1)):

    regularizer = regularizers.l2(l2)
    optimizer = Adam(lr=lr)
    kwargs = {
        'padding': 'same',
        'kernel_regularizer': regularizer
    }

    ####################
    ### BRANCH MODEL ###
    ####################

    inp = Input(shape=img_shape) # (384, 384, 1)
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # (96, 96, 64)

    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # (48 x 48 x 64)
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x) # (48, 48, 128)

    for _ in range(6): # 4
        x = sub_block(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # (24, 24, 128)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x) # (24, 24, 256)

    for _ in range(6): # 4
        x = sub_block(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # (12, 12, 256)
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x) # (12, 12, 512) # 384

    for _ in range(6): # 4
        x = sub_block(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # (6, 6, 512) # 384
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), activation='relu', **kwargs)(x) # (6, 6, 1024) # 512

    for _ in range(6): # 4
        x = sub_block(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x) # (1024) # 512

    branch_model = Model(inp, x)


    ##################
    ### HEAD MODEL ###
    ##################

    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])

    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)

    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)

    head_model = Model([xa_inp, xb_inp], x, name='head')

    ##############################
    ### SIAMESE NEURAL NETWORK ###
    ##############################

    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)

    model.compile(optimizer, loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    return model, branch_model, head_model

########################
### HELPER FUNCTIONS ###
########################

def sub_block(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)
    y = Add()([x, y])
    y = Activation('relu')(y)
    return y

def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 0.9
    similarity = K.square(y_pred) * y_true
    dissimilarity = K.square(K.maximum(margin - y_pred, 0)) * (1 - y_true)
    loss = K.mean(similarity + dissimilarity) / 2
    return loss

def euclidian_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepDims=True)
    distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return distance

def contrastive_acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)
