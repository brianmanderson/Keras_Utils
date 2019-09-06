from keras.utils import Sequence, np_utils
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.models import load_model, Model
import keras.backend as K
from keras.layers.core import Lambda
import tensorflow as tf
from keras.backend import resize_images, categorical_crossentropy
from keras.layers import Input, Dropout, SpatialDropout2D, ConvLSTM2D, TimeDistributed, UpSampling2D, Concatenate, \
    SpatialDropout3D, BatchNormalization, Activation, Add, Conv3D, Flatten, UpSampling3D, \
    MaxPooling3D, ZeroPadding3D, Conv2D, Multiply, MaxPooling2D, Reshape, AveragePooling2D
from tensorflow.compat.v1 import Graph, Session, ConfigProto, GPUOptions
from TensorflowUtils import load_obj, save_obj, np, remove_non_liver, get_metrics, turn_pickle_into_text, \
    normalize_images, plot_scroll_Image, visualize, plt
from skimage.measure import block_reduce
import math, warnings, cv2, os, copy, time, glob
from scipy.ndimage import interpolation
import scipy.ndimage.filters as filts
from skimage.morphology import label
# from Predict_On_VGG_Unet_Module_Class import Prediction_Model_Class
from v3_model import Deeplabv3, BilinearUpsampling
from tensorflow.python.client import device_lib


def fill_in_overlapping_missing_pixels(liver, mask, slice_thickness, pixel_spacing_x, pixel_spacing_y):
    zeros = np.expand_dims(np.zeros(liver.shape), axis=-1)
    output = np.argmax(np.concatenate([zeros, mask[..., 1:]], axis=-1), axis=-1)
    summed_image = np.sum(mask[..., 1:], axis=-1).astype('int')
    liver[summed_image>0] = 1
    overlap_locations = np.where(summed_image > 1)
    output[overlap_locations] = 0  # Remove overlapping sections
    removed_overlap = np_utils.to_categorical(output, mask.shape[-1])
    kernel = np.ones([3, 3, 3]) / 9
    kernel[0, :, :] = 0
    kernel[2, :, :] = 0
    only_edges = np.zeros(removed_overlap.shape)
    for i in range(1, removed_overlap.shape[-1]):
        only_edges[..., i] = filts.convolve(removed_overlap[..., i], kernel) * removed_overlap[..., i]
    data_points = np.where((liver == 1) & ((summed_image == 0) | (summed_image > 1)))
    points_to_fill = np.concatenate([np.expand_dims(data_points[0], axis=-1),
                                     np.expand_dims(data_points[1], axis=-1),
                                     np.expand_dims(data_points[2], axis=-1)], axis=-1)
    points_to_fill = points_to_fill.astype('int')
    space_info = np.asarray([slice_thickness, pixel_spacing_x, pixel_spacing_y])
    output_data = np.zeros([points_to_fill.shape[0], mask.shape[-1] - 1])
    for i in range(1, only_edges.shape[-1]):
        print(i)
        points_in_mask = np.where((only_edges[..., i] > 0) & (only_edges[..., i] < 1))
        points_in_mask = np.concatenate([np.expand_dims(points_in_mask[0], axis=-1),
                                         np.expand_dims(points_in_mask[1], axis=-1),
                                         np.expand_dims(points_in_mask[2], axis=-1)], axis=-1)
        points_in_mask = points_in_mask.astype('int')
        dif = (points_to_fill[:, None] - points_in_mask).astype('float16')
        difference = np.multiply(dif, space_info).astype('float32')
        difference = np.sqrt(np.sum((difference) ** 2, axis=-1).astype('float32')).astype('float32')
        difference = np.min(difference, axis=-1).astype('float32')
        output_data[:, i - 1] = difference

    values = np.argmin(output_data, axis=1) + 1
    output[data_points] = values  # For each point, what mask does it belong to? Takes the earliest index [7,5,5] will be 1
    output = np_utils.to_categorical(output, removed_overlap.shape[-1])
    output[..., 0] = liver
    return output

def freeze_until_name(model,name):
    set_trainable = False
    for layer in model.layers:
        if layer.name == name:
            set_trainable = True
        layer.trainable = set_trainable
    return model

def freeze_names(model,desc):
    for layer in model.layers:
        if layer.name.find(desc) == -1:
            layer.trainable = False
        else:
            layer.trainable = True
    return model

class Pyramid_Pool_3D(object):

    def __init__(self, start_block=32, channels=1, filter_vals=None,num_of_classes=2,num_layers=2, resolutions=[64,128,128]):
        self.start_block = start_block
        self.resolutions = resolutions
        if filter_vals is None:
            filter_vals = [5,5,5]
        self.pool_size = (2,2,2)
        self.filter_vals = filter_vals
        self.channels = channels
        self.num_of_classes = num_of_classes
        self.num_layers = num_layers
        self.activation = 'relu'
        self.unet()

    def residual_block(self,output_size,x, drop=0.0, short_cut=False, prefix=''):
        inputs = x
        x = Conv3D(output_size, self.filters, activation=None, padding='same',
                   name=self.image_resolution + '_conv' + str(self.layer) + '_0_UNet' + prefix)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv3D(output_size, self.filters, activation=None, padding='same',
                   name=self.image_resolution + '_conv' + str(self.layer) + '_1_UNet' + prefix)(x)
        x = BatchNormalization()(x)
        x = SpatialDropout3D(drop)(x)

        x = Conv3D(output_size, kernel_size=(1, 1, 1), strides=(1,1,1), padding='same',
                           name=self.image_resolution + '_conv' + str(self.layer) + '_2_UNet' + prefix)(x)
        x = BatchNormalization()(x)
        if short_cut:
            x = Add(name=self.image_resolution + '_skip' + str(self.layer) + '_UNet' + prefix)([x,inputs])
        x = Activation(self.activation)(x)

        return x

    def unet(self):
        high_image = Input([self.resolutions[0], self.resolutions[1], self.resolutions[2], self.channels])
        medium_image = Input([self.resolutions[0], self.resolutions[1], self.resolutions[2], self.channels])
        low_image = Input([self.resolutions[0], self.resolutions[1], self.resolutions[2], self.channels])
        self.data_dict = {'high':{'Input':high_image},'medium':{'Input':medium_image},'low':{'Input':low_image}}
        self.filters = (self.filter_vals[0], self.filter_vals[1], self.filter_vals[2])
        self.layer = 1

        '''
        First, do the low resolution block
        '''
        self.image_resolution = 'low'
        x = self.data_dict[self.image_resolution]['Input']
        self.res_block(x)
        '''
        Then, concatenate results and do medium block
        '''
        self.image_resolution = 'medium'
        x = Concatenate()([self.data_dict['medium']['Input'],self.data_dict['low']['last_layer']])
        self.res_block(x)
        '''
        Lastly, concatenate results and do high block
        '''
        self.image_resolution = 'high'
        x = Concatenate()([self.data_dict['high']['Input'], self.data_dict['medium']['last_layer']])
        self.res_block(x)
        self.model = Model(inputs=[self.data_dict['high']['Input'], self.data_dict['medium']['Input'],
                                   self.data_dict['low']['Input']],
                           outputs=[self.data_dict['high']['last_layer'],self.data_dict['medium']['last_layer'],
                                    self.data_dict['low']['last_layer']],
                           name='Multi_Scale_BMA')

    def res_block(self, x):
        drop_out = 0.0
        for self.layer in range(self.num_layers):
            short_cut = False
            self.start_block = int(self.start_block * 2)
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Down_0')
            drop_out = 0.2
            short_cut = True
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Down_1')
            self.data_dict[self.image_resolution][str(self.layer)] = x
            x = MaxPooling3D()(x)

        for self.layer in range(self.num_layers-1,-1,-1):
            short_cut = False
            x = UpSampling3D()(x)
            x = Concatenate()([x,self.data_dict[self.image_resolution][str(self.layer)]])
            self.start_block = int(self.start_block / 2)
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Up_0')
            drop_out = 0.2
            short_cut = True
            x = self.residual_block(int(self.start_block), x, drop=drop_out, short_cut=short_cut, prefix='_Up_1')
        x = Conv3D(self.num_of_classes, (1, 1, 1), padding='same', activation='softmax', name=self.image_resolution + '_last_layer')(x)
        self.data_dict[self.image_resolution]['last_layer'] = x


class Attrous_3D(object):

    def __init__(self, image_size=25, batch_size=32, channels=3, filter_vals=None,num_of_classes=2,num_layers=2):
        if filter_vals is None:
            filter_vals = [5,5,5]
        self.pool_size = (2,2,2)
        self.filter_vals = filter_vals
        self.channels = channels
        self.image_size = image_size
        self.num_of_classes = num_of_classes
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.activation = 'relu'
        self.unet()

    def atrous_conv_block(self, prefix,x,rate,drop=0.0, kernel_size=3, stride=1, epsilon=1e-3):
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding3D((pad_beg, pad_end, pad_end))(x)
            depth_padding = 'valid'
        x = Conv3D((kernel_size, kernel_size, kernel_size), strides=(stride, stride, stride), dilation_rate=(rate, rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_3Dconv')(x)
        x = BatchNormalization(name=prefix + '_3Dconv_BN', epsilon=epsilon)(x)
        x = Activation('relu')(x)
        if drop > 0.0:
            x = SpatialDropout3D(drop)(x)
        return x

    def unet(self):
        x = input_image = Input([16, 32, 32, 1])
        self.filters = (self.filter_vals[0], self.filter_vals[1], self.filter_vals[2])
        self.layer = 1
        layer_vals = {}
        block = 30
        drop_out = 0.0
        for i in range(self.num_layers):
            block += 10
            rate_1 = self.atrous_conv_block('block_' + str(i), rate=1, drop=drop_out, x=x)
            rate_2 = self.atrous_conv_block('block_' + str(i), rate=2, drop=drop_out, x=x)
            x = Concatenate()([rate_1,rate_2])

            drop_out = 0.2
        x = self.residual_block(int(150), x_concat, drop=drop_out)
        x = Conv3D(self.num_of_classes, (1, 1, 1), padding='same', name='last_layer')(x)
        x = Activation('softmax')(x)
        self.model = Model(inputs=[fine_image,coarse_image], outputs=[x], name='DeepMedic')


class TDLSTM_Conv(object):
    def __init__(self, input_batch=16,input_image=256,input_channels=33, start_block=64, layers=2):
        self.input_image = input_image
        self.input_batch = input_batch
        self.input_channels = input_channels
        self.layer_start = start_block
        self.num_classes = 2
        self.layers = layers
        self.block = 0
        self.activation = 'elu'
        self.conv_number = 2
        self.unet_network()


    def conv_block(self, x):
        for i in range(self.conv_number):
            x = ConvLSTM2D(filters=int(self.layer_start), kernel_size=(3,3), padding='same', return_sequences=True,
                           activation=None, name='conv_block_' + str(self.desc) + str(self.i) + '_' + str(i))(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if self.drop_out_spatial > 0.0:
                x = SpatialDropout3D(self.drop_out_spatial)(x)
            if self.drop_out > 0.0:
                x = Dropout(self.drop_out)(x)
            self.i += 1
        return x

    def up_sample_out(self,x):
        pool_size = int(self.input_image/int(x.shape[3]))
        if pool_size != 1:
            x = TimeDistributed(UpSampling2D((pool_size, pool_size)))(x)
        filters = 64
        for i in range(2):
            x = ConvLSTM2D(filters=int(filters), kernel_size=(3,3), padding='same', return_sequences=True,
                           activation=None, name='up_sample_conv_block_' + str(self.desc) + str(self.i) + '_' + str(i))(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            if self.drop_out_spatial > 0.0:
                x = SpatialDropout3D(self.drop_out_spatial)(x)
            if self.drop_out > 0.0:
                x = Dropout(self.drop_out)(x)
            filters /= 2
        self.layer_up += 1
        self.i += 1
        return x
    def unet_network(self):
        input_image = Input([self.input_batch,self.input_image,self.input_image,self.input_channels], name='input')

        x = input_image
        net = {}
        output_net = {}
        self.drop_out = 0.0
        self.drop_out_spatial = 0.2
        self.desc = 'down_'
        self.i = 0
        for self.block in range(self.layers):
            x = self.conv_block(x)
            net['conv ' + str(self.block)] = x
            x = TimeDistributed(MaxPooling2D((2,2), (2,2)))(x)
            self.layer_start *= 2

        self.layer_up = 0
        x = self.conv_block(x)

        self.desc = 'up_'
        for self.block in range(self.layers-1,-1,-1):
            # output_net['up_conv' + str(self.layer_up)] = self.up_sample_out(x)
            self.layer_start /= 2
            x = TimeDistributed(UpSampling2D((2,2)))(x)
            x = Concatenate(name='concat' + str(self.block) + '_Unet')([x, net['conv ' + str(self.block)]])
            x = self.conv_block(x)

        # keys = list(output_net.keys())
        # for key in keys:
        #     x = Concatenate()([x,output_net[key]])

        output = ConvLSTM2D(filters=self.num_classes, kernel_size=(3, 3), padding='same', return_sequences=True,
                       activation='softmax', name='output')(x)
        output = Flatten()(output)

        self.created_model = Model(input_image, outputs=[output])
        return None
    def network(self):
        input_image = Input([self.input_batch,self.input_image,self.input_image,self.input_channels], name='input')

        x = input_image
        self.drop_out = 0.0
        self.drop_out_spatial = 0.2
        self.desc = 'down_'
        self.i = 0
        x = self.conv_block(x)

        output = ConvLSTM2D(filters=self.num_classes, kernel_size=(3, 3), padding='same', return_sequences=True,
                       activation='softmax', name='output')(x)
        output = Flatten()(output)

        self.created_model = Model(input_image, outputs=[output])
        return None

class VGG16_class(object):

    def __init__(self):
        self.trainable = False
        self.activation = 'relu'
        self.filters = 64
        self.block = 1
        self.drop_out = 0.0

    def conv_block(self, x):
        for layer in range(1,self.layers+1):
            x = Conv2D(self.filters, (3, 3), activation=None, padding='same', name='block' + str(self.block) + '_conv'+str(layer),
                       trainable=self.trainable)(x)
            x = Activation(self.activation)(x)
            x = BatchNormalization()(x)
            if self.drop_out > 0.0:
                x = SpatialDropout2D(0.2)(x)
        return x

    def VGG16(self,include_top=True, weights='imagenet',
              input_tensor=None, input_shape=None,
              pooling=None,
              classes=1000, only_vals=False, use_3d_unet=False, trainable=True):
        """Instantiates the VGG16 architecture.

        Optionally loads weights pre-trained
        on ImageNet. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.

        The model and the weights are compatible with both
        TensorFlow and Theano. The data format
        convention used by the model is the one
        specified in your Keras config file.

        # Arguments
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
            weights: one of `None` (random initialization),
                  'imagenet' (pre-training on ImageNet),
                  or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 input channels,
                and width and height should be no smaller than 48.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `imagenet` '
                             '(pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')
        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top,
                                          weights=weights)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        # Block 1
        net = {}
        layout = [2,2,3,3,3]
        x = img_input
        for self.block in range(1,6):
            if self.block > 2 and trainable:
                self.trainable = True # Freeze the first 2 layers
            if self.trainable:
                self.drop_out = 0.0
            self.layers = layout[self.block-1]
            x = self.conv_block(x)
            if self.filters < 512:
                self.filters *= 2
            # if use_3d_unet:
            #     net['conv_block' + str(self.block) + '_pool'] = x
            net['block' + str(self.block) + '_pool'] = x
            if self.block < 5:
                x = MaxPooling2D((2, 2), strides=(2, 2), name='block' + str(self.block) + '_pool')(x)

        if only_vals:
            return img_input, x, net
        if include_top:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='vgg16')

        # load weights
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        file_hash='64373286793e3c8b2b4e3219cbf3544b')
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash='6d6bbae143d832006294945121d1f1fc')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                layer_utils.convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first':
                if include_top:
                    maxpool = model.get_layer(name='block5_pool')
                    shape = maxpool.output_shape[1:]
                    dense = model.get_layer(name='fc1')
                    layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image data format convention '
                                  '(`image_data_format="channels_first"`). '
                                  'For best performance, set '
                                  '`image_data_format="channels_last"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
        elif weights is not None:
            model.load_weights(weights)

        return model


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def get_start_stop(pred, is_liver=True):
    preds = copy.deepcopy(np.argmax(pred,axis=-1))
    preds[preds>0.5] = 1
    preds[preds<1] = 0
    preds = preds.astype('int')
    if is_liver:
        labels = label(preds,neighbors=4)
        start = 0
        stop = labels.shape[0]
        area = []
        max_val = 0
        for i in range(1,labels.max()+1):
            new_area = labels[labels == i].shape[0]
            area.append(new_area)
            if new_area == max(area):
                max_val = i
        labels[labels != max_val] = 0
        labels[labels > 0] = 1
    else:
        labels = preds
    for i in range(labels.shape[0]):
        if np.any(labels[i,:,:].astype('int'),1).max():
            start = i
            break
    for i in range(labels.shape[0]-1,0,-1):
        if np.any(labels[i,:,:].astype('int'),1).max():
            stop = i
            break
    return start, stop


def exp_decay(fraction, initial_lrate=0.1):
    k = 0.5
    lrate = initial_lrate * np.exp(-k*fraction)
    # lrate = initial_lrate * 0.1 ** fraction
    return lrate


class half_decay(object):
    name = 'half_decay'

    def __init__(self, initial_learning_rate=0.001):
        self.initial_learning_rate = initial_learning_rate

    def decay(self, fraction):
        return self.initial_learning_rate * 0.5 ** fraction


class step_decay(object):
    name = 'step_decay'

    def __init__(self, initial_learning_rate=0.001):
        self.lr = initial_learning_rate

    def decay(self, epoch):
        if epoch % 1 == 0 and epoch != 0:
            self.lr /= 2
        return self.lr


def step_decay_old(epoch,lr):
    if epoch % 1 == 0 and epoch != 0:
        lr /= 2
    return lr


class New_Learning_Scheduler(LearningRateScheduler):

    def __init__(self, steps_per_epoch, epochs, decay_function, initial_lrate, epoch_i=0,verbose=0, fraction=1):
        epochs += 1
        super(LearningRateScheduler, self).__init__()
        self.lr = copy.deepcopy(initial_lrate)
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.lrs = []
        x = []
        for i in range(steps_per_epoch * epochs):
            x.append(i/steps_per_epoch)
            fraction_step = i / (steps_per_epoch * fraction)
            if decay_function.name != 'half_decay':
                self.lr = decay_function.decay(fraction_step)
            else:
                self.lr = decay_function.decay(fraction_step)
            # self.lrs.append(decay_function(i / (steps_per_epoch * fraction), initial_lrate)
            self.lrs.append(self.lr)
        self.verbose = verbose
    def on_batch_begin(self, step, logs=None):
        K.set_value(self.model.optimizer.lr, self.lrs[step + self.steps_per_epoch * self.epochs_done])
        if self.verbose > 0:
            print('\nStep %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (step + 1, self.lrs[step]))
    def on_epoch_begin(self, epoch, logs=None):
        self.epochs_done = epoch

        K.set_value(self.model.optimizer.lr, self.lrs[self.steps_per_epoch * epoch])
    def on_epoch_end(self, epoch, logs=None):
        pass


class Resize_Images_Keras():
    def __init__(self,num_channels=1,image_size=256):
        if tf.__version__ == '1.14.0':
            device = tf.compat.v1.device
        else:
            device = tf.device
        with device('/gpu:0'):
            self.graph1 = Graph()
            with self.graph1.as_default():
                gpu_options = GPUOptions(allow_growth=True)
                self.session1 = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with self.session1.as_default():
                    self.input_val = Input((image_size, image_size, num_channels))
                    self.out = resize_images(self.input_val, 2, 2, 'channels_last')
    def resize_images(self,images):
        with self.graph1.as_default():
            with self.session1.as_default():
                x = self.session1.run(self.out,feed_dict={self.input_val:images})
        return x


class Save_History_Class_2D(Callback):

    def __init__(self,path, test_patient_generator, image_size=512, num_classes=6,
                 pat_path='G:\\CNN\\data\\Data_Kelly\\Test\\', volume_threshold=9999999, epoch_count=1):
        super(Save_History_Class_2D, self).__init__()
        print(image_size)
        self.slice_thickness = 2.5
        self.epoch_count = epoch_count
        self.volume_threshold = volume_threshold
        self.pixel_spacing = 0.97
        self.path = path
        self.test_pat = test_patient_generator
        self.pat_path = pat_path
        if self.pat_path.find('Saved_Test_Set.pkl') == -1:
            self.pat_path = os.path.join(self.pat_path,'Saved_Test_Set.pkl')
        self.pat_set = load_obj(self.pat_path)
        print(self.pat_path)
        self.pat_info = [[3,0.68359],[5,0.78125], [2.5,0.78125], [2.5,0.8593], [2.5,0.97656], [5,0.78125],
                         [3,0.87], [2.5,0.74218], [2.5,0.820312]]
        self.image_size = image_size
        self.resize_images = Resize_Images_Keras(num_channels=num_classes)
        self.num_of_classes = num_classes
        self.save_pats = False

    def get_pat_info(self):
        if self.pat_set == {}:
            for i in range(len(self.test_pat)):
                self.pat_set[i] = self.test_pat.__getitem__(i)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_count == 0:
            # previous = load_obj(self.path)
            # val = ['loss','acc']
            # for name in val:
            #     logs[name] = previous[epoch + 1][name]
            outs_dict = {}
            for pat_num in range(len(self.test_pat)):
                if pat_num not in self.pat_set.keys():
                    self.save_pats = True
                    print('loading object')
                    self.pat_set[pat_num] = self.test_pat.__getitem__(pat_num)
                image_full_size, y = copy.deepcopy(self.pat_set[pat_num])
                if image_full_size.shape[1] != self.image_size and self.image_size == 256:
                    x = block_reduce(image_full_size,(1,2,2,1),np.average)
                else:
                    x = image_full_size
                if x[:,:,:,0].min() > -100:
                    x[:, :, :, 0] -= 123.68
                    x[:, :, :, 1] -= 116.78
                    x[:, :, :, 2] -= 103.94
                step = 5
                pred_out = np.zeros([x.shape[0], x.shape[1], x.shape[2],self.num_of_classes])
                start_vgg = 0
                while start_vgg < x.shape[0]:
                    temp_images = x[start_vgg:start_vgg + step, :, :, :]
                    pred_temp = self.model.predict(temp_images)
                    if len(pred_temp.shape) == 2:
                        pred_temp = np.reshape(pred_temp, [temp_images.shape[0], temp_images.shape[1], temp_images.shape[2],
                                                           self.num_of_classes])
                    pred_out[start_vgg:start_vgg + temp_images.shape[0], :, :, :] = pred_temp
                    start_vgg += step
                # pred_out = np.reshape(self.model.predict(x),[image_full_size.shape[0],256,256,self.num_of_classes])
                truth = y
                if pred_out.shape[1] != truth.shape[1]:
                    pred_out = self.resize_images.resize_images(pred_out)
                slice_thickness, pixel_spacing = [5,0.97]
                output = {}
                for xxx in range(1,truth.shape[-1]):
                    pred = copy.deepcopy(pred_out[:,:,:,xxx])
                    pred[pred > pred.max()/2] = 1
                    pred = remove_non_liver(pred, 0.5, volume_threshold=self.volume_threshold)
                    output_temp = get_metrics(pred, truth[:,:,:,xxx], slice_thickness, pixel_spacing)
                    k = output_temp['Overlap_Results']
                    k.update(output_temp['Surface_Results'])
                    output[xxx] = k
                outs_dict[pat_num] = output
            if self.save_pats:
                save_obj(self.pat_set, self.pat_path)
                self.save_pats = False
            logs['epoch'] = epoch + 1
            for val in output.keys():
                for key in output[val].keys():
                    if key.find('std') != -1:
                        continue
                    avg = []
                    for i in outs_dict.keys():
                        avg.append(outs_dict[i][val][key])
                    logs[str(val) + str(key)] = sum(avg)/len(avg)
                    logs[str(val) + str(key) + '_std'] = np.std(avg)
                    print(str(val) + '_' + str(key) + ' : ' + str(sum(avg)/len(avg)))
            # previous = copy.deepcopy(self.model.history.history)
            previous = load_obj(self.path)
            # logs['learning_rate'] = previous[epoch + 1]['learning_rate']
            logs['learning_rate'] = float(K.get_value(self.model.optimizer.lr))
            previous[epoch + 1] = logs

            # if os.path.exists(self.path):
            #     new_obj = load_obj(self.path)
            #     for key in new_obj.keys():
            #         previous[key] += new_obj[key]
            save_obj(previous,self.path)
            turn_pickle_into_text(previous,self.path.replace('.pkl','.txt'))
        return logs


class Add_DSC_To_Output(Callback):

    def __init__(self,test_patient_generator, image_size=512, epoch_count=1):
        super(Add_DSC_To_Output, self).__init__()
        print(image_size)
        self.test_pat = test_patient_generator
        self.epoch_count=epoch_count
        self.pat_set = {}
        self.smooth = 0.01

    def get_pat_info(self):
        if self.pat_set == {}:
            for i in range(len(self.test_pat)):
                self.pat_set[i] = self.test_pat.__getitem__(i)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_count == 0:
            output = []
            for pat_num in range(len(self.test_pat)):
                if pat_num not in self.pat_set.keys():
                    self.save_pats = True
                    print('loading object')
                    self.pat_set[pat_num] = self.test_pat.__getitem__(pat_num)
                image_full_size, y = copy.deepcopy(self.pat_set[pat_num])
                pred_out = self.model.predict(image_full_size)
                truth = y
                for xxx in range(1,truth.shape[-1]):
                    pred = copy.deepcopy(pred_out[...,xxx])
                    pred[pred > pred.max()/2] = 1
                    intersection = np.sum(np.abs(truth[...,xxx] * pred_out[...,xxx]))
                    output.append((2. * intersection + self.smooth) / (np.sum((pred_out[...,xxx])) + np.sum(truth[..., xxx]) + self.smooth))
            logs['self_dsc'] = np.mean(output)
        return logs


class Predict_On_VGG_UNet_Model(Callback):
    def __init__(self,model_load_path_VGG = '\\\\mymdafiles\\di_data1\\Morfeus\\Liver_Auto_Contour\\Liver_Model\\Model\\'
                                            'vgg-weights-improvement-adam_fixed_rotation-40.hdf5',
                 model_base_vgg_UNet='\\\\mymdafiles\\di_data1\\Morfeus\\Liver_Auto_Contour\\Liver_Model\\Model\\vgg-unet-'
                                     'weights-improvement-adam_no_decay_3x3x3_fixed_rotation-60.hdf5',use='VGG_UNet',
                 vgg_model=None, period=10):
        super(Predict_On_VGG_UNet_Model,self).__init__()
        use = use.lower()
        self.period = period
        self.use_vgg = False
        self.use_unet = False
        if use.find('vgg') != -1:
            self.use_vgg = True
        if use.find('unet') != -1:
            self.use_unet = True
        self.Prediction_Model = Prediction_Model_Class(model_load_path_VGG=model_load_path_VGG,
                                                       model_base_vgg_UNet=model_base_vgg_UNet,
                                                       vgg_model=vgg_model, use_unet=self.use_unet, use_vgg=self.use_vgg)
        # global pause_val
        # pause_val = False

    def on_batch_begin(self, batch, logs=None):
        if batch % self.period == 0:
            self.Prediction_Model.check_status_and_predict()
        return None


class Save_History_Class_3D(Callback):

    def __init__(self, path, test_patient_generator, pred_generator, use_vgg_pred=True, batch_size=32,image_size=256,
                 pat_path='G:\\CNN\\data\\Data_Kelly\\Test\\', num_of_classes=2, get_start_stop_val=True, epoch_count=1,
                 use_arg_max=True, use_vgg=True, expanded_vgg=False, is_single_structure=True):
        self.expanded_vgg = expanded_vgg
        self.is_single_structure = is_single_structure
        super(Save_History_Class_3D, self).__init__()
        self.use_arg_max = use_arg_max
        self.epoch_count = epoch_count
        self.get_start_start = get_start_stop_val
        self.num_of_classes = num_of_classes
        self.slice_thickness = 2.5
        self.pixel_spacing = 0.97
        self.path = path
        self.vgg_model = pred_generator
        self.test_pat = test_patient_generator
        self.use_vgg_pred = use_vgg_pred
        self.use_vgg = use_vgg
        if self.use_vgg_pred:
            print('Using VGG model')
        else:
            print('Not using VGG model')
        self.batch_size = batch_size
        self.pat_path = pat_path
        if self.pat_path.find('Saved_Test_Set.pkl') == -1:
            self.pat_path += 'Saved_Test_Set.pkl'
        self.pat_set = load_obj(self.pat_path)
        self.pat_info = [[3,0.68359],[5,0.78125], [2.5,0.78125], [2.5,0.8593], [2.5,0.97656], [5,0.78125],
                         [3,0.87], [2.5,0.74218], [2.5,0.820312]]
        self.image_size = image_size
        self.resize_images = Resize_Images_Keras(num_channels=self.num_of_classes)
        self.save_pats = False

    def get_pat_info(self):
        if self.pat_set == {}:
            for i in range(len(self.test_pat)):
                self.pat_set[i] = self.test_pat.__getitem__(i)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_count == 0:
            outs_dict = {}
            batch_size = self.batch_size
            for pat_num in range(len(self.test_pat)):
                if pat_num not in self.pat_set.keys():
                    self.save_pats = True
                    print('loading object')
                    x,y = self.test_pat.__getitem__(pat_num)
                    if len(x.shape) == 5:
                        x = x[0,:,:,:,:]
                        y = y[0,:,:,:,:]
                    self.pat_set[pat_num] = x,y
                image_full_size, y = copy.deepcopy(self.pat_set[pat_num])
                if not self.use_vgg and min(image_full_size[:,:,:,0]) < -120:
                    image_full_size[:, :, :, 0] += 123.68
                    image_full_size[:, :, :, 1] += 116.78
                    image_full_size[:, :, :, 2] += 103.94
                if image_full_size.shape[1] != self.image_size:
                    x = block_reduce(image_full_size,(1,2,2,1),np.average)
                else:
                    x = image_full_size
                start = 0
                stop = image_full_size.shape[0]
                if self.use_vgg:
                    vgg_train_images = copy.deepcopy(x)
                    vgg_train_images[:, :, :, 0] -= 123.68
                    vgg_train_images[:, :, :, 1] -= 116.78
                    vgg_train_images[:, :, :, 2] -= 103.94
                    step = 30
                    pred_0 = np.zeros([vgg_train_images.shape[0],vgg_train_images.shape[1],vgg_train_images.shape[2],self.num_of_classes])
                    pred = np.zeros([vgg_train_images.shape[0], vgg_train_images.shape[1], vgg_train_images.shape[2],
                                       int(self.vgg_model.vgg_model_base.outputs[1].shape[-1])])
                    start_vgg = 0
                    while start_vgg < vgg_train_images.shape[0]:
                        temp_images = vgg_train_images[start_vgg:start_vgg + step, :, :, :]
                        pred_temp_0, pred_temp_1 = self.vgg_model.predict(temp_images)
                        if len(pred_temp_1.shape) == 2:
                            pred_temp_0 = np.reshape(pred_temp_0, [temp_images.shape[0], self.image_size, self.image_size,self.num_of_classes])
                        pred_0[start_vgg:start_vgg + temp_images.shape[0], :, :, :] = pred_temp_0
                        pred[start_vgg:start_vgg + temp_images.shape[0], :, :, :] = pred_temp_1
                        start_vgg += step
                    start, stop = get_start_stop(pred_0)
                    if self.image_size == 512:
                        images = image_full_size
                        pred = self.resize_images.resize_images(pred)
                    else:
                        images = x
                    if self.expanded_vgg and self.use_vgg_pred:
                        images = np.concatenate([np.expand_dims(images[:, :, :, 0], axis=-1), pred],axis=-1)
                    elif self.use_vgg_pred:
                        if self.use_arg_max:
                            images[:, :, :, 1] = np.argmax(pred, axis=-1) * (255 / (self.num_of_classes - 1))
                        else:
                            images[:, :, :, 1] = pred[:, :, :, 1] * (255 / (self.num_of_classes - 1))
                else:
                    if self.image_size == 512:
                        images = image_full_size
                    else:
                        images = x
                images = np.expand_dims(images,axis=0)
                added_top = 0
                if stop - start < batch_size:
                    added_top = abs(stop-start-batch_size)
                    for i in range(added_top):
                        images = np.concatenate((np.expand_dims(images[0, :, :, :], axis=0), images), axis=0)
                pred_out = np.zeros(images.shape)
                pred_out = pred_out[:, :, :, :, :self.num_of_classes]
                print('Making UNet Predictions')
                z_start, z_stop, row_start, row_stop, col_start, col_stop = get_bounding_box(x,y,random_start=False)
                z_start -= 64
                row_start -= 64
                col_start -= 64
                row_start_base = copy.deepcopy(row_start)
                col_start_base = copy.deepcopy(col_start)
                while z_start < z_stop:
                    z_start += 64
                    row_start = copy.deepcopy(row_start_base)
                    while row_start < row_stop:
                        row_start += 64
                        col_start = copy.deepcopy(col_start_base)
                        while col_start < col_stop:
                            col_start += 64
                            print(z_start,row_start,col_start)
                            data = make_resolution_levels(images,np.expand_dims(y,axis=0),
                                                          [z_start,z_start+64,row_start,row_start+64,
                                                           col_start,col_start+64],
                                                          resolutions=[[64,64,64],[128,128,128],[256,256,256]])
                            for key in data:
                                data[key]['image'] = np.expand_dims(data[key]['image'],axis=0)
                            pred_out_dict = self.model.predict([data['high']['image'], data['medium']['image'], data['low']['image']])
                            output_shape = pred_out[:, z_start: z_start + 64, row_start: row_start + 64, col_start: col_start + 64,:].shape
                            output_shape = [i for i in output_shape]
                            pred_out[:, z_start: z_start + 64, row_start: row_start + 64, col_start: col_start + 64,:] = pred_out_dict[0][:,0:output_shape[1],0:output_shape[2],0:output_shape[3],:]
                while start < stop:
                    if start + batch_size > stop:
                        start = (stop - batch_size)
                    temp_images = images[:, start:start + self.batch_size, :, :, :]
                    if temp_images.shape[-1] == 33:
                        input_val = {'UNet_Input_image': temp_images[:, :, :, :, :1],
                                     'UNet_Input_conv': temp_images[:, :, :, :, 1:]}
                    else:
                        input_val = temp_images
                    pred = self.model.predict(input_val)
                    if len(pred.shape) == 2:
                        pred = np.reshape(pred, [temp_images.shape[0], temp_images.shape[1], temp_images.shape[2],
                                                 temp_images.shape[3], 2])
                    pred_out[:, start:start + self.batch_size, :, :, :] = \
                        pred[:, :, :, :, :]
                    start += batch_size

                if added_top != 0:
                    pred_out = pred_out[:, added_top:, :, :, :]
                pred_out = pred_out[0, :, :, :, :]  # pred_out[pred_out > .5] = 1.0
                truth = y
                if pred_out.shape[1] != truth.shape[1]:
                    pred_out = self.resize_images.resize_images(pred_out)
                output = {}
                for xxx in range(1,truth.shape[-1]):
                    pred = pred_out[:,:,:,xxx]
                    if pred.max() > 0.5:
                        pred[pred > pred.max() / 2] = 1
                        if self.is_single_structure:
                            pred = remove_non_liver(pred, 0.5)
                    else:
                        pred[pred > pred.max() / 2] = 1
                    pred[pred < 1] = 0
                    slice_thickness, pixel_spacing = [3,0.97]
                    output_temp = get_metrics(pred, truth[:,:,:,xxx], slice_thickness, pixel_spacing)
                    k = output_temp['Overlap_Results']
                    k.update(output_temp['Surface_Results'])
                    output[xxx] = k
                    print(k['dice'])
                outs_dict[pat_num] = output
            if self.save_pats:
                save_obj(self.pat_set, self.pat_path)
                self.save_pats = False
            logs['epoch'] = epoch + 1
            for val in output.keys():
                for key in output[val].keys():
                    if key.find('std') != -1:
                        continue
                    avg = []
                    for i in outs_dict.keys():
                        avg.append(outs_dict[i][val][key])
                    logs[str(val) + str(key)] = sum(avg)/len(avg)
                    logs[str(val) + str(key) + '_std'] = np.std(avg)
                    print(str(val) + '_' + str(key) + ' : ' + str(sum(avg)/len(avg)))
            # previous = copy.deepcopy(self.model.history.history)
            previous = load_obj(self.path)
            logs['learning_rate'] = float(K.get_value(self.model.optimizer.lr))
            previous[epoch + 1] = logs
            save_obj(previous, self.path)
            try:
                turn_pickle_into_text(previous, self.path.replace('.pkl', '.txt'))
            except:
                print('issue with pickle to text')
        return logs


class Save_History_Class_3D_new(Callback):

    def __init__(self, test_patient_generator,num_of_classes=2, epoch_count=1):
        self.num_of_classes = num_of_classes
        self.epoch_count = epoch_count
        self.test_patient_generator = test_patient_generator
        self.is_single_structure = True
        super(Save_History_Class_3D_new, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_count == 0:
            outs_dict = {}
            for pat_num in range(len(self.test_patient_generator)):
                images,truth = self.test_patient_generator.__getitem__(pat_num)
                pred_out = self.model.predict(images)
                output = {}
                for xxx in range(1,truth.shape[-1]):
                    pred = pred_out[:,:,:,xxx]
                    if pred.max() > 0.5:
                        pred[pred > pred.max() / 2] = 1
                        if self.is_single_structure:
                            pred = remove_non_liver(pred, 0.5)
                    else:
                        pred[pred > pred.max() / 2] = 1
                    pred[pred < 1] = 0
                    slice_thickness, pixel_spacing = [3,0.97]
                    output_temp = get_metrics(pred, truth[:,:,:,xxx], 3, 0.97)
                    k = output_temp['Overlap_Results']
                    k.update(output_temp['Surface_Results'])
                    output[xxx] = k
                    print(k['dice'])
                outs_dict[pat_num] = output

            logs['epoch'] = epoch + 1
            for val in output.keys():
                for key in output[val].keys():
                    if key.find('std') != -1:
                        continue
                    avg = []
                    for i in outs_dict.keys():
                        avg.append(outs_dict[i][val][key])
                    logs[str(val) + str(key)] = sum(avg)/len(avg)
                    logs[str(val) + str(key) + '_std'] = np.std(avg)
                    print(str(val) + '_' + str(key) + ' : ' + str(sum(avg)/len(avg)))
            logs['learning_rate'] = float(K.get_value(self.model.optimizer.lr))
        return logs


class ModelCheckpoint_new(ModelCheckpoint):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, model=None):
        self.is_gpu_model = False
        if model:
            self.save_model = model
            self.is_gpu_model = True
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    def set_path(self, path):
        self.path = path
    def get_current_best(self):
        self.metric = []
        epoch_list = list(self.logs.keys())
        for i in epoch_list[:epoch_list.index(self.epoch+1)+1]:
            self.metric.append(self.logs[i][self.monitor])
    def on_epoch_end(self, epoch, logs=None):
        if not self.is_gpu_model:
            self.save_model = self.model
        self.epoch = epoch + 1
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            filepath = self.filepath.replace('{epoch:02d}',str(epoch + 1))
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            # if self.save_best_only:
            #     self.logs = load_obj(self.path)
            #     self.get_current_best()
            #     if len(self.metric) > 1:
            #         self.best = max(self.metric[:-1])
            #     else:
            #         self.best = 0.0
            #     self.epochs_since_last_save = 0
            #     current = self.metric[-1]
            #     if current is None:
            #         warnings.warn('Can save best model only with %s available, '
            #                       'skipping.' % (self.monitor), RuntimeWarning)
            #     else:
            #         if self.metric[-1] >= self.best:
            #             if self.verbose > 0:
            #                 print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
            #                       ' saving model to %s'
            #                       % (epoch + 1, self.monitor, self.best,
            #                          current, filepath))
            #             self.best = current
            #             if self.save_weights_only:
            #                 self.save_model.save_weights(filepath, overwrite=True)
            #             else:
            #                 self.save_model.save(filepath, overwrite=True)
            #         else:
            #             if self.verbose > 0:
            #                 print('\nEpoch %05d: %s did not improve from %0.5f' %
            #                       (epoch + 1, self.monitor, self.best))
            else:
                self.epochs_since_last_save = 0
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.save_model.save_weights(filepath, overwrite=True)
                else:
                    self.save_model.save(filepath, overwrite=True)


def shuffle_item(item):
    perm = np.arange(len(item))
    np.random.shuffle(perm)
    item = np.asarray(item)[perm]
    perm = np.arange(len(item))
    np.random.shuffle(perm)
    item = list(np.asarray(item)[perm])
    return item


def get_bounding_box_indexes(annotation):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: the min and max z, row, and column numbers bounding the image
    '''
    annotation = np.squeeze(annotation)
    if annotation.dtype != 'int':
        annotation[annotation>0.1] = 1
        annotation = annotation.astype('int')
    indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
    min_z_s, max_z_s = indexes[0], indexes[-1]
    '''
    Get the row values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_r_s, max_r_s = indexes[0], indexes[-1]
    '''
    Get the col values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_c_s, max_c_s = indexes[0], indexes[-1]
    return min_z_s, int(max_z_s + 1), min_r_s, int(max_r_s + 1), min_c_s, int(max_c_s + 1)

def pad_images(images,annotations,output_size=None,value=0):
    if not output_size:
        print('did not provide a desired size')
        return images, annotations
    holder = output_size - np.asarray(images.shape)
    if np.max(holder) == 0:
        return images, annotations
    val_differences = [[max([int(i/2 - 1),0]), max([int(i/2 - 1),0])] for i in holder]
    if np.max(val_differences) > 0:
        images, annotations = np.pad(images, val_differences, 'constant', constant_values=(value)), \
                        np.pad(annotations, val_differences, 'constant', constant_values=(0))
    holder = output_size - np.asarray(images.shape)
    final_pad = [[0, i] for i in holder]
    if np.max(final_pad) > 0:
        images, annotations = np.pad(images, final_pad, 'constant', constant_values=(value)), \
                        np.pad(annotations, final_pad, 'constant', constant_values=(0))
    return images, annotations

def pull_cube_from_image(images, annotation, desired_size=(16,32,32), samples=10):
    output_images = np.ones([samples,desired_size[0],desired_size[1],desired_size[2],1])*np.min(images)
    output_annotations = np.zeros([samples, desired_size[0], desired_size[1], desired_size[2], annotation.shape[-1]])
    pat_locations, z_locations, r_locations, c_locations = np.where(annotation[...,-1] == 1)
    for i in range(samples):
        index = np.random.randint(len(z_locations))
        z_start = max([0, int(z_locations[index] - desired_size[0] / 2)])
        z_stop = min([z_start + desired_size[0], images.shape[1]])
        r_start = max([0, int(r_locations[index] - desired_size[1] / 2)])
        r_stop = min([r_start + desired_size[1], images.shape[2]])
        c_start = max([0, int(c_locations[index] - desired_size[2] / 2)])
        c_stop = min([c_start + desired_size[2], images.shape[3]])
        output_images[i, :z_stop - z_start, :r_stop - r_start, :c_stop - c_start, ...] = images[pat_locations[index],z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
        output_annotations[i, :z_stop - z_start, :r_stop - r_start, :c_stop - c_start, ...] = annotation[pat_locations[index],z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
    return output_images, output_annotations


def center_image_based_on_annotation(images,annotation,mask,layers=2,extensions=(0,0,0)):
    '''
    :param images: 1, images, rows, columns, channels
    :param annotation: 1, images, rows, cols, channels
    :param mask: images, rows, cols
    :param layers:
    :param extensions:
    :return:
    '''
    if mask.dtype != 'int':
        mask[mask>0.1] = 1
        mask = mask.astype('int')
    mask = remove_non_liver(mask)
    z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(np.expand_dims(mask,axis=-1))
    max_image_number = 150
    if z_stop - z_start > max_image_number:
        dif = int((max_image_number - (z_stop - z_start)))
        if np.random.random() > 0.5:
            z_stop += dif
        else:
            z_start -= dif
    power_val = 2 ** layers
    z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
    z_total, r_total, c_total = z_total + extensions[0], r_total + extensions[1], c_total + extensions[2]
    remainder_z, remainder_r, remaineder_c = power_val - z_total % power_val if z_total % power_val != 0 else 0, \
                                             power_val - r_total % power_val if r_total % power_val != 0 else 0, \
                                             power_val - c_total % power_val if c_total % power_val != 0 else 0
    min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r + extensions[1], c_total + remaineder_c + extensions[2]
    dif_z = min_images - (z_stop - z_start + 1)
    dif_r = min_rows - (r_stop - r_start + 1)
    dif_c = min_cols - (c_stop - c_start + 1)
    extension = min([min([z_start, images.shape[1] - z_stop]), int(dif_z / 2)])  # Keep everything centered
    z_start, z_stop = z_start - extension, z_stop + extension
    # self.z_start, self.z_stop = int(self.z_start - mult*extension), int(self.z_stop + mult*extension)
    extension = min([min([r_start, images.shape[2] - r_stop]), int(dif_r / 2)])  # Keep everything centered
    r_start, r_stop = r_start - extension, r_stop + extension
    # self.r_start, self.r_stop = int(self.r_start - mult*extension), int(self.r_stop + mult*extension)
    extension = min([min([c_start, images.shape[3] - c_stop]), int(dif_c / 2)])  # Keep everything centered
    # self.c_start, self.c_stop = int(self.c_start - mult * extension), int(self.c_stop + mult * extension)
    c_start, c_stop = c_start - extension, c_stop + extension
    if min_images - (z_stop - z_start) == 1:
        if z_start > 0:
            z_start -= 1
            # self.z_start -= mult
        elif z_stop < images.shape[0]:
            z_stop += 1
            # self.z_stop += mult
    if min_rows - (r_stop - r_start) == 1:
        if r_start > 0:
            r_start -= 1
            # self.r_start -= mult
        elif r_stop < images.shape[1]:
            r_stop += 1
            # self.r_stop += mult
    if min_cols - (c_stop - c_start) == 1:
        if c_start > 0:
            c_start -= 1
            # self.c_start -= mult
        elif c_stop < images.shape[2]:
            c_stop += 1
            # self.c_stop += mult
    images, annotation = images[:, z_start:z_stop, r_start:r_stop, c_start:c_stop], \
                    annotation[:, z_start:z_stop, r_start:r_stop, c_start:c_stop, :]
    images, annotation = pad_images(images, annotation, [1, min_images, min_rows, min_cols, images.shape[-1]],value=-3.55)
    return images, annotation


def cartesian_to_polar(xyz):
    '''
    :param x: x_values in single array
    :param y: y_values in a single array
    :param z: z_values in a single array
    :return: polar coordinates in the form of: radius, rotation away from the z axis, and rotation from the y axis
    '''
    # xyz = np.stack([x, y, z], axis=-1)
    input_shape = xyz.shape
    reshape = False
    if len(input_shape) > 2:
        reshape = True
        xyz = np.reshape(xyz,[np.prod(xyz.shape[:-1]),3])
    polar_points = np.empty(xyz.shape)
    # ptsnew = np.hstack((xyz, np.empty(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    polar_points[:,0] = np.sqrt(xy + xyz[:,2]**2)
    polar_points[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    polar_points[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    if reshape:
        polar_points = np.reshape(polar_points,input_shape)
    return polar_points

def polar_to_cartesian(polar_xyz):
    '''
    :param polar_xyz: in the form of radius, elevation away from z axis, and elevation from y axis
    :return: x, y, and z intensities
    '''
    cartesian_points = np.empty(polar_xyz.shape)
    from_y = polar_xyz[:,2]
    xy_plane = np.sin(polar_xyz[:,1])*polar_xyz[:,0]
    cartesian_points[:,2] = np.cos(polar_xyz[:,1])*polar_xyz[:,0]
    cartesian_points[:,0] = np.sin(from_y)*xy_plane
    cartesian_points[:,1] = np.cos(from_y)*xy_plane
    return cartesian_points


def make_resolution_levels(image, annotation, indexes, resolutions=[[64, 64, 64], [128, 128, 128], [256, 256, 256]]):
    '''
    :param image: Image in form of [z_images, rows, cols, channels]
    :param annotation: Annotations in from of [z_images, rows, cols, classes(1 per channel)
    :param indexes: z, row, and col start and stop values
    :param resolutions: Resolutions to sample back down to high resolution (being the first one)
    :return:
    '''
    if len(image.shape) > 4:
        image = np.squeeze(image,axis=0)
    if len(annotation.shape) > 4:
        annotation = np.squeeze(annotation, axis=0)
    x, y = image, annotation
    sub_sample_size = resolutions[0]
    z_start, z_stop, row_start, row_stop, col_start, col_stop = indexes
    resolutions_out = {'high':{'image':[],'annotation':[]},'medium':{'image':[],'annotation':[]},'low':{'image':[],'annotation':[]}}
    key_list = list(resolutions_out.keys())
    for resolution in resolutions:
        z_size, row_size, col_size = resolution
        z_shift, row_shift, col_shift = z_size - (z_stop - z_start), row_size - (row_stop - row_start),\
                                        col_size - (col_stop - col_start)
        z_shift, row_shift, col_shift = int(z_shift/2), int(row_shift/2), int(col_shift/2)
        z_start, z_stop, row_start, row_stop, col_start, col_stop = z_start - z_shift, z_stop + z_shift, \
                                                                    row_start - row_shift, row_stop + row_shift, \
                                                                    col_start - col_shift, col_stop + col_shift
        i = 0
        while z_stop - z_start < z_size: # If they are off by 1, fix it
            if i == 0:
                z_start -= 1
                i = 1
            else:
                i = 0
                z_stop += 1
        i = 0
        while row_stop - row_start < row_size:
            if i == 0:
                row_start -= 1
                i = 1
            else:
                i = 0
                row_stop += 1
        i = 0
        while col_stop - col_start < col_size:
            if i == 0:
                col_start -= 1
                i = 1
            else:
                i = 0
                col_stop += 1
        z_start_add = abs(z_start) if z_start < 0 else 0
        z_stop_add = z_stop - x.shape[0] if z_stop > x.shape[0] else 0
        row_start_add = abs(row_start) if row_start < 0 else 0
        row_stop_add = row_stop - x.shape[1] if row_stop > x.shape[1] else 0
        col_start_add = abs(col_start) if col_start < 0 else 0
        col_stop_add = col_stop - x.shape[2] if col_stop > x.shape[2] else 0
        z_start = 0 if z_start < 0 else z_start
        row_start = 0 if row_start < 0 else row_start
        col_start = 0 if col_start < 0 else col_start
        z_stop = x.shape[0] if z_stop > x.shape[0] else z_stop
        row_stop = x.shape[1] if row_stop > x.shape[1] else row_stop
        col_stop = x.shape[2] if col_stop > x.shape[2] else col_stop

        temp_x, temp_y = x[z_start:z_stop, row_start:row_stop, col_start:col_stop, :],y[z_start:z_stop, row_start:row_stop, col_start:col_stop, :]
        if max([z_start_add, z_stop_add, row_start_add, row_stop_add, col_start_add, col_stop_add]) > 0:
            val_differences = [[z_start_add, z_stop_add],[row_start_add, row_stop_add],[col_start_add, col_stop_add],[0,0]]
            temp_x, temp_y = np.pad(temp_x,val_differences, 'constant', constant_values=(0)), \
                             np.pad(temp_y,val_differences, 'constant', constant_values=(0))
        reduction_factors = [int(resolution[i] / sub_sample_size[i]) for i in range(len(resolution))] + [1]  # for channel dimension
        if max(reduction_factors) > 1:
            temp_y = block_reduce(temp_y.astype('int'), tuple(reduction_factors), np.max).astype('int')
            temp_x = block_reduce(temp_x, tuple(reduction_factors), np.average)
        resolutions_out[key_list[resolutions.index(resolution)]]['image'],resolutions_out[key_list[resolutions.index(resolution)]]['annotation'] = temp_x, temp_y
    return resolutions_out

def get_bounding_box(train_images_out_base, train_annotations_out_base, include_mask=True,
                     image_size=512, sub_sample=[64,64,64], random_start=True):
    '''
    :param train_images_out_base: shape[1, #images, image_size, image_size, channels]
    :param train_annotations_out_base: shape[1, #images, image_size, image_size, #classes]
    :param include_mask:
    :param image_size:
    :param sub_sample: the box dimensions to include the organ
    :param random_start: Makes a random sub section
    :return: list of indicies which indicate the bounding box of the organ
    '''
    if len(train_images_out_base.shape) == 4:
        train_images_out_base = np.expand_dims(train_images_out_base, axis=0)
        train_annotations_out_base = np.expand_dims(train_annotations_out_base, axis=0)
    train_images_out = train_images_out_base
    train_annotations_out = train_annotations_out_base
    min_row, min_col, min_z, max_row, max_col, max_z = 0, 0, 0, image_size, image_size, train_images_out.shape[1]
    if include_mask:
        mask_comparison = np.squeeze((np.argmax(train_annotations_out, axis=-1)),axis=0)
        itemindex = np.where(mask_comparison > 0)
        min_z, max_z = min(itemindex[0]), max(itemindex[0])
        min_row, max_row = min(itemindex[1]), max(itemindex[1])
        min_col, max_col = min(itemindex[2]), max(itemindex[2])
        if random_start:
            min_row = min_row - int(sub_sample[1] / 2) if min_row - int(sub_sample[1] / 2) > 0 else 0
            min_col = min_col - int(sub_sample[2] / 2) if min_col - int(sub_sample[2] / 2) > 0 else 0
            min_z = min_z - int(sub_sample[0]/2) if min_z - int(sub_sample[0]/2) > 0 else 0
            max_row = max_row + int(sub_sample[1]/2) if max_row + sub_sample[1]/2 < image_size else image_size
            max_col = max_col + int(sub_sample[2]/2) if max_col + sub_sample[2]/2 < image_size else image_size
            max_z = max_z + sub_sample[0]/2 if max_z + sub_sample[0]/2 < train_images_out.shape[1] else train_images_out.shape[1]
            got_region = False
            while not got_region:
                z_start = np.random.randint(min_z, max_z - sub_sample[0]) if max_z - sub_sample[0] > min_z else min_z
                row_start = np.random.randint(min_row,max_row - sub_sample[1])
                col_start = np.random.randint(min_col,max_col - sub_sample[2])
                if z_start < 0:
                    z_start = 0
                col_stop = col_start + sub_sample[2]
                row_stop = row_start + sub_sample[1]
                z_stop = z_start + sub_sample[0] if z_start + sub_sample[0] <= train_images_out.shape[1] else train_images_out.shape[1]
                # train_images_out = train_images_out[:, z_start:z_stop, row_start:row_stop, col_start:col_stop, :]
                # train_annotations_out = train_annotations_out[:, z_start:z_stop, row_start:row_stop, col_start:col_stop, :]
                if not include_mask:
                    got_region = True
                elif np.any(mask_comparison[z_start:z_stop, row_start:row_stop, col_start:col_stop] > 0):
                    got_region = True
            return z_start, z_stop, row_start, row_stop, col_start, col_stop
        else:
            return min_z, max_z, min_row, max_row, min_col, max_col
    else:
        return min_z, max_z, min_row, max_row, min_col, max_col



def make_necessary_folders(base_path):
    if not os.path.exists(os.path.join(base_path,'Keras','Models','v3_Model')):
        os.makedirs(os.path.join(base_path,'Keras','Models','v3_Model'))
    if not os.path.exists(os.path.join(base_path,'Keras','Text_Output','v3')):
        os.makedirs(os.path.join(base_path,'Keras','Text_Output','v3'))
    return None


class VGG_Model_Pretrained():
    def __init__(self,model_path,num_classes=2,gpu=0,image_size=512,graph1=Graph(),session1=Session(config=ConfigProto(gpu_options=GPUOptions(allow_growth=True), log_device_placement=False)), Bilinear_model=None):
        self.image_size=image_size
        print('loaded vgg model ' + model_path)
        self.num_classes = num_classes
        self.graph1 = graph1
        self.session1 = session1
        if tf.__version__ == '1.14.0':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Restrict TensorFlow to only use the first GPU
                try:
                    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
                except:
                    xxx = 1
            with graph1.as_default():
                with session1.as_default():
                    print('loading VGG Pretrained')
                    self.vgg_model_base = load_model(model_path, custom_objects={'BilinearUpsampling':Bilinear_model,'dice_coef_3D':dice_coef_3D})
        else:
            with tf.device('/gpu:' + str(gpu)):
                with graph1.as_default():
                    with session1.as_default():
                        print('loading VGG Pretrained')
                        self.vgg_model_base = load_model(model_path, custom_objects={'BilinearUpsampling':Bilinear_model,'dice_coef_3D':dice_coef_3D})
        # with K.tf.device('/gpu:0'):
        #     self.graph1 = Graph()
        #     with self.graph1.as_default():
        #         gpu_options = GPUOptions(allow_growth=True)
        #         self.session1 = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #         with self.session1.as_default():

    def predict(self,images):
        num_outputs = len(self.vgg_model_base.outputs)
        return self.vgg_model_base.predict(images)
        # try:
        #     pred_0 = np.zeros([images.shape[0], images.shape[1], images.shape[2], self.num_classes])
        #     if num_outputs == 2:
        #         outputs = int(self.vgg_model_base.outputs[1].shape[-1])
        #         pred_1 = np.zeros([images.shape[0], self.image_size, self.image_size, outputs])
        # except:
        #     outputs = self.num_classes
        #     pred_0 = np.zeros([images.shape[0], self.image_size, self.image_size, outputs])
        # with self.graph1.as_default():
        #     with self.session1.as_default():
        #         step = 30
        #         start = 0
        #         for i in range(int(images.shape[0] / step) + 1):
        #             if start > images.shape[0]:
        #                 break
        #             temp_images = images[start:start+step,:,:,:]
        #             temp_output = self.vgg_model_base.predict(temp_images)
        #             if num_outputs == 2:
        #                 temp_pred_0, temp_pred_1 = temp_output
        #             else:
        #                 temp_pred_0 = temp_output
        #             if len(temp_pred_0.shape) == 2:
        #                 temp_pred_0 = np.reshape(temp_pred_0, [temp_images.shape[0], temp_images.shape[1], temp_images.shape[2], self.num_classes])
        #             pred_0[start:start + temp_images.shape[0], :, :, :] = temp_pred_0
        #             if num_outputs == 2:
        #                 pred_1[start:start + temp_images.shape[0], :, :, :] = temp_pred_1
        #             start += step
        #         if num_outputs == 2:
        #             return pred_0, pred_1
        #         else:
        #             return pred_0


class Predict_On_Models():
    images = []

    def __init__(self,vgg_model, UNet_model, num_classes=2, use_unet=True, batch_size=32, is_CT=True, image_size=256,
                 step=30, vgg_normalize=True, verbose=True):
        self.step = step
        self.image_size = image_size
        self.vgg_model = vgg_model
        self.UNet_Model = UNet_model
        self.batch_size = batch_size
        self.use_unet = use_unet
        self.num_classes = num_classes
        self.is_CT = is_CT
        self.vgg_normalize = vgg_normalize
        self.verbose = verbose

    def make_3_channel(self):
        if self.images.shape[-1] != 3:
            if self.images.shape[-1] != 1:
                self.images = np.expand_dims(self.images, axis=-1)
            images_stacked = np.concatenate((self.images, self.images), axis=-1)
            self.images = np.concatenate((self.images, images_stacked), axis=-1)

    def normalize_images(self):
        self.images = normalize_images(images=self.images,
                                       lower_threshold=-100, upper_threshold=300, is_CT=self.is_CT)

    def resize_images(self):
        if self.images.shape[1] != self.image_size:
            self.images = block_reduce(self.images, (1, 2, 2, 1), np.average)

    def vgg_pred_model(self):
        start = 0
        new_size = [self.images.shape[0], self.images.shape[1], self.images.shape[2], self.num_classes]
        self.vgg_pred = np.zeros(new_size)
        self.vgg_images = copy.deepcopy(self.images)
        if self.vgg_normalize:
            if self.vgg_images[:,:,:,0].min() > -50:
                self.vgg_images[:, :, :, 0] -= 123.68
                self.vgg_images[:, :, :, 1] -= 116.78
                self.vgg_images[:, :, :, 2] -= 103.94
        stop = self.vgg_images.shape[0]
        if not self.is_CT:
            for i in range(self.vgg_images.shape[0]):
                val = self.vgg_images[i,0,0,0]
                if not math.isnan(val) and self.vgg_images[i,:,:,:].max() > 100:
                    start = i
                    break
            for i in range(self.vgg_images.shape[0]-1,-1,-1):
                val = self.vgg_images[i,0,0,0]
                if not math.isnan(val) and self.vgg_images[i,:,:,:].max() > 100:
                    stop = i
                    break

        step = self.step
        total_steps = int(self.vgg_images.shape[0]/step) + 1
        for i in range(int(self.vgg_images.shape[0]/step) + 1):
            if start >= stop:
                break
            if start + step > stop:
                step = stop - start
            self.vgg_pred[start:start + step,:,:, :] = self.vgg_model.predict(self.vgg_images[start:start+step,:,:,:])
            start += step
            if self.verbose:
                print(str((i + 1)/total_steps * 100) + ' % done predicting')

    def make_predictions(self):
        self.make_3_channel()
        # self.normalize_images()
        self.resize_images()
        self.vgg_pred_model()
        images = self.images
        if not self.use_unet:
            self.pred = self.vgg_pred
        else:
            start, stop = get_start_stop(copy.deepcopy(self.vgg_pred))
            # images[:, :, :, 1] = np.argmax(self.vgg_pred, axis=-1) * (255 / (self.num_classes - 1))
            images[:, :, :, 1] = self.vgg_pred[:,:,:,1] * (255 / (self.num_classes - 1))
            shift_size = int(self.batch_size/2)
            start -= int(shift_size * 2)
            stop += int(shift_size * 2)
            added_top = 0
            if start < 0:
                added_top = abs(start)
                start += added_top
                for i in range(added_top):
                    images = np.concatenate((np.expand_dims(images[0, :, :, :], axis=0), images), axis=0)
            start_i = copy.deepcopy(start)
            added_bottom_batch = 0
            while start_i + int(self.batch_size * 2 / 4) <= stop:
                added_bottom_batch += 1
                start_i += self.batch_size
            added_bottom_batch += 1
            added_bottom = start + added_bottom_batch * self.batch_size - images.shape[0]

            if added_bottom > 0:
                for i in range(added_bottom):
                    images = np.concatenate((images, np.expand_dims(images[-1, :, :, :], axis=0)), axis=0)
            else:
                added_bottom = 0
            images = np.expand_dims(images, axis=0)

            images[0, :, :, :, 0] -= 123.68
            images[0, :, :, :, 1] -= 116.78
            images[0, :, :, :, 2] -= 103.94
            pred_out = np.zeros(images.shape)
            pred_out = pred_out[:, :, :, :, :-1]
            print('Making UNet Predictions')
            while start + int(self.batch_size * 2 / 4) <= stop:
                temp_images = images[:, start:start + self.batch_size, :, :, :]
                pred = self.UNet_Model.predict(temp_images)
                if len(pred.shape) == 2:
                    pred = np.reshape(pred, [temp_images.shape[0], temp_images.shape[1], temp_images.shape[2],
                                             temp_images.shape[3], self.num_classes])
                pred_out[:, start + int(shift_size / 2):start + int(shift_size * 3 / 2), :, :, :] = \
                    pred[:, int(shift_size / 2):int(shift_size * 3 / 2), :, :, :]
                start += shift_size
            print('Finished UNet')
            if added_top != 0:
                pred_out = pred_out[:, added_top:, :, :, :]
            if added_bottom != 0:
                pred_out = pred_out[:, :-added_bottom, :, :, :]
            pred_out = pred_out[0, :, :, :, :]# pred_out[pred_out > .5] = 1.0
        # pred_out[pred_out < 1.0] = 0.0
            self.pred = pred_out


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[...,1:] * y_pred[...,1:])
    union = K.sum(y_true[...,1:]) + K.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)

def single_dice(y_true, y_pred, smooth = 0.0001):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_3D_np(y_true, y_pred, smooth=0.0001):
    intersection = np.sum(y_true[...,1:] * y_pred[...,1:])
    union = np.sum(y_true[...,1:]) + np.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_2D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[:,:,:,1:] * y_pred[:,:,:,1:])
    union = K.sum(y_true[:,:,:,1:]) + K.sum(y_pred[:,:,:,1:])
    classes = (2. * intersection + smooth) / (union + smooth)
    intersection = K.sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
    union = K.sum(y_true[:,:,:,0]) + K.sum(y_pred[:,:,:,0])
    background = (2. * intersection + smooth) / (union + smooth)
    return background + 5*classes

def jaccard_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[:,:,:,:,1:] * y_pred[:,:,:,:,1:])
    union = K.sum(y_true[:,:,:,:,1:]) + K.sum(y_pred[:,:,:,:,1:])
    classes = (intersection + smooth) / (union - intersection)
    intersection = K.sum(y_true[:,:,:,:,0] * y_pred[:,:,:,:,0])
    union = K.sum(y_true[:,:,:,:,0]) + K.sum(y_pred[:,:,:,:,0])
    background = (intersection + smooth) / (union - intersection)
    return background + 5*classes

def jaccard_coef_2D(y_true, y_pred, smooth=0.0001):
    intersection = K.sum(y_true[:,:,:,1:] * y_pred[:,:,:,1:])
    union = K.sum(y_true[:,:,:,1:]) + K.sum(y_pred[:,:,:,1:])
    return (intersection + smooth) / (union - intersection)

def jaccard_coef_loss(y_true, y_pred):
    return 1-jaccard_coef_3D(y_true, y_pred)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def pad_depth(x, desired_channels):
    y = K.zeros_like(x, name='pad_depth1')
    new_channels = desired_channels - x.shape.as_list()[1]
    y = y[:new_channels,:,:]
    return Concatenate([x,y], name='pad_depth2')

def masked_mse(mask_value):
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, 0.0), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.sum(mask_true, axis=-1)
        return masked_mse
    f.__name__ = 'Masked MSE (mask_value={})'.format(mask_value)
    return f
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)*100, axis=0) # increase the loss

def dice_coef(y_true, y_pred, smooth=0.1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true[...,1] * y_pred[...,1]))
    return (2. * intersection + smooth) / (K.sum((y_true[...,1])) + K.sum(y_pred[...,1]) + smooth)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

class Up_Sample_Class(object):
    def __init__(self):
        xxx = 1
        self.main()
    def main(self):
        input = x = Input(shape=(None, None, None, 3))
        x = UpSampling3D((2,2,2))(x)
        model = Model(inputs=input, outputs=x)
        self.created_model = model


def masked_mean_squared_error(y_true,y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    return K.sum(masked_squared_error, axis=0) / K.sum(mask_true, axis=0)


def weighted_mse(y_true, y_pred, weights):
    return K.mean(K.abs(y_true - y_pred) * weights, axis=-1)

def weighted_mse_polar(y_true, y_pred, weights=K.variable(np.array([1,2,2]))):
    return K.mean(K.abs(y_true - y_pred) * weights, axis=-1)

def categorical_crossentropy_masked(y_true, y_pred, mask, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(y_pred.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(y_pred.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    # scale preds so that the class probs of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred)
    loss = loss * mask
    loss = -K.sum(loss, -1)
    return loss

def expand_dims(x):
    return K.expand_dims(x,0)

def expand_dims_output_shape(input_shape):
    return (1, input_shape[0], input_shape[1], input_shape[2], input_shape[3])

def squeeze_dims(x):
    return K.squeeze(x,0)
def squeezed_dims_output_shape(input_shape):
    return (input_shape[1], input_shape[2], input_shape[3], input_shape[4])

ExpandDimension = lambda axis: Lambda(lambda x: K.expand_dims(x, axis))
SqueezeDimension = lambda axis: Lambda(lambda x: K.squeeze(x, axis))

def create_3D_Addition(model, desired_layer_name=None):
    all_layers = model.layers
    layer = [layer for layer in all_layers if layer.name == desired_layer_name][0]
    layer_output = layer.output  # We already have the input
    x = ExpandDimension(0)(layer_output)
    x = Conv3D(16,(3,3,3),activation='elu',padding='same')(x)
    output = Conv3D(2,(1,1,1),activation='softmax')(x)
    output = SqueezeDimension(0)(output)
    activation_model = Model(inputs=model.input, outputs=output)
    return activation_model
if __name__ == '__main__':
    xxx = 1