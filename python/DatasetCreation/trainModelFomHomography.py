from skimage.draw import polygon2mask, polygon, polygon_perimeter
from makeRandomHomography import RandomHomographyCreator as rhc
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import gc

from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow as tf

# CONSTANTS

NUM_IMAGES = 100
image_size = 128
SEED = 42
partitions = 8
curr_path = os.path.dirname(os.path.realpath(__file__))


# FUNCTIONS

def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "Images", "Masks")
    mask = tf.io.read_file(mask_path)

    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask != 0, np.dtype('uint8').type(1), mask)

    return {'image': image, 'segmentation_mask': mask}

def parse_generated_images(image, mask) -> dict:
    return {'image': image, 'segmentation_mask': mask}

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[image_size, image_size, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # show_predictions()
        gc.collect()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# MODEL

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[image_size, image_size, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# tf.keras.utils.plot_model(model, show_shapes=True)

# PROGRAM

# Prepare training dataset
test_data = tf.data.Dataset.list_files(curr_path + "/../../Datasets/BlenderDataset/Images/" + "*.png", seed=SEED)
test_data = test_data.map(parse_image)

services = []
names = []

pistes = open(curr_path + "/coords_pistes.txt", "r")
coords_pista = pistes.readline()
while coords_pista and not "#DESCARTATS" in coords_pista:
    coords = np.array(coords_pista.split("//")[0].replace(",","").split(), dtype=np.float64).reshape(4,2)
    services.append(rhc(coords, image_size*2, partitions))
    names.append(coords_pista.split("//")[1].replace("\n","").replace("\r",""))
    coords_pista = pistes.readline()
pistes.close()

iteration = 1

while True:
    images = []
    masks = []
    it = 0
    while it < NUM_IMAGES:
        index = 0 #random.choice(range(len(services)))
        result, cords, horizon_mask = services[index].getRandomHomography()

        if(result is not None):
            mask = np.transpose(polygon2mask(result.shape[:-1], cords))
            images.append(cv2.resize(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), (image_size, image_size)))
            masks.append(cv2.resize(mask.astype(np.uint8), (image_size, image_size)))
            it += 1
        else:
            print("Error creating homography, skipping image")
    images = tf.data.Dataset.from_tensor_slices(np.array(images))
    masks = tf.data.Dataset.from_tensor_slices(np.array(masks).reshape(NUM_IMAGES, image_size, image_size, 1).astype(np.uint8))
    # for im, mask in zip(images, masks):
    #     cv2.imshow("True Mask", cv2.resize(mask.astype(np.uint8)*255, (256, 256)))
    #     cv2.imshow("Result", cv2.resize(im, (256, 256)))
    #     cv2.waitKey()

    train_data = tf.data.Dataset.zip((images, masks))
    train_data = train_data.map(parse_generated_images)

    train = train_data.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test = test_data.map(load_image_test)

    for image, mask in train.take(1):
        sample_image, sample_mask = image, mask
    # display([sample_image, sample_mask])

    BUFFER_SIZE = 100
    TRAIN_LENGTH = len(train)
    BATCH_SIZE = 5
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_dataset = train.take(BUFFER_SIZE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    
    EPOCHS = 5
    VAL_SUBSPLITS = 10
    VALIDATION_STEPS = len(test)//BATCH_SIZE//VAL_SUBSPLITS

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[DisplayCallback()])

    # loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']

    # epochs = range(EPOCHS)

    # plt.figure()
    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()

    # show_predictions(test_dataset, 3)

    model.save(curr_path+'/Models/ModelTest'+ str(iteration) +'.h5')
    iteration += 1


    

