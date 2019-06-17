import matplotlib  # knihovna pro tvorbu grafù
import matplotlib.pyplot as plt
import datetime  # datum a èas
import imageio  # umožòuje èíst obrázek a provádìt s ním další operace
import glob  # najde všechny cesty odpovídající vzoru, aèkoliv výsledky se vrací v neuspoøádaném poøadí
import os.path  # pro ètení a zapisování souborù
import numpy as np  # pro vìdecké výpoèty, mj. poèítání s maticemi
from keras.applications.vgg19 import VGG19  # z knihovny Keras importuje neuronovou sí
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.layers import Dense, Dropout  # GlobalAveragePooling2D,
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, Flatten, Activation

matplotlib.use('Agg')  # tvorba grafù bez geometrie v obrázkovém formátu, napøíklad .png (anti grain geometry engine)


EPOCHS = 20
BATCHSIZE = 15
LEFT = 64
RIGHT = 64
MODELDIR = "C:/Users/Terezka/Desktop/model/"  #složka, kam se ukládá model
DATADIR = "C:/Users/Terezka/Desktop/obrazky/train/" #složka s obrázkama 64x64

VALIDATIONDIR = "C:/Users/Terezka/Desktop/obrazky/validation/" #validaèní data? 

# Used for giving the model a meaningful name. Update manually
IDENTNAME = "VGG19_finalimage_random" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

NUMDATAFILES = len([name for name in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, name))])
# cesty ke složkám, len=délka vstupní promìnné
NUMVALIDATIONFILES = len(
    [name for name in os.listdir(VALIDATIONDIR) if os.path.isfile(os.path.join(VALIDATIONDIR, name))])
DATASTEPS = round(NUMDATAFILES / BATCHSIZE)
# round=celá èísla, zaokrouhlené dìlení

# ** ** ** ** Tereza ** * Lots of ways of doing the generation of data.This is one method.It depends on the
# batches having the name "_noskred_" or "_skred_" skred = Avalanche in Norwegian.From this is generates the
# "ground truth".Other alternatives are sorting the images in two directories.


def generator(batch_size, datapath):
    from random import shuffle  # importuje funkci která míchá

    target = glob.glob(datapath + '*.png')  # najde všechny png soubory

    n_samples = len(target)  # kolik tam je souborù
    n_batches = n_samples // batch_size
    b = n_batches

    while True:
        
        if b == n_batches:  # vždycky pomíchá soubory
            shuffle(target)
            b = 0
            print("epoch finished - " + datapath)

        # initialize current batch
        batch_features = np.zeros((batch_size, LEFT, RIGHT, 3))  # matice s nulama velkou jako obrázek
        batch_labels = np.zeros((batch_size, 2))  #
        target_b = target[b * batch_size:(b + 1) * batch_size]  # indices of the current batch, target je cíl

        # populate current batch
        for i, t in enumerate(target_b):
            batch_features[i, :, :, :] = imageio.imread(t)[:, :, :3]  # odstraní prùhlednost aby zùstalo jenom rgb

            batch_labels[i, :] = np.array([1, 0]) if "_noavalanche_" in t else np.array([0, 1])

        b += 1

        yield batch_features, batch_labels


for _ in range(3):

    base_model = VGG19(weights=None, include_top=False, pooling='avg', input_shape=(LEFT, RIGHT, 3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Print the layers
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.output_shape)  # vypisuje vìci dole
    plot_model(model, show_shapes=True, to_file=MODELDIR + IDENTNAME + '_model.png')

    # sys.exit()

    # we chose to train the top  inception blocks, i.e. we will freeze
    # the first 5 layers and unfreeze the rest:
    for layer in model.layers[:10]:
        layer.trainable = True
    for layer in model.layers[10:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    
    from keras.optimizers import Adam
    
    optimizer = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(generator(BATCHSIZE, DATADIR), steps_per_epoch=DATASTEPS,
                                  validation_data=generator(NUMVALIDATIONFILES, VALIDATIONDIR), validation_steps=1,
                                  epochs=EPOCHS, verbose=1, class_weight={0: 1, 1: 1})

    # Save model and weights....
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(MODELDIR + IDENTNAME + '_model.yaml', "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(MODELDIR + IDENTNAME + '_weights.h5')
    print("Saved model to disk")

    # plot metrics
    # summarize history for accuracy
    print("Plotting accuracy and loss")
    fig1 = plt.figure()
    plt.plot(history.history['acc'])
    # If we have test data, this can be enabled as well
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(MODELDIR + IDENTNAME + '_accuracy.png')

    # summarize history for loss
    fig2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('chyba sítì')
    plt.xlabel('epocha')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(MODELDIR + IDENTNAME + '_chyba.png')

    # Summary
    result = sorted(zip(history.history['val_acc']), reverse=True)[:3]

    print("\nToto jsou tøi nejlepší výsledky:")
    print(str(round(float(result[0][0]), 5)) + "\t" + str(round(float(result[1][0]), 5)) + "\t" + str(
        round(float(result[2][0]), 5)))
    print("\nMaximum bylo dosaženo po " + str(
        history.history['val_acc'].index(max(history.history['val_acc'])) + 1) + " epochách.")

    with open('C:/Users/Terezka/Desktop/Obrazky/skript.txt', 'a') as the_file:
        the_file.write('\n\n' + IDENTNAME + '\n')
        the_file.write('Poèet epoch' + str(EPOCHS) + '\n')
        the_file.write('Batchsize' + str(BATCHSIZE) + '\n')
        the_file.write('Velikost' + str(LEFT) + 'X' + str(RIGHT) + '\n')
        the_file.write(str(datetime.datetime.now()) + '\n')
        the_file.write("Maximum bylo dosaženo po " + str(
            history.history['val_acc'].index(max(history.history['val_acc'])) + 1) + " epochách.\n")
        the_file.write('Tohle byly tøi nejlepší výsledky:' + '\n')
        the_file.write(str(round(float(result[0][0]), 5)) + '\t' + str(round(float(result[1][0]), 5)) + '\t' + str(
            round(float(result[2][0]), 5)) + '\n')
