
from keras_vggface.vggface import VGGFace
from keras.engine import Model
from keras import models, layers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime
from PIL import Image
import argparse
import json
import os



class CustomVGGModel(object):
    def __init__(self, model, height, width, channels, nb_class, nb_freeze,**kwargs):
        self.model = model
        self.height = height
        self.width = width
        self.channels = channels

        self.nb_class = nb_class
        self.nb_freeze = nb_freeze

        self.hidden_dim_1 = 512
        self.hidden_dim_2 = 256
        self.hidden_dim_2 = 128
    
    def model_build(self):

        # Carga del modelo por defecto
        """
        self.vggface = VGGFace(include_top=False, 
                                 model=self.model, 
                                 input_shape=(
                                     self.height, 
                                     self.width, 
                                     self.channels
                                     )
                                )
        """
        self.vggface = VGGFace(include_top=True, 
                                 model=self.model, 
                                 input_shape=(
                                     self.height, 
                                     self.width, 
                                     self.channels
                                     )
                                )
        # Añadimos capas de clasificación a la red convolucional
        """
        if self.model == 'vgg16':
            self.last_layer = self.vggface.get_layer('pool5').output
        elif self.model == 'resnet50':
            self.last_layer = self.vggface.get_layer('avg_pool').output
        elif self.model == 'senet50':
            self.last_layer = self.vggface.get_layer('avg_pool').output
        """
        if self.model == 'vgg16':
            self.last_layer = self.vggface.get_layer('flatten').output
        elif self.model == 'resnet50':
            self.last_layer = self.vggface.get_layer('flatten_1').output
        elif self.model == 'senet50':
            self.last_layer = self.vggface.get_layer('flatten_1').output
        """
        self.x = Flatten(name='flatten')(self.last_layer)
        """
        self.x = self.last_layer
        self.x = Dense(self.hidden_dim_1, activation='relu', name='fc1')(self.x)
        self.x = Dense(self.hidden_dim_2, activation='relu', name='fc2')(self.x)
        self.out = Dense(self.nb_class, activation='softmax', name='out')(self.x)
        # Now a custom model has been created based on our architecture
        # supported in Visual Geometry Group
        self.custom_model = Model(inputs=self.vggface.input, outputs=self.out)
        
        return self.custom_model
    
    def set_trainable(self):
        # If we want to set the first self.nb_freeze layers of the network to be frozen
        # the rest should remain trainable. 
        # Our dataset is small and very different from human faces!
        # It will be hard to find a balance between the number of layers to train and freeze. 
        # If you go to deep your model can overfit, if you stay in the shallow end 
        # of your model you won’t learn anything useful.
        if self.nb_freeze is None:
            # Mantenemos como entrenables todas las capas
            for layer in self.custom_model.layers[:]:
                layer.trainable=True
        else:
            if self.nb_freeze != 0:
                # Entendemos que un número positivo indica que 
                # queremos congelar las self.nb_freeze primeras capas
                # Entendemos que un número negativo indica que 
                # queremos congelar todas las capas salvo las self.nb_freeze últimas capas
                for layer in self.custom_model.layers[:self.nb_freeze]:
                    layer.trainable=False
            elif self.nb_freeze == 0:
                # Entendemos que un 0 indica que no queremos congelar nada
                for layer in self.custom_model.layers[:]:
                    layer.trainable=True

    def custom_model_summary(self):
        print(self.custom_model.summary())
        print('Status de las diferentes capas del modelo:')
        for i,layer in enumerate(self.custom_model.layers):
            print(i, layer.name, layer.trainable)
    
    def compiler(self, optimizer, loss, metrics):
        self.custom_model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        return self.custom_model


class Generator(object):
    def __init__(self, height, width, bs, dir, **kwargs):
        self.height = height
        self.width = width
        self.batch_size = bs
        self.dir = dir # 'path-to-the-main-data-folder'

        self.generator=ImageDataGenerator(preprocessing_function=preprocess_input,
                            rotation_range=20,
                            width_shift_range=self.width*0.005,
                            height_shift_range=self.height*0.005,
                            # shear_range=0.01,
                            horizontal_flip=True,
                            validation_split=0.2
                            ) 

    def dataset_generator(self):

        train_generator=self.generator.flow_from_directory(self.dir,
                                                target_size=(self.height,self.width),
                                                color_mode='rgb',
                                                batch_size=self.batch_size,
                                                class_mode='categorical',
                                                shuffle=True,
                                                subset='training')

        test_generator=self.generator.flow_from_directory(self.dir,
                                                 target_size=(self.height,self.width),
                                                 color_mode='rgb',
                                                 batch_size=self.batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True,
                                                 subset='validation')

        step_size_train=train_generator.n//train_generator.batch_size
        step_size_test=test_generator.n//test_generator.batch_size

        return train_generator, test_generator, step_size_train, step_size_test


class FitGenerator(object):
    def __init__(self, model, train_generator, test_generator, step_size_train, step_size_test, epochs,**kwargs):
        self.model = model
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.step_size_train = step_size_train
        self.step_size_test = step_size_test
        self.epochs = epochs
        self.epochs_to_wait = 15

    def train(self, farm):

        CheckpointsMkdir(farm).check()
        LogsMkdir().check()
    
        early_stopping_callback = EarlyStopping(monitor='val_loss', 
                                                patience=self.epochs_to_wait)
        checkpoint_callback = ModelCheckpoint(filepath=os.path.join('./checkpoints', farm, 'chckpt.best.h5'), 
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min')
        
        tensorboard_callback = TensorBoard(log_dir='./logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
                                          histogram_freq=0, # It is general issue with keras/tboard that you cannot get histograms with a validation_generator
                                          write_graph=True,
                                          write_images=True,
                                          )

        self.model.fit_generator(generator=self.train_generator, \
                               steps_per_epoch=self.step_size_train, \
                               validation_data=self.test_generator, \
                               validation_steps=self.step_size_test, \
                               epochs=self.epochs, \
                               verbose=1, \
                               callbacks=[
                                   early_stopping_callback, 
                                   checkpoint_callback,
                                   tensorboard_callback
                                   ])


class CheckpointsMkdir(object):
    def __init__(self, farm):
        self.farm = farm
        self.bool = os.path.isdir(os.path.join('./checkpoints', self.farm))
    def check(self):
        if not self.bool:
            os.mkdir(os.path.join('./checkpoints', self.farm))
            #print('Checkpoints folder created. \n Skipping creation.')
        else:
            pass
            #print('Checkpoints folder already exists. \n Skipping creation.')


class LogsMkdir(object):
    def __init__(self):
        self.bool = os.path.isdir('./logs')
    def check(self):
        if not self.bool:
            os.mkdir('./logs')
            #print('Logging folder created.')
        else:
            pass
            #print('Logging folder already exists. \n Skipping creation.')


class InputShape(object):
    def __init__(self, image, **kwargs):
        self.image = image
    def _shape(self):
        return Image.open(self.image).size # returns (width, height) tuple

class ClassParser(object):
    def __init__(self, input, farm, dir, **kwargs):
        self.input = input
        self.farm = farm
        self.dir = dir
    def save(self):
        CheckpointsMkdir(self.farm).check()
        filepath = os.path.join(self.dir, self.farm) + '\\labels.json'
        with open(filepath, 'w') as fp:
            json.dump(self.input, 
                      fp, 
                      sort_keys=True, 
                      indent=4)

def main():

    parser = argparse.ArgumentParser(
        description="Convolutional Neural Network training routine for cattle recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('--granja', '-ng', type=str, default='nome', help='Nome da granja')
    parser.add_argument('--model', '-m', type=str, default='resnet50', help='Modelo a ser usado para treinamento. Valores suportados: vgg16, resnet50 ou senet50. Valor padrão: vgg16')
    parser.add_argument('--epochs', '-ep', type=int, default=20, help='Número de épocas para treinar o modelo')
    parser.add_argument('--batch_size', '-bs', type=int, default=30, help='Tamanho do lote das amostras para treinar o modelo')

    args = parser.parse_args()

    if not args.granja:
        parser.print_help()
        raise ValueError('Você deve proporcionar o nome da granja após --granja')

    train_generator, \
    test_generator, \
    step_size_train, \
    step_size_test = Generator(height=224, 
                               width=224, 
                               bs=args.batch_size, 
                               dir='./dataset').dataset_generator()

    deep_model = CustomVGGModel(model=args.model, 
                                height=224, 
                                width=224, 
                                channels=3, 
                                nb_class=train_generator.class_indices.__len__(), # That way we get the number of classes automatically
                                nb_freeze=None)
    
    # Guardamos las clases, las vamos a necesitar para testeo

    ClassParser(input=train_generator.class_indices,
                farm=args.granja,
                dir='./checkpoints').save()
    
    deep_model.model_build()
    deep_model.set_trainable()
    deep_model.custom_model_summary()

    compiled_deep_model= deep_model.compiler(optimizer='Adam', 
                                             loss='categorical_crossentropy', 
                                             metrics=['accuracy'])

    FitGenerator(model=compiled_deep_model, 
                 train_generator=train_generator, 
                 test_generator=test_generator, 
                 step_size_train=step_size_train, 
                 step_size_test=step_size_test, 
                 epochs=args.epochs).train(args.granja)

if __name__ == '__main__':
    main()

# TODO: gridsearch