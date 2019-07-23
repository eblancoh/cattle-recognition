from keras.utils import plot_model
from keras_vggface.vggface import VGGFace 
import argparse
import os

class Visualize(object):
    def __init__(self, model, **kwargs):
        self.model = model
    def model_graph(self):
        if self.model == 'vgg16':
            model = VGGFace(model='vgg16')
        elif self.model == 'resnet50':
            model = VGGFace(model='resnet50')
        elif self.model == 'senet50':
            model = VGGFace(model='senet50')
        
        GraphsMkdir().check()
        plot_model(model, to_file='../graphs/' + self.model + '.png', show_shapes=True)

class GraphsMkdir(object):
    def __init__(self):
        self.bool = os.path.isdir('../graphs')
    def check(self):
        if not self.bool:
            os.mkdir('../graphs')
            #print('Checkpoints folder created. \n Skipping creation.')
        else:
            pass
            #print('Checkpoints folder already exists. \n Skipping creation.')


def main():
    parser = argparse.ArgumentParser(
        description="Convolutional Neural Network drawing routine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('--model', '-m', type=str, default='vgg16', help='Modelo da arquitetura convolucional. Valores suportados: vgg16, resnet50 ou senet50. Valor padrão: vgg16')
    args = parser.parse_args()

    Visualize(model=args.model).model_graph()

if __name__ == '__main__':
    main()