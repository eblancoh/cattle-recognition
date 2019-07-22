from keras.utils import plot_model
from keras_vggface import VGGFace

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
        plot_model(model, to_file=self.model + '.png', show_shapes=True)