# https://fairyonice.github.io/Saliency-Map-with-keras-vis.html
# https://fairyonice.github.io/Grad-CAM-with-keras-vis.html

from keras import activations
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, json
from keras.preprocessing import image
from keras_vggface import utils
from vis.visualization import visualize_cam
from vis.utils import utils as vis_utils
import argparse


class ModelLoad(object):
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
    def model_loader(self):
        try:
            print("Trying to restore last checkpoint ...")
            model = load_model(filepath=self.filepath, compile=True)
            print("Restored model from:", self.filepath)

            return model
        except OSError:
            # If the above failed for some reason, simply
            print("Unable to open file name = %s, No such file or directory", self.filepath)
        
class ImageScore(object):
    def __init__(self, model, img_path, farm, version, **kwargs):
        self.model = model
        self.img_path = img_path
        self.farm = farm
        self.version = version

    def scores(self):
        
        with open(os.path.join('./checkpoints', self.farm, 'labels.json')) as json_file:
            classes = json.load(json_file)
        inv_classes = {v: k for k, v in classes.items()}

        self._img = image.load_img(self.img_path, target_size=(224, 224))
        self.img = image.img_to_array(self._img)
        self.img = np.expand_dims(self.img, axis=0)
        self.img = utils.preprocess_input(self.img, version=self.version) # or version=2
        #Use utils.preprocess_input(x, version=1) for VGG16
        #Use utils.preprocess_input(x, version=2) for RESNET50 or SENET50

        scores = self.model.predict(x=self.img, verbose=1)
        
        preds = dict()
        for i in range(len(classes)):
            preds[inv_classes[i]] = scores[0][i].item()
        
        preds = json.dumps(preds, indent=4, sort_keys=True)
        return scores, preds, self.img, self._img, inv_classes


class GradCAM(object):
    def __init__(self, arch, model, img, _img, farm, scores, inv_classes, **kwargs):
        self.arch = arch
        self.model = model
        self.img = img
        self._img = _img
        self.farm = farm
        self.scores = scores
        self.inv_classes = inv_classes

    def plot_map(self):
        # Utility to search for layer index by name. 
        # Alternatively we can specify this as -1 since it corresponds to the last layer.
        if self.arch == 'vgg16':
            layer_idx = vis_utils.find_layer_idx(self.model, layer_name='conv5_3')
        elif self.arch == 'resnet50': 
            layer_idx = vis_utils.find_layer_idx(self.model, layer_name='conv5_3_3x3')
        elif self.arch == 'senet50': 
            layer_idx = vis_utils.find_layer_idx(self.model, layer_name='conv5_3_3x3') # o también conv5_3_1x1_up ó conv5_3_1x1_down
    
        class_idxs_sorted = np.argsort(self.scores.flatten())[::-1]
        class_idx  = class_idxs_sorted[0]
        seed_input = self.img
        grad_top  = visualize_cam(self.model, layer_idx, class_idx, seed_input, 
                                   penultimate_layer_idx = layer_idx,
                                   backprop_modifier = None,
                                   grad_modifier = None)

        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        axes[0].imshow(self._img)
        axes[1].imshow(self._img)
        i = axes[1].imshow(grad_top/255., cmap="jet", alpha=0.8)
        fig.colorbar(i)
        plt.suptitle("Pr[img|label={}] = {:5.2f}".format(
                      self.inv_classes[class_idx],
                      self.scores[0,class_idx]))
        plt.show()

def main():
    parser = argparse.ArgumentParser(
    description="Convolutional Neural Network testing routine for cattle recognition",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    fromfile_prefix_chars='@')
    parser.add_argument('--granja', '-ng', help='Nome da granja')
    parser.add_argument('--img', '-img', type=str, default=None, help='Localização da imagem para obter previsões.')
    parser.add_argument('--model', '-m', type=str, default='resnet50', help='Modelo a ser usado para treinamento. Valores suportados: vgg16, resnet50 ou senet50. Valor padrão: vgg16')
    args = parser.parse_args()

    if not args.granja:
        parser.print_help()
        raise ValueError('Você deve proporcionar o nome da granja após --granja')
    if not args.img:
        parser.print_help()
        raise ValueError('Please, provide a relative path for the location of the image to be sampled after --img')

    # Cargamos modelo desde checkpoint
    model = ModelLoad(filepath=os.path.join('./checkpoints', args.granja, 'chckpt.best.h5')).model_loader()

    scores, preds, img, _img, inv_classes = ImageScore(model=model, 
                       img_path=args.img, 
                       farm=args.granja, 
                       version=1).scores()
    GradCAM(arch=args.model, 
            model=model, 
            img=img,
            _img=_img,
            farm=args.granja,
            inv_classes=inv_classes,
            scores=scores).plot_map()

if __name__ == '__main__':
    main()