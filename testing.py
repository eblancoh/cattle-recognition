from keras.models import load_model
from keras.preprocessing import image
from keras_vggface import utils
import os
import json
import numpy as np
import argparse
import operator


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
    def __init__(self, model, img, farm, version, **kwargs):
        self.model = model
        self.img = img
        self.farm = farm
        self.version = version

    def scores(self):
        
        with open(os.path.join('./checkpoints', self.farm, 'labels.json')) as json_file:
            classes = json.load(json_file)
        inv_classes = {v: k for k, v in classes.items()}

        self.img = image.load_img(self.img, target_size=(224, 224))
        self.img = image.img_to_array(self.img)
        self.img = np.expand_dims(self.img, axis=0)
        self.img = utils.preprocess_input(self.img, version=self.version) # or version=2
        #Use utils.preprocess_input(x, version=1) for VGG16
        #Use utils.preprocess_input(x, version=2) for RESNET50 or SENET50

        scores = self.model.predict(x=self.img, verbose=1)[0]
        
        preds = dict()
        for i in range(len(classes)):
            preds[inv_classes[i]] = scores[i].item()
        
        # Ordenamos de mayor a menor score
        preds = {k: preds[k] for k in sorted(preds, key=preds.get, reverse=True)}
        preds = json.dumps(preds, indent=4, sort_keys=False)
        #print(preds)
        return preds

def main():
    parser = argparse.ArgumentParser(
        description="Convolutional Neural Network testing routine for cattle recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('--granja', '-ng', help='Nome da granja')
    parser.add_argument('--img', '-img', type=str, default=None, help='Localização da imagem para obter previsões.')
    
    args = parser.parse_args()

    if not args.granja:
        parser.print_help()
        raise ValueError('Você deve proporcionar o nome da granja após --granja')
    if not args.img:
        parser.print_help()
        raise ValueError('Você deve proporcionar o destino da imagem após --img')


    # Cargamos modelo desde checkpoint
    model = ModelLoad(filepath=os.path.join('./checkpoints', args.granja, 'chckpt.best.h5')).model_loader()

    preds = ImageScore(model=model, 
                       img=args.img, 
                       farm=args.granja, 
                       version=1).scores()

if __name__=='__main__':
    main()