# Cattle Recognition - Identificación de Ganado con Convolutional Neural Networks (CCNs)

<p align="center">
<img width="500" height="300" src=./support/cow_meme.jpg>
</p>
Este proyecto tiene como misión hacer uso de redes convolucionales para la identificación de ganado (vacas, cerdos, etc.) haciendo uso de redes neuronales profundas convolucionales. Las capas convolucionales actúan como extractores de características.

Con la intención de ahorrar tiempo, se pretende hacer uso de modelos ya entrenados de Deep Learning dedicados a reconocimiento facial de seres humanos para la identificación de ganado. *(Pendiente de probar).*

Para ello, se realiza Transfer Learning y Fine-Tuning de los modelos de Oxford VGGFace a través del endopoint de TensorFlow. El código de este repositorio soporta tanto los modelos **VGG16** como **RESNET50** o **SENET50**:

```python
from keras_vggface.vggface import VGGFace

# Based on VGG16 architecture -> old paper(2015)
vggface = VGGFace(model='vgg16') # or VGGFace() as default

# Based on RESNET50 architecture -> new paper(2017)
vggface = VGGFace(model='resnet50')

# Based on SENET50 architecture -> new paper(2017)
vggface = VGGFace(model='senet50')
```
Tanto en su versión:
 
```python 
include_top=False
``` 
que permiten quedarte sólo con la parte convolucional de la red para hacer un stacking superior de un clasificador a placer

Como en la versión 
```python 
include_top=True
```
la cual incluye la parte de clasificación original con todas las capas densas, lo que lo hace más pesado.

## Ejemplo de uso

### 1. Data Cleaning (opcional)

Para evitar la baja varianza entre imágenes se emplea la medida del índice de similitud estructural (SSIM) para medir la similitud entre fotografías. Esto ayuda a evitar datos muy similares (casi idénticos en las particiones de datos de validación y entrenamiento.

En los subdirectorios anidados de `./dataset` se checkea una imagen contra todas las demás y se van eliminando aquellas que sean similares por encima de un valor de similitud (entre `0` y `1`) indicado por el usuario.

```bash
$ python ssim.py --dir "dir/to/images" --threshold 0.95
```

### 2. Entrenamiento

Ejemplo del entrenamiento de un dataset de imágenes:
```bash
$ python training.py --granja test --model resnet50 --epochs 20 --batch_size 30
```
La rutina realiza el entrenamiento, guarda el mejor checkpoint y devuelve un reporte de clasificación sobre el test dataset.

#### 2.1. Customización de arquitectura para Transfer Learning
El script `training.py` se lanza tal y como se muestra arriba. Este script entrena una Red Neuronal convolucional que puede ser `vgg16`, `resnet50` o `senet50` y que acaba en un clasificador que, por defecto, tiene la siguiente implementación `Sequential` de Keras:

```python
self.x = self.last_layer
self.x = Dense(self.hidden_dim_1, activation='relu', name='fc1')(self.x)
self.x = Dense(self.hidden_dim_2, activation='relu', name='fc2')(self.x)
self.x = Dense(self.hidden_dim_3, activation='relu', name='fc3')(self.x)
self.out = Dense(self.nb_class, activation='softmax', name='out')(self.x)
```
Clasificador de capas densas totalmente conectadas que se meten tras la capa `flatten` de los modelos convolucionales.

`TODO:` ver si la regularización por Dropout mejora la performance de los modelos.

#### 2.2. Congelación de capas para Fine-Tuning
Normalmente, para Transfer Learning y Fine-Tuning de modelos con dataset pequeños, lo que se hace es congelar la arquitectira transferida y entrenar sólamente el clasificador customizado por nosotros. El número de capas a definir como entrenables se especifica en la función `main()` en la línea `277`:
* `nb_freeze = None` indica que no se congela ninguna capa. Todas las capas son entrenables.
* `nb_freeze = 10` indica que se congelan las 10 primeras capas. Las restantes son entrenables por defecto.
* `nb_freeze = -4` indica que se congelan todas menos las 4 últimas capas. Las restantes son entrenables por defecto.

#### 2.3. Dataset
El dataset sobre el que se desea entrenar debe situarse en la carpeta `./dataset`. Para cada clase, se deben agrupar todas las imágenes en subdirectorios. 

Los batches de entrenamiento, validación, así como el núemro de clases a predecir y, por lo tanto, la arquitectura de salida del modelo, están definidas tanto por el generador `ImageDataGenerator()` como por la función `flow_from_directory()`.

Sobre el dataset disponible se hace data augmentation:
* **Rotación aleatoria** de cada imagen de hasta `20 grados`;
* **Desplazamiento en altura y anchura** de hasta el `5%` de la dimensión de la imagen;
* **Horizontal flipping**.

#### 2.4. Logueo del entrenamiento

Para el entrenamiento se han definido dos callbacks: 
* **EarlyStopping** para evitar overfitting o alargar innecesariamente el tiempo de entrenamiento. 
* **TensorBoard Callback** que permite logar precisión y funciones de pérdida para su visualización en browser de las curvas de aprendizaje y del grafo creado y entrenado.

```bash
$ cd logs/folder_tbd/
$ tensorboard --logdir=./ --port 6006
```

De manera que con sólo ir a tu navegador a `http://localhost:6006/` se puede visualizar cómo ha transcurrido el entrenamiento. Ver el [siguiente artículo](https://itnext.io/how-to-use-tensorboard-5d82f8654496) para aclarar dudas.

### 3. Testeo
De cara al testeo de un modelo ya entrenado con una imagen de muestra, se ejecuta el script `testing.py`:
```bash
$ python testing.py --granja test  --img "path/to/img"
```

Esta rutina devuelve las predicciones de una imagen en base a todas las clases soportadas por el modelo. La rutina devuelve:

```bash
$ python testing.py --granja test  --img "path/to/img"
{
    "class 0": score 0,
    "class 1": score 1,
    ...
    "class N": score N
}
```

### 4. Grad-CAM Checking

Un inconveniente frecuentemente citado del uso de redes neuronales es que entender exactamente lo que están modelando es muy difícil. Esto se complica aún más utilizando redes profundas. Sin embargo, esto ha comenzado a atraer una gran cantidad de interés de investigación, especialmente para las CNNs para garantizar que la "atención" de la red se centre en las características reales y discriminativas del propio animal, en lugar de otras partes de la imagen que puedan contener información discriminatoria (por ejemplo, una etiqueta de clase, una marca de tiempo impresa en la imagen o un borde sin interés).

Para comprobar que nuestro modelo se centra en partes interesantes de la imagen, se puede elegir una imagen de prueba y comprobar qué regiones son de interés para la red neuronal, se puede ejecutar:

```bash
$ python grad_CAM.py --granja test --model resnet50 --img "path/to/img"
```
Lo cual te devuelve un mapa de calor sobre la imagen de las regiones de interés.

## Enlaces de Soporte e Interés:

* [Keras Framework](www.keras.io)
* [Oxford VGGFace Website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
* [Arkhi et al.: Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
* [Towards on-farm pig face recognition using convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0166361517304992?via%3Dihub#fig0025)
* [VGGFace implementation with Keras Framework](https://github.com/rcmalli/keras-vggface)
* [Deep Learning For Beginners Using Transfer Learning In Keras](https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e)
* [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)
* [Transfer Learning using Keras](https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8)
* [Tutorial on using Keras `flow_from_directory()` and generators](https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)
* [Transfer learning and Image classification using Keras on Kaggle kernels](https://towardsdatascience.com/transfer-learning-and-image-classification-using-keras-on-kaggle-kernels-c76d3b030649)
* [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
* [Gradient-weighted Class Activation Mapping - Grad-CAM](https://medium.com/@mohamedchetoui/grad-cam-gradient-weighted-class-activation-mapping-ffd72742243a)
* [Keras implementation of GradCAM](https://github.com/eclique/keras-gradcam/blob/master/gradcam_vgg.ipynb)
* [Grad-CAM with keras-vis](https://fairyonice.github.io/Grad-CAM-with-keras-vis.html)
* [A Python module for computing the Structural Similarity Image Metric (SSIM)](https://github.com/jterrace/pyssim)

# Licencia
This is free and unencumbered software released into the public domain. Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.