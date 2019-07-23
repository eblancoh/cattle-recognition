
Oxford VGGFace Implementation using Keras Functional Framework v2+

Models are converted from original caffe networks.
It supports only Tensorflow backend.
You can also load only feature extraction layers with VGGFace(include_top=False) initiation.
When you use it for the first time, weights are downloaded and stored in ~/.keras/models/vggface folder.

Traerme los checkpoints a local

# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
# https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
# https://towardsdatascience.com/transfer-learning-and-image-classification-using-keras-on-kaggle-kernels-c76d3b030649

https://medium.com/@mohamedchetoui/grad-cam-gradient-weighted-class-activation-mapping-ffd72742243a
https://github.com/eclique/keras-gradcam/blob/master/gradcam_vgg.ipynb

# https://github.com/rcmalli/keras-vggface

cd path/to/file
tensorboard --logdir=./

https://fairyonice.github.io/Grad-CAM-with-keras-vis.html