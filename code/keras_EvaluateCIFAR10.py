import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop

 
def load_and_scale_imgs():
   img_names = ['cat-standing.jpg', 'dog.jpg']
 
   imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                        (2, 0, 1)).astype('float32')
           for img_name in img_names]
   return np.array(imgs) / 255

def load_model(model_def_fname, model_weight_fname):
   model = model_from_json(open(model_def_fname).read())
   model.load_weights(model_weight_fname)
   return model

imgs = load_and_scale_imgs()
model = load_model('cifar10_architecture.json', 'cifar10_weights.h5')
model.summary()

# train
optim = RMSprop()
#optim = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optim,
	metrics=['accuracy'])
 
predictions = model.predict_classes(imgs)
print(predictions)

print ('airplane	0\
 automobile	1\
 bird	2\
 cat	3\
 deer	4\
 dog	5\
 frog	6\
 horse	7\
 ship	8\
 truck	9')