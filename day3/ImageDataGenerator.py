from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

base_dir ='./datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    #constant(cval) reflect wrap
)

from tensorflow.keras.preprocessing import image
train_cats_dir = os.path.join(train_dir, 'cats')
fnames = sorted([os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)])
img_path = fnames[3]

img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
print(x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgpot = plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i % 8 == 0:
        break
plt.show()
