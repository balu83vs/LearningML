import torch

from torchvision import transforms          # актуальная библиотека
from torchvision.transforms import v2       # новая библиотека

import matplotlib.pyplot as plt  # для отображения картинок
from PIL import Image

import numpy as np


# преобразуем изображение в массив данных numpy
image = np.array(Image.open('/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/transformations/image.JPG'))


#################################### Использование модуля transforms ##############################################

# ToTensor()
"""
Трансформация ToTensor() 
    - правильно изменяет размер массива
    - приводит к правильному типу данных
    - в правильном интервале
"""

transform = transforms.ToTensor()   # создаем трансформацию
img_totensor = transform(image)     # преобразуем изображение в тензор


###### неправильный способ перемещение цветовых каналов со 2-й оси на 0-ю ######
# с помошью метода reshape
"""
img_c2 = image
img_c0 = image.reshape([3, 1667, 1667])

_, ax = plt.subplots(2,1)

ax[0].imshow(img_c2[...,0])
ax[1].imshow(img_c0[0,...])
plt.show()
"""
# получается неправильно отмасштабированное изображение


# Normalize
"""
Трансформация Normalize()
    - предназначена для нормирования данных
"""

 # передаем среднее значение для каждого канала и стандартное отклонение для каждого канала
transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
img_norm = transform(img_totensor) # в данную трансформацию можно передавать только тензоры типа float


# Compose
"""
Класс Compose необходим для последовательного применения трансформаций
"""

# трансформация для изображений с 3-я каналами
transform_3color = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

# трансформация для изображений с 1-м каналом (Grayscale)
transform_1color = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

# передаем в трансформацию изображение в формате PIL
img_totensor_norm = transform_3color(Image.open('/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/transformations/image.JPG'))
# передаем массив данных numpy
img_totensor_norm_2 = transform_3color(image)



#################################### Использование модуля V2 ##############################################

# ToTensor()
transform_v2 = v2.ToTensor()                # создаем трансформацию
img_totensor_v2 = transform_v2(image)       # преобразуем изображение в тензор


# ToImage
"""
ToImage преобразует данные в тензор и перемещаает цветовые каналы на 0 ось
не меняет тип данных и не переопределяет интервал
"""
transform_v2 = v2.ToImage()                  # создаем трансформацию
img_toimage_v2 = transform_v2(image)         # преобразуем изображение в тензор


# ToDtype
"""
ToDtype меняет тип данных в тензоре
"""
transform_v2 = v2.ToDtype(torch.float32, scale=True)     # создаем трансформацию (scale - True меняет интервал после преобразования)
img_todtype_v2 = transform_v2(img_toimage_v2)            # преобразуем изображение в тензор


# Normalize
"""
Трансформация Normalize()
    - предназначена для нормирования данных
"""

 # передаем среднее значение для каждого канала и стандартное отклонение для каждого канала
transform_v2 = v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
img_norm_v2 = transform_v2(img_todtype_v2) # в данную трансформацию можно передавать только тензоры типа float


# Compose
"""
Класс Compose необходим для последовательного применения трансформаций
"""

# трансформация для изображений с 3-мя каналами
transform_v2_3color = v2.Compose(
    [
        v2.ToImage(),
        # v2.Grayscale(),  если используем трансформацию в Imagefolder (иначе будет всегда 3 канала цвета)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

# трансформация для изображений с 1-м каналом (Grayscale)
transform_v2_1color = v2.Compose(
    [
        v2.ToImage(),
        # v2.Grayscale(), если используем трансформацию в Imagefolder (иначе будет всегда 3 канала цвета)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,))
    ])

# передаем в трансформацию изображение в формате PIL
img_totensor_norm_v2 = transform_v2_3color(Image.open('/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/transformations/image.JPG'))
# передаем массив данных numpy
img_totensor_norm_2_v2 = transform_v2_3color(image)



######################## Создание собственного класса трансформации ###################################################

class MyTransform(torch.nn.Module):
    """
    Рыба класса трансформации
    """
    def forward(self, sample):
        pass


class MyNormalize(torch.nn.Module):
    """
    Класс нормализации с наследованием
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        sample = (sample - self.mean) / self.std
        
        return sample


class MyNormalize_2:
    """
    Класс нормализации без наследованием
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = (sample - self.mean) / self.std
        
        return sample
    



if __name__ == '__main__':

    plt.axis('off')
    plt.imshow(Image.open('/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/transformations/image.JPG'))
    #plt.show()

    print('Исходный массив')
    print(type(image))
    print(image.shape)                  # не правильное положение цветовых каналов на 2 оси
    print(image.dtype)
    print(f'min = {image.min()}, max = {image.max()}')
    print()

    print("Преобразование массива в тензор")
    print(type(img_totensor))
    print(img_totensor.shape)           # правильное положение цветовых каналов на 0 оси
    print(img_totensor.dtype)
    print(f'min = {img_totensor.min()}, max = {img_totensor.max()}')
    print()

    print('Нормализованный массив')
    print(type(img_norm))
    print(img_norm.shape)
    print(img_norm.dtype)
    print(f'min = {img_norm.min()}, max = {img_norm.max()}')
    print()

    print('Переведен в тензор и нормализованный массив')
    print(type(img_totensor_norm))
    print(img_totensor_norm.shape)
    print(img_totensor_norm.dtype)
    print(f'min = {img_totensor_norm.min()}, max = {img_totensor_norm.max()}')
    print()
    print('Переведен в тензор и нормализованный массив')
    print(type(img_totensor_norm_2))
    print(img_totensor_norm_2.shape)
    print(img_totensor_norm_2.dtype)
    print(f'min = {img_totensor_norm_2.min()}, max = {img_totensor_norm_2.max()}')
    print()

    print("Преобразование массива в тензор V2")
    print(type(img_totensor_v2))
    print(img_totensor_v2.shape)
    print(img_totensor_v2.dtype)
    print(f'min = {img_totensor_v2.min()}, max = {img_totensor_v2.max()}')
    print()

    print("Преобразование массива только в тензор V2")
    print(type(img_toimage_v2))
    print(img_toimage_v2.shape)
    print(img_toimage_v2.dtype)
    print(f'min = {img_toimage_v2.min()}, max = {img_toimage_v2.max()}')
    print()

    print("Изменение типа данных тензора V2")
    print(type(img_todtype_v2))
    print(img_todtype_v2.shape)
    print(img_todtype_v2.dtype)
    print(f'min = {img_todtype_v2.min()}, max = {img_todtype_v2.max()}')
    print()

    print('Нормализованный массив v2')
    print(type(img_norm_v2))
    print(img_norm_v2.shape)
    print(img_norm_v2.dtype)
    print(f'min = {img_norm_v2.min()}, max = {img_norm_v2.max()}')
    print()

    print('Переведен в тензор и нормализованный массив')
    print(type(img_totensor_norm_v2))
    print(img_totensor_norm_v2.shape)
    print(img_totensor_norm_v2.dtype)
    print(f'min = {img_totensor_norm_v2.min()}, max = {img_totensor_norm_v2.max()}')
    print()
    print('Переведен в тензор и нормализованный массив')
    print(type(img_totensor_norm_2_v2))
    print(img_totensor_norm_2_v2.shape)
    print(img_totensor_norm_2_v2.dtype)
    print(f'min = {img_totensor_norm_2_v2.min()}, max = {img_totensor_norm_2_v2.max()}')