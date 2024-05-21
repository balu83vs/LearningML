from dataset_myselfclass import DatasetReg

from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt  # для отображения картинок

path = '/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/regressions/dataset'


# подключаем файлы из других папок
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../transformations')) # поднимаемся на уровень выше и заходим в папку transformations
from transform import transform_1color, transform_v2_1color


if __name__ == "__main__":

    # без применения трансформаций
    #dataset = DatasetReg(path)

    # с трансформациями
    dataset = DatasetReg(path, transform=transform_v2_1color)

    # размер dataset
    #print(len(dataset))

    # отображаем изображение и красной точкой указываем центр белого квадрата
    #img, coord = dataset[98850]
    #print(f'Координаты центра {coord}')
    #plt.scatter(coord[1], coord[0], marker='o', color = 'red')
    #plt.imshow(img, cmap='gray')
    #plt.show()


    img, coord = dataset[2]

    print('img:')
    print(f'    {type(img)}')
    print(f'    {img.shape}')
    print(f'    {img.dtype}')
    print(f'    min = {img.min()}, max = {img.max()}')
    print('coord:')
    print(f'    {type(coord)}')
    print(f'    {coord.shape}')
    print(f'    {coord.dtype}')

    # разбиваем данные на тренировочные, валидационные и тестовые
    train_set, val_set, test_set = random_split(dataset, [0.7, 0.1, 0.2])
    print(f'Длина тренировочных данных = {len(train_set)}')
    print(f'Длина валидационных данных = {len(val_set)}')
    print(f'Длина тестовых данных = {len(test_set)}')


    # разбиваем данные на батчи нужного размера
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    for i, (samples, target) in enumerate(train_loader):
        if i < 3:
            print(f'Номер batch = {i+1}')
            print(f'    размер samples = {samples.shape}')
            print(f'    размер target = {target.shape}')

    print('\n   .................   \n')
    print(f'Номер batch = {i+1}')
    print(f'    размер samples = {samples.shape}')
    print(f'    размер target = {target.shape}')
