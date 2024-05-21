import matplotlib.pyplot as plt # для отображения картинок

from dataset_mnistdataset import MNISTDataset
from torch.utils.data import DataLoader, random_split

# подключаем файлы из других папок
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../transformations')) # поднимаемся на уровень выше и заходим в папку transformations
from transform import transform_1color, transform_v2_1color


if __name__ == "__main__":

    train_path = '/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/classifications/mnist/training'
    test_path = '/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/classifications/mnist/testing'

    
    """
    # без трансформаций
    train_data = MNISTDataset(train_path)
    test_data = MNISTDataset(test_path)
    """

    # с трансформациями
    train_data = MNISTDataset(train_path, transform=transform_v2_1color)
    test_data = MNISTDataset(test_path, transform=transform_v2_1color)
    

    # проверяем свойства classes и class_to_idx
    #print(train_data.classes)
    #print(train_data.class_to_idx)

    # проверка метода getitem
    #print(train_data[2564])


    # проверка соответствия картинки классу
    """
    img, one_hot_position = train_data[2564]

    cls = train_data.classes[one_hot_position]

    print(f'Класс - {cls}')
    plt.imshow(img, cmap='gray')
    plt.show()
    """
    img, cls = test_data[2]

    print('img:')
    print(f'    {type(img)}')
    print(f'    {img.shape}')
    print(f'    {img.dtype}')
    print(f'    min = {img.min()}, max = {img.max()}')
    print('cls:')
    print(f'    {cls}')


    # разделение тренировочных данных на тренировочные и валидационные
    train_data, val_data = random_split(train_data, [0.8, 0.2])

    print(f'Длина тренировочных данных = {len(train_data)}')
    print(f'Длина валидационных данных = {len(val_data)}')
    print(f'Длина тестовых данных = {len(test_data)}')


    # разбиваем данные на batch
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    