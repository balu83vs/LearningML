from torchvision.datasets import ImageFolder # готовый класс для подготовки dataset
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt


train_path = '/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/classifications/mnist/training'
test_path = '/home/vitaliisokolov/py_learning_linux/LearningML/PyTorch/classifications/mnist/testing'

train_data = ImageFolder(root=train_path)
test_data = ImageFolder(root=test_path)

# проверяем свойства classes и class_to_idx
    #print(train_data.classes)
    #print(train_data.class_to_idx)

# проверка соответствия картинки классу
img, one_hot_position = train_data[2564]

cls = train_data.classes[one_hot_position]

print(f'Класс - {cls}')
plt.imshow(img, cmap='gray')


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