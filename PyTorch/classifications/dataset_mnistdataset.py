from torch.utils.data import Dataset
##############################################################

import os
import numpy as np

from PIL import Image           # для загрузки изображений
##############################################################

class MNISTDataset(Dataset):
    """
    Класс формирования dataset
    """

    def __init__(self, path: str, transform=None):
        self.path = path                # путь хранения данных 
        self.transform = transform      # трансформации

        self.len_dataset = 0            # размер dataset
        self.data_list = [] 

        # формируем данные о папках и файлах(путь к папке, список папок, список файлов) внутри папки (training и testing)
        for path_dir, dir_list, file_list in os.walk(self.path):
            if path_dir == self.path:                                        # если папка корневая
                self.classes = sorted(dir_list)                                 # локальная переменная со списком папок    
                # формируем словарь, в котором каждой папке сопоставлена ее позиция в one_hot векторе                   
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes) 
                }
                continue

            cls = path_dir.split('/')[-1]                                   # формируем имя вложенных папок (class_0, class_1 ...)

            # проходим по всем файлам    
            for name_file in file_list:                                     
                file_path = os.path.join(path_dir, name_file)               # путь до каждого файла 
                self.data_list.append((file_path, self.class_to_idx[cls]))  # кортеж (путь к файлу + позиция в one_hot векторе)

            self.len_dataset += len(file_list)
            

    def __len__(self):
        return self.len_dataset


    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        #sample = np.array(Image.open(file_path))        # картинка представленная в виде numpy массива
        sample = Image.open(file_path)                   # картинка представленная в виде PIL изображения (лучше для трансформаций)

        if self.transform is not None:                  # если заданы трансформации
            sample = self.transform(sample)             # применяем к изображению
                                    
        return sample, target                           # возвращаем изображение и кортеж (путь к файлу + позиция в one_hot векторе) 