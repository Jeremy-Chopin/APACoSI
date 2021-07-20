import numpy as np
import os
from skimage.segmentation import mark_boundaries
from PIL import Image


class Slicer3D:
    def __init__(self, initial_image = None, segmentation_map = None, manual_segmentation = None, slices_path = None, classes = None, step = 1):
        self.inital_image = initial_image
        self.segmentation_map = segmentation_map
        self.manual_segmentation = manual_segmentation
        self.slices_path = slices_path
        self.classes=classes
        self.step =step

    def set_initial_image(self, inital_image):
        self.inital_image = initial_image

    def set_segmentation_map(self, segmentation_map):
        self.segmentation_map = segmentation_map 
    
    def set_slices_path(self, slices_path):
        self.slices_path = slices_path

    def set_step(self, step):
        self.step = step
    
    def set_classes(self, step):
        self.classes = classes

    def export_slices(self):
        x,y,z = self.inital_image.shape
        index = 0

        slice_dir_path = os.path.join(self.slices_path, "slices")
        
        if os.path.isdir(slice_dir_path) == False:
            os.mkdir(slice_dir_path)

        for i in range(0, z, self.step):
            slice_img = self.inital_image[:,:,i]
            
            im = Image.fromarray(np.uint8((slice_img)))

            save_path = os.path.join(slice_dir_path, str(index) + '.png')
            im.save(save_path)

            index += self.step

    def export_slices_multi_class(self):
        x,y,z = self.inital_image.shape

        if self.classes:
            nb_classes = len(self.classes)
        else:
            nb_classes = len(np.unique(self.segmentation_map))

        index = 0

        slice_dir_path = os.path.join(self.slices_path, "slices_multi")
        
        if os.path.isdir(slice_dir_path) == False:
            os.mkdir(slice_dir_path)

        for c in range(0, nb_classes):

            if self.classes:
                slice_class_dir_path = os.path.join(slice_dir_path, self.classes[c])
            else:
                slice_class_dir_path = os.path.join(slice_dir_path, str(c))

            if os.path.isdir(slice_dir_path) == False:
                os.mkdir(slice_class_dir_path)

            slice_img = np.where(slice_img == c, 1, 0)
            
            for i in range(0, z, self.step):
                slice_img = self.inital_image[:,:,i]
                
                im = Image.fromarray(np.uint8((slice_img)))

                save_path = os.path.join(self.slice_class_dir_path, str(index) + '.png')
                im.save(save_path)

                index += step 

    def export_slices_boundaries(self, dir_name):
        x,y,z = self.inital_image.shape

        if self.classes:
            nb_classes = len(self.classes)
        else:
            nb_classes = len(np.unique(self.manual_segmentation))

        slice_dir_path = os.path.join(self.slices_path, dir_name)
        
        if os.path.isdir(slice_dir_path) == False:
            os.mkdir(slice_dir_path)

        for c in range(0, nb_classes):

            if self.classes:
                slice_class_dir_path = os.path.join(slice_dir_path, self.classes[c])
            else:
                slice_class_dir_path = os.path.join(slice_dir_path, str(c))

            if os.path.isdir(slice_class_dir_path) == False:
                os.mkdir(slice_class_dir_path)

            seg_class = np.where(self.segmentation_map == c, 1, 0)

            seg_manual_class = np.where(self.manual_segmentation == c, 1, 0)

            index = 0
            for i in range(0, y, self.step):
                slice_img = self.inital_image[:,i,:]

                slice_seg_class = seg_class[:,i,:]

                slice_seg_manual_class = seg_manual_class[:,i,:]
                
                boundaries = mark_boundaries(slice_img, slice_seg_class, color=(1,0,1))
                boundaries = mark_boundaries(boundaries, slice_seg_manual_class, color=(1,1,0))

                im = Image.fromarray(np.uint8(boundaries*255))

                save_path = os.path.join(slice_class_dir_path, str(index) +'.png')

                im = im.rotate(90)
                im.save(save_path)

                index += self.step