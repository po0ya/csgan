# Copyright 2018 The CSGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the super class for all dataset classes. LazyDataset is also defined here."""

import os
import numpy as np
import scipy
import tensorflow as tf
import scipy.misc


class Dataset(object):
    """The base class for handling datasets.

    Attributes:
            name: Name of the dataset.
            data_dir: The directory where the dataset resides.
    """

    def __init__(self, name):
        """The datsaet default constructor.

            Args:
                name: A string, name of the dataset.     
        """

        self.data_dir = os.path.join("./data", name)
        self.name = name

    def load(self, split='train'):
        """ Abstract function specific to each dataset."""

        raise NotImplementedError

    def transform(self, images, int_type=None):
        """Normalizes the input to either [0, 1] or [-1, 1].

        Args:
            images: The input images of type np.ndarray of type np.float32.
            int_type: Type of transformation: 0 -> [-1, 1], and 1 -> [0, 1]

        Returns:
            transformed_images: The normalized images of type np.ndarray
        """

        if int_type is None:
            flags = tf.app.flags.FLAGS
            int_type = flags.input_transform_type

        if int_type == 0:
            transformed_images = images / 127.5 - 1
        else:
            transformed_images = images / 255.

        return transformed_images

    def inv_transform(self, transformed_images, int_type=None):
        """Inverse-transform to bring back the pixel values to [0, 255].

        Args:
            transformed_images: The transformed input images of type np.ndarray of type np.float32.
            int_type: Type of transformation: 0 -> [-1, 1], and 1 -> [0, 1]

        Returns:
            original_images: The recovered images in [0, 255] of type np.ndarray.
        """

        if int_type is None:
            f = tf.app.flags.FLAGS
            int_type = f.input_transform_type

        if int_type == 0:
            original_images = (transformed_images + 1.0) * 127.5
        else:
            original_images = transformed_images * 255.0
        original_images[original_images < 0.0] = 0.0
        original_images[original_images > 255] = 255

        return original_images


class LazyDataset(object):
    """The Lazy Dataset class.
    Instead of loading the whole dataset into memory, this class loads images only when their index is accessed.

        Attributes:
            fps: String list of file paths.
            center_crop_dim: An integer for the size of center crop (after loading the images).
            resize_size: The final resize size (after loading the images).
    """

    def __init__(self, fps, center_crop_dim, resize_size):
        """LazyDataset constructor.

        Args:
            fps: File paths.
            center_crop_dim: The dimension of the center cropped square.
            resize_size: Final size to resize the center crop of the images.
        """

        self.fps = fps
        self.center_crop_dim = center_crop_dim
        self.resize_size = resize_size

    def get_image(self, image_path, input_height, input_width, resize_height=64, resize_width=64, is_crop=True,
                  is_grayscale=False):
        """Retrieves an image at a given path and resizes it to the specified size.

        Args:
            image_path: Path to image.
            input_height: Original image height.
            input_width: Original image width.
            resize_height: Height to resize to.
            resize_width: Width to resize to.
            is_crop: If True, center-crop the image.
            is_grayscale: If True, load grayscale image.

        Returns:
            Loaded and transformed image.
        """

        # Read image at image_path.
        image = self.imread(image_path, is_grayscale)

        def transform(image, crop_height, crop_width, resize_height=64, resize_width=64, is_crop=True):
            """Transforms an image by first applying an optional center crop, then resizing it.

            Args:
                image: Input image.
                crop_height: The height of the crop.
                crop_width: The width of the crop.
                resize_height: The resize height after cropping.
                resize_width: The resize width after cropping.
                is_crop: If True, first apply a center crop.

            Returns:
                The cropped and resized image.
            """

            def center_crop(image, crop_h, crop_w, resize_h=64, resize_w=64):
                """Performs a center crop followed by a resize.

                Args:
                    image: Image of type np.ndarray
                    crop_h: The height of the crop.
                    crop_w: The width of the crop.
                    resize_h: The resize height after cropping.
                    resize_w: The resize width after cropping.

                Returns:
                    The cropped and resized image of type np.ndarray.
                """
                if crop_w is None:
                    crop_w = crop_h
                h, w = image.shape[:2]
                j = int(round((h - crop_h) / 2.))
                i = int(round((w - crop_w) / 2.))
                # Crop then resize.
                return scipy.misc.imresize(image[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])

            # Optionally crop the image. Then resize it.
            if is_crop:
                cropped_image = center_crop(image, crop_height, crop_width, resize_height, resize_width)
            else:
                cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
            return np.array(cropped_image)

        # Return transformed image.
        return transform(image, input_height, input_width, resize_height=resize_height, resize_width=resize_width,
                         is_crop=is_crop)

    def imread(self, path, is_grayscale=False):
        """Reads an image given a path.

        Args:
            path: Path to image.
            is_grayscale: If True, will convert images to grayscale.

        Returns:
            Loaded image.
        """

        if is_grayscale:
            return scipy.misc.imread(path, flatten=True).astype(np.float)
        else:
            return scipy.misc.imread(path).astype(np.float)

    def __len__(self):
        """Gives the number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """

        return len(self.fps)

    def __getitem__(self, index):
        """Loads and returns images specified by index.

        Args:
            index: Indices of images to load.

        Returns:
            Loaded images.

        Raises:
            TypeError: If index is neither of: int, slice, np.ndarray.
        """

        # Case of a single integer index.
        if isinstance(index, int):
            return self.get_image(self.fps[index], self.center_crop_dim, self.center_crop_dim,
                                  resize_width=self.resize_size, resize_height=self.resize_size)
        # Case of a slice or array of indices.
        elif isinstance(index, slice) or isinstance(index, np.ndarray):
            if isinstance(index, slice):
                if index.start is None:
                    index = range(index.stop)
                elif index.step is None:
                    index = range(index.start, index.stop)
                else:
                    index = range(index.start, index.stop, index.step)
            return np.array([self.get_image(self.fps[i], self.center_crop_dim, self.center_crop_dim,
                                            resize_height=self.resize_size, resize_width=self.resize_size)
                             for i in index])
        else:
            raise TypeError("Index must be an integer or a slice.")

    def get_subset(self, indices):
        """Gets a subset of the images

        Args:
            indices: The indices of the images that are needed. It's like lazy indexing without loading. 

        Raises:
            TypeError if index is not a slice.
        """
        if isinstance(indices, int):
            self.fps = self.fps[indices]
        elif isinstance(indices, slice) or isinstance(indices, np.ndarray):
            self.fps = [self.fps[i] for i in indices]
        else:
            raise TypeError("Index must be an integer or a slice.")
