import numpy as np

def _random_sample(img_shape=(64, 64)):
    '''
    Generate an image with black background and two white points on it. Image
    contains only one channel.
    
    Random coordinates are taken from an uniform distribution.
    
    # Arguments
        output_dim: A tuple of two integers which denote height and width of an
           output image.
    
    # Returns
        An image of three dimensions (img_shape[0], img_shape[1], 1). The last
        dimension is added to allow image to be an input of convolutional layers.
    '''
    img = np.zeros(img_shape, dtype=np.float32)
    coords = []

    for _ in range(2):
        y, x = (np.random.randint(dim) for dim in img_shape)
        coords.append(np.array([y, x]))
        img[y, x] = 1.0

    label = np.linalg.norm(coords[0] - coords[1])
    
    return img[:, :, np.newaxis], label

def generate_data(dim, size):
    '''
    Generate a dataset where:
    * Inputs are black images (grayscale) with two white points, and
    * Labels are Euclidean distance between these two points.

    # Arguments
        dim: Two dimensions of images to generate (height, width).
        size: A size of dataset to generate.

    # Returns
        A tuple of two numpy arrays where the first one is network inputs array,
        and the second one is a labels array.
    '''
    X, y = (np.array(arr) for arr in zip(*[_random_sample(dim) for _ in range(size)]))

    return X, y

def augment_with_mesh(X):
    '''
    Augment the network input data with a mesh grid of x and y coordinates.

    # Arguments
        X: Original input data.

    # Returns
        A new numpy array augmented along the last axis with mesh grid of x and y
        coordinates. Dimensionality transformation is next: input array
        (num_samples, height, width, 1) -> (num_samples, height, width, 3).
    '''
    assert X.shape[-1] == 1, 'The last dimension is expected to be 1, bu actually {}'.format(X.shape[-1])

    xs = np.linspace(-0.5, 0.5, X.shape[2])
    ys = np.linspace(-0.5, 0.5, X.shape[1])
    xv, yv = np.meshgrid(xs, ys)
    xv, yv = xv[:, :, np.newaxis], yv[:, :, np.newaxis]
    mesh = np.concatenate((xv, yv), axis=2)[np.newaxis, :, :, :]
    mesh = np.tile(mesh, (X.shape[0], 1, 1, 1))
    print(mesh.shape)
    print(X.shape)

    return np.concatenate((X, mesh), axis=3)

def flatten(X):
    '''Flatten an image to ease use as an input to a dense layer.
    
    Shape transformation: (num_samples, height, width, N) -> (num_samples, height * width * N),
    where N is a number of channels.

    # Arguments
        X: A dataset to transform.

    # Returns
        A dataset of two dimensions.
    '''
    return np.array([x.ravel() for x in X])

def load_dataset(input_and_labels_files):
    '''Load the network input and labels from a given iterable of
    file names.
    
    # Arguments
        input_and_labels_files: A tuple of two string, first of which is a file
            where a network input array is stored, and the second one is a
            labels array file name.
    
    # Returns
        A generator of two numpy arrays--network inputs and labels.
    '''
    return (np.load(path) for path in input_and_labels_files)

def dump_dataset(arrays, files):
    '''Dump network inputs and labels arrays to given files.
    
    # Arguments
        arrays: A tuple of two numpy arrays where the first one is network
            inputs array and the second one is labels array.
    '''
    for array, file_name in zip(arrays, files):
        np.save(file_name, array)
