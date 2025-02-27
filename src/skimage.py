from skimage.transform import resize

def _interpolate(image, scale_factor, order):
    new_shape = (
        int(image.shape[0] * scale_factor),
        int(image.shape[1] * scale_factor),
        image.shape[2],
    )
    return resize(image, new_shape, order=order, anti_aliasing=False)

def nearest_neighbor(image, scale_factor):
    return _interpolate(image, scale_factor, order=0)
    
def bilinear(image, scale_factor):
    return _interpolate(image, scale_factor, order=1)

def bicubic(image, scale_factor):
    return _interpolate(image, scale_factor, order=3)

