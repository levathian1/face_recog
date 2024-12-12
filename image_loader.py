from PIL import Image

def load_img(path):
    """
        Loads an image and converts if not already in RGB format

        Parameter:
            path: Image path

        Return:
            PIL Image
    """
    loaded_img = Image.open(path)
    return loaded_img.convert('RGB')