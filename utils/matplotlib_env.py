import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image


def show_one_image_by_matplotlib(image):
    """ show image by matplotlib
    Args:
        image: Tensor or Array or PILImage

    Examples:
        filename = './left.png'
        image = Image.open(filename)
        show_image_by_matplotlib(image)
    """
    plt.figure()
    plt.title("Matplotlib image show") 
    plt.imshow(image) 
    plt.show()

def show_multi_image_by_matplotlib(images,rows,colums):
    """ show images by matplotlib
    Args:
        images: list and every item is Tensor or Array or PILImage
        rows: int
        colums: int
    Examples:
        images = []
        filename = './data/20200915214148920_image1.yuv_disp_color.png'
        images.append(Image.open(filename))
        filename = './data/20200915214148920_image1.yuv_disp_color_ins.png left.png'
        images.append(Image.open(filename))
        show_multi_image_by_matplotlib(images,1,2)
    """
    plt.figure()
    for row in range(rows):
        for col in range(colums):
            plt.subplot(rows,colums,row*colums+col+1)
            plt.imshow(images[row*colums+col])
    plt.show()

if __name__ == "__main__":
    images = []
    filename = './data/20200915214148920_image1.yuv_disp_color.png'
    images.append(Image.open(filename))
    filename = './data/20200915214148920_image1.yuv_disp_color_ins.png'
    images.append(Image.open(filename))
    show_multi_image_by_matplotlib(images,1,2)