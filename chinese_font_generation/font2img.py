# Generate the dataset
import os
from PIL import Image, ImageDraw, ImageFont


# generate image from a given word by using the font file
def generate_img(
    word="的",
    font_path="font_files/MSYHBD.TTF",
    font_size=64,
    img_size=[64, 64],
):
    w = img_size[0]
    h = img_size[1]
    background_color = 255
    img = Image.new("L", (w, h), background_color)

    font = ImageFont.truetype(font_path, font_size)

    word_position = (0, 0)
    word_color = 0
    draw = ImageDraw.Draw(img)
    draw.text(word_position, word, font=font, fill=word_color)

    return img


def generate_data(
    data_path="data/YingZhangXingShu",
    font_path="font_files/YingZhangXingShu.ttf",
    font_size=64,
    img_size=[64, 64],
    words_path="font_files/3500.txt",
):
    # create data folder
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # generate images
    words = open(words_path, "r", encoding="utf-8").readlines()[0]
    num_samples = len(words)
    for i in range(num_samples):
        word = words[i].strip()
        img_name = str(i) + ".jpg"
        save_path = os.path.join(data_path, img_name)
        img = generate_img(word, font_path, font_size, img_size)
        # save image
        img.save(save_path)


if __name__ == "__main__":
    generate_data()