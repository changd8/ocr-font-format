import pygame as pg
import json
import string


def ocr_dataset(text_corpus):
    pg.init()
    pg.font.init()

    fonts = {'calibri', 'arial', 'timesnewroman'}
    font_style = {'normal': (False, False), 'bold': (True, False), 'italic': (False, True)}
    style_labels = {'normal': 0, 'bold': 1, 'italic': 2}
    words_data = list(string.ascii_lowercase)
    words_data.extend(list(string.ascii_uppercase))

    with open(text_corpus) as f:
        lines = f.readlines()

    for line in lines[:100]:
        tokens = line.split()
        for token in tokens:
            words_data.append(token)

    dataset = []
    for i, token in enumerate(words_data):
        # save dataset splits after 1000 words
        if i == 1000:
            train_output_file = './data/train.json'
            train_split = dataset[:7200]
            with open(train_output_file, 'w') as output_json:
                json.dump(train_split, output_json)

            val_output_file = './data/val.json'
            val_split = dataset[7200:8100]
            with open(val_output_file, 'w') as output_json:
                json.dump(val_split, output_json)

            test_output_file = './data/test.json'
            test_split = dataset[8100:]
            with open(test_output_file, 'w') as output_json:
                json.dump(test_split, output_json)

            print(len(train_split), len(val_split), len(test_split))
            return

        if i % 50 == 0:
            print(i)

        for font in fonts:
            for style, (bold, italic) in font_style.items():
                pg_font = pg.font.SysFont(font, 80, bold=bold, italic=italic)

                # create image of the word of this style
                text = pg_font.render(token, False, (0, 0, 0))
                rect = text.get_rect()
                pg.draw.rect(text, (0, 0, 0), rect, 1)
                screen = pg.display.set_mode((rect[2], rect[3]))
                screen.fill((255, 255, 255))
                screen.blit(text, (0, 0))
                image_id = 'word_{}_{}_{}.png'.format(i, font, style)
                pg.image.save(screen, './data/images/{}'.format(image_id))
                datum = {'word_id': i,
                         'font': font,
                         'word': token,
                         'style': style,
                         'img_id': image_id,
                         'label': style_labels[style]}
                dataset.append(datum)

# https://sourceforge.net/p/apertium/svn/62515/tree/incubator/apertium-eng-kaz/dev/news.en
ocr_dataset('./data/text-corpus.txt')

