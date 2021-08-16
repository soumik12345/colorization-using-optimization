from unittest import TestCase

from colorization import Colorizer, IterativeColorizer


class TestColorizer(TestCase):

    def test_output_shape(self):
        colorizer = Colorizer(
            gray_image_file='./data/original/example.png',
            visual_clues_file='./data/visual-clues/example.png'
        )
        result = colorizer.colorize()
        assert colorizer.original_gray_image.shape[0] == result.shape[0]
        assert colorizer.original_gray_image.shape[1] == result.shape[1]


class TestIterativeColorizer(TestCase):

    def test_output_shape(self):
        colorizer = IterativeColorizer(
            original_image='./data/original/example.png',
            visual_clues='./data/visual-clues/example.png'
        )
        colorizer.colorize(epochs=600, log_interval=100)
        for result in colorizer.result_history:
            assert colorizer.image_original.shape[0] == result.shape[0]
            assert colorizer.image_original.shape[1] == result.shape[1]
