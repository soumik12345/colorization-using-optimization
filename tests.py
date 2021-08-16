from unittest import TestCase

from colorization import Colorizer, IterativeColorizer


class TestColorizer(TestCase):

    def test_output_shape(self):
        colorizer = Colorizer(
            gray_image_file='./data/example.bmp',
            visual_clues_file='./data/example_marked.bmp'
        )
        result = colorizer.colorize()
        assert colorizer.original_gray_image.shape[0] == result.shape[0]
        assert colorizer.original_gray_image.shape[1] == result.shape[1]


class TestIterativeColorizer(TestCase):

    def test_output_shape(self):
        colorizer = IterativeColorizer(
            original_image='./data/example.bmp',
            visual_clues='./data/example_marked.bmp'
        )
        colorizer.colorize(epochs=600, log_interval=100)
        for result in colorizer.result_history:
            assert colorizer.image_original.shape[0] == result.shape[0]
            assert colorizer.image_original.shape[1] == result.shape[1]
