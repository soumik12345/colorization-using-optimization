import cv2
import click

from colorization import Colorizer, IterativeColorizer


@click.command()
@click.option('--original_image', help='Original Image Path')
@click.option('--visual_clue', help='Visual Clue Image Path')
@click.option('--result_path', default='./result', help='Colorized Image Path (without file extensions)')
@click.option('--use_itercative', '-i', is_flag=True, help='Use Iterative Mode')
@click.option('--epochs', default=500, help='Number of epochs for Iterative Mode')
@click.option('--log_intervals', default=100, help='Log Interval')
def colorize(original_image, visual_clue, result_path, use_itercative, epochs, log_intervals):
    if use_itercative:
        colorizer = Colorizer(
            gray_image_file=original_image,
            visual_clues_file=visual_clue
        )
        colorizer.plot_inputs()
        result = colorizer.colorize()
        colorizer.plot_results(result)
        cv2.imwrite(
            result_path + '.png',
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        )
    else:
        colorizer = IterativeColorizer(
            original_image=original_image,
            visual_clues=visual_clue
        )
        colorizer.plot_inputs()
        colorizer.colorize(
            epochs=epochs, log_interval=log_intervals
        )
        colorizer.plot_results(log_intervals=log_intervals)
        for i, result in enumerate(colorizer.result_history):
            cv2.imwrite(
                result_path + '{}.png'.format(i + 1),
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            )


if __name__ == '__main__':
    colorize()
