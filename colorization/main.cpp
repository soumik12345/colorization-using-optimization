#include <string>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "colorization.h"


int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cerr << argv[0] << " [input-image] [visual-clues] [result] [gamma] [threshold]" << std::endl;
        return 0;
    }

    double gamma = 2.0;
    if (argc >= 5)
        gamma = std::stod(argv[4]);

    int threshold = 10;
    if (argc >= 6)
        threshold = std::stoi(argv[5]);

    std::string inputImagePath{argv[1]};
    std::string visualCluesPath{argv[2]};
    std::string resultPath{argv[3]};

    cv::Mat image = cv::imread(inputImagePath);
    cv::Mat visual_clues = cv::imread(visualCluesPath);

    if (image.empty()) {
        std::cerr << "Failed to read file from " << inputImagePath << std::endl;
        return 0;
    }

    if (visual_clues.empty()) {
        std::cerr << "Failed to read file from " << visualCluesPath << std::endl;
        return 0;
    }

    assert(image.size() == visual_clues.size());
    cv::Mat mask = colorization::getVisualClueMask(image, visual_clues, threshold);
    cv::Mat colorImage = colorization::colorize(image, visual_clues, mask, gamma);
    cv::imwrite(resultPath, colorImage);
    return 0;
}
