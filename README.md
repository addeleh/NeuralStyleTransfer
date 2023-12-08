# Neural Style Transfer

The TensorFlow Implementation of *A Neural Algorithm of Artistic Style* (Gatys et al.)

## Introduction

Have you ever wondered what the Mona Lisa would look like if painted by Vincent Van Gogh? Or what you would look like if you were painted by Wassily Kandinsky? These are questions that can both be answered by Neural Style Transfer. Neural Style Transfer is a deep learning method for image stylization that was introduced in the 2015 paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576.pdf) (Gatys et al.). It takes in a content image and a style image, and generates an output image that looks like the content image illustrated in the style of the style image. See the example below.

| Content Image                        | Style Image                                          | Output Image                                      |
|-----------------------|---------------------------|-----------------------|
| ![](Source%20Images/mona%20lisa.jpg) | ![](Source%20Images/starry%20night.jpg){width="374"} | ![](Example%20Outputs/starry%20lisa%20output.png) |

This kind of image stylization can be used with any content or style images, including photographs, and has many applications in art, film, and social media. Neural Style Transfer was the first algorithm to successfully accomplish this task. Although many algorithms have come after that produce higher quality outputs, Neural Style Transfer is still the simplest and the fastest to implement.

Neural Style Transfer is implemented by optimizing the output image to match the content features of the content image and the style features of the style image. These content and style features are extracted from the images using a convolutional neural network.

This repository implements the [TensorFlow Core Tutorial](%5Bhttps://www.tensorflow.org/tutorials/generative/style_transfer?hl=en) for neural style transfer, which follows the algorithm outlined in the original [paper](https://arxiv.org/pdf/1508.06576.pdf). In the jupyter notebook, I have restructured the tutorial's code so that the entire neural style transfer algorithm can be run with a single function call.

In this blog post, I will explain the theory behind neural style transfer, including the network architecture and the loss function.

## Network Architecture

Neural Style Transfer uses a pre-trained convolutional neural network (CNN) to identify and extract style and content features. In the TensorFlow Core implementation, they use the VGG19 model which is a 19-layer CNN that was trained to perform 1000-class image classification on the ImageNet dataset. VGG19 has 16 convolutional layers and 3 fully-connected layers; Neural Style Transfer only needs the convolutional layers.

Using a CNN that was trained on such a large dataset is critical to the quality of feature extraction. Intuitively, since the network is trained on a large number and variety of images, it is more likely to have learned something about the low-level and high-level features of those images and is more likely to be able to identify similar features in other images.
