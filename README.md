# Neural Style Transfer

The TensorFlow Implementation of *A Neural Algorithm of Artistic Style* (Gatys et al.)

## Introduction

Neural Style Transfer is a deep learning method for image stylization that was introduced in the 2015 paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576.pdf) (Gatys et al.). It takes in a content image and a style image, and generates an output image that looks like the content image has been illustrated in the style of the style image. This is implemented by optimizing the output image to match the content features of the content image and the style features of the style image. These features are extracted from the images using a convolutional neural network.

\

This notebook performs neural style transfer using the code in [this TensorFlow Core Tutorial]([https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en),](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en),) which follows the original neural style transfer algorithm outlined in \<a href="[https://arxiv.org/abs/1508.06576"](https://arxiv.org/abs/1508.06576") class="external"\>A Neural Algorithm of Artistic Style\</a\> (Gatys et al.).

\

I have restructured the TensorFlow Core Tutorial's code to make it easier to interact with. The algorithm takes place in one function below. The function takes in arguments for the path to the style image, the path to the content image, the weight for style loss, and the weights for content loss. This should enable you to engage with neural style transfer by simply calling this function.
