# Neural Style Transfer

Using The TensorFlow Implementation of *A Neural Algorithm of Artistic Style* (Gatys et al.)

## Introduction

Have you ever wondered what the Mona Lisa would look like if painted by Vincent Van Gogh? Or what you would look like if you were painted by Wassily Kandinsky? These are questions that can both be answered by Neural Style Transfer. Neural Style Transfer is a deep learning method for image stylization that was introduced in the 2015 paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576.pdf) (Gatys et al.). It takes in a content image and a style image, and generates an output image that looks like the content image illustrated in the style of the style image. See the example below.

| Content Image                                        | Style Image                                             | Output Image                                                    |
|----------------------|-----------------------|---------------------------|
| <img src="Source Images/mona lisa.jpg" width="300"/> | <img src="Source Images/starry night.jpg" width="400"/> | <img src="Example Outputs/starry lisa output.png" width="300"/> |

This kind of image stylization can be used with any content or style images, including photographs, and has many applications in art, film, and social media. Neural Style Transfer was the first algorithm to successfully accomplish this task. Although many algorithms have come after that produce higher quality outputs, Neural Style Transfer is still the simplest and the fastest to implement.

Neural Style Transfer is implemented by optimizing the output image to match the content features of the content image and the style features of the style image. These content and style features are extracted from the images using a convolutional neural network.

This repository implements the [TensorFlow Core Tutorial](%5Bhttps://www.tensorflow.org/tutorials/generative/style_transfer?hl=en) for neural style transfer, which follows the algorithm outlined in the original [paper](https://arxiv.org/pdf/1508.06576.pdf). In the jupyter notebook, I have restructured the tutorial's code so that the entire neural style transfer algorithm can be run with a single function call.

In this blog post, I will explain the theory behind neural style transfer, including the network architecture and the loss function.

## Network Architecture: Pre-trained CNN

<img src="Supplemental Images/VGG19 CNN.png" width="500"/>

Neural Style Transfer uses a pre-trained convolutional neural network (CNN) to identify and extract style and content features from the style and content images. The TensorFlow Core implementation uses VGG19, which is a 19-layer CNN that was trained to perform 1000-class image classification on the ImageNet dataset. VGG19 has 16 convolutional layers and 3 fully-connected layers. For neural style transfer, we only care about the convolutional layers and can ignore the three fully connected years.

```         
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
```

Using a CNN that was trained on such a large dataset is critical to the quality of feature extraction. Since the network is trained on a large number and variety of images, it is more likely to have learned something about the features of those images and is more likely to be able to identify those features in subsequent images that we feed through the network.

## Style and Content Layers

Once we have a pre-trained CNN, we need to identify which convolutional layers can represent the style and content of an image. In general, earlier convolutions are typically able to identify low level features like textures and edges, which are more closely related to the style of an image, and the later convolutions are typically able to identify more high level features like objects, faces, hands, which are more typically related to the content of an image. For Neural Style Transfer, this means that we can typically choose layers that appear later in the network to represent content, and layers that appear earlier in the network to represent style.

<img src="Supplemental Images/VGG19 style and content.png" width="500"/>

In the Tensorflow implementation, they chose five layers to represent style and one layer to represent content. These layers are highlighted in the figure above. As you can see, they chose one of the last layers to be the content layer, and they chose style layers throughout to pick up both low level and high level style features. Later, we will see why it's important to have multiple style layers from throughout the network.

```         
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
```

## Content Loss

Now that we know how style and content layers are chosen, we need to understand how those layers are used to preserve style and content features in the output image. To understand how content features are preserved, we can think about it this way:

1.  The content image is fed through VGG19.
2.  The activations of the content layer (in response to the content image) are set as the target activations for the content layer.
3.  The pixels of the output image are optimized so that when the output image is fed through VGG19, the activations of the content layer (in response to it) are very similar to the target activations.

<img src="Supplemental Images/Content Loss.png"/>

Mathematically, we preserve content features by minimizing the difference between the activations of the content layer for the content image and output image. The chosen content layer has 512 feature maps, and the feature maps are all of size 14 by 14. To calculate the difference between all of these values, the algorithm turns each 14 by 14 feature map into a vector of length 196, stacks these 512 vectors into a matrix, and takes the differences between these matrices. That's how we get the formula for content loss.

## Style Loss

Style loss is computed differently from content loss. For style loss, we don't want to take the difference between the activations in the style layers because then the output image will look very similar to the style image. Instead, we want to look at the relationships between the feature maps in each style layer and take the difference between these relationships. We want the relationships between feature maps within style layers to be the same for the style image and the output image.

<img src="Supplemental Images/Style Loss.png"/>

Mathematically, we do the same thing as we did for content loss where we transform all the feature maps in a certain layer into matrices. Unlike content loss, however, we are then interested in calculating the Gram matrix by multiplying the matrices by their transpose. To understand why, remember that every row in this matrix represents a feature map. By multiplying the matrix by its transpose, we get a resulting 512 by 512 matrix where the (i,j)-th entry is the product of the i-th and j-th feature maps. In essence, we multiply each feature map by every other feature map in a particular layer. The resulting Gram matrix then provides an abstract representation of the relationship between every pair of feature maps. By taking the difference between the Gram matrices of the style image and output image, we can ensure that the relationships between feature maps within each style layers is as similar as possible for both the style image and the output image. That's how we get the style loss formula.

## Total Loss and "Training"

Total loss is a weighted sum of the style loss and the content loss: $Loss =\alpha*\mathcal{L}_{s}+\beta*\mathcal{L}_{c}$.

This is minimized during each so-called training step. Note that there is no model being trained. Instead, you could say that it's the output image that is being trained, since its pixel values change to minimize this loss. In the TensorFlow implementation, the output image is initialized to the content image. Then, training is run to impose the style on the output image. We can see how style is gradually imposed on the output image in the figure below.

<img src="Example Outputs/varying training steps.png"/>

## Style and Content Weights

You can also play with the weights on style loss and content loss in the total loss function, depending on whether you want to incorporate more of the content or more of the style into the output image.

<img src="Example Outputs/varying style:content ratios.png"/>

The figure above demonstrates the effect of varying the style and content weights. The x-axis represents the style-to-content ratio ($\frac{\alpha}{\beta}$). On the left, the style to content ratio is low, so the content is more prominent in the output image. On the right, the style to content ratio is high, so the style comes through more in the output image.

## Low-Level and High-Level Features

This algorithm can also demonstrate the location of high-level and low-level features in the network by using different convolutional layers as style layers.

<img src="Example Outputs/varying style layers.png"/>

In the figure above, the x-axis represents the different style layers that were used to generate the output image. On the left, only style layers from earlier convolutions were used, meaning that the outputs display more low-level style features. Notice in these images how there are more solid colors and defined edges. On the right, the images were generated using style layers from both earlier and later convolutions, meaning that there are more high-level style features in the generated images. Notice how you can now see Van Gogh's notorious wavy lines. In order to capture the entirety of the style, it appears to be important to use style layers from across the network, which is why the authors of the TensorFlow implementation chose to use the five style layers on the right. They are evenly dispersed throughout the network and successfully capture both low- and high-level style features.

## Personal Examples

Now that you know how neural style transfer works under the hood, here are some examples of the images that I generated.

<img src="Example Outputs/fun example.png"/>

Feel free to try it yourself using the jupyter notebook I have prepared. Have fun!
