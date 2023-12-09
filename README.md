# Neural Style Transfer

Using The TensorFlow Implementation of *A Neural Algorithm of Artistic Style* (Gatys et al.)

## Introduction

Have you ever wondered what the Mona Lisa would look like if painted by Vincent Van Gogh? Or what you would look like if you were painted by Wassily Kandinsky? These are questions that can both be answered by Neural Style Transfer. Neural Style Transfer is a deep learning method for image stylization that was introduced in the 2015 paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576.pdf) (Gatys et al.). It takes in a content image and a style image, and generates an output image that looks like the content image illustrated in the style of the style image. See the example below.

| Content Image                                        | Style Image                                             | Output Image                                                    |
|------------------------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------|
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
vgg.trainable = False
```

Using a CNN that was trained on such a large dataset is critical to the quality of feature extraction. Since the network is trained on a large number and variety of images, it is more likely to have learned something about the features of those images and is more likely to be able to identify those features in subsequent images that we feed through the network.

## Style and Content Layers

Once we have a pre-trained CNN, we need to identify which convolutional layers can represent the style and content of an image. In general, earlier convolutions are typically able to identify low level features like textures and edges, which are more closely related to the style of an image, and the later convolutions are typically able to identify more high level features like objects, faces, hands, which are more typically related to the content of an image. For Neural Style Transfer, this means that we can typically choose layers that appear later in the network to represent content, and layers that appear earlier in the network to represent style.

<p style="text-align:center;">

<img src="Supplemental Images/VGG19 style and content.png" width="500"/>

</p>

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

Style loss is computed differently from content loss. For style loss, we don't want to take the difference between the activations in the style layers because then the output image will look very similar to the style image. Instead, we want to look at the relationships between the feature maps in each style layer and takne the difference between these relationships. We want the relationships between feature maps to be the same for when the style image and the output image are fed through VGG 19.

To look at these relationships mathematically we do the same thing as we did for content loss where we transform all the feature maps in a certain layer into these matrices and we get something called the Gram matrix by multiplying this matrix by its transpose. Now remember that every row in this matrix represents a feature map.

So by multiplying this matrix by its transpose we're essentially multiplying each feature map by each other feature map and getting some sort of an, an understanding of the relationship between these feature maps. By then taking the difference between these gram matrices we can ensure that for each style layer the relationship.

Between every single feature map in those style layers is as similar as possible for both the style image and the output image. That's how you get the style loss formula. Total loss is an awaited sum of the style loss and the content loss, and that's what you minimize to get an output image that has the content of the content image but the style of the style image.

You can also play with the weights on the style loss and the content loss, depending on whether you want to incorporate more of the content or more of the style into the output image. This figure from the original paper on neural style transfer demonstrates the impact of the styling content weights really well.

On the Xaxis is the content to style ratio. On the left hand side, the content to style ratio is much lower, so the style comes through much more in the output image. But on the right hand side, the content to style ratio is higher, so the content comes through more in the output image.

The Y axis also demonstrates something interesting that we talked about earlier. It demonstrates the difference between high level and low level features. On the Y axis are the different style layers that they included in the model when they created these output images. At the top, only style layers from earlier convolutions are included, so the images focus more on low level features, but at the bottom, the images were generated using style layers from both earlier and later convolutions, and so you can see more high level features in the generated images.

Now that you know how neural style transfer works, here are some examples of the images that I generated.
