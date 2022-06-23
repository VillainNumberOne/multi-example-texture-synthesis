<div align="center">

# `MExTS`
`multi example-based texture synthesis`

A package that allows to generate textures of arbitrary size with the ability to edit stylistic attributes in a meaningfull way

<img src="https://sun9-34.userapi.com/impf/ojZjce-0oA6ba-Ah5M1SlBd17OOTPeiYDLvkNg/vMigSPLP66c.jpg?size=2560x1298&quality=96&sign=08d5dc66d3a29e2f900f29ad769d569f&type=album" width="1000">

</div>

### Installation

    pip install mexts

## Features 

### 1. Simple generation

Generate arbitrary-sized textures from a single example.

The image below shows a comparison between real and generated textures. Top row - real; bottom row - generated.

<div align="center">
<img src="https://sun9-77.userapi.com/impf/ygYmk7VmXMGtA118978XzGuaS5snUfsVP0_dnw/awy8SWQ6BNA.jpg?size=2560x640&quality=96&sign=7393451b5e6616b2fb8fab8974a77d19&type=album" width="1000">
</div>

The texture synthesis algorithm is hybrid: it combines iterative optimization [1] with MSInit [2] and AdaIN-autoencoder [3]. The algorithm allows to reach a compromise between quality and time, so it is possible to generate good-quality textures as well as rough-quality previews of final results.

<div align="center">
<img src="https://sun9-55.userapi.com/impf/tjwDKTFwDFST5SCj5vXFYFUv28quVe3xuY62Ag/-JiBXQG9g7I.jpg?size=1461x519&quality=96&sign=6defecd7748ed514ec290cd91d217ba2&type=album" width="500">
</div>

Initialization of texture generator:
```python
from mexts.feature_extractor import FeatureExtractor
from mexts.adain_autoencoder import AdaINAutoencoder
from mexts.gen import TextureGen
import torch

device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
FE = FeatureExtractor().to(device)
AA = AdaINAutoencoder().to(device)
TG = TextureGen(FE, AA, device)
```

Loading real image and obtaining it's style reperesentation vector:
```python
image = load_from_url("image_url").resize((256, 256))
t = FE.get_style_representation(image_to_tensor(image).to(device), K=2)
```

The image below shows a conceptual scheme of the algorithm for obtaining style representation vectors.

<div align="center">
<img src="https://sun9-77.userapi.com/impf/wLaNd-sRuUwYRldcbY5B_a9KEd--cAHWYXMZZw/qq2IA9R9hkU.jpg?size=1002x826&quality=96&sign=2f51be756eb117b0ddfc72e6b2ed6df4&type=album" width="500">
</div>


Generating a preview:
```python
result = TG.run(
    style_tensor=t,
    size=(256, 256),
    n_iter=0,
    alpha=1
)
preview = tensor_to_image(result[0])
```

Generating final results:
```python
result = TG.run(
    style_tensor=t,
    size=(256, 256),
    n_iter=10,
    alpha=0
)
final = tensor_to_image(result[0])
```


### 2. Manipulating generated texture style with multiple examples
Algebraic operations with style representation vectors can give predictable results.

#### 2.1. Interpolation
It is possible to "interpolate" between two real textures. 

$$
{\overrightarrow{t}}^{*} = \lambda{\overrightarrow{t}}_{1} + (1 - \lambda){\overrightarrow{t}}_{2},\text{  }\lambda \in (0,1)
$$

Code:
```python
l = 0.5
result_style_tensor = t1 * (1 - l) + t2 * l
```

<div align="center">
<img src="https://sun9-76.userapi.com/impf/sTuiVZK7RSayTQJ5EG_oXtY1nA2QVyjjivFSDw/4gAn-pAXs1Y.jpg?size=1657x732&quality=96&sign=d12f5e84029a45de97cd006a841d30ff&type=album" width="500">
</div>

#### 2.2. Extraction of stylistic attributes and linear operations

This part was inspired by the papers "Deep Feature Interpolation for Image Content Changes" [4] and "Interpreting the Latent Space of GANs for Semantic Face Editing" [5].
<br>
To extract individual stylistic attributes, such as "grass between rocks" two methods are adopted.
The first one, naïve, includes finding mean style representation vectors for two sets of images: one that shows a particular attribute and one that doesn't. Then the difference between these two vectors represents style difference. The conceptual scheme of this method is shown in the image below.

<div align="center">
<img src="https://sun9-84.userapi.com/impf/_f2M6ut1Fw0aJxp8duHEhaa4yiEtBhBgQGwnrw/YE98r-83K8U.jpg?size=1345x593&quality=96&sign=6eaa7aac66998262460cf2f817218017&type=album" width="500">
</div>

Code:
```python
from mexts.style_features_manipulation import style_attribute_extraction_svm
style_difference = style_attribute_extraction_means(style_tensor_set1, style_tensor_set2)
```

The second method is based on the assumption that for any binary stylistic attribute, there exists a hyperplane on the one side of which the attribute appears, and on the other doesn't. Then the normal to this hyperplane represents style difference. The conceptual scheme of this method is shown in the image below.

<div align="center">
<img src="https://sun9-9.userapi.com/impf/YVxk9pt7IcXYcJ_1bzXnuCsmquqoffybOGkAZg/iEoTzSEzUcY.jpg?size=1327x583&quality=96&sign=ab367f7a9c110ca4e84c61cc15279f16&type=album" width="500">
</div>

Code:
```python
from mexts.style_features_manipulation import style_attribute_extraction_svm
style_difference = style_attribute_extraction_svm(style_tensor_set1, style_tensor_set2)
```
Obtained style difference vector can be added to a target texture vector:
$$
{\overrightarrow{t}}^{*} = {\overrightarrow{t}}_{1} + \lambda\overrightarrow{d}\
$$

The image below shows a comparison between these two methods.
<div align="center">
<img src="https://sun9-88.userapi.com/impf/7tYU5WFpIiuTaZxSSBZAoDYoC1ejBYpQ3RVVWw/dAuWz3oyXZg.jpg?size=1031x278&quality=96&sign=d7c89f08777ed5cff7c5fffcfdf326fa&type=album" width="1000">
</div>

### 3. GUI

To use GUI, you can clone this repository and run setup.py after that launch the app by running gui/main.py. If you installed the package via pip, you may download only the folder gui/.

<div align="center">
<img src="https://sun9-78.userapi.com/impf/icI4Gwi3_eSOYk7oS2xV0MHXWgLqRAMdEHwwZQ/1okcQ2KnYxE.jpg?size=1332x804&quality=96&sign=84754890365e5aa7b153d79704fabcb8&type=album" width="500">
</div>


## Acknowledgement

[1] Leon A. Gatys, Alexander S. Ecker, Matthias Bethge (2015). Texture synthesis and the controlled generation of natural stimuli using convolutional neural net-works. CoRR, abs/1505.07376.

[2] Gonthier, N., Gousseau, Y., & Ladjal, S. (2020). High resolution neural texture synthesis with long range constraints. arXiv. https://doi.org/10.48550/ARXIV.2008.01808

[3] Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. arXiv. https://doi.org/10.48550/ARXIV.1703.06868 

[4] 15.	Upchurch, P., Gardner, J., Pleiss, G., Pless, R., Snavely, N., Bala, K., & Wein-berger, K. (2016). Deep Feature Interpolation for Image Content Changes. arXiv. https://doi.org/10.48550/ARXIV.1611.05507

[5] 16.	Shen, Y., Gu, J., Tang, X., & Zhou, B. (2019). Interpreting the Latent Space of GANs for Semantic Face Editing. arXiv. https://doi.org/10.48550/ARXIV.1907.10786

Also used some code and state-dicts from [this implementation](https://github.com/naoto0804/pytorch-AdaIN) of [3]
