# STEREO 360 IMAGE PROCESSING

Let’s consider a stereo 360° image/video.

See [image-stereo-360](https://www.couleur.org/JS-Web/Sprint2023/image-stereo-360.html) and [video-stereo-360](https://www.couleur.org/JS-Web/Sprint2023/video-stereo-360.html)

## Equirectangular vs cube map

See: [programming](http://www.paul-reed.co.uk/programming.html)

Code: [Cubemaps-Equirectangular-DualFishEye](https://github.com/PaulMakesStuff/Cubemaps-Equirectangular-DualFishEye)
 
Implement the following image transformations with OpenCV and CUDA:

• Equirectangular => Cube map

• Cube map => Equirectangular

on a stereo 360° (on the left or the right image from a static image or a video).

## Data

```wget https://drive.google.com/file/d/1pSnnK-QOVsgqJO1_8rbqXhD-GTU1Hm1O/view?usp=sharing```

```unzip data.zip```

### Equirectangular => Cube map

to compile the cuda and openCV file run:
```
/usr/local/cuda/bin/nvcc image.cu `pkg-config opencv4 --cflags --libs` imagecpp-linux.cpp -o imagecuda
```

to execute:
```
./imagecuda <equirectangular-image-path>
```
for example: ```./imagecuda_from_cube ./data/image-360-resized.jpg```

### Cube map => Equirectangular

to compile the cuda and openCV file run:
```
/usr/local/cuda/bin/nvcc image-from-cube.cu `pkg-config opencv4 --cflags --libs` imagecpp-from-cube-to-equi.cpp -o imagecuda_from_cube
```

to execute:
```
./imagecuda_from_cube <cubemap-image-path>
```
for example: ```./imagecuda_from_cube image-cube.jpg```

## Future work possibilities:

### Equirectangular image filtering

What is happening when you apply the denoising method you developed for PW2 on an equirectangular image?

Display it with: [image-360 Three.js](https://www.couleur.org/JS-Web/Sprint2023/image-360.html)

### Cube map image filtering

Now apply the same filtering methods on each face of the corresponding cube map and transform the resulting images in its equirectangular
representation.

Display it with: [image-360 Three.js](https://www.couleur.org/JS-Web/Sprint2023/image-360.html)

### Video processing

Apply this cube map based method on videos and stereo videos.


