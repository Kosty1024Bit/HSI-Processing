# Moffett Field labeling
"Moffett Field" hyperspectral image labeling.
![rgb_mf](https://user-images.githubusercontent.com/32631025/130922046-b5047a45-c37f-43c6-bbb0-5baf80286ff3.png)

## Useful links:
[Download HSI Moffett Field](https://drive.google.com/file/d/1xbTM2D-HpMVYf1BUtXefKqqokfEVz9OA/view?usp=sharing)

[Download HSI Cuprite97 (Yandex Disk)](https://disk.yandex.ru/d/YIMepFJvW0TBnw)
```python
hsi_tif = tfl.TiffFile(r'C:\Users\konst\jupyter_notebook\HSI\data envil\Cuprite97\Cuprite97.tif')
hsi = hsi_tif.asarray().copy()
```
___
[www.spectralpython.net](https://www.spectralpython.net/)

Spectral Python (SPy) is a pure Python module for processing hyperspectral image data. It has functions for reading, displaying, manipulating, and classifying hyperspectral imagery.
