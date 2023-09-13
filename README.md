# HSI Processing
"Moffett Field" hyperspectral image labeling.
![rgb_mf](https://user-images.githubusercontent.com/32631025/130922046-b5047a45-c37f-43c6-bbb0-5baf80286ff3.png)

## Useful links:
[Download HSI Moffett Field](https://drive.google.com/file/d/1xbTM2D-HpMVYf1BUtXefKqqokfEVz9OA/view?usp=sharing)

[Download HSI Cuprite97 (Yandex Disk)](https://disk.yandex.ru/d/YIMepFJvW0TBnw)
```python
#How to open in Python
import numpy as np
import tifffile as tfl
hsi_tif = tfl.TiffFile('./Cuprite97/Cuprite97.tif')
hsi = hsi_tif.asarray().copy()
```
[Download HSI Urban_F210 (Yandex Disk)](https://disk.yandex.ru/d/WL4q_BmPl8lgxw)
```python
#How to open in Python
import numpy as np
import spectral.io.envi as envi
hsi_envi = envi.open('./Urban_F210/Urban_F210.hdr',
                     './Urban_F210/Urban_F210.img')
hsi = np.array(hsi_envi.open_memmap(writble = True), dtype = float)
```
___
[www.spectralpython.net](https://www.spectralpython.net/)

Spectral Python (SPy) is a pure Python module for processing hyperspectral image data. It has functions for reading, displaying, manipulating, and classifying hyperspectral imagery.
