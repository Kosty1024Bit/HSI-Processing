# HSI Processing
"Moffett Field" hyperspectral image.
![rgb_mf](https://user-images.githubusercontent.com/32631025/130922046-b5047a45-c37f-43c6-bbb0-5baf80286ff3.png)

## Useful links:
[Download HSI Moffett Field (Yandex Disk)](https://disk.yandex.ru/d/spFt8e40w839OQ)

[Download HSI jasperRidge2_F224 (Yandex Disk)](https://disk.yandex.ru/d/LdILQV3pF945mQ)

[Download HSI samson_1 (Yandex Disk)](https://disk.yandex.ru/d/7CaNWUNfHUp_RA)

[Download HSI indian pine (Yandex Disk)](https://disk.yandex.ru/d/Vvpef_-KETbPFg)

[Download HSI Washington_DC_Mall (Yandex Disk)](https://disk.yandex.ru/d/Xcy7TNYpCaVG9g)

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
