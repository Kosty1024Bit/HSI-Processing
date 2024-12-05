# Добро пожаловатиь в HSI-Processing!
![rgb_mf](https://user-images.githubusercontent.com/32631025/130922046-b5047a45-c37f-43c6-bbb0-5baf80286ff3.png)

## Установка
1. Убедитесь, что у вас установлен Python 3.10+ и `pip`.

2. Установка пакета напрямую из GitHub

```bash
pip install git+https://github.com/Kosty1024Bit/HSI-Processing.git
```

3. Клонирование репозитория и локальная установка

```bash
git clone git+https://github.com/Kosty1024Bit/HSI-Processing.git
cd HSI-Processing
pip install -e .
```

## Полезные ссылки:
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

Spectral Python (SPy) — это модуль Python для обработки данных гиперспектральных изображений. Он имеет функции для чтения, отображения, обработки и классификации гиперспектральных изображений.


## Лицензия

Этот проект распространяется под лицензией [Apache License 2.0](LICENSE).
