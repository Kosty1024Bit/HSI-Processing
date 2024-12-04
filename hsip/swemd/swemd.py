from numba import jit
import numpy as np
from tqdm import tqdm

def SWEMD(data: np.ndarray, number_of_modes:int = 4, windows_size: list = [3], verbose: bool = True):
    '''
    Рассчитывает и возвращает IMF и окна для каждого каждого сигнала, указанного в `data`.

    Параметры
    ----------
    x : np.ndarray
        Массив размерности 3 (height * width * bands), размерности 2 (n_samples * bands), или просто один образец.
    number_of_modes : int, по умолчанию=4
        Количество IMF, которые необходимо вычислить для входного сигнала.
    windows_size : list или tuple of int, по умолчанию=3
        Размер окон для каждой моды, начиная с первой. Если передан список, то каждый элемент указывает размер окна для соответствующего IMF. 
        Если список переданных размеров меньше, чем указано в `number_of_modes`, то последующие размеры скользящих окон будут вычислены автоматически.
        Если в списке указан элемент со значением -1, то размер этого окна будет также вычислен автоматически.
        Если передано целое число, это будет размер скользящего окна только для первого IMF.
        
    Возвращает
    ---------
    IMFs : list
        Эмпирические моды для каждого образца.
    err_windows_size : list
        Размеры окон для каждого уровня эмпирических мод.

    Примеры
    --------
    Вычисление EMD: длины окон равны трем для 1-ой и 2-ой моды, 5 для 3-ий моды и автоматически для остальных.

    >>> import numpy as np
    >>> from hsip.swemd.swemd import SWEMD
    >>> data = np.random.rand(1000, 1000, 100) * 10  # Пример спектральных данных
    >>> IMFs, windows = SWEMD(data, number_of_modes=8, windows_size=[3, 3, 5])
    >>> print(IMFs.shape, windows.shape)
    (8, 1000, 1000, 100), (8, 1000, 1000)
    '''

    if data.dtype != np.float64:
        data = np.float64(data)
    
    if len(data.shape) == 3:
        height, width, bands = data.shape
        total_signals = height * width
        IMFs = np.zeros(shape=(number_of_modes, height, width, bands), dtype=np.float64)
        out_windows_size = np.zeros(shape=(number_of_modes, height, width), dtype=int)
    
        with tqdm(total=total_signals, disable=not verbose) as pbar:
            for i in range(height):
                for j in range(width):
                    IMFs[:, i, j, :], out_windows_size[:, i, j] = SWEMD_signal(iSample=data[i, j], number_of_modes=number_of_modes, windows_size=windows_size)
                    pbar.update(1)
    
    elif len(data.shape) == 2:
        n_samples, bands = data.shape
        IMFs = np.zeros(shape=(number_of_modes, n_samples, bands), dtype=np.float64)
        out_windows_size = np.zeros(shape=(number_of_modes, n_samples), dtype=int)

        for i in tqdm(range(n_samples), disable=not verbose):
            IMFs[:, i, :], out_windows_size[:, i] = SWEMD_signal(iSample=data[i], number_of_modes=number_of_modes, windows_size=windows_size)
                
    elif len(data.shape) == 1:
        IMFs, out_windows_size = SWEMD_signal(iSample=data, number_of_modes=number_of_modes, windows_size=windows_size)
        
    else:
        raise ValueError(f'It is allowed that the input array "data" has 1, 2 or 3 dimensions, but was transferred with {len(HSI.shape)} dimensions.')
        
    return IMFs, out_windows_size

@jit(nopython=True)
def SWEMD_signal(iSample: np.ndarray, number_of_modes: int = 4, windows_size: list = [3]):
    '''
    Возвращает IMF для одномерного образца.

    Параметры
    ----------
    x : np.ndarray
        Одномерный сигнал.
    number_of_modes : int, по умолчанию=4
        Количество IMF, которые необходимо вычислить для входного сигнала.
    windows_size : list или tuple of int, по умолчанию=3
        Размер окон для каждой моды, начиная с первой. Если передан список, то каждый элемент указывает размер окна для соответствующего IMF. 
        Если список переданных размеров меньше, чем указано в `number_of_modes`, то последующие размеры скользящих окон будут вычислены автоматически.
        Если в списке указан элемент со значением -1, то размер этого окна будет также вычислен автоматически.
        Если передано целое число, это будет размер скользящего окна только для первого IMF.
        
    Возвращает
    ---------
    IMFs : list
        Эмпирические моды для каждого образца.
    err_windows_size : list
        Размеры окон для каждого уровня эмпирических мод.

    Примеры
    --------
    Вычисление EMD: длины окон равны трем для 1-ой и 2-ой моды, 5 для 3-ий моды и автоматически для остальных.

    >>> import numpy as np
    >>> from hsip.swemd.swemd import SWEMD_signal
    >>> data = np.random.rand(100) * 10  # Пример спектральных данных
    >>> IMFs, windows = SWEMD(data, number_of_modes=8, windows_size=[3, 3, 5])
    >>> print(IMFs.shape, windows.shape)
    (100), (8)
    '''

    #Warning! Legacy code follows!

    if isinstance(windows_size, int):
        windows_size = [windows_size]

    if windows_size[0] is None:
        windows_size[0] = 3
    windowSize = windows_size[0]
    
    sampleSize = int(iSample.shape[0])
    bound = int(windowSize / 2)
    windowSum = float(0.0)
    
    empModeSample = np.zeros(shape = sampleSize, dtype = np.float64)
    sample        = iSample.copy()
    rSample       = np.zeros(shape = sampleSize, dtype = np.float64)
    
    isDmax    = False
    isDmin    = False
    dSize     = int(sampleSize)
    dMaxCount = int(0)
    dMinCount = int(0)
    
    resEmpModes = np.zeros(shape = (number_of_modes, iSample.shape[0]), dtype = np.float64)

    for num_imf in range(number_of_modes):
        
        #print('windowSize', windowSize)
        
        for i in range(sampleSize):
            for j in range(int(windowSize)):
                
                if (i - bound + j < 0):
                    windowSum = windowSum + sample[0]
                    continue
                
                if (i - bound + j > sampleSize - 1):
                    windowSum += sample[sampleSize - 1]
                    continue
                
                windowSum += sample[i - bound + j]
                
            rSample[i] = windowSum / windowSize
            empModeSample[i] = sample[i] - rSample[i]
            windowSum = 0.0  
            
        dSize = sampleSize
        dMaxCount = 0
        dMinCount = 0
        
        localMaxs = np.empty(shape = 0, dtype = np.int64)
        localMins = np.empty(shape = 0, dtype = np.int64)
        
        for i in range(sampleSize):
            for j in range(int(windowSize)):
                
                if (i - bound + j == i) or (i - bound + j < 0) or (i - bound + j > sampleSize - 1):
                    continue
                    
                if empModeSample[i] > empModeSample[i - bound + j]:
                    if isDmin == False:
                        isDmax = True;
                        continue
                    else:
                        isDmax = False;
                        isDmin = False;
                        break
                
                if empModeSample[i] < empModeSample[i - bound + j]:
                    if isDmax == False:
                        isDmin = True
                        continue
                    else:
                        isDmax = False
                        isDmin = False
                        break
                
                isDmax = False
                isDmin = False
                break

            if isDmax == True:
                localMaxs = np.append(localMaxs, i + 1)

            if isDmin == True:
                localMins = np.append(localMins, i + 1)

            isDmax = False;
            isDmin = False;
                
        dMaxCount = len(localMaxs)
        dMinCount = len(localMins)
        maxD = int(0)
        
        if dMaxCount >= 2:
            maxD = np.max(np.diff( localMaxs ))
            if maxD < 0: maxD = 0
                
        if dMinCount >= 2:
            maxD_min = np.min(np.diff( localMins ))
            if maxD_min < maxD: maxD = maxD_min
            
        dSize = maxD
        
        resEmpModes[num_imf] = empModeSample
        
        if len(windows_size) - 1 > num_imf:
            if windows_size[num_imf + 1] != -1:
                windowSize = windows_size[num_imf + 1]  # for next imf
            else:
                windowSize = int(2 * (dSize / 2) + 1)  # for next imf
                windows_size[num_imf + 1] = windowSize
                
        elif len(windows_size) - 1 == num_imf: 
            windowSize = int(2 * (dSize / 2) + 1)  # for next imf

        else:
            windows_size.append(windowSize)
            windowSize = int(2 * (dSize / 2) + 1)  # for next imf
            
        bound = int(windowSize / 2)
        
        sample = rSample.copy()
        rSample = np.zeros(shape = sampleSize, dtype = np.float64)
    
    return resEmpModes, windows_size