from numba import jit
import numpy as np
import matplotlib.pyplot as plt


def maximum_filter(layer, sigma = 3):
    mean_value = layer.mean()
    std_value = layer.std()
    threshold = mean_value + (std_value * sigma)
    
    layer[layer > threshold] = threshold
    
    return layer

def HSI_to_RGB(HSI, bands):
    height, width, _ = HSI.shape
    RGB_image_size = (height, width, 3)
    RGB_image = np.zeros(shape = RGB_image_size, dtype = float)
        
    for i, b in enumerate(bands):
        layer = np.mean((HSI[..., b-1], HSI[..., b], HSI[..., b+1]),
                        axis = 0)
        layer[layer < 0] = 0
        layer = maximum_filter(layer, 3)
            
        min_value = layer.min()
        max_value = layer.max()
        
        layer = (layer - min_value) / (max_value - min_value)
        RGB_image[..., i] = layer
    
    return np.uint8(RGB_image * 255)

def RGB_labels(RGB_image, labels):
    height, width, _ = RGB_image.shape
    classes = np.unique(labels)
    n_classes = len(classes)
    RGB_syntes = np.zeros(shape = (height, width, 3), dtype = np.uint8)
    
    for i in range(n_classes):
        mask = labels == classes[i]
        color_class = [np.mean(RGB_image[mask, b]) for b in range(3)]
        RGB_syntes[mask] = color_class
    
    return RGB_syntes

def EMD(HSI, number_of_modes = 4, start_mode = 0):

    if len(HSI.shape) == 3:
        height, width, bands = HSI.shape
    
        IMFs = np.zeros(shape = (number_of_modes, height, width, bands),
                        dtype = np.float32)
        windows_size = np.zeros(shape = (number_of_modes, height, width), dtype = int)
    
        for i in range(height):
            for j in range(width):
                #IMFs[:, i, j, :], windows_size[i, j] = np.float32(do_EMD(np.float64(HSI[i, j]), number_of_modes + start_mode)[start_mode:])
                
                em, ws = do_EMD(np.float64(HSI[i, j]), number_of_modes + start_mode)

                IMFs[:, i, j, :] = np.float32(em[start_mode:])
                windows_size[:, i, j] = ws[start_mode:]
            

            pix = (i * width) + height
            print('\r', end = '')
            print("обработано " + str(pix) + " пикселей из " +
                str(height * width), end = '')
    
    elif len(HSI.shape) == 2:
        size, bands = HSI.shape

        IMFs = np.zeros(shape = (number_of_modes, size, bands),
                        dtype = float)
        windows_size = np.zeros(shape = (number_of_modes, size), dtype = int)

        for i in range(size):
            IMFs[:, i, :], windows_size[:, i] = do_EMD(np.float64(HSI[i]), number_of_modes + start_mode)[start_mode:]

            if i % 100 == 0:
                print('\r', end = '')
                print("обработано " + str(i + size) + " пикселей из " +
                    str(size), end = '')

    return IMFs, windows_size

@jit(nopython = True, cache = True)
def do_EMD(iSample, number_of_modes = 4, windowSize = 3):
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

    windows_size = []
    
    resEmpModes = np.zeros(shape = (number_of_modes, iSample.shape[0]), dtype = np.float64)

    for num_imf in range(1, number_of_modes + 1):
        
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
            
        #plot(iSample)
        #plot(empModeSample)
        #plot(rSample)
            
            
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
        
        resEmpModes[num_imf - 1] = empModeSample
        
        windows_size.append(windowSize)

        windowSize = int(2 * (dSize / 2) + 1)
        bound = int(windowSize / 2)
        
        
        sample = rSample.copy()
        rSample = np.zeros(shape = sampleSize, dtype = np.float64)
    
    return resEmpModes, windows_size

def do_kmeans(X, K, max_iters = 100):
    np.random.seed(42)
    
    # Инициализация центроидов случайным образом
    centroids = X[np.random.choice(range(len(X)), size = K, replace=False)]
    
    for i in range(max_iters):
        # Расчет расстояний между точками и центроидами
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Определение ближайшего центроида для каждой точки
        labels = np.argmin(distances, axis=1)
        
        # Обновление центроидов на основе средних значений в каждом кластере
        new_centroids = np.array([X[labels==k].mean(axis=0) for k in range(K)])
        
        # Проверка условия сходимости
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
        # Напечатать номер завершённой итерации
        print('\r', end = '')
        print("Итерация № " + str(i + 1), end = '')
        
    print(". Завершено")
        
    return centroids, labels

def KMeans(HSI, number_of_clusters = 8):
    height, width, bands = HSI.shape
    HSI = np.reshape(HSI, (height * width, bands))
    
    centroids, labels = do_kmeans(HSI, number_of_clusters)
    
    return centroids, np.reshape(labels, (height, width))

def calc_ref(reference, hsi, win_size, image = None):
    reference_pix = np.zeros(shape = (reference.shape[0], hsi.shape[2]), dtype = int)
    
    for r in range(reference.shape[0]):
        y_up = reference[r][0] - win_size
        y_dw = reference[r][0] + win_size + 1
        x_lt = reference[r][1] - win_size
        x_rt = reference[r][1] + win_size + 1
        
        for b in range(hsi.shape[2]):
            reference_pix[r, b] = hsi[y_up:y_dw, x_lt:x_rt , b].mean()
        
        if image is not None:
            print(r)
            plt.figure(figsize = (6, 6))
            plt.imshow(image[y_up:y_dw, x_lt:x_rt])
            plt.show()
            
    return reference_pix

def show(image, title = "", figsize = (6, 6), cmap = None):
    plt.figure(figsize = figsize)
    plt.title(title, fontsize = 16)
    plt.imshow(image, cmap = cmap)
    plt.axis('off')
    plt.show()