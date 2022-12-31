import matplotlib.pyplot as plt
# import pytesseract
import cv2
import easyocr


def open_img(img_path):
    carplate_img = cv2.imread(img_path) #читаем картинку
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB) #меняес цветовую палитру

    plt.axis('off') #выключаем оси
    plt.imshow(carplate_img) #рисуем картинку
    # plt.show() #выводим на экран

    return carplate_img

#обнаружение и извлечение номера с картинки
def carplate_extract(img, carplate_haar_cascade): #принимает картинку и путь до обученной модели
    # метод позволяет обнаружить объекты разных размеров на изображении и возвращает список из границ прямоугольника
    # scaleFactor уменьшит изображение на 10 процентов, для лучшего сопоставления номера
    # minNeighboards отвечает за качество обнаруживаемых объектов, чем больше значение, тем меньше обнаружений
    carplate_rects = carplate_haar_cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5)

    #проходим по списку координат x, y, ширина, высота
    for x, y, w, h in carplate_rects:
        #выделяем нужно нам расстояние по вертикали и горизонтали по координатам
        carplate_img = img[y : y+h, x : x+w]

    return carplate_img

#увелисивает изображение, для лучшего распознования
def enlarge_img(img, scale_percent): #принимает картинку и как сильно увеличится картинка
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    dim = (width, height)
    plt.axis('off')
    #изменяем размер, передаем картинку, новые размеры, и используем метод сжатия картинки
    resizes_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resizes_img

def main():
    #вызываем функцию
    carplate_img_rgb = open_img(img_path=r'D:\PyCharm Community Edition 2022.2.3\pythonProject\car_detection\car\one.jpg')

    #пердобученная  модель
    carplate_haar_cascade = cv2.CascadeClassifier('D:\PyCharm Community Edition 2022.2.3\pythonProject\car_detection\haar_cascade\haarcascade_russian_plate_number.xml')

    #извлекаем координаты
    carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)
    #формируем и увеличиваем картинку
    carplate_extract_img = enlarge_img(carplate_extract_img, 150)
    #рисуем
    plt.imshow(carplate_extract_img)
    # plt.show()

    #сделаем номер в оттенках серого, чтобы было четче
    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
    #рисуем
    plt.imshow(carplate_extract_img_gray, cmap='gray')
    plt.axis('off')
    plt.show()

    #преобразуем номер в текс
    reaader = easyocr.Reader(["ru", "en"])
    result = reaader.readtext(carplate_extract_img_gray, detail=0)
    print(result)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    # q = pytesseract.image_to_string(carplate_extract_img_gray, config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    # print(q)

if __name__ == '__main__':
    main()
