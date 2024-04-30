from PIL import Image
import matplotlib.pyplot as plt




def save_image(img, save_path, save_format):
    img.save(save_path, format=save_format)
    print(f"Resim başarıyla kaydedildi: {save_path}")
def display_image(img):
    img.show()

#renkli resmin gri resme çevirme
def convert_to_gray(img):
    width, height = img.size

    # Eğer resim zaten gri seviyeli ise, aynı resmi döndür


    gray_img = Image.new('L', (width, height))

    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((i, j))
            gray_value = int((r + g + b) / 3)  # Renkli değerlerin ortalamasını al
            gray_img.putpixel((i, j), gray_value)

    return gray_img







def convert_to_bw(img, threshold):
    width, height = img.size

    # Eğer resim zaten siyah-beyaz ise, aynı resmi döndür
    if img.mode == '1':
        return img

    bw_img = Image.new('1', (width, height))

    for i in range(width):
        for j in range(height):
            # Eğer resim gri ise, sadece tek bir renk kanalını kullan
            if img.mode == 'L':
                gray_value = img.getpixel((i, j))
            else:
                r, g, b = img.getpixel((i, j))
                gray_value = int((r + g + b) / 3)  # Renkli değerlerin ortalamasını al

            bw_value = 0 if gray_value < threshold else 255  # Eşik değeri ile karşılaştır
            bw_img.putpixel((i, j), bw_value)
    display_image(bw_img)
    return bw_img


# Bu fonksiyon, her pikselin konumunu iki katına çıkaran veya yarı yarıya indirir
def zoom_in_out(img, zoom_factor):
    width, height = img.size

    # Zoom in
    zoomed_in_img = Image.new('RGB', (width * zoom_factor, height * zoom_factor))
    for i in range(width * zoom_factor):
        for j in range(height * zoom_factor):
            original_x = i // zoom_factor
            original_y = j // zoom_factor
            r, g, b = img.getpixel((original_x, original_y))
            zoomed_in_img.putpixel((i, j), (r, g, b))

    # Zoom out
    zoomed_out_img = Image.new('RGB', (width // zoom_factor, height // zoom_factor))
    for i in range(width // zoom_factor):
        for j in range(height // zoom_factor):
            original_x = i * zoom_factor
            original_y = j * zoom_factor
            r, g, b = img.getpixel((original_x, original_y))
            zoomed_out_img.putpixel((i, j), (r, g, b))

    display_image(zoomed_in_img)
    display_image(zoomed_out_img)

    return zoomed_in_img, zoomed_out_img



def crop_image(img, start_x, start_y, width, height):
    original_width, original_height = img.size

    # Sınırları kontrol et ve gerekirse düzelt
    start_x = max(0, min(start_x, original_width - 1))
    start_y = max(0, min(start_y, original_height - 1))
    end_x = min(start_x + width, original_width)
    end_y = min(start_y + height, original_height)

    # Kırpma işlemi
    cropped_width = end_x - start_x
    cropped_height = end_y - start_y

    cropped_img = Image.new('RGB', (cropped_width, cropped_height))
    for i in range(cropped_width):
        for j in range(cropped_height):
            r, g, b = img.getpixel((start_x + i, start_y + j))
            cropped_img.putpixel((i, j), (r, g, b))

    display_image(cropped_img)

    return cropped_img





def create_histogram(img):
    # Convert the image to grayscale
    gray_img = convert_to_gray(img)

    width, height = gray_img.size
    histogram = [0] * 256

    for i in range(width):
        for j in range(height):
            gray_value = gray_img.getpixel((i, j))
            histogram[gray_value] += 1

    # Histogram visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.bar(range(256), histogram, color='grey', alpha=0.7)
    plt.title('Histogram')
    plt.show()

    return img
def equalize_histogram(img):
    # Resmi gri seviyeye dönüştür
    img_gray = convert_to_gray(img)

    # Resmin genişliği ve yüksekliği
    width, height = img_gray.size

    # Histogram oluştur
    histogram = [0] * 256
    for y in range(height):
        for x in range(width):
            pixel_value = img_gray.getpixel((x, y))
            histogram[pixel_value] += 1

    # Kumulatif dağılım fonksiyonunu hesapla
    cumulative_distribution = [sum(histogram[:i + 1]) for i in range(len(histogram))]

    # Histogram eşitleme işlemini uygula
    equalized_img = Image.new('L', (width, height))
    for y in range(height):
        for x in range(width):
            pixel_value = img_gray.getpixel((x, y))
            new_pixel_value = int((cumulative_distribution[pixel_value] / (width * height - 1)) * 255)
            equalized_img.putpixel((x, y), new_pixel_value)

    display_image(equalized_img)

    return equalized_img


def quantize_image(img, num_tones):
    width, height = img.size

    quantized_img = Image.new('RGB', (width, height))

    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((i, j))

            # Her bir renk kanalı için nicemleme işlemi
            quantized_value_r = int(r * (num_tones - 1) / 255)
            quantized_value_g = int(g * (num_tones - 1) / 255)
            quantized_value_b = int(b * (num_tones - 1) / 255)

            # Yeni renkleri uygula
            quantized_img.putpixel((i, j), (int(quantized_value_r * 255 / (num_tones - 1)),
                                            int(quantized_value_g * 255 / (num_tones - 1)),
                                            int(quantized_value_b * 255 / (num_tones - 1))))

    quantized_img.show()
    return quantized_img



import math

def apply_gaussian_blur(img, sigma):
    width, height = img.size
    blurred_img = Image.new('L', (width, height))
    img=convert_to_gray(img)

    # Gaussian filtre matrisini oluştur
    kernel_size = int(6 * sigma + 1)
    kernel = [math.exp(-0.5 * ((x - kernel_size // 2) / sigma) ** 2) for x in range(kernel_size)]
    kernel_sum = sum(kernel)

    # Filtre uygulama
    for i in range(width):
        for j in range(height):
            pixel_acc = 0
            weight_acc = 0

            for x in range(kernel_size):
                for y in range(kernel_size):
                    xi = i + x - kernel_size // 2
                    yj = j + y - kernel_size // 2

                    if 0 <= xi < width and 0 <= yj < height:
                        pixel = img.getpixel((xi, yj))
                        weight = kernel[x] * kernel[y]
                        pixel_acc += pixel * weight
                        weight_acc += weight

            # Normalizasyon ve yeni piksel değeri
            pixel_new = int(pixel_acc / weight_acc)

            blurred_img.putpixel((i, j), pixel_new)

    blurred_img.show()
    return blurred_img





def sharpen_image(img):
    width, height = img.size
    sharpened_img = Image.new('RGB', (width, height))

    for i in range(1, width-1):
        for j in range(1, height-1):
            # Orta piksel ve komşu pikselleri al
            center_pixel = img.getpixel((i, j))
            left_pixel = img.getpixel((i-1, j))
            right_pixel = img.getpixel((i+1, j))
            top_pixel = img.getpixel((i, j-1))
            bottom_pixel = img.getpixel((i, j+1))

            # Piksel değerlerini ağırlıklı topla
            new_r = center_pixel[0] * 5 - left_pixel[0] - right_pixel[0] - top_pixel[0] - bottom_pixel[0]
            new_g = center_pixel[1] * 5 - left_pixel[1] - right_pixel[1] - top_pixel[1] - bottom_pixel[1]
            new_b = center_pixel[2] * 5 - left_pixel[2] - right_pixel[2] - top_pixel[2] - bottom_pixel[2]

            # Sınırları 0-255 arasında tut
            new_r = max(0, min(new_r, 255))
            new_g = max(0, min(new_g, 255))
            new_b = max(0, min(new_b, 255))

            # Yeni piksel değerlerini atama
            sharpened_img.putpixel((i, j), (int(new_r), int(new_g), int(new_b)))

    # Sonucu göster
    sharpened_img.show()

    return sharpened_img




def detect_edges(img):
    #kenar bulma filtresi
    width, height = img.size

    edges_img = convert_to_gray(img)

    # Sobel operatörü için kenar tespiti filtreleri
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for i in range(1, width-1):
        for j in range(1, height-1):
            # Piksel etrafındaki komşu pikselleri al
            pixels = [img.getpixel((i+x, j+y))[0] for x in range(-1, 2) for y in range(-1, 2)]

            # Sobel filtresi uygula
            gradient_x = sum([pixels[idx] * sobel_x[idx//3][idx%3] for idx in range(9)])
            gradient_y = sum([pixels[idx] * sobel_y[idx//3][idx%3] for idx in range(9)])

            # Kenar gücünü hesapla
            edge_strength = int((gradient_x**2 + gradient_y**2)**0.5)

            edges_img.putpixel((i, j), edge_strength)

    edges_img.show()
    return edges_img


def mean_filter(image, filter_size):
    width, height = image.size
    filtered_image = convert_to_gray(image)
    image=convert_to_gray(image)
    # Filtre boyutunu kontrol et (tek sayı olmalı)
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number.")

    half_size = filter_size // 2

    for i in range(half_size, width - half_size):
        for j in range(half_size, height - half_size):
            # Filtre bölgesini seç
            region = [image.getpixel((x, y)) for x in range(i - half_size, i + half_size + 1)
                                                for y in range(j - half_size, j + half_size + 1)]

            # Filtre bölgesinin ortalamasını al ve yeni piksel değerini ata
            new_pixel_value = int(sum(region) / len(region))
            filtered_image.putpixel((i, j), new_pixel_value)
    filtered_image.show()
    return filtered_image

def get_neighbors_gray(img, x, y, radius):
    width, height = img.size
    neighbors = []

    for i in range(max(0, x - radius), min(width, x + radius + 1)):
        for j in range(max(0, y - radius), min(height, y + radius + 1)):
            neighbors.append(img.getpixel((i, j)))

    # Ortalama al ve döndür
    return int(sum(neighbors) / len(neighbors))








def apply_median_filter(input_image, filter_size):
    width, height = input_image.size
    output_image = Image.new('L', (width, height))
    input_image=convert_to_gray(input_image)

    # Filtre boyutunu kontrol et (tek sayı olmalı)
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number.")

    half_size = filter_size // 2

    for i in range(half_size, width - half_size):
        for j in range(half_size, height - half_size):
            # Filtre bölgesini seç
            region = []
            for x in range(i - half_size, i + half_size + 1):
                for y in range(j - half_size, j + half_size + 1):
                    region.append(input_image.getpixel((x, y)))

            # Filtre bölgesini sırala
            sorted_region = sorted(region)

            # Ortanca değeri seç
            median_value = sorted_region[len(sorted_region) // 2]

            # Yeni piksel değerini ata
            output_image.putpixel((i, j), median_value)
    output_image.show()
    return output_image








def apply_contraharmonic_mean_filter(img, Q):
    width, height = img.size
    filtered_img = Image.new('L', (width, height))
    img = convert_to_gray(img)

    # Filtre boyutu (3x3)
    filter_size = 3

    for i in range(width):
        for j in range(height):
            # Pikselin etrafındaki bölgeyi al
            neighbors = get_neighbors_gray_harmonic(img, i, j, filter_size // 2)

            # Q. kuvvetini al ve ortalamayı hesapla
            numerator = sum([x ** (Q + 1) for x in neighbors])
            denominator = sum([x ** Q for x in neighbors])

            # Paydanın sıfır olması durumunu kontrol et
            filtered_value = numerator / (denominator + 1e-6) if denominator != 0 else 0

            # Yeni piksel değerini atayarak filtre uygula
            filtered_img.putpixel((i, j), int(filtered_value))

    filtered_img.show()
    return filtered_img

def get_neighbors_gray_harmonic(img, x, y, radius):
    # Belirli bir pikselin etrafındaki gri tonlamalı pikselleri alır
    width, height = img.size
    neighbors = []

    for i in range(max(0, x - radius), min(width, x + radius + 1)):
        for j in range(max(0, y - radius), min(height, y + radius + 1)):
            pixel_value = img.getpixel((i, j))

            # If the image is RGB, convert to grayscale by taking the intensity value
            if isinstance(pixel_value, tuple):
                pixel_value = sum(pixel_value) // len(pixel_value)

            neighbors.append(pixel_value)

    return neighbors






def dilation(image, kernel_size=(3, 3)):
    # Resmi yükle ve siyah-beyaz çevir


    width, height = image.size

    # Genişletme için kullanılacak kernel
    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]

    # Genişletme işlemi
    result = Image.new('L', (width, height))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Her bir piksel için kernel boyunca dolaş
            region = [image.getpixel((j + x, i + y)) for x in range(-1, 2) for y in range(-1, 2)]
            max_value = max([region[k] + kernel[x][y] for k in range(9) for x in range(3) for y in range(3)])
            result.putpixel((j, i), max_value)

    # Sonucu göster ve kaydet
    result.show()

    return result









def erosion(image, kernel_size=(3, 3)):
    width, height = image.size


    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]

    # Erozyon işlemi
    result = Image.new('L', (width, height))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Her bir piksel için kernel boyunca dolaş
            region = [image.getpixel((j + x, i + y)) for x in range(-1, 2) for y in range(-1, 2)]
            min_value = min([region[k] * kernel[x][y] for k in range(9) for x in range(3) for y in range(3)])
            result.putpixel((j, i), min_value)

    # Sonucu göster ve kaydet
    result.show()
    return result




def skeletonize(image):


    width, height = image.size

    # İskelet çıkarma için kullanılacak kernel
    skeleton_kernel = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]

    # İskelet çıkarma işlemi
    result = Image.new('L', (width, height))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Her bir piksel için kernel boyunca dolaş
            region = [image.getpixel((j + x, i + y)) for x in range(-1, 2) for y in range(-1, 2)]

            # Kernel içinde en az bir siyah piksel varsa, bu pikseli siyah yap
            if any([region[k] == 0 and skeleton_kernel[x][y] == 1 for k in range(9) for x in range(3) for y in range(3)]):
                result.putpixel((j, i), 0)
            else:
                result.putpixel((j, i), 255)

    # Sonucu göster ve kaydet
    result.show()


    return result



def main():
    a = 1
    img_path = input("Lütfen resmin yolunu girin: ")
    original_img = Image.open(img_path)
    onizleme_img = Image.open(img_path)
    display_image(original_img)

    current_img = original_img  # Başlangıçta, mevcut resim orijinal resimdir.

    if a == 1:
        print("1. Önişlem Menüsü-1")
        print("9. devam")
        choice = int(input("Lütfen bir menü seçeneği girin (1-9): "))
        if choice == 1:
            while True:
                print("a. Renkli resmi Gri seviye resme dönüştürme")
                print("b. Gri resmi Siyah Beyaz resme dönüştürme (Eşik girerek)")
                print("c. Zoom in - Zoom out")
                print("d. Resimden istenilen bölgenin kesilip alınması")
                print("e. Menüden Çık")

                sub_choice = input("Lütfen bir seçenek girin (a-e): ")

                if sub_choice == 'a':
                    onizleme_img = convert_to_gray(current_img)
                    display_image(onizleme_img)

                elif sub_choice == 'b':
                    threshold = int(input("Eşik değerini girin: "))
                    onizleme_img = convert_to_bw(current_img, threshold)
                elif sub_choice == 'c':
                    _, onizleme_img = zoom_in_out(current_img,5)
                elif sub_choice == 'd':
                    start_x = int(input("Başlangıç X koordinatını girin: "))
                    start_y = int(input("Başlangıç Y koordinatını girin: "))
                    width = int(input("Bölgenin genişliğini girin: "))
                    height = int(input("Bölgenin yüksekliğini girin: "))
                    onizleme_img = crop_image(current_img, start_x, start_y, width, height)
                elif sub_choice == 'e':
                    current_img = onizleme_img
                    a = 2
                    break
                else:
                    print("Geçersiz seçenek!")
        if choice == 9:
            a = 2

    if a == 2:
        print("2. Önişlem Menüsü-2")
        print("9. devam")
        choice = int(input("Lütfen bir menü seçeneği girin (1-2): "))
        if choice == 2:
            while True:
                print("a. Histogram oluşturma")
                print("b. Histogram Eşitleme")
                print("c. Görüntü Nicemleme (Quantization)")
                print("d. Menüden Çık")

                sub_choice = input("Lütfen bir seçenek girin (a-d): ")

                if sub_choice == 'a':
                    onizleme_img = create_histogram(current_img)
                elif sub_choice == 'b':
                    onizleme_img = equalize_histogram(current_img)
                elif sub_choice == 'c':
                    num_tones = int(input("Ton sayısını girin: "))
                    onizleme_img = quantize_image(current_img, num_tones)
                elif sub_choice == 'd':
                    current_img = onizleme_img
                    a=3
                    break
                else:
                    print("Geçersiz seçenek!")

        if choice == 9:
            a = 3
    if a == 3:
        print("3. Filtreleme Menüsü")
        print("9. devam")
        choice = int(input("Lütfen bir menü seçeneği girin (1-2): "))
        if choice == 3:
            while True:
                print("a. Gaussian Bulanıklaştırma filtresi")
                print("b. Keskinleştirme filtresi")
                print("c. Kenar bulma filtresi")
                print("d. Ortalama (Mean) filtresi")
                print("e. Ortanca (Median) filtresi")
                print("f. Kontra Harmonik Ortalama filtresi")
                print("g. Menüden Çık")

                sub_choice = input("Lütfen bir seçenek girin (a-g): ")

                if sub_choice == 'a':
                    sigma = float(input("Standart sapma (σ) değerini girin: "))
                    onizleme_img = apply_gaussian_blur(current_img, sigma)
                elif sub_choice == 'b':
                    onizleme_img = sharpen_image(current_img)
                elif sub_choice == 'c':
                    onizleme_img = detect_edges(current_img)
                elif sub_choice == 'd':
                    size = int(input("filtre büyüklüğünü girin:"))
                    onizleme_img = mean_filter(current_img,size)
                elif sub_choice == 'e':
                    size = int(input("filtre büyüklüğünü girin:"))
                    onizleme_img = apply_median_filter(current_img,size)
                elif sub_choice == 'f':
                    Q = float(input("Q değerini girin: "))
                    onizleme_img = apply_contraharmonic_mean_filter(current_img, Q)
                elif sub_choice == 'g':
                    current_img = onizleme_img
                    a=4
                    break
                else:
                    print("Geçersiz seçenek!")
        if choice == 9:
            a = 4
    if a == 4:
        print("4. Morfolojik İşlemler")
        print("9. devam")
        choice = int(input("Lütfen bir menü seçeneği girin (1-2): "))
        if choice == 4:
            while True:
                print("a. Siyah beyaz resimde genişletme")
                print("b. Siyah beyaz resimde erozyon")
                print("c. İskelet çıkartma (Skeletonization)")
                print("d. Menüden Çık")

                sub_choice = input("Lütfen bir seçenek girin (a-d): ")

                if sub_choice == 'a':

                    onizleme_img = dilation(current_img)
                elif sub_choice == 'b':
                    onizleme_img = erosion(current_img,(3,3))
                elif sub_choice == 'c':
                    onizleme_img = skeletonize(current_img)
                elif sub_choice == 'd':
                    current_img = onizleme_img
                    a = 5
                    break
                else:
                    print("Geçersiz seçenek!")

        elif choice == 5:
            pass

        else:
            print("Geçersiz seçenek!")
        if choice == 9:
            a = 5

    if a == 5:
        print("5. Kaydet ve Çık")
        print("9. Çık")
        choice = int(input("Lütfen bir menü seçeneği girin (5): "))
        if choice == 5:
            save_path = input("Lütfen resmi kaydetmek istediğiniz dosya yolunu girin (örneğin: kaydedilen_resim.jpg): ")
            save_format = input("Lütfen kaydetme formatını girin (örneğin: JPEG): ")

            # Eğer kullanıcının girdiği format desteklenmiyorsa varsayılan olarak JPEG kullanabilirsiniz
            if save_format not in ["JPEG", "PNG", "BMP"]:
                print("Geçersiz kaydetme formatı. Varsayılan olarak JPEG kullanılacak.")
                save_format = "JPEG"

            save_image(onizleme_img, save_path, save_format)
        if choice == 9:
            print("kaydetmeden çıkıldı")





if __name__ == "__main__":
    main()
