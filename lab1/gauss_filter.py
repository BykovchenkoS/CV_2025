import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


class GaussianFilter:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        Создание гауссова ядра

        Args:
            size: Размер ядра (нечетный)
            sigma: Стандартное отклонение

        Returns:
            Нормализованное гауссово ядро
        """
        if size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным")

        kernel = np.zeros((size, size))
        center = size // 2

        # Вычисление значений ядра
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        # Нормализация
        return kernel / np.sum(kernel)

    def native_gaussian_filter(self, image: np.ndarray, kernel_size: int,
                               sigma: float) -> np.ndarray:
        """
        Нативная реализация фильтра Гаусса

        Args:
            image: Входное изображение
            kernel_size: Размер ядра
            sigma: Стандартное отклонение

        Returns:
            Отфильтрованное изображение
        """
        # Создание ядра
        kernel = self.gaussian_kernel(kernel_size, sigma)
        pad = kernel_size // 2

        # Обработка в зависимости от количества каналов
        if len(image.shape) == 3:
            height, width, channels = image.shape
            # Добавление границ
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            result = np.zeros_like(image, dtype=np.float64)

            # Применение фильтра для каждого канала
            for c in range(channels):
                for i in range(height):
                    for j in range(width):
                        region = padded[i:i + kernel_size, j:j + kernel_size, c]
                        result[i, j, c] = np.sum(region * kernel)
        else:
            height, width = image.shape
            padded = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
            result = np.zeros_like(image, dtype=np.float64)

            for i in range(height):
                for j in range(width):
                    region = padded[i:i + kernel_size, j:j + kernel_size]
                    result[i, j] = np.sum(region * kernel)

        return np.clip(result, 0, 255).astype(np.uint8)

    def opencv_gaussian_filter(self, image: np.ndarray, kernel_size: int,
                               sigma: float) -> np.ndarray:
        """
        Реализация фильтра Гаусса с использованием OpenCV

        Args:
            image: Входное изображение
            kernel_size: Размер ядра
            sigma: Стандартное отклонение

        Returns:
            Отфильтрованное изображение
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def compare_performance(self, image_path: str, kernel_sizes: list,
                            sigma: float = 1.0) -> dict:
        """
        Сравнение производительности двух реализаций

        Args:
            image_path: Путь к изображению
            kernel_sizes: Список размеров ядер для тестирования
            sigma: Стандартное отклонение

        Returns:
            Словарь с результатами тестирования
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = {}

        for size in kernel_sizes:
            print(f"Testing kernel size: {size}x{size}")

            # Нативная реализация
            start_time = time.time()
            native_result = self.native_gaussian_filter(image_rgb, size, sigma)
            native_time = time.time() - start_time

            # OpenCV реализация
            start_time = time.time()
            opencv_result = self.opencv_gaussian_filter(image_rgb, size, sigma)
            opencv_time = time.time() - start_time

            results[size] = {
                'native_time': native_time,
                'opencv_time': opencv_time,
                'native_result': native_result,
                'opencv_result': opencv_result
            }

            print(f"Native: {native_time:.4f}s, OpenCV: {opencv_time:.4f}s, "
                  f"Speedup: {native_time / (opencv_time+0.000001):.1f}x")

        return results

    def visualize_results(self, original: np.ndarray, results: dict):
        """
        Визуализация результатов

        Args:
            original: Исходное изображение
            results: Результаты тестирования
        """
        n_sizes = len(results)
        fig, axes = plt.subplots(2, n_sizes + 1, figsize=(9, 6))

        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        axes[1, 0].axis('off')

        # Результаты для разных размеров ядра
        for idx, (size, result_data) in enumerate(results.items(), 1):
            # Нативная реализация
            axes[0, idx].imshow(result_data['native_result'])
            axes[0, idx].set_title(f'Native {size}x{size}\n'
                                   f'{result_data["native_time"]:.3f}s')
            axes[0, idx].axis('off')

            # OpenCV реализация
            axes[1, idx].imshow(result_data['opencv_result'])
            axes[1, idx].set_title(f'OpenCV {size}x{size}\n'
                                   f'{result_data["opencv_time"]:.3f}s')
            axes[1, idx].axis('off')

        plt.tight_layout()
        plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
