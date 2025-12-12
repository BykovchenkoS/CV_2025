import cv2


class FeatureBasedTracker:
    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, min_points_for_bbox=5):
        """
        :param max_corners: Максимальное количество углов для отслеживания.
        :param quality_level: Минимальный порог качества для углов.
        :param min_distance: Минимальное расстояние между углами.
        :param block_size: Размер окна для вычисления производных.
        :param min_points_for_bbox: Минимальное число точек для отрисовки bounding box.
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.min_points_for_bbox = min_points_for_bbox

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.previous_frame = None
        self.previous_points = None
        self.tracking_initialized = False

    def _detect_initial_features(self, frame_gray):
        corners = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
        return corners

    def initialize_tracker(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous_points = self._detect_initial_features(frame_gray)
        if self.previous_points is not None and len(self.previous_points) >= self.min_points_for_bbox:
            self.tracking_initialized = True
            self.previous_frame = frame_gray
            return True
        return False

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.tracking_initialized:
            self.previous_points = self._detect_initial_features(frame_gray)
            if self.previous_points is not None and len(self.previous_points) >= self.min_points_for_bbox:
                self.tracking_initialized = True
                self.previous_frame = frame_gray
                return self._draw_features_and_bbox(frame.copy(), self.previous_points)
            else:
                return frame.copy()
        else:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_frame, frame_gray, self.previous_points, None, **self.lk_params
            )

            if new_points is None or status is None or len(new_points[status == 1]) < self.min_points_for_bbox:
                self.tracking_initialized = False
                self.previous_frame = None
                self.previous_points = None
                return frame.copy()

            good_new = new_points[status == 1]
            self.previous_points = good_new.reshape(-1, 1, 2)
            self.previous_frame = frame_gray

            return self._draw_features_and_bbox(frame.copy(), self.previous_points)

    def _draw_features_and_bbox(self, frame, points):
        points_flat = points.reshape(-1, 2)
        if len(points_flat) < self.min_points_for_bbox:
            return frame

        for x, y in points_flat:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        x_coords = points_flat[:, 0]
        y_coords = points_flat[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        cv2.putText(frame, 'Tracked object', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame


class VideoTracker:
    def __init__(self, video_path, tracker: FeatureBasedTracker, max_window_width=800):
        """
        :param video_path: Путь к файлу
        :param tracker: Объект трекера
        :param max_window_width: Максимальная ширина окна (высота вычисляется автоматически)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = tracker
        self.max_window_width = max_window_width

        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {video_path}")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.original_width > max_window_width:
            scale_factor = max_window_width / self.original_width
            self.window_width = max_window_width
            self.window_height = int(self.original_height * scale_factor)
        else:
            self.window_width = self.original_width
            self.window_height = self.original_height

        print(f"Оригинальный размер видео: {self.original_width}x{self.original_height}")
        print(f"Размер окна: {self.window_width}x{self.window_height}")

        cv2.namedWindow('Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tracker', self.window_width, self.window_height)

    def _resize_frame(self, frame):
        if self.original_width <= self.max_window_width:
            return frame
        resized_frame = cv2.resize(frame,(self.window_width, self.window_height), interpolation=cv2.INTER_LINEAR)
        return resized_frame

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                tracked_frame = self.tracker.process_frame(frame)
                resized_frame = self._resize_frame(tracked_frame)
                cv2.imshow('Tracker', resized_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+'):
                    self.max_window_width = min(self.max_window_width + 100, 1920)
                    if self.original_width > self.max_window_width:
                        scale_factor = self.max_window_width / self.original_width
                        self.window_width = self.max_window_width
                        self.window_height = int(self.original_height * scale_factor)
                    cv2.resizeWindow('Tracker', self.window_width, self.window_height)
                elif key == ord('-'):
                    self.max_window_width = max(self.max_window_width - 100, 320)
                    if self.original_width > self.max_window_width:
                        scale_factor = self.max_window_width / self.original_width
                        self.window_width = self.max_window_width
                        self.window_height = int(self.original_height * scale_factor)
                    cv2.resizeWindow('Tracker', self.window_width, self.window_height)
                elif key == ord('f'):
                    if cv2.getWindowProperty('Tracker', cv2.WND_PROP_FULLSCREEN) == 0:
                        cv2.setWindowProperty('Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty('Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Tracker', self.window_width, self.window_height)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "./mona-lisa.avi"
    tracker = FeatureBasedTracker(min_points_for_bbox=10, max_corners=500, quality_level=0.01, min_distance=3, block_size=5)
    video_tracker = VideoTracker(video_file, tracker, max_window_width=800)
    ret, first_frame = video_tracker.cap.read()
    if ret:
        success = tracker.initialize_tracker(first_frame)
        if success:
            print("Трекер успешно инициализирован по первому кадру")
            video_tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Не удалось инициализировать трекер по первому кадру")

    video_tracker.run()
