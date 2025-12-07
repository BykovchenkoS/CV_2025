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
            # Расчёт optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_frame, frame_gray, self.previous_points, None, **self.lk_params
            )

            if new_points is None or status is None or len(new_points[status == 1]) < self.min_points_for_bbox:
                # Трекинг потерян — сброс
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

        # bounding box
        x_coords = points_flat[:, 0]
        y_coords = points_flat[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return frame


class VideoTracker:
    def __init__(self, video_path, tracker: FeatureBasedTracker):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = tracker

        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {video_path}")

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                tracked_frame = self.tracker.process_frame(frame)
                cv2.imshow('Tracker', tracked_frame)

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    video_file = "./IMG_7315.MP4"
    tracker = FeatureBasedTracker(min_points_for_bbox=5, max_corners=100, block_size=7)
    video_tracker = VideoTracker(video_file, tracker)
    video_tracker.run()
