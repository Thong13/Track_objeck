import cv2
import numpy as np

def track_object():
    # Hàm để tìm vật thể màu trong frame và vẽ viền màu vàng
    def find_colored_objects(frame, color_lower, color_upper, min_area_threshold=100):
        # Chuyển đổi frame sang không gian màu HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tạo mask để chỉ xem vùng nào có màu nằm trong phạm vi màu cần tìm
        mask = cv2.inRange(hsv_frame, color_lower, color_upper)

        # Tìm contours (biên) của các vật thể màu
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        colored_objects = []

        for contour in contours:
            # Tính diện tích của vật thể
            area = cv2.contourArea(contour)

            # Bỏ qua các vật thể nhỏ (có diện tích nhỏ hơn ngưỡng)
            if area < min_area_threshold:
                continue

            # Vẽ hình bao quanh vật thể màu với viền màu vàng (BGR: 0, 255, 255)
            cv2.drawContours(frame, [contour], 0, (0, 255, 255), 2)

            # Tính toán giá trị màu RGB của vật thể và kích thước
            x, y, w, h = cv2.boundingRect(contour)
            roi_frame = frame[y:y + h, x:x + w]
            b, g, r = cv2.mean(roi_frame)[:3]

            colored_objects.append((r, g, b, w, h, x, y))  # Thêm giá trị màu, kích thước, và tọa độ vào danh sách

        return frame, colored_objects

    # Mở camera
    cap = cv2.VideoCapture(1)  # 1 là số thứ tự của camera (thường là camera mặc định)

    object_color = None  # Biến để lưu trữ màu của vật thể lớn nhất
    object_found = False  # Biến để kiểm tra xem có vật thể nào trong khung hình hay không

    # Khởi tạo biến để theo dõi tốc độ của vật thể
    previous_frame = None
    previous_position_x = None
    previous_position_y = None

    # Kích thước thực tế của vật (4x4x2 cm)
    real_width_cm = 4
    real_height_cm = 4
    real_depth_cm = 2

    # Định nghĩa các biến font và font_scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # Điều chỉnh kích thước font tại đây

    frame_buffer = []  # Danh sách lưu frame (30 frame gần nhất)
    frame_buffer_size = 30

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        # Kiểm tra xem frame đã được đọc thành công
        if not ret:
            break

        # Định nghĩa phạm vi màu đỏ trong không gian màu HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        # Định nghĩa phạm vi màu xanh trong không gian màu HSV
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Tìm và theo dõi các vật thể màu đỏ trong frame
        frame, red_objects = find_colored_objects(frame, lower_red, upper_red, min_area_threshold=500)

        # Tìm và theo dõi các vật thể màu xanh trong frame
        frame, blue_objects = find_colored_objects(frame, lower_blue, upper_blue, min_area_threshold=500)

        if red_objects or blue_objects:
            # Tìm vật thể lớn nhất trong các vật thể màu đỏ và màu xanh
            all_objects = red_objects + blue_objects
            largest_object = max(all_objects, key=lambda x: x[3] * x[4])  # Sắp xếp theo diện tích

            # Xác định màu của vật thể lớn nhất
            if largest_object in red_objects:
                object_color = "Red"
            else:
                object_color = "Blue"

            # Lấy kích thước của vật thể lớn nhất
            object_width, object_height = largest_object[3], largest_object[4]

            # Xác định trạng thái của vật thể
            if object_width > 1.5 * object_height:
                object_orientation = "Horizontal"
            elif object_height > 1.5 * object_width:
                object_orientation = "Vertical"
            else:
                object_orientation = "Opposite"

            object_found = True
        else:
            object_found = False

        if object_found:
            # Lấy tọa độ của vật
            x, y = largest_object[5], largest_object[6]

            if previous_position_x is not None and previous_position_y is not None:
                # Tính tốc độ di chuyển (tính theo pixel/giây)
                time_elapsed = len(frame_buffer) / 30  # Thời gian trôi qua cho 30 frame
                delta_x = x - previous_position_x
                delta_y = y - previous_position_y

                speed_x = abs(delta_x) / time_elapsed  # Chuyển tốc độ âm thành dương
                speed_y = abs(delta_y) / time_elapsed

                # Tính tỷ lệ pixel/cm
                pixel_per_cm_x = object_width / real_width_cm
                pixel_per_cm_y = object_height / real_height_cm

                # Tính tốc độ theo cm/giây
                speed_x_cm_per_second = speed_x / pixel_per_cm_x
                speed_y_cm_per_second = speed_y / pixel_per_cm_y

                # Hiển thị thông tin
                cv2.putText(frame, f'Largest Object Color: {object_color}', (10, 30), font, font_scale, (0, 255, 0), 1)
                cv2.putText(frame, f'Object Size in pixel (Width x Height): {object_width} x {object_height}', (10, 70),
                            font, font_scale, (0, 255, 0), 1)
                cv2.putText(frame, f'Object Orientation: {object_orientation}', (10, 110), font, font_scale, (0, 255, 0), 1)
                cv2.putText(frame, f'Speed in X direction: {speed_x_cm_per_second:.2f} cm/s', (10, 150), font, font_scale,
                            (0, 255, 0), 1)
                cv2.putText(frame, f'Speed in Y direction: {speed_y_cm_per_second:.2f} cm/s', (10, 190), font, font_scale,
                            (0, 255, 0), 1)
                cv2.putText(frame, f'Position: x {x / pixel_per_cm_x:.2f} cm, y {y / pixel_per_cm_y:.2f} cm', (10, 230),
                            font, font_scale, (0, 255, 0), 1)

                print(f'Object Size (Width x Height): {object_width} x {object_height}')
                print(f'Tọa độ X: {x / pixel_per_cm_x:.2f} cm, Y: {y / pixel_per_cm_y:.2f} cm')

            # Cập nhật tọa độ trước đó
            previous_position_x, previous_position_y = x, y

        # Lưu frame vào buffer và giới hạn kích thước của buffer
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > frame_buffer_size:
            frame_buffer.pop(0)

        # Hiển thị frame
        cv2.imshow('Object Tracking', frame)

        # Thoát vòng lặp khi nhấn phím "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm để thực hiện theo dõi vật thể
track_object()
