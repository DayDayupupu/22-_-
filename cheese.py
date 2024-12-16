import pyrealsense2 as rs
import numpy as np
import cv2

# 配置相机
pipeline = rs.pipeline()
config = rs.config()

# 设置流参数，尝试使用更低的分辨率以提高帧率
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 640x480 @ 30fps
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 开始流
pipeline.start(config)

# 深度图像滤波器
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.laser_power, 360)  # 激光功率（可调）

try:
    while True:
        # 等待新帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 将深度图像转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())


        # 将深度图像归一化到 8 位图像
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = np.uint8(depth_image_normalized)

        # 反转黑白
        depth_image_inverted = 255 - depth_image_normalized

        # 显示图像
        cv2.imshow('Color Image', np.asanyarray(color_frame.get_data()))
        cv2.imshow('Depth Image', depth_image_inverted)  # 显示反转后的深度图像

        # 按 's' 保存图像
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('data/data_solo/color_image.png', np.asanyarray(color_frame.get_data()))
            cv2.imwrite('data/data_solo/depth_image.png', depth_image_normalized)  # 保存滤波后的深度图像
            cv2.imwrite('data/data_solo/depth_image_raw.png', depth_image)  # 保存原始深度图像
            print("图像已保存！")
        elif key == ord('q'):
            break

finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
