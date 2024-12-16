import pyrealsense2 as rs
import numpy as np
import cv2

# 配置相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 设置彩色流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    # 设置深度流

# 开始流
pipeline.start(config)

# 获取深度流的内参
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()

# 获取焦距和光心参数
focal_length = intrinsics.fx  # 焦距 fx
baseline = 0.1  # 基线，以米为单位

print(f"相机内参: 焦距 (fx, fy) = ({intrinsics.fx}, {intrinsics.fy}), 光心 (cx, cy) = ({intrinsics.ppx}, {intrinsics.ppy})")

# 深度图像滤波器
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.laser_power, 360)

# 设定深度阈值
depth_threshold = 0.5  # 小于0.5米的深度值将被忽略

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

        # 将深度值转换为视差
        disparity_image = np.zeros_like(depth_image, dtype=np.float32)
        valid_depth = (depth_image > depth_threshold) & (depth_image > 0)  # 有效深度值
        disparity_image[valid_depth] = (focal_length * baseline) / depth_image[valid_depth]

        # 将视差图归一化到 8 位图像
        disparity_image_normalized = cv2.normalize(disparity_image, None, 0, 255, cv2.NORM_MINMAX)
        disparity_image_normalized = np.uint8(disparity_image_normalized)

        # 显示图像
        cv2.imshow('Color Image', np.asanyarray(color_frame.get_data()))
        cv2.imshow('Disparity Image', disparity_image_normalized)

        # 按 's' 保存图像
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('data/data_solo/color_image.png', np.asanyarray(color_frame.get_data()))
            cv2.imwrite('data/data_solo/disparity_image.png', disparity_image_normalized)  # 保存视差图
            print("图像已保存！")
        elif key == ord('q'):
            break

finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
