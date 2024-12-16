import pyrealsense2 as rs

# 配置相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)

# 开始流
pipeline.start(config)

try:
    # 获取当前活动配置
    profile = pipeline.get_active_profile()

    # 获取深度流的内参
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = depth_stream.get_intrinsics()

    print(f"焦距 (fx, fy): ({intrinsics.fx}, {intrinsics.fy})")
    print(f"光心 (cx, cy): ({intrinsics.ppx}, {intrinsics.ppy})")

finally:
    # 停止流
    pipeline.stop()
