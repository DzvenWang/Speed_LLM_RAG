import ctypes
import time
from datetime import datetime, timedelta


# ======================
# 配置区
# ======================

CLICK_INTERVAL = 10      # 每 10 秒点击一次
RUN_HOURS = 5            # 运行几个小时
START_DELAY = 5          # 启动后等待 5 秒，方便切换窗口


# ======================
# Windows 鼠标点击
# ======================

user32 = ctypes.windll.user32

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


def get_screen_center():
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    center_x = screen_width // 2
    center_y = screen_height // 2

    return center_x, center_y


def click_at(x, y):
    user32.SetCursorPos(x, y)

    # 左键按下
    user32.mouse_event(0x0002, 0, 0, 0, 0)

    # 左键松开
    user32.mouse_event(0x0004, 0, 0, 0, 0)


def main():
    print(f"程序将在 {START_DELAY} 秒后开始。")
    print(f"运行时长：{RUN_HOURS} 小时")
    print("按 Ctrl + C 可以停止。")

    time.sleep(START_DELAY)

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=RUN_HOURS)

    center_x, center_y = get_screen_center()

    print(f"开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预计结束：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"屏幕中心：({center_x}, {center_y})")

    try:
        while datetime.now() < end_time:
            click_at(center_x, center_y)

            now = datetime.now()
            remaining_seconds = int((end_time - now).total_seconds())

            print(
                f"[{now.strftime('%H:%M:%S')}] 已点击，"
                f"剩余约 {remaining_seconds // 60} 分钟"
            )

            if remaining_seconds <= 0:
                break

            time.sleep(min(CLICK_INTERVAL, remaining_seconds))

        print("运行时间已到，程序结束。")

    except KeyboardInterrupt:
        print("\n已手动停止。")


if __name__ == "__main__":
    main()