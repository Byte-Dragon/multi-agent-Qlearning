# 获取调用者信息
import sys
import datetime

caller_frame = sys._getframe(1)  # 1表示上一层调用者
timestamp = datetime.datetime.now()
print(f"函数 my_function 被调用于 {timestamp}，由 {caller_frame.f_code.co_name} "
      f"在文件 {caller_frame.f_code.co_filename} 第 {caller_frame.f_lineno} 行调用。")