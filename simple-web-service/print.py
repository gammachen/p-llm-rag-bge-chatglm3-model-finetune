import argparse
import time
import sys
from pathlib import Path

def slow_print_file(file_path, delay=0.3, line_delay=0.01, char_delay=0.001):
    """
    缓慢输出文件内容
    :param file_path: 要输出的文件路径
    :param delay: 文件开始前的延迟(秒)
    :param line_delay: 行与行之间的延迟(秒)
    :param char_delay: 字符之间的延迟(秒)
    """
    try:
        file = Path(file_path)
        if not file.exists():
            print(f"Error: 文件 {file_path} 不存在")
            return
        
        time.sleep(delay)  # 开始前的延迟
        
        with file.open('r', encoding='utf-8') as f:
            for line in f:
                # 逐个字符输出
                for char in line:
                    print(char, end='', flush=True)
                    time.sleep(char_delay)
                time.sleep(line_delay)  # 行间延迟
    
    except Exception as e:
        print(f"\n发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='缓慢输出Python文件内容')
    parser.add_argument('file', help='要输出的Python文件路径')
    parser.add_argument('--delay', type=float, default=0.3, help='开始前的延迟(秒)')
    parser.add_argument('--line-delay', type=float, default=0.1, help='行间延迟(秒)')
    parser.add_argument('--char-delay', type=float, default=0.02, help='字符间延迟(秒)')
    
    args = parser.parse_args()
    
    slow_print_file(
        args.file,
        delay=args.delay,
        line_delay=args.line_delay,
        char_delay=args.char_delay
    )

if __name__ == '__main__':
    main()
