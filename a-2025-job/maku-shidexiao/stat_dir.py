import os
import ast


class PyFileStats:
    def __init__(self):
        self.py_file_count = 0
        self.class_count = 0
        self.function_count = 0
        self.code_line_count = 0

    def analyze_directory(self, directory):
        """分析指定目录下的所有Python文件"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    self.py_file_count += 1
                    filepath = os.path.join(root, file)
                    self._analyze_file(filepath)

    def _analyze_file(self, filepath):
        """分析单个Python文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                self.code_line_count += len(content.splitlines())

                # 使用ast模块解析Python代码
                tree = ast.parse(content)

                # 统计类和函数/方法
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self.class_count += 1
                        # 类中的方法
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                self.function_count += 1
                    elif isinstance(node, ast.FunctionDef):
                        self.function_count += 1
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")

    def display_results(self):
        """显示统计结果"""
        print("\nPython文件统计结果:")
        print(f"Python文件数量: {self.py_file_count}")
        print(f"类数量: {self.class_count}")
        print(f"方法/函数数量: {self.function_count}")
        print(f"代码总行数: {self.code_line_count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方法: python stats.py <目录路径>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"错误: {directory} 不是有效的目录")
        sys.exit(1)

    stats = PyFileStats()
    stats.analyze_directory(directory)
    stats.display_results()


