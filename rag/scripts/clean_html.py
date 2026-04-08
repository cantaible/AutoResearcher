"""HTML 清洗模块：将 HTML 内容转为干净的纯文本。"""
from bs4 import BeautifulSoup
import html2text


def clean_html(html_content: str) -> str:
    """将 HTML 转为纯文本，保留段落结构。"""
    if not html_content:
        return ""

    # html2text 转换器：输出纯文本，不保留 markdown 格式
    converter = html2text.HTML2Text()
    converter.ignore_links = True      # 去掉链接 URL
    converter.ignore_images = True     # 去掉图片标签
    converter.ignore_emphasis = True   # 去掉加粗/斜体标记
    converter.body_width = 0           # 不自动换行

    text = converter.handle(html_content)

    # 压缩连续空行为单个空行
    lines = text.splitlines()
    cleaned = []
    prev_empty = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(stripped)
            prev_empty = False

    return "\n".join(cleaned).strip()


if __name__ == "__main__":
    # 快速测试
    sample = '<p>Hello <b>world</b></p><p>Second paragraph</p>'
    print(clean_html(sample))
