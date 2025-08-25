import requests
from lxml import html
from datetime import datetime


def fetch_chinadaily_economy_articles():
    # 请求头设置
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 目标URL
    url = "https://www.chinadaily.com.cn/business/economy"

    try:
        # 发送HTTP请求
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功

        # 解析HTML内容
        tree = html.fromstring(response.content)

        # 提取文章标题 - 从h4 > a标签中获取
        titles = tree.xpath('//div[@class="mb10 tw3_01_2"]//h4/a/text()')

        # 提取文章发布时间 - 从span.tw3_01_2_t > b标签中获取
        dates = tree.xpath('//div[@class="mb10 tw3_01_2"]//span[@class="tw3_01_2_t"]/b/text()')

        # 清理数据
        titles = [title.strip() for title in titles if title.strip()]
        dates = [date.strip() for date in dates if date.strip()]

        # 确保标题和日期数量一致
        min_length = min(len(titles), len(dates))
        titles = titles[:min_length]
        dates = dates[:min_length]

        # 打印结果
        print(f"中国日报经济频道首页文章 (共{min_length}篇):\n")
        for i, (title, date) in enumerate(zip(titles, dates), 1):
            print(f"{i}. [{date}] {title}")

        return list(zip(titles, dates))

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return []
    except Exception as e:
        print(f"解析过程中发生错误: {e}")
        return []


if __name__ == "__main__":
    fetch_chinadaily_economy_articles()