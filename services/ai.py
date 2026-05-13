import os
import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# 載入 .env
load_dotenv()

# OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------- 共用 headers ----------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------- 股票名稱資料 ----------

@lru_cache(maxsize=1)
def stock_name():
    """
    取得全部股票的股號、股名、產業別
    """

    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.encoding = "big5"

    soup = BeautifulSoup(response.text, "html.parser")

    stock_company = soup.find_all("tr")

    data = []

    for row in stock_company[2:]:

        tds = row.find_all("td")

        if len(tds) < 5:
            continue

        stock_info = tds[0].text.strip()

        if "　" not in stock_info:
            continue

        split_data = stock_info.split("　")

        if len(split_data) < 2:
            continue

        stock_id = split_data[0].strip()
        stock_name = split_data[1].strip()
        industry = tds[4].text.strip()

        if len(stock_id) == 4:
            data.append(
                (stock_id, stock_name, industry)
            )

    df = pd.DataFrame(
        data,
        columns=["股號", "股名", "產業別"]
    )

    return df


def get_stock_name(stock_id):

    df = stock_name()

    try:
        return (
            df.set_index("股號")
            .loc[stock_id, "股名"]
        )
    except Exception:
        return stock_id


# ---------- 股價資料 ----------

def stock_price(stock_id="大盤", days=30):

    ticker = "^TWII" if stock_id == "大盤" else f"{stock_id}.TW"

    end = dt.date.today()

    start = end - dt.timedelta(days=days)

    df = yf.download(
        ticker,
        start=start,
        auto_adjust=False,
        progress=False,
        multi_level_index=False
    )

    if df.empty:
        return {}

    df.columns = [
        "調整後收盤價",
        "收盤價",
        "最高價",
        "最低價",
        "開盤價",
        "成交量"
    ]

    data = {
        "日期": df.index.strftime("%Y-%m-%d").tolist(),
        "收盤價": np.round(
            df["收盤價"], 2
        ).tolist(),
        "每日報酬": np.round(
            df["收盤價"].pct_change(),
            4
        ).fillna(0).tolist(),
        "漲跌價差": np.round(
            df["調整後收盤價"].diff(),
            2
        ).fillna(0).tolist()
    }

    return data


# ---------- 基本面 ----------

def stock_fundamental(stock_id="大盤"):

    if stock_id == "大盤":
        return None

    ticker = f"{stock_id}.TW"

    stock = yf.Ticker(ticker)

    try:

        quarterly_financials = stock.quarterly_financials

        revenue = quarterly_financials.loc[
            "Total Revenue"
        ]

        eps = quarterly_financials.loc[
            "Basic EPS"
        ]

        revenue_growth = np.round(
            revenue.pct_change(-1),
            2
        ).dropna()

        eps_growth = np.round(
            eps.pct_change(-1),
            2
        ).dropna()

        dates = [
            date.strftime("%Y-%m-%d")
            for date in quarterly_financials.columns
        ]

        data = {
            "季日期": dates[:3],
            "營收成長率": revenue_growth.tolist()[:3],
            "EPS": np.round(
                eps, 2
            ).tolist()[:3],
            "EPS季增率": eps_growth.tolist()[:3]
        }

        return data

    except Exception as e:

        return {
            "error": str(e)
        }


# ---------- 新聞 ----------

def stock_news(stock_name="大盤"):

    if stock_name == "大盤":
        stock_name = "台股"

    api_url = (
        "https://ess.api.cnyes.com/"
        f"ess/api/v1/news/keyword?q={stock_name}"
        "&limit=5&page=1"
    )

    response = requests.get(
        api_url,
        headers=HEADERS,
        timeout=30
    )

    json_data = response.json()

    items = json_data["data"]["items"]

    data = []

    for item in items:

        try:

            news_id = item["newsId"]

            title = item["title"]

            publish_at = item["publishAt"]

            utc_time = dt.datetime.utcfromtimestamp(
                publish_at
            )

            formatted_date = utc_time.strftime(
                "%Y-%m-%d"
            )

            article_url = (
                f"https://news.cnyes.com/news/id/{news_id}"
            )

            article = requests.get(
                article_url,
                headers=HEADERS,
                timeout=30
            )

            soup = BeautifulSoup(
                article.content,
                "html.parser"
            )

            p_elements = soup.find_all("p")

            content = ""

            for paragraph in p_elements[4:]:

                content += (
                    paragraph.get_text(strip=True)
                    + "\n"
                )

            data.append({
                "新聞日期": formatted_date,
                "新聞標題": title,
                "新聞內容": content[:1000]
            })

        except Exception:
            continue

    return data


# ---------- Prompt ----------

def generate_content_msg(stock_id):

    stock_name_value = (
        get_stock_name(stock_id)
        if stock_id != "大盤"
        else "大盤"
    )

    price_data = stock_price(stock_id)

    news_data = stock_news(stock_name_value)

    content_msg = f"""
請依據以下資料進行完整股票分析：

股票名稱：
{stock_name_value}

近期價格資訊：
{price_data}

"""

    if stock_id != "大盤":

        fundamental_data = stock_fundamental(
            stock_id
        )

        content_msg += f"""

基本面資訊：
{fundamental_data}

"""

    content_msg += f"""

近期新聞資訊：
{news_data}

請完成：

1. 股價趨勢分析
2. 基本面分析
3. 新聞事件分析
4. 未來可能趨勢
5. 投資風險
6. 短中長期觀察重點

請使用：
- 繁體中文
- 專業證券分析師語氣
- 條列式
- 提及重要數據
- 不要過度樂觀
"""

    return content_msg


# ---------- OpenAI ----------

def get_reply(messages):

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.5
        )

        reply = (
            response.choices[0]
            .message.content
        )

        return reply

    except OpenAIError as err:

        return f"OpenAI Error: {str(err)}"

    except Exception as e:

        return f"System Error: {str(e)}"


# ---------- 主功能 ----------

def stock_gpt(stock_id):

    content_msg = generate_content_msg(
        stock_id
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位專業台股證券分析師，"
                "會綜合股價、基本面與新聞，"
                "產生完整投資分析報告。"
            )
        },
        {
            "role": "user",
            "content": content_msg
        }
    ]

    reply = get_reply(messages)

    return reply
