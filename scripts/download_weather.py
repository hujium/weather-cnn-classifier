#!/usr/bin/env python3
"""
天气图片批量下载器 - 使用国内源（360图片搜索）
6类天气各100张：晴天、多云、阴天、雨天、雪天、雾天
"""

import os
import json
import time
import random
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import ssl
import sys

# 配置
BASE_DIR = "/storage/emulated/0/Documents/ClawOutPut/weather-cnn/dataset/all"
TIMEOUT = 15
MAX_RETRIES = 2
TARGET_PER_CLASS = 100

# SSL context
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# 天气分类及搜索关键词
WEATHER_CLASSES = {
    "sunny": {
        "name": "晴天",
        "keywords": ["晴天风景", "阳光明媚风景", "蓝天白云", "晴空万里", "sunny sky landscape"]
    },
    "cloudy": {
        "name": "多云",
        "keywords": ["多云天气风景", "云层天空", "多云风景", "白云天空", "cloudy sky"]
    },
    "overcast": {
        "name": "阴天",
        "keywords": ["阴天风景", "灰暗天空", "阴沉天气", "乌云密布", "overcast sky"]
    },
    "rainy": {
        "name": "雨天",
        "keywords": ["下雨风景", "雨天风景", "暴雨风景", "rainy weather", "雨中风景"]
    },
    "snowy": {
        "name": "雪天",
        "keywords": ["雪景风景", "大雪风景", "雪天风景", "snowy landscape", "冰雪风景"]
    },
    "foggy": {
        "name": "雾天",
        "keywords": ["雾天风景", "大雾风景", "晨雾风景", "foggy landscape", "雾气风景"]
    }
}


def search_360(keyword, page=1, count=30):
    """从360图片搜索获取图片URL列表"""
    url = f"https://image.so.com/j?q={urllib.parse.quote(keyword)}&pn={page}&sn=0&kn={count}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Referer': 'https://image.so.com/',
        'X-Requested-With': 'XMLHttpRequest',
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            urls = []
            for item in data.get('list', []):
                # 优先取原图img字段
                img_url = item.get('img') or item.get('thumb') or ''
                if img_url and img_url.startswith('http'):
                    urls.append(img_url)
            return urls
    except Exception as e:
        print(f"    ⚠️ 搜索失败: {e}")
        return []


def download_image(url, save_path, retry=0):
    """下载单张图片"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Referer': 'https://image.so.com/',
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ssl_ctx) as resp:
            data = resp.read()
            # 最小 8KB
            if len(data) < 8192:
                return False
            # 检查文件头判断是否是图片
            is_image = False
            if data[:3] == b'\xff\xd8\xff':  # JPEG
                is_image = True
            elif data[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                is_image = True
            elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':  # WebP
                is_image = True
            elif data[:2] == b'BM':  # BMP
                is_image = True
            if is_image:
                with open(save_path, 'wb') as f:
                    f.write(data)
                return True
            return False
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(1)
            return download_image(url, save_path, retry + 1)
        return False


def download_class(class_name, class_info, target=TARGET_PER_CLASS):
    """下载某一类天气的图片"""
    class_dir = os.path.join(BASE_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    existing = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    if existing >= target:
        print(f"  ✅ {class_info['name']} 已有 {existing} 张，跳过")
        return existing

    need = target - existing
    print(f"\n🌤️  下载 {class_info['name']} ({class_name})，需要 {need} 张...")

    all_urls = []
    for kw in class_info['keywords']:
        print(f"  🔍 搜索: {kw}")
        for page in range(1, 8):  # 搜7页
            urls = search_360(kw, page=page, count=30)
            all_urls.extend(urls)
            if len(all_urls) >= need * 4:
                break
            time.sleep(random.uniform(0.8, 1.5))
        if len(all_urls) >= need * 4:
            break

    # 去重
    seen = set()
    unique_urls = []
    for u in all_urls:
        h = hashlib.md5(u.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_urls.append(u)

    print(f"  📋 获取到 {len(unique_urls)} 个唯一URL")
    random.shuffle(unique_urls)

    count = existing
    for i, url in enumerate(unique_urls):
        if count >= target:
            break
        filename = f"{class_name}_{count + 1:04d}.jpg"
        save_path = os.path.join(class_dir, filename)
        if download_image(url, save_path):
            count += 1
            if count % 10 == 0:
                print(f"  ✅ {class_info['name']}: {count}/{target}")
        if i % 8 == 0:
            time.sleep(0.3)

    print(f"  📊 {class_info['name']} 最终: {count} 张")
    return count


def main():
    print("=" * 60)
    print("🌦️  天气图片批量下载器（国内源 - 360图片）")
    print("   6类天气 × 100张 = 600张")
    print("=" * 60)

    results = {}
    for class_name, class_info in WEATHER_CLASSES.items():
        count = download_class(class_name, class_info)
        results[class_name] = count

    print("\n" + "=" * 60)
    print("📊 下载统计:")
    total = 0
    emoji_map = {"sunny": "☀️", "cloudy": "⛅", "overcast": "🌥️", "rainy": "🌧️", "snowy": "❄️", "foggy": "🌫️"}
    for name, count in results.items():
        print(f"  {emoji_map.get(name, '📷')} {WEATHER_CLASSES[name]['name']}: {count} 张")
        total += count
    print(f"  📸 总计: {total} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()
