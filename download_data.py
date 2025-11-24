"""
下载过去一年的监测数据并保存为 CSV 文件
"""
import requests
import csv
import time
from datetime import datetime, timedelta
import os

# API 基础地址
BASE_URL = "my_url"

# 接口列表
APIS = {
    "测缝计": "/jmData",
    "测斜探头": "/jmBus",
    "水准仪": "/jmLevel",
    "水位计": "/jmWlg"
}

def get_timestamp_range():
    """获取过去一年的时间戳范围"""
    now = datetime.now()
    one_year_ago = now - timedelta(days=365)
    
    timestamp1 = int(one_year_ago.timestamp())
    timestamp2 = int(now.timestamp())
    
    return timestamp1, timestamp2

def fetch_data(api_path, timestamp1, timestamp2, batch_days=30):
    """获取数据（分批次下载，避免单次请求数据量过大）"""
    url = f"{BASE_URL}{api_path}"
    
    print(f"正在获取数据: {url}")
    print(f"时间范围: {datetime.fromtimestamp(timestamp1)} 到 {datetime.fromtimestamp(timestamp2)}")
    
    all_data = []
    current_start = timestamp1
    
    # 按批次获取数据（每批30天）
    batch_num = 1
    total_days = (timestamp2 - timestamp1) / (24 * 60 * 60)
    total_batches = int(total_days / batch_days) + 1
    
    while current_start < timestamp2:
        batch_end = min(current_start + batch_days * 24 * 60 * 60, timestamp2)
        
        print(f"  批次 {batch_num}/{total_batches}: {datetime.fromtimestamp(current_start).strftime('%Y-%m-%d')} 到 {datetime.fromtimestamp(batch_end).strftime('%Y-%m-%d')}")
        
        params = {
            "timestamp1": current_start,
            "timestamp2": batch_end
        }
        
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            
            # 处理响应数据格式
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'code' in data and data.get('code') == 200:
                    data = data.get('data', [])
            
            if not isinstance(data, list):
                print(f"  警告: 返回数据格式不是列表，类型: {type(data)}")
                data = []
            
            if data:
                all_data.extend(data)
                print(f"  获取到 {len(data)} 条数据（累计: {len(all_data)} 条）")
            else:
                print(f"  本批次无数据")
            
        except requests.exceptions.RequestException as e:
            print(f"  请求失败: {e}")
        except Exception as e:
            print(f"  处理数据时出错: {e}")
        
        current_start = batch_end
        batch_num += 1
        
        # 添加延迟，避免请求过快
        time.sleep(0.5)
    
    print(f"总共获取 {len(all_data)} 条数据")
    return all_data

def save_to_csv(data, filename, data_type):
    """保存数据为 CSV 文件"""
    if not data:
        print(f"没有数据可保存: {filename}")
        return
    
    # 确保输出目录存在
    output_dir = "data_export"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    # 获取所有字段名
    if data:
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))
    else:
        fieldnames = []
    
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in data:
            # 处理嵌套数据
            clean_row = {}
            for key, value in row.items():
                if value is None:
                    clean_row[key] = ''
                elif isinstance(value, (dict, list)):
                    clean_row[key] = str(value)
                else:
                    clean_row[key] = value
            writer.writerow(clean_row)
    
    print(f"数据已保存到: {filepath}")
    print(f"共保存 {len(data)} 条记录\n")

def main():
    """主函数"""
    print("=" * 60)
    print("开始下载过去一年的监测数据")
    print("=" * 60)
    print()
    
    # 获取时间戳范围
    timestamp1, timestamp2 = get_timestamp_range()
    
    print(f"时间范围: {datetime.fromtimestamp(timestamp1).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"        到: {datetime.fromtimestamp(timestamp2).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 遍历所有接口
    for data_type, api_path in APIS.items():
        print(f"\n{'='*60}")
        print(f"处理: {data_type}")
        print(f"{'='*60}")
        
        # 获取数据
        data = fetch_data(api_path, timestamp1, timestamp2)
        
        if data:
            # 生成文件名
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp_str}.csv"
            
            # 保存为 CSV
            save_to_csv(data, filename, data_type)
        else:
            print(f"未获取到 {data_type} 的数据\n")
        
        # 添加延迟，避免请求过快
        time.sleep(2)
    
    print("=" * 60)
    print("所有数据下载完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

