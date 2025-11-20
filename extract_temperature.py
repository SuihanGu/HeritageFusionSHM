"""
从测斜探头数据中提取环境温度数据
仅保存 number=00476464 的数据，提取 data3（温度）和 time 列
"""
import csv
import os
from datetime import datetime

def extract_temperature_data(input_file, output_file):
    """
    从测斜探头数据中提取环境温度数据
    只保存 number=00476464 的数据，提取 data3 和 time 列
    """
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return
    
    print(f"正在处理文件: {input_file}")
    
    extracted_rows = []
    total_rows = 0
    extracted_count = 0
    
    # 目标设备编号
    target_number = "00476464"
    
    # 读取CSV文件
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            number = row.get('number', '').strip()
            
            # 只处理 number=00476464 的数据
            if number == target_number:
                data3 = row.get('data3', '')
                time = row.get('time', '')
                
                if time:  # 确保有时间字段
                    extracted_rows.append({
                        'time': time,
                        'data3': data3
                    })
                    extracted_count += 1
    
    if not extracted_rows:
        print(f"警告: 未找到 number={target_number} 的数据")
        return
    
    # 保存提取的数据
    fieldnames = ['time', 'data3']
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_rows)
    
    print(f"提取完成!")
    print(f"  总行数: {total_rows}")
    print(f"  提取行数: {extracted_count}")
    print(f"  输出文件: {output_file}")
    print()

def main():
    """主函数"""
    print("=" * 60)
    print("提取测斜探头环境温度数据")
    print("=" * 60)
    print()
    
    # 输入和输出目录
    input_dir = "data_export_processed"
    output_dir = "data_export_processed"
    
    # 查找所有处理后的测斜探头CSV文件
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    tilt_files = [f for f in os.listdir(input_dir) if f.startswith("processed_测斜探头") and f.endswith(".csv")]
    
    if not tilt_files:
        print(f"在 {input_dir} 目录下未找到处理后的测斜探头CSV文件")
        return
    
    print(f"找到 {len(tilt_files)} 个测斜探头文件\n")
    
    # 处理每个文件
    for filename in tilt_files:
        input_file = os.path.join(input_dir, filename)
        
        # 生成输出文件名：环境温度_原文件名
        base_name = filename.replace("processed_测斜探头_", "")
        output_filename = f"环境温度_{base_name}"
        output_file = os.path.join(output_dir, output_filename)
        
        extract_temperature_data(input_file, output_file)
    
    print("=" * 60)
    print("所有文件处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()




