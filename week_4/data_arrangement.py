import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re

# 載入 XML
tree = ET.parse('O-A0038-003.xml')
root = tree.getroot()

# 命名空間字典
namespaces = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}

# 找出 Content 節點 (帶命名空間)
content_elem = root.find('.//ns:Content', namespaces)

if content_elem is None or content_elem.text is None:
    raise ValueError('找不到包含溫度數據的 Content 節點或內容為空')

content_text = content_elem.text

# 用正則提取所有科學計數法格式的浮點數字串
numbers_str_list = re.findall(r'-?\d+\.\d+E[+-]?\d+', content_text)

raw_values = [float(x) for x in numbers_str_list]

if len(raw_values) < 67 * 120:
    raise ValueError('溫度資料長度不足')

# 轉成 120 行(緯度)、67 列(經度)的 array
grid_data = np.array(raw_values[:67*120]).reshape((120, 67))

lon0, lat0 = 120.00, 21.88
dx, dy = 0.03, 0.03

classification = []
regression = []

for j in range(120):
    for i in range(67):
        lon = round(lon0 + i * dx, 4)
        lat = round(lat0 + j * dy, 4)

        val = grid_data[j, i]
        label = 0 if val == -999 else 1
        classification.append((lon, lat, label))
        if val != -999:
            regression.append((lon, lat, val))

# 儲存CSV
pd.DataFrame(classification, columns=['lon','lat','label']).to_csv('classification.csv', index=False)
pd.DataFrame(regression, columns=['lon','lat','value']).to_csv('regression.csv', index=False)

print('分類資料集 classification.csv 與回歸資料集 regression.csv 已成功輸出')
