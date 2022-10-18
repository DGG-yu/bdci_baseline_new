### BaseLine代码理解



[TOC]



# 1. 目录树

```
├── dataset                                             # 数据文件夹
│   └── bdci                                            # 数据集文件夹
│       ├── train_bdci.json           # 比赛提供的原始数据
│       ├── rel2id.json               # 关系到id的映射
│       ├── evalA.json                # A榜评测集
│       ├── train.json                # 经过处理的训练集(data_generator.py生成的文件)
│       └── test.json           	  # 经过处理的测试集(data_generator.py生成的文件)
├── output                                              # 模型输出保存路径 
│   ├── pytorch_model.bin                     # 训练好的模型
│   ├── dev_pred.json                         # 验证集预测结果
│   ├── log.txt                               # 模型训练日志
│   ├── config.txt                            # 模型训练参数(predict.py生成的文件)
│   ├── test_pred.json                        # 测试集预测结果(predict.py生成的文件)
│   └── evalResult.json                       # 模型预测结果(predict.py生成的文件)
├── pretrain_models                                     # 预训练模型文件夹
│   ├── config.json                           # 模型配置？
│   ├── pytorch_model.bin                     # 预训练模型
│   └── vocab.txt                    		  # 词典？ 
├── requirements.txt                                    # 依赖库
├── data_generator.py                                   # 数据处理工具
├── data_utils.py                                       # 数据处理类
├── train.py                                            # 训练代码
├── predict.py                                          # 预测代码
├── model.py                                            # 模型类
└── util.py                                             # 工具类
```



# 2. 文件解析

## 2.1 dataset 数据文件夹

- 存放所有数据集的文件夹，其子文件夹bdci为Big Data & Computing Intelligence缩写。

### 2.1.1 train_bdci.json # 比赛提供的原始数据

1. 数据示例如下（共1491条）：

```json
{"ID": "AT0001", "text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。", "spo_list": [{"h": {"name": "发动机", "pos": [28, 31]}, "t": {"name": "熄火", "pos": [31, 33]}, "relation": "部件故障"}]}
```

2. 每条数据都包含：**“ID”**、**“text”**、**“spo_list”** 这三个字段。

   - **ID**：数据编号

   - **text**：数据文本

   - **spo_list**：文本中关系的三元组（s：主，p：谓，o：宾）
- **h**：头实体；
  
- **t**：尾实体；
  
- **name**：实体名；
  
- **pos**：实体在文本中的位置，前开后闭；
  
- **relation**：关系名

### 2.1.2 rel2id.json # 关系到id的映射

1. 代码如下：

```json
[
  {
    "0": "部件故障",
    "1": "性能故障",
    "2": "检测工具",
    "3": "组成"
  },
  {
    "部件故障": 0,
    "性能故障": 1,
    "检测工具": 2,
    "组成": 3
  }
]
```

2. 关系类型：

   |   主体   |   客体   |       关系       |  主体示例  |  客体示例  |
   | :------: | :------: | :--------------: | :--------: | :--------: |
   | 部件单元 | 故障状态 | **==部件故障==** |  发动机盖  |    抖动    |
   | 性能表征 | 故障状态 | **==性能故障==** |    液面    |    变低    |
   | 检测工具 | 性能表征 | **==检测工具==** | 漏电测试仪 |    电流    |
   | 部件单元 | 部件单元 |   **==组成==**   |   断路器   | 换流变压器 |

### 2.1.3 evalA.json # A榜评测集

1. 用于比赛A榜的评测。
2. 包含**“ID”**和**“text”**两个字段。
3. 数据示例：

```json
{"ID": "AE0001", "text": "三、故障排除(1)发生电缆分支箱带电后,先断开电缆分支箱电源。(2)及时汇报上级部门,并由95598抢修平台发布停电信息。(3)分析故障原因。一般情况下设备外壳带电是由于相线接地、零线断线或三相负荷不平衡和接地电阻不符合要求引起。应先分析用户侧用电使用情况,如用户侧用电正常,则说明相线不完全接地或三相负荷不平衡和接地电阻不符合要求引起,如用户侧用电不正常,则说明相线完全接地或零线断线等。(4)根据分析结果制定查找故障点方法,正确填写配电故障紧急抢修单,选择查找故障所需的工器具和材料。(5)查找故障点时为确保作业人身安全,操作时应按带电作业的安全要求进行。(6)查找故障点。首先确认该电缆分支箱已停电,出线负荷已断开;检查电缆分支箱内的设备(导线)与分支箱外壳有无明显导通,可用万用表欧姆挡测量各相与分支箱外壳是否导通;再检查零线连接是否牢固或有无断线;最后用接地电阻摇表测量接地电阻是否合格,确定故障点位置。(7)落实安全组织措施后,工作许可人应做好线路停电、验电、挂设接地线、悬挂标志牌等安全技术措施,并向工作负责人办理许可手续。(8)施工前工作负责人向全体工作班成员进行\"三交三查\",班组人员确认签名。(9)故障点处理。使用合格的工器具,做好防触电等安全措施,正确处理故障点。(10)工作负责人对施工质量进行验收,并符合设计要求。(11)拆除现场安全围栏等设施,收回工器具、材料并清理现场,工作负责人召开站班会,组织抢修人员撤离作业现场。(12)工作负责人向工作许可人汇报工作结束,工作许可人拆除所有安全措施后,按操作步骤进行送电。(13)工作总结,并由 95598 抢修平台发布送电信息。"}
```

### 2.1.4 train.json # 经过处理的训练集

1. 该文件是执行[data_generator.py](#2.4.2 data_generator.py # 数据处理工具)时所生成的文件
2. 将[train_bdci.json](#2.1.1 train_bdci.json # 比赛提供的原始数据)中数据集进行处理后生成的文件，详细请看[2.4.2.2 train_generator函数](#2.4.2.2 train_generator函数)
3. 数据示例：

```json
  {
    "id": "AT0001",
    "text": "62号汽车故障报告综合情况:故障现象:加速后，丢开油门，发动机熄火。",
    "spos": [
      [
        [
          28,
          31,
          "发动机"
        ],
        "部件故障",
        [
          31,
          33,
          "熄火"
        ]
      ]
    ]
  },
```

### 2.1.5 test.json # 经过处理的测试集

1. 该文件是执行[data_generator.py](#2.4.2 data_generator.py # 数据处理工具)时所生成的文件
2. 将[evalA.json](#2.1.3 evalA.json # A榜评测集)中数据集进行处理后生成的文件，详细请看[2.4.2.3 test_generator函数](#2.4.2.3 test_generator函数)
3. 数据示例：

```
{
  "id": "AE0001_0",
  "text": "三、故障排除(1)发生电缆分支箱带电后,先断开电缆分支箱电源。(2)及时汇报上级部门,并由95598抢修平台发布停电信息。(3)分析故障原因。一般情况下设备外壳带电是由于相线接地、零线断线或三相负荷不平衡和接地电阻不符合要求引起。应先分析用户侧用电使用情况,如用户侧用电正常,则说明相线不完全接地或三相负荷不平衡和接地电阻不符合要求引起,如用户侧用电不正常,则说明相线完全接地或零线断线等。"
},
{
  "id": "AE0001_1",
  "text": "(4)根据分析结果制定查找故障点方法,正确填写配电故障紧急抢修单,选择查找故障所需的工器具和材料。(5)查找故障点时为确保作业人身安全,操作时应按带电作业的安全要求进行。(6)查找故障点。"
},
{
  "id": "AE0001_2",
  "text": "首先确认该电缆分支箱已停电,出线负荷已断开;检查电缆分支箱内的设备(导线)与分支箱外壳有无明显导通,可用万用表欧姆挡测量各相与分支箱外壳是否导通;再检查零线连接是否牢固或有无断线;最后用接地电阻摇表测量接地电阻是否合格,确定故障点位置。(7)落实安全组织措施后,工作许可人应做好线路停电、验电、挂设接地线、悬挂标志牌等安全技术措施,并向工作负责人办理许可手续。"
},
{
  "id": "AE0001_3",
  "text": "(8)施工前工作负责人向全体工作班成员进行\"三交三查\",班组人员确认签名。(9)故障点处理。使用合格的工器具,做好防触电等安全措施,正确处理故障点。(10)工作负责人对施工质量进行验收,并符合设计要求。(11)拆除现场安全围栏等设施,收回工器具、材料并清理现场,工作负责人召开站班会,组织抢修人员撤离作业现场。"
},
{
  "id": "AE0001_4",
  "text": "(12)工作负责人向工作许可人汇报工作结束,工作许可人拆除所有安全措施后,按操作步骤进行送电。(13)工作总结,并由 95598 抢修平台发布送电信息。"
},
```

## 2.2 output 文件夹

- baseline并不包含此文件夹，它是在执行完train.py以及predict.py后所生成的。包含预测模型的各项报告。

### 2.2.1 pytorch_model.bin # 训练好的模型

### 2.2.2 dev_pred.json # 验证集预测结果

### 2.2.3 log.txt  # 模型训练日志



### 2.2.4 config.txt # 模型训练参数 

1. 该文件是执行[predict.py](#2.4.5 predict.py # 预测代码)时所生成的文件
2. 模型训练时的参数配置，详见[main函数](#2.4.5.1 main函数)
3. 配置参考如下：

```xml
base_path = ./dataset
bert_config_path = ./pretrain_models/config.json
bert_model_path = ./pretrain_models/pytorch_model.bin
bert_vocab_path = ./pretrain_models/vocab.txt
cuda_id = 1
dataset = bdci
fix_bert_embeddings = False
max_len = 200
output_path = output
rounds = 4
test_batch_size = 1
```

### 2.2.5 test_pred.json # 测试集预测结果

1. 该文件是执行[predict.py](#2.4.5 predict.py # 预测代码)时所生成的文件
2. 将[test.json](#2.1.5 test.json # 经过处理的测试集)中数据集进行预测后生成的文件，详细请看[2.4.5.4 Predict_data函数](#2.4.5.4 Predict_data函数)
3. 数据示例：

```json
{"ID": "AE0001_0", "text": "三、故障排除(1)发生电缆分支箱带电后,先断开电缆分支箱电源。(2)及时汇报上级部门,并由95598抢修平台发布停电信息。(3)分析故障原因。一般情况下设备外壳带电是由于相线接地、零线断线或三相负荷不平衡和接地电阻不符合要求引起。应先分析用户侧用电使用情况,如用户侧用电正常,则说明相线不完全接地或三相负荷不平衡和接地电阻不符合要求引起,如用户侧用电不正常,则说明相线完全接地或零线断线等。", "spo_list": [{"h": {"name": "电缆分支箱", "pos": [11, 16]}, "t": {"name": "带电", "pos": [16, 18]}, "relation": "部件故障"}, {"h": {"name": "电缆分支箱", "pos": [11, 16]}, "t": {"name": "接地", "pos": [87, 89]}, "relation": "部件故障"}, {"h": {"name": "相线", "pos": [85, 87]}, "t": {"name": "带电", "pos": [16, 18]}, "relation": "部件故障"}, {"h": {"name": "相线", "pos": [85, 87]}, "t": {"name": "接地", "pos": [87, 89]}, "relation": "部件故障"}, {"h": {"name": "零线", "pos": [90, 92]}, "t": {"name": "断线", "pos": [92, 94]}, "relation": "部件故障"}, {"h": {"name": "三相", "pos": [95, 97]}, "t": {"name": "负荷不平衡", "pos": [97, 102]}, "relation": "部件故障"}, {"h": {"name": "接地电阻", "pos": [103, 107]}, "t": {"name": "不符", "pos": [107, 109]}, "relation": "性能故障"}, {"h": {"name": "相线", "pos": [141, 143]}, "t": {"name": "不完全接地", "pos": [143, 148]}, "relation": "部件故障"}, {"h": {"name": "三相", "pos": [149, 151]}, "t": {"name": "负荷不平衡", "pos": [151, 156]}, "relation": "部件故障"}, {"h": {"name": "三相", "pos": [149, 151]}, "t": {"name": "接地电阻不符合要求", "pos": [157, 166]}, "relation": "部件故障"}, {"h": {"name": "零线", "pos": [189, 191]}, "t": {"name": "断线", "pos": [191, 193]}, "relation": "部件故障"}]}
{"ID": "AE0001_1", "text": "(4)根据分析结果制定查找故障点方法,正确填写配电故障紧急抢修单,选择查找故障所需的工器具和材料。(5)查找故障点时为确保作业人身安全,操作时应按带电作业的安全要求进行。(6)查找故障点。", "spo_list": []}
{"ID": "AE0001_2", "text": "首先确认该电缆分支箱已停电,出线负荷已断开;检查电缆分支箱内的设备(导线)与分支箱外壳有无明显导通,可用万用表欧姆挡测量各相与分支箱外壳是否导通;再检查零线连接是否牢固或有无断线;最后用接地电阻摇表测量接地电阻是否合格,确定故障点位置。(7)落实安全组织措施后,工作许可人应做好线路停电、验电、挂设接地线、悬挂标志牌等安全技术措施,并向工作负责人办理许可手续。", "spo_list": [{"h": {"name": "电缆分支箱", "pos": [5, 10]}, "t": {"name": "停电", "pos": [11, 13]}, "relation": "部件故障"}, {"h": {"name": "出线负荷", "pos": [14, 18]}, "t": {"name": "断开", "pos": [19, 21]}, "relation": "性能故障"}]}
{"ID": "AE0001_3", "text": "(8)施工前工作负责人向全体工作班成员进行\"三交三查\",班组人员确认签名。(9)故障点处理。使用合格的工器具,做好防触电等安全措施,正确处理故障点。(10)工作负责人对施工质量进行验收,并符合设计要求。(11)拆除现场安全围栏等设施,收回工器具、材料并清理现场,工作负责人召开站班会,组织抢修人员撤离作业现场。", "spo_list": []}
{"ID": "AE0001_4", "text": "(12)工作负责人向工作许可人汇报工作结束,工作许可人拆除所有安全措施后,按操作步骤进行送电。(13)工作总结,并由 95598 抢修平台发布送电信息。", "spo_list": []}
```

### 2.2.6 evalResult.json # 模型预测结果

1. 该文件是执行[predict.py](#2.4.5 predict.py # 预测代码)时所生成的文件
2. 将[test_pred.json](#2.2.5 test_pred.json # 测试集预测结果)中数据集进行合并后生成的文件，详细请看[2.4.5.5 correct_id函数](#2.4.5.5 correct_id函数)
3. 数据示例：

```json
{"ID": "AE0001", "text": "三、故障排除(1)发生电缆分支箱带电后,先断开电缆分支箱电源。(2)及时汇报上级部门,并由95598抢修平台发布停电信息。(3)分析故障原因。一般情况下设备外壳带电是由于相线接地、零线断线或三相负荷不平衡和接地电阻不符合要求引起。应先分析用户侧用电使用情况,如用户侧用电正常,则说明相线不完全接地或三相负荷不平衡和接地电阻不符合要求引起,如用户侧用电不正常,则说明相线完全接地或零线断线等。(4)根据分析结果制定查找故障点方法,正确填写配电故障紧急抢修单,选择查找故障所需的工器具和材料。(5)查找故障点时为确保作业人身安全,操作时应按带电作业的安全要求进行。(6)查找故障点。首先确认该电缆分支箱已停电,出线负荷已断开;检查电缆分支箱内的设备(导线)与分支箱外壳有无明显导通,可用万用表欧姆挡测量各相与分支箱外壳是否导通;再检查零线连接是否牢固或有无断线;最后用接地电阻摇表测量接地电阻是否合格,确定故障点位置。(7)落实安全组织措施后,工作许可人应做好线路停电、验电、挂设接地线、悬挂标志牌等安全技术措施,并向工作负责人办理许可手续。(8)施工前工作负责人向全体工作班成员进行\"三交三查\",班组人员确认签名。(9)故障点处理。使用合格的工器具,做好防触电等安全措施,正确处理故障点。(10)工作负责人对施工质量进行验收,并符合设计要求。(11)拆除现场安全围栏等设施,收回工器具、材料并清理现场,工作负责人召开站班会,组织抢修人员撤离作业现场。(12)工作负责人向工作许可人汇报工作结束,工作许可人拆除所有安全措施后,按操作步骤进行送电。(13)工作总结,并由 95598 抢修平台发布送电信息。", "spo_list": [{"h": {"name": "电缆分支箱", "pos": [11, 16]}, "t": {"name": "带电", "pos": [16, 18]}, "relation": "部件故障"}, {"h": {"name": "电缆分支箱", "pos": [11, 16]}, "t": {"name": "接地", "pos": [87, 89]}, "relation": "部件故障"}, {"h": {"name": "相线", "pos": [85, 87]}, "t": {"name": "带电", "pos": [16, 18]}, "relation": "部件故障"}, {"h": {"name": "相线", "pos": [85, 87]}, "t": {"name": "接地", "pos": [87, 89]}, "relation": "部件故障"}, {"h": {"name": "零线", "pos": [90, 92]}, "t": {"name": "断线", "pos": [92, 94]}, "relation": "部件故障"}, {"h": {"name": "三相", "pos": [95, 97]}, "t": {"name": "负荷不平衡", "pos": [97, 102]}, "relation": "部件故障"}, {"h": {"name": "接地电阻", "pos": [103, 107]}, "t": {"name": "不符", "pos": [107, 109]}, "relation": "性能故障"}, {"h": {"name": "相线", "pos": [141, 143]}, "t": {"name": "不完全接地", "pos": [143, 148]}, "relation": "部件故障"}, {"h": {"name": "三相", "pos": [149, 151]}, "t": {"name": "负荷不平衡", "pos": [151, 156]}, "relation": "部件故障"}, {"h": {"name": "三相", "pos": [149, 151]}, "t": {"name": "接地电阻不符合要求", "pos": [157, 166]}, "relation": "部件故障"}, {"h": {"name": "零线", "pos": [189, 191]}, "t": {"name": "断线", "pos": [191, 193]}, "relation": "部件故障"}, {"h": {"name": "电缆分支箱", "pos": [294, 299]}, "t": {"name": "停电", "pos": [300, 302]}, "relation": "部件故障"}, {"h": {"name": "出线负荷", "pos": [303, 307]}, "t": {"name": "断开", "pos": [308, 310]}, "relation": "性能故障"}]}
```



## 2.3 pretrain_models # 预训练模型文件夹

- 用于存放预训练模型的文件夹
- 所采用的预训练模型为：RoBERTa-wwm-large
  - **RoBERTa**：“Robust optimize bert approach” 相对于Bert的改进：更多的数据、更多的训练步数、更大的批次(8000)，用字节进行编码以解决未发现词的问题。去除了Next Sentence任务。
  - **wwm**：“whole word masking”（对全词进行mask），谷歌2019年5月31日发布，对bert的升级，主要更改了原预训练阶段的训练样本生成策略。改进：用mask标签替换一个完整的词而不是字。
  - **large**：？（没找到啥意思）
- 下载地址：链接: https://pan.baidu.com/s/1COlBY1k9yHoGXAZdEpdbWA?pwd=dgre 提取码: dgre

### 2.3.1 config.json  # 

### 2.3.2 pytorch_model.bin # 预训练模型

- RoBERTa-wwm-large本体

### 2.3.3 vocab.txt  # 



## 2.4 主要执行文件

### 2.4.1 requirements.txt # 依赖库

1. 代码如下：

```xml
bert4keras==0.11.3
numpy==1.23.2
torch==1.10.1+cu113
tqdm==4.62.3
transformers==4.5.0
```

2. 除了导入以上依赖还需导入**tensorflow**
3. 命令行如下：

```pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --default-time=2000
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-time=2000 tensorflow
```

4. 其他问题：如果“torch==1.10.1+cu113”安装报错可通过conda命令安装。mac环境没有此依赖。

```conda
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### 2.4.2 data_generator.py # 数据处理工具

#### 2.4.2.1 main函数

```python
if __name__ == '__main__':
    #配置文件路径
    file_train_bdci = 'dataset/bdci/train_bdci.json'
    file_train = 'dataset/bdci/train.json'
    file_evalA = 'dataset/bdci/evalA.json'
    file_test = 'dataset/bdci/test.json'

    #训练集生成器
    train_generator(file_train_bdci, file_train)
    #测试集生成器
    test_generator(file_evalA, file_test)
```

#### 2.4.2.2 train_generator函数

```python
"""
训练集生成器

Parameters:
    file_train_bdci - 训练集(dataset/bdci/train_bdci.json)
    file_train - 经过处理的训练集(dataset/bdci/train.json)
"""
def train_generator(file_train_bdci, file_train):
    #加载训练集
    with open(file_train_bdci, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        result_arr = []
        #生成经过处理的训练集
        with open(file_train, 'w', encoding='utf-8') as fw:
            for line in lines:
                line = line.strip()
                if line == "":
                    continue

                dic_single = {}
                text_count = []

                #首先读取train_bdci.json中的"ID"、"text"和"spo_list"信息
                line = json.loads(line)
                line_id = line['ID']
                line_text = line['text']
                spo_list = line['spo_list']

                dic_single['id'] = line_id
                dic_single['text'] = line_text
                dic_single['spos'] = []

                if line_text in result_arr:
                    continue
                    
				#判断text内容长度是否大于200
                if len(line_text) > 200:
                    #大于200，则需要进行剪裁，根据结束符"，"、"。"、"！"等划分句子
                    line_arr = []
                    text = ''
                    for l in range(len(line_text)):
                        text += line_text[l]
                        if line_text[l] in ['，', '。', '！', '？', '、']:
                            line_arr.append(text)
                            text = ''
                        if l == len(line_text) - 1 and text != line_arr[-1]:
                            line_arr.append(text)

                    text_new = ''
                    n = 0
                    
                    #遍历每个分割出来的句子，添加到text_new中。
                    for i in range(len(line_arr)):
                        dic_single_new = {}
                        text_original = text_new
                        text_new += line_arr[i]
                        #如果text_new累计句子的长度没超过200，
                        #则直接通过get_spos获取text_new的spo_list
                        if len(text_new) <= 200:
                            if i == len(line_arr) - 1:
                                # id_new = line_id + '_' + str(n)
                                # out = {'id': id_new, 'text': text_new}
                                text_count.append(text_new)
                                #调用get_spos()函数
                                spos_new1 = get_spos(dic_single, spo_list, text_count)
                                dic_single_new['id'] = line_id
                                dic_single_new['text'] = text_new
                                dic_single_new['spos'] = spos_new1
                                result_arr.append(dic_single_new)
                            else:
                                continue
                        #如果中间text_new累计长度超过200，
                    	#则将之前长度未超过200的句子通过get_spos获取spo_list，
                        else:
                            # id_new = line_id + '_' + str(n)
                            # out = {'id': id_new, 'text': text_original}
                            text_count.append(text_original)
                            spos_new1 = get_spos(dic_single, spo_list, text_count)
                            dic_single_new['id'] = line_id
                            dic_single_new['text'] = text_original
                            dic_single_new['spos'] = spos_new1
                            result_arr.append(dic_single_new)
                            #然后text_new从该句开始，继续计数
                            text_new = line_arr[i]
                            n += 1
                            if i == len(line_arr) - 1:
                                # id_new = line_id + '_' + str(n)
                                # out = {'id': id_new, 'text': text_new}
                                text_count.append(text_new)
                                spos_new1 = get_spos(dic_single, spo_list, text_count)
                                dic_single_new['id'] = line_id
                                dic_single_new['text'] = text_new
                                dic_single_new['spos'] = spos_new1
                                result_arr.append(dic_single_new)
                #<=200，则直接将spos信息记录到输出结果result_arr中
                else:
                    for spo in spo_list:
                        h = spo['h']
                        t = spo['t']
                        relation = spo['relation']

                        arr_h = []
                        arr_h.append(h['pos'][0])
                        arr_h.append(h['pos'][1])
                        arr_h.append(h['name'])

                        arr_t = []
                        arr_t.append(t['pos'][0])
                        arr_t.append(t['pos'][1])
                        arr_t.append(t['name'])

                        arr_spo = []
                        arr_spo.append(arr_h)
                        arr_spo.append(relation)
                        arr_spo.append(arr_t)
                        dic_single['spos'].append(arr_spo)

                    result_arr.append(dic_single)

            print('train:', len(result_arr))
            result_json = json.dumps(result_arr, ensure_ascii=False, indent=2)
            fw.write(result_json)
```

#### 2.4.2.3 test_generator函数

```python
"""
测试集生成器

Parameters:
    file_evalA - 测试集(dataset/bdci/evalA.json)
    file_test - 经过处理的测试集(dataset/bdci/test.json)
"""
def test_generator(file_evalA, file_test):
    #加载测试集，数据结构为：{id: "", text: ""}
    with open(file_evalA, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        result_arr = []
		#生成经过处理的测试集
        with open(file_test, 'w', encoding='utf-8') as fw:
            for line in lines:
                line = json.loads(line)
                line_id = line['ID']
                line_text = line['text']
				#判断text长度是否大于200
                if len(line_text) > 200:
                    #大于200，则需要进行剪裁，根据结束符"，"、"。"、"！"等划分句子
                    line_arr = []
                    text = ''
                    for l in range(len(line_text)):
                        text+=line_text[l]
                        if line_text[l] in ['，', '。', '！', '？', '、']:
                            line_arr.append(text)
                            text = ''
                        if l == len(line_text) - 1 and text != line_arr[-1]:
                            line_arr.append(text)

                    text_new = ''
                    n = 0
                    #遍历每个分割出来的句子，添加到text_new中。
                    for i in range(len(line_arr)):
                        text_original = text_new
                        text_new+=line_arr[i]
                        #如果text_new累计句子的长度没超过200，
                        #则直接添加到输出结果result_arr中
                        if len(text_new) <= 200:
                            if i == len(line_arr) - 1:
                                id_new = line_id + '_' + str(n)
                                out = {'id': id_new, 'text': text_new}
                                result_arr.append(out)
                            else:
                                continue
                        #如果中间text_new累计长度超过200，
                        #则将之前长度未超过200的句子，添加到输出结果result_arr中
                        else:
                            id_new = line_id + '_' + str(n)
                            out = {'id':id_new, 'text':text_original}
                            result_arr.append(out)
                            #然后text_new从该句开始，继续计数
                            text_new = line_arr[i]
                            n+=1
                            if i == len(line_arr) - 1:
                                id_new = line_id + '_' + str(n)
                                out = {'id': id_new, 'text': text_new}
                                result_arr.append(out)
                #<=200，则直接将信息记录到输出结果result_arr中
                else:
                    out = {'id': line_id, 'text': line_text}
                    result_arr.append(out)

            print('test:', len(result_arr))
            result_json = json.dumps(result_arr, ensure_ascii=False, indent=2)
            fw.write(result_json)
```

#### 2.4.2.4 get_spos函数

```python
"""
将原句子spos信息输入到dic_single中，然后处理每个三元组中pos属性的起始和终止的偏移信息

Parameters:
    dic_single - [id: "", text: "", spos: [start: "", end: "", 实体名: ""]]
    spo_list - 训练集train_bdci.json中的spo列表
    text_count - 截取的句子，输出为spos信息
"""
def get_spos(dic_single, spo_list, text_count):
    dic_single_copy = copy.deepcopy(dic_single)
    for spo in spo_list:
        h = spo['h']
        t = spo['t']
        relation = spo['relation']

        arr_h = []
        arr_h.append(h['pos'][0])
        arr_h.append(h['pos'][1])
        arr_h.append(h['name'])

        arr_t = []
        arr_t.append(t['pos'][0])
        arr_t.append(t['pos'][1])
        arr_t.append(t['name'])

        arr_spo = []
        arr_spo.append(arr_h)
        arr_spo.append(relation)
        arr_spo.append(arr_t)
        dic_single_copy['spos'].append((arr_spo))

    spos_new = sorted(dic_single_copy['spos'], key=lambda x: x[0])
    spos_new = sorted(spos_new, key=lambda x: x[2])

    spos_new1 = []
    s_idx = 0
    e_idx = 0

    for s in text_count[:-1]:
        s_idx+=len(s)
    for e in text_count:
        e_idx += len(e)

    # print(spos_new)
    for spo in spos_new:
        # print(spo)
        if spo[0][0] >= s_idx and spo[-1][1] <= e_idx:
            spo[0][0] = spo[0][0] - s_idx
            spo[0][1] = spo[0][1] - s_idx
            spo[2][0] = spo[2][0] - s_idx
            spo[2][1] = spo[2][1] - s_idx
            spos_new1.append(spo)
        else:
            continue

    # print(spos_new1)
    # exit()
    return spos_new1
```



### 2.4.3 data_utils.py # 数据处理类

#### 2.4.3.1 DataGenerator类

```python
"""
本类为bert4keras，一个基于[keras](https://keras.io/)的预训练模型加载框架所实现的数据生成类
"""
class DataGenerator(object):
    """
    初始化步长steps为(数据长度÷批长度)，将buffer_size设置为(1000×批长度)
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size # 5019//6==836
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000 # 6*1000==6000

    def __len__(self):
        return self.steps
    
    """
    是作者实现好、定义好的生成器，可以每次从“原始数据列表（全部训练数据或valid数据）”中拿出一条数据，每次返回值有两个(is_end, d_current)，如果这个列表全部拿完了，is_end则为True，否则为False，来指示是否“全部训练(or 验证)数据”被读完。注意，这里所获取的每条d_current都还是原始数，一般都是“人可读”的标注数据。
    """
    def sample(self, random=False):
        #首先判断是否需要随机输出数据
        if random:
            #如果需要随机输出，则判断步长self.steps是否为空
            if self.steps is None:
                #如果步长self.steps为null，
                #则使用caches数组和一个isfull标识符，每次添加一个数据进入到数组中，
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            #如果buffer_size已满，
                            #则使用随机数生成的方式，随机输出caches中的一个数据
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)
			#如果步长self.steps不为null，
            #则使用np.random.shuffle函数，将所有数据随机排序，然后输出
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        #不需要随机输出，则使用迭代器输出数据
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next
		
        #最后输出is_end和数据d_current
        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d
```



### 2.4.4 train.py # 训练代码

#### 2.4.4.1 main函数

```python
if __name__ == '__main__':
	#创建一个解析器，添加参数 同predict
    parser = argparse.ArgumentParser(description='Model Controller')
	#指定GPU
    parser.add_argument('--cuda_id', default="2", type=str)

    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--train', default="train", type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--val_batch_size', default=4, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    parser.add_argument('--bert_vocab_path', default="./pretrain_models/vocab.txt", type=str)
    parser.add_argument('--bert_config_path', default="./pretrain_models/config.json", type=str)
    parser.add_argument('--bert_model_path', default="./pretrain_models/pytorch_model.bin", type=str)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--min_num', default=1e-7, type=float)
    parser.add_argument('--base_path', default="./dataset", type=str)
    parser.add_argument('--output_path', default="output", type=str)

    args = parser.parse_args()
    
    #调用训练函数
    train()
```

#### 2.4.4.2 train函数

```python
def train():
    #随机数种子
    set_seed()
   	#设置使用的显卡
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id
    #默认output
    output_path=os.path.join(args.output_path)
    #默认./dataset/bdci/train.json
    train_path=os.path.join(args.base_path,args.dataset,"train.json")
    #默认./dataset/bdci/rel2id.json
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    #默认output/dev_pred.json
    dev_pred_path=os.path.join(output_path,"dev_pred.json")
    #默认output/log.txt
    log_path=os.path.join(output_path,"log.txt")

    #label
    label_list=["N/A","SMH","SMT","SS","MMH","MMT","MSH","MST"]
    # id2label:{'0': 'N/A', '1': 'SMH', '2': 'SMT', '3': 'SS', '4': 'MMH', '5': 'MMT', '6': 'MSH', '7': 'MST'}
    # label2id:{'N/A': 0, 'SMH': 1, 'SMT': 2, 'SS': 3, 'MMH': 4, 'MMT': 5, 'MSH': 6, 'MST': 7}
    id2label,label2id={},{}

    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i
	#从路径./dataset/bdci/train.json加载训练集
    train_data = json.load(open(train_path,'r',encoding='utf-8'))
	#np.array()转换为数组
    all_data = np.array(train_data)
	#数组顺序打乱
    random.shuffle(all_data)
	#划分训练集与验证集
    num = len(all_data)
    split_num = int(num*0.8)
    train_data = all_data[ : split_num]
    dev_data = all_data[split_num : ]

    id2predicate, predicate2id = json.load(open(rel2id_path,'r',encoding='utf-8'))
	#直接传入一个vocab.txt的路径创建分词器
    tokenizer = Tokenizer(args.bert_vocab_path)
	#BertConfig是Bert模型的配置类
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)#=4？
    config.num_label=len(label_list)#=8？
    config.rounds=args.rounds#default=4
    #嵌入层，将token变成矩阵或张量
    config.fix_bert_embeddings=args.fix_bert_embeddings
	#定义了一个训练模型，这个模型就是GRTE吧，from_pretrained()是一个黑盒，具体模型要看class GRTE
    train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")
	#创建输出路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)
    #加载数据集，按batchsize取出数据
    dataloader = data_generator(args,train_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.batch_size,random=True)
    dev_dataloader=data_generator(args,dev_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.val_batch_size,random=False,is_train=False)
    #定义一个变量t_total来计算总共要训练的次数，用于后面对学习率进行优化：
    t_total = len(dataloader) * args.num_train_epochs
	#偏差，权重
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    #使用Adamw优化器来进行梯度下降：
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    #选择好优化器之后，我们创建一个scheduler对象，使用warmup来对学习率进行优化：
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )
    #记录最优的F1_SCORE
    best_f1 = -1.0
    #iteration表示1次迭代
    step = 0
    #交叉熵损失函数
    crossentropy=nn.CrossEntropyLoss(reduction="none")

    for epoch in range(args.num_train_epochs):
        #使用model.train()声明当前模型处于训练状态
        train_model.train()
        #计算所有epoch的loss总和
        epoch_loss = 0
        print('current epoch: %d\n' % (epoch))
        with tqdm(total=dataloader.__len__(), desc="train") as t:
            #将训练集dataloader放入迭代器里，然后按照batch的大小取出数据
            for i, batch in enumerate(dataloader):
                #前向传播
                batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
                batch_token_ids, batch_mask,batch_label,batch_mask_label= batch
                table = train_model(batch_token_ids, batch_mask)
                table=table.reshape([-1,len(label_list)])
                batch_label=batch_label.reshape([-1])

                loss=crossentropy(table,batch_label.long())
                loss=(loss*batch_mask_label.reshape([-1])).sum()
                #反向传播
                loss.backward()
                step += 1
                #计算损失函数
                epoch_loss += loss.item()
                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                #更新参数
                optimizer.step()
                #更新学习率
                scheduler.step()
                #清空梯度信息
                train_model.zero_grad()

                t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                t.update(1)

        f1, precision, recall = evaluate(args,tokenizer,id2predicate,id2label,label2id,train_model,dev_dataloader,dev_pred_path)
		#保存F1_SCORE的最高的模型到我们设定好的路径
        if f1 > best_f1:
            # Save model checkpoint
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))

        epoch_loss = epoch_loss / dataloader.__len__()
		#打开训练日志，输出日志中每个epoch的epoch_loss, f1, precision, recall, best_f1
        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f\t" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)

	#输出最终模型（保留的最好的？）测试的f1, precision, recall, best_f1
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,id2label,label2id,train_model,dev_dataloader,dev_pred_path)
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)
```

#### 2.4.4.3 evaluate函数

```python
"""
评估模型
计算f1 score,precision,recall
具体实现不懂
"""
def evaluate(args,tokenizer,id2predicate,id2label,label2id,model,dataloader,evl_path):

    """
    X=TP,Y=TP+FP,Z=TP+FN
    1/F1=1/2*(1/precision+1/recall)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    #进度条
    pbar = tqdm()

    for batch in dataloader:

        batch_ex=batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch

        batch_spo=extract_spo_list(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex,batch_token_ids, batch_mask)
        for i,ex in enumerate(batch_ex):
            one = batch_spo[i]
            R = set([(tuple(item[0]), item[1], tuple(item[2])) for item in one])
            T = set([(tuple(item[0]), item[1], tuple(item[2])) for item in ex['spos']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': ex['text'],
                'spos_list': list(T),
                'spos_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
    pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall
```





### 2.4.5 predict.py # 预测代码



#### 2.4.5.1 main函数

```python
"""
main函数，负责配置参数，执行预测
"""
if __name__ == '__main__':
	#创建一个解析对象
    parser = argparse.ArgumentParser(description='Model Controller')
    """
    向该对象中添加参数
        cuda_id 1	表示选择的设备torch.cuda.set_device(int(args.cuda_id))
        dataset bdci 数据为bdci数据集
        rounds 4 执行次数为4
        test_batch_size 1 测试的批大小为1（指一次性有多少个测试数据输入到网络）
        fix_bert_embeddings False 修复BERT嵌入？ 没懂什么意思？？
        bert_vocab_path 预训练BERT词典的路径
        bert_config_path 预训练BERT配置文件的路径
        bert_model_path 预训练BERT模型的路径
        max_len 200 最大长度为200 ？什么长度？
        base_path 基础路径（应该是指数据集的路径）
        output_path 输出路径（执行结果的报告输出到output目录）
    """
    parser.add_argument('--cuda_id', default="1", type=str)
    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    parser.add_argument('--bert_vocab_path', default="./pretrain_models/vocab.txt", type=str)
    parser.add_argument('--bert_config_path', default="./pretrain_models/config.json", type=str)
    parser.add_argument('--bert_model_path', default="./pretrain_models/pytorch_model.bin", type=str)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--base_path', default="./dataset", type=str)
    parser.add_argument('--output_path', default="output", type=str)
	#通过args使用以上参数
    args = parser.parse_args()
    
	#执行预测函数
    predict()
```

#### 2.4.5.2 predict函数

```python
"""
predict函数,模型预测
"""
def predict():
    try:
        #作用：设置当前设备。
        #参数：device(int)表示选择的设备。如果此参数为负，则此函数是无操作的
        torch.cuda.set_device(int(args.cuda_id))
    except:
        #处理异常
        os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id
        
	#拼接地址
    #output_path=output
    output_path=os.path.join(args.output_path)
    #test_path=bdci/test.json
    test_path=os.path.join(args.base_path,args.dataset,"test.json")
    #rel2id_path=bdci/rel2id.json
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    #test_pred_path=output/test_pred.json
    test_pred_path = os.path.join(output_path, "test_pred.json")
    
    """
    label 单元格填充的标签集合
    每个字段由三个字母组成（详见https://www.jianshu.com/p/5ebe68f24848）：
    	第一个字母：Subject是多个token(M),还是单个token(S)
    	第二个字母：Object是多个token(M),还是单个token(S)
    	第三个字母：token pair(w_i,w_j)是两个实体的开头(H)或结尾(T)
    	不存在上述关系的用N/A填充
    """
    label_list=["N/A","SMH","SMT","SS","MMH","MMT","MSH","MST"]
    
    """
    声明两个集合id2label(id到label的映射)和label2id(label到id的映射)
    id-label={'0':'N/A','1':'SMH','2':'SMT','3':'SS','4':'MMH','5':'MMT'
    			,'6':'MSH','7':'MST'}
    label-id={'N/A':0,'SMH':1,'SMT':2,'SS':3,'MMH':4,'MMT':5,'MSH':6,'MST':7}
    """
    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i
	
    #打开并读取测试数据的json文件(bdci/test.json)
    test_data = json.load(open(test_path, 'r', encoding= 'utf-8'))
    #打开并读取关系到id的映射的json文件，按照对应映射关系分别存进两个变量中
    id2predicate, predicate2id = json.load(open(rel2id_path, 'r', encoding= 'utf-8'))

    """
    Tokenizer为Bert原生分词器,纯Python实现，代码修改自keras_bert的tokenizer实现
    将预训练BERT词典传入分词器
    """
    tokenizer = Tokenizer(args.bert_vocab_path)
    #获取预训练模型BERT的配置？
    config = BertConfig.from_pretrained(args.bert_config_path)
    #获取rel2id.json的长度
    config.num_p=len(id2predicate)
    #获取label_list的长度
    config.num_label=len(label_list)
    #预测轮数
    config.rounds=args.rounds
    #修复BERT嵌入？ 没懂什么意思？？
    config.fix_bert_embeddings=args.fix_bert_embeddings
	#加载预训练模型
    train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    #将训练模型加载到相应的设备(cuda)中
    train_model.to("cuda")
	
    #如果不存在output目录那就创建一个
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #调用util类中的打印配置函数，生成config.txt文件
    print_config(args)

    #调用util类中的数据生成器,将测试数据(bdci/test.json)导入数据生成器中
    test_dataloader=data_generator(args,test_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)

    #将训练好的模型的参数重新加载到新的模型之中(output/pytorch_model.bin)
    #map_location参数是用于重定向，比如此前模型的参数是在cpu中的，我们希望将其加载到cuda:0
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    
    #调用测试评估函数
    evaluate_test(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path)
    
    #调用准确度函数？
    correct_id(args, test_pred_path)
```

#### 2.4.5.3 evaluate_test函数：

```python
"""
测试评估

Parameters:
    args - mian函数输入参数
    tokenizer - 分词器
    id2predicate - id到关系的映射
    id2label - id到label的映射
    label2id - label到id的映射
    train_model - 训练好的模型
    test_dataloader - 测试数据生成器(包含bdci/test.json中的数据)
    test_pred_path - 预测测试集路径(output/test_pred.json)
"""
def evaluate_test(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path):
    #打开测试集预测结果(output/test_pred.json)
    f = open(test_pred_path, 'w', encoding='utf-8')

    """
    遍历测试数据生成器中每一条元素
    @variable batch(截取前):
    	[array([[ 101, ...,  102]]), array([[0, ..., 0]]), [{'id': 'AE0006', 'text': '故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失'}]
    @variable batch_ex:
    	[{'id': 'AE0006', 'text': '故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失'}]
    @variable batch(截取后):
    	[tensor([[ 101,...,  102]],  device='cuda:0', dtype=torch.int32), tensor([[0,..., 0]], device='cuda:0', dtype=torch.int32)]
    @variable batch_token_ids: 
    	tensor([[ 101,...,  102]],  device='cuda:0', dtype=torch.int32)
    @variable batch_mask: 
    	tensor([[0,..., 0]], device='cuda:0', dtype=torch.int32)
    @variable batch_spo:
    	[[((19, 22, '转向机'), '部件故障', (27, 29, '缺油')), ((23, 26, '转向轴'), '部件故障', (27, 29, '缺油'))]]
    	
	batch[-1] 取最后一个元素
	batch[:-1] 除了最后一个取全部
	batch[::-1] 取从后向前（相反）的元素
	batch[n::-1] 取从下标为n的元素开始向前的元素
	"""
    for batch in test_dataloader:
        batch_ex=batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch

        #调用util类中抽取spo列表函数，获得batch的spo列表
        batch_spo=extract_spo_list(args, tokenizer, id2predicate,id2label,label2id, train_model, batch_ex,batch_token_ids, batch_mask)
        #调用预测数据函数
        predict_data(batch_ex, batch_spo, f)
		#关闭预测测试数据集文件
    f.close()
```

#### 2.4.5.4 Predict_data函数

```python
"""
预测数据
将test.json中的数据进行关系预测，写入test_pred.json(测试集预测结果)文件中

Parameters:
    batch_ex - test.json中的一条数据，batch[]的最后一项数据
    batch_spo - batch_ex的spo列表
    f - 测试集预测结果文件(output/test_pred.json)
"""
def predict_data(batch_ex, batch_spo, f):
  	#断言
    assert len(batch_ex) == len(batch_spo)

    """
    预测spo_list
    组装:"ID","text","spo_list"
    向测试集预测结果文件里写数据，如：
    {"ID": "AE0006", 
     "text": "故障现象：转向时有“咯噔”声原因分析：转向机与转向轴处缺油解决措施：向此处重新覆盖一层润滑脂后，故障消失", 
     "spo_list": [
          {"h": {"name": "转向机", "pos": [19, 22]}, 
            "t": {"name": "缺油", "pos": [27, 29]}, 
            "relation": "部件故障"
          }, 
          {"h": {"name": "转向轴", "pos": [23, 26]}, 
            "t": {"name": "缺油", "pos": [27, 29]}, 
            "relation": "部件故障"
          }
       ]
    }
    """
    for i in range(len(batch_ex)):
        sample = {}
        spo_list = []
        ex = batch_ex[i]
        spo = batch_spo[i]
        sample['ID'] = ex['id']
        sample['text'] = ex['text']

        for s in spo:
            h = {}
            t = {}

            h['name'] = s[0][-1]
            h['pos'] = [s[0][0], s[0][1]]
            t['name'] = s[2][-1]
            t['pos'] = [s[2][0], s[2][1]]

            spo_list.append({'h':h, 't':t, 'relation':s[1]})

        sample['spo_list'] = spo_list
        s = json.dumps(sample, ensure_ascii=False)

        f.write(s + '\n')
```

#### 2.4.5.5 correct_id函数

```python
"""
准确度？
将output/test_pred.json中的多条数据合并成一条，并写入evalResult.json(模型预测结果)文件
	如将：
    {"ID": "AE0001_0", "text":"1","spo_list": [{1}]}
    {"ID": "AE0001_1", "text":"2","spo_list": [{2}]}
    {"ID": "AE0001_2", "text":"3","spo_list": [{3}]}
	合并为：
	{"ID": "AE0001", "text":"123","spo_list": [{1},{2},{3}]}
	
Parameters:
    args - mian函数输入参数
    test_pred_path - 测试集预测结果路径(output/test_pred.json)
"""
def correct_id(args, test_pred_path):
  	#获取output路径
    output_path = os.path.join(args.output_path)
    #获取模型预测结果集路径(output/evalResult.json)
    test_result_path = os.path.join(output_path, "evalResult.json")

    #创建新行集
    lines_new = {}
    #打开测试集预测结果
    with open(test_pred_path, 'r', encoding='utf-8') as f:
      	#按行读
        lines = f.readlines()
        #打开模型预测结果集文件
        with open(test_result_path, 'w', encoding='utf-8') as fw:
          	#遍历每行
            for line in lines:
              	#去除每行结尾的\n
                line = json.loads(line.strip('\n'))
                #拆分出样本id，取_之前的字段（AE0001_0取AE0001）
                sample_id = line['ID'].split('_')[0]
                #将该样本加入新行集，格式为{sample_id: [line]}
                if sample_id not in lines_new:
                    lines_new[sample_id] = []
                    lines_new[sample_id].append(line)
                else:
                    lines_new[sample_id].append(line)

            #合并"text"和"spo_list"
            for key, value in lines_new.items():
                if len(value) > 1:
                    text = ''
                    sample = {}
                    spo_lists = []
                    sample['ID'] = key
                    for i in range(len(value)):
                        text += value[i]['text']

                        total_len = len(text)
                        current_len = len(value[i]['text'])
                        for s in value[i]['spo_list']:
                            h = {}
                            t = {}

                            h['name'] = s['h']['name']
                            h['pos'] = [s['h']['pos'][0] + (total_len - current_len), 
                                        s['h']['pos'][1] + (total_len - current_len)]
                            t['name'] = s['t']['name']
                            t['pos'] = [s['t']['pos'][0] + (total_len - current_len), 
                                        s['t']['pos'][1] + (total_len - current_len)]

                            spo_lists.append({'h':h,'t':t,'relation':s['relation']})

                    print(spo_lists)

                    sample['text'] = text
                    sample['spo_list'] = spo_lists
                    result = json.dumps(sample, ensure_ascii=False)
                    fw.write(result + '\n')
                else:
                    result = json.dumps(value[0], ensure_ascii=False)
                    fw.write(result + '\n')
```

#### 2.4.5.6 所用到的util工具类函数

1.   [print_config函数](#2.4.7.1 print_config(args)函数)
2.   [extract_spo_list函数](#2.4.7.8 extract_spo_list()函数)
3.   [data_generator类](#2.4.7.10 data_generator类)



### 2.4.6 model.py # 模型类



### 2.4.7 util.py # 工具类

#### 2.4.7.1 print_config(args)函数

```python
def print_config(args):
    config_path=os.path.join(args.output_path,"config.txt")
    with open(config_path,"w",encoding="utf-8") as f:
        for k,v in sorted(vars(args).items()):
            print(k,'=',v,file=f)
```



#### 2.4.7.8 extract_spo_list()函数

```python
def extract_spo_list(args, tokenizer, id2predicate, id2label, label2id, model, batch_ex, batch_token_ids, batch_mask):

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        table = model(batch_token_ids, batch_mask)
        table = table.cpu().detach().numpy()

    def get_pred_id(table, all_tokens):

        B, L, _, R, _ = table.shape
        res = []
        for i in range(B):
            res.append([])
        table = table.argmax(axis=-1)  # BLLR
        all_loc = np.where(table != label2id["N/A"])
        res_dict = []
        for i in range(B):
            res_dict.append([])
        for i in range(len(all_loc[0])):
            token_n = len(all_tokens[all_loc[0][i]])
            if token_n - 1 <= all_loc[1][i] \
                    or token_n - 1 <= all_loc[2][i] \
                    or 0 in [all_loc[1][i], all_loc[2][i]]:
                continue
            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])

        for i in range(B):
            for l1, l2, r in res_dict[i]:
                if table[i, l1, l2, r] == label2id["SS"]:
                    res[i].append([l1, l1, r, l2, l2])
                elif table[i, l1, l2, r] == label2id["SMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "SMT"] and l1_ == l1 and l2_ > l2:
                            res[i].append([l1, l1, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MMT"] and l1_ > l1 and l2_ > l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MSH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MST"] and l1_ > l1 and l2_ == l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
        return res

    all_tokens = []
    for ex in batch_ex:
        tokens = tokenizer.tokenize(ex["text"], maxlen=args.max_len)
        all_tokens.append(tokens)

    res_id = get_pred_id(table, all_tokens)
    batch_spo = [[] for _ in range(len(batch_ex))]
    for b, ex in enumerate(batch_ex):
        text = ex["text"]
        tokens = all_tokens[b]
        mapping = tokenizer.rematch(text, tokens)
        for sh, st, r, oh, ot in res_id[b]:
            s = (mapping[sh][0], mapping[st][-1])
            o = (mapping[oh][0], mapping[ot][-1])
            batch_spo[b].append(
                ((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)], (o[0], o[1] + 1, text[o[0]:o[1] + 1])) #？
            )
    return batch_spo
```

  

#### 2.4.7.10 data_generator类

```python
"""
数据生成器类
继承DataGenerator类(data_utils.py)
"""
class data_generator(DataGenerator):
    """
    初始化
    Parameters:
        args - 参数配置对象
        train_data - 测试数据文件(bdci/test.json)
        tokenizer - 分词器
        predicate_map - 两种映射组成的map [predicate2id,id2predicate]
        label_map - 两种映射组成的map [label2id,id2label]
        batch_size - 预测时的批大小 args.test_batch_size
        random - 随机 False
        is_train - xx False
    """
    def __init__(self, args, train_data, tokenizer, predicate_map, label_map, batch_size, random=False, is_train=True):
        super(data_generator, self).__init__(train_data, batch_size)
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.predicate2id, self.id2predicate = predicate_map
        self.label2id, self.id2label = label_map
        self.random = random
        self.is_train = is_train
        
    """
    还没看
    """
    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label = []
        batch_mask_label = []
        batch_ex = []
        for is_end, d in self.sample(self.random):
            if self.is_train:
                if judge(d) == False:
                    continue
            token_ids, mask = self.tokenizer.encode(
                d['text'], maxlen=self.max_len
            )
            if self.is_train:
                entities = []
                for spo in d['spos']:
                    entities.append(tuple(spo[0]))
                    entities.append(tuple(spo[2]))
                entities = sorted(list(set(entities)))
                one_info = get_token_idx(d['text'], entities, self.tokenizer)
                spoes = {}
                for ss, pp, oo in d['spos']:
                    s_key = (ss[0], ss[1])
                    p = self.predicate2id[pp]
                    o_key = (oo[0], oo[1])
                    s = tuple(one_info[s_key])
                    o = copy.deepcopy(one_info[o_key])
                    o.append(p)
                    o = tuple(o)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)

                if spoes:
                    label = np.zeros([len(token_ids), len(token_ids), len(self.id2predicate)])
                    for s in spoes:
                        s1, s2 = s
                        try:
                            for o1, o2, p in spoes[s]:
                                try:
                                    if s1 == s2 and o1 == o2:
                                        label[s1, o1, p] = self.label2id["SS"]
                                    elif s1 != s2 and o1 == o2:
                                        label[s1, o1, p] = self.label2id["MSH"]
                                        label[s2, o1, p] = self.label2id["MST"]
                                    elif s1 == s2 and o1 != o2:
                                        label[s1, o1, p] = self.label2id["SMH"]
                                        label[s1, o2, p] = self.label2id["SMT"]
                                    elif s1 != s2 and o1 != o2:
                                        label[s1, o1, p] = self.label2id["MMH"]
                                        label[s2, o2, p] = self.label2id["MMT"]
                                except:

                                    print(d, spoes)
                        except Exception as e:
                            print(one_info, d['text'])
                            assert 0

                    mask_label = np.ones(label.shape)
                    mask_label[0, :, :] = 0
                    mask_label[-1, :, :] = 0
                    mask_label[:, 0, :] = 0
                    mask_label[:, -1, :] = 0

                    for a, b in zip([batch_token_ids, batch_mask, batch_label, batch_mask_label, batch_ex],
                                    [token_ids, mask, label, mask_label, d]):
                        a.append(b)

                    if len(batch_token_ids) == self.batch_size or is_end:
                        batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                        batch_label = mat_padding(batch_label)
                        batch_mask_label = mat_padding(batch_mask_label)
                        yield [
                            batch_token_ids, batch_mask,
                            batch_label,
                            batch_mask_label, batch_ex
                        ]
                        batch_token_ids, batch_mask = [], []
                        batch_label = []
                        batch_mask_label = []
                        batch_ex = []
            else:
                for a, b in zip([batch_token_ids, batch_mask, batch_ex], [token_ids, mask, d]):
                    a.append(b)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    yield [
                        batch_token_ids, batch_mask, batch_ex
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_ex = []
```





# 3. 原理分析



# 4. 未做完的工作

1. model.py没看
1. util.py没看完
1. 对data_generator，train，predict进行讨论，修改
1. 完善概念
1. 理解原理
1. 服务器跑通20轮训练
1. 改进baseline...
