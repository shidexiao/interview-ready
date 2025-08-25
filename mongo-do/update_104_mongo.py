from pymongo import MongoClient

# 定义连接信息
host = "127.0.0.1"
port = 31018
username = "third_data_mg"
password = "36sT6$YoeC89M6G"
auth_db = "risk_ext_data"

# 创建连接 URI
uri = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_db}"

# 使用 pymongo 连接到 MongoDB
client = MongoClient(uri)

# 选择数据库
db = client[auth_db]

# 测试连接，比如列出数据库中的集合
print(db.list_collection_names())

# 选择数据库
db = client["risk_ext_data"]
collection = db['baihang_jxrh4000']
documents = collection.find()


def is_valid_set(data):
    ASSET_TRIGGER_A_range = {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                             'B8'}
    INCOME_GRIDA_range = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
    IND_TRIGGER_3_range = {'0', '1'}

    asset = data.get("ASSET_TRIGGER_A", "")
    income = data.get("INCOME_GRIDA", "")
    ind_trigger = data.get("IND_TRIGGER_3", "")

    bill_count = 0

    if asset in ASSET_TRIGGER_A_range:
        bill_count = bill_count + 1
    if income in INCOME_GRIDA_range:
        bill_count = bill_count + 1
    if ind_trigger in IND_TRIGGER_3_range:
        bill_count = bill_count + 1

    if asset in ASSET_TRIGGER_A_range or income in INCOME_GRIDA_range or ind_trigger in IND_TRIGGER_3_range:
        return True, bill_count
    return False, bill_count

for document in documents:
    # print(document)
    raw_res_decrypt = document['raw_res_decrypt']
    respBody = raw_res_decrypt['respBody']
    respBody_score = respBody['SCORE']
    is_valid, bill_count = is_valid_set(respBody_score)
    if is_valid:
        status = '1'
    else:
        status='2'
    doc_id = document['_id']

    print(f"{doc_id}, {document['status']},{document.get('bill_count')} - {status}, {bill_count}")
    # 创建更新操作
    update_query = {"$set": {"status": status,"bill_count":bill_count}}  # 替换为实际的更新内容

    # 执行更新
    result = collection.update_one({"_id": doc_id}, update_query)

    # 输出更新结果
    print(f"更新文档 {doc_id} 的结果：", result.modified_count)


