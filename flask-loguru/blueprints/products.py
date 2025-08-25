from flask import Blueprint, request
from loguru import logger

products_bp = Blueprint('products', __name__)

@products_bp.route('/list', methods=['GET'])
def list_products():
    logger.debug("Fetching product list")
    # 模拟返回产品列表
    return {"products": ["Product1", "Product2", "Product3"]}

@products_bp.route('/add', methods=['POST'])
def add_product():
    product = request.json.get('name')
    logger.debug(f"Adding product: {product}")
    # 处理添加产品的逻辑
    return f"Product {product} added", 201
