from flask import Blueprint, request, current_app
from loguru import logger

products_bp = Blueprint('products', __name__)


# 为 products 模块添加一个独立的日志文件
logger.add("logs/products.log", level="DEBUG", rotation="1 week", retention="1 month")

@products_bp.route('/list', methods=['GET'])
def list_products():
    current_app.logger()
    logger.debug("Fetching product list")
    return {"products": ["Product1", "Product2", "Product3"]}

@products_bp.route('/add', methods=['POST'])
def add_product():
    product = request.json.get('name')
    logger.debug(f"Adding product: {product}")
    return f"Product {product} added", 201
