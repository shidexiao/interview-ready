from flask import Blueprint, request
from loguru import logger

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    logger.debug(f"Login attempt with data: {request.json}")
    # 处理登录逻辑
    return "Login successful", 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    logger.debug("User logged out")
    # 处理登出逻辑
    return "Logout successful", 200
