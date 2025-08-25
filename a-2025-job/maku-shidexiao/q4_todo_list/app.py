import datetime
from flask import Flask, jsonify, request, render_template
from models import db, Todo

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'

db.init_app(app)


@app.route('/')
def index():
    return render_template('index.html')


# API端点
@app.route('/api/todos', methods=['GET', 'POST'])
def todos():
    if request.method == 'GET':
        todos = Todo.query.filter_by(is_deleted=False).all()
        return jsonify([todo.to_dict() for todo in todos])

    elif request.method == 'POST':
        data = request.get_json()
        new_todo = Todo(content=data['content'])
        db.session.add(new_todo)
        db.session.commit()
        return jsonify(new_todo.to_dict()), 201


@app.route('/api/todos/<int:id>', methods=['PUT', 'DELETE'])
def todo(id):
    todo = Todo.query.get_or_404(id)

    if request.method == 'PUT':
        data = request.get_json()

        if 'content' in data:
            todo.content = data['content']
        if 'completed' in data:
            todo.completed = data['completed']

        todo.updated_at = datetime.datetime.utcnow()  # 每次更新都自动更新时间
        db.session.commit()

        return jsonify(todo.to_dict())

    elif request.method == 'DELETE':
        # 软删除：标记为已删除而非真正删除
        todo.is_deleted = True
        todo.updated_at = datetime.datetime.utcnow()  # 更新修改时间
        db.session.commit()
        return '', 204