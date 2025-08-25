from flask import Flask, request, jsonify
import grpc
import calculator_pb2
import calculator_pb2_grpc

app = Flask(__name__)


# 连接到 gRPC 服务
def get_grpc_stub():
    channel = grpc.insecure_channel("localhost:50051")
    return calculator_pb2_grpc.CalculatorStub(channel)


# Flask 路由，处理 HTTP 请求并调用 gRPC
@app.route("/add", methods=["GET"])
def add():
    num1 = float(request.args.get("num1", 0))
    num2 = float(request.args.get("num2", 0))

    # 调用 gRPC 服务
    stub = get_grpc_stub()
    request_data = calculator_pb2.AddRequest(num1=num1, num2=num2)
    response = stub.Add(request_data)

    return jsonify({"result": response.result})


if __name__ == "__main__":
    app.run(port=5001, host="0.0.0.0")
