#! /usr/bin/env python

import io
from flask import Flask, render_template, request
from model import load_saved_model, predict
from PIL import Image

app = Flask(__name__)
net = load_saved_model()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/eval", methods=["POST"])
def eval():
    image = Image.open(io.BytesIO(request.data)).convert("L")
    result = predict(net, image)

    # Dictionary return values are implicitly converted to JSON,
    # which the client can parse to get the result.
    return {"result": result}


if __name__ == "__main__":
    app.run(host="0.0.0.0")
