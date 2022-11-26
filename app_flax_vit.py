#! /usr/bin/env python
import os
import io
from flask import Flask, render_template, request
from model_flax_vit import load_saved_model
from PIL import Image

HF_VIT_MODEL_NAME = os.environ.get("HF_VIT_MODEL_NAME", "google/vit-base-patch16-224")

app = Flask(__name__)
pipeline = load_saved_model(HF_VIT_MODEL_NAME)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/eval", methods=["POST"])
def eval():
    image = Image.open(io.BytesIO(request.data)).convert("L")
    _, label_text = pipeline.predict(image)

    # Dictionary return values are implicitly converted to JSON,
    # which the client can parse to get the result.
    return {"result": label_text}


if __name__ == "__main__":
    app.run(host="0.0.0.0")
