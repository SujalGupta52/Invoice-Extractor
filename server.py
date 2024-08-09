from flask import Flask, make_response, render_template, request
import os
import pypdfium2 as pdfium
import os
from llm_controller import LLM_controller
from PIL import Image
import pytesseract
import numpy as np


app = Flask(__name__, static_url_path="", static_folder="./static")

DIR = "invoices"
MODEL_PATH = "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

llm = LLM_controller(
    model_path=f"{MODEL_PATH}",
    verbose=True,
    temperature=0.1,
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files["file"]
        f.save(os.path.join("./invoices", f.filename))
        if f.filename.endswith(".pdf"):
            pdf = pdfium.PdfDocument(f"{DIR}/{f.filename}")
            page = pdf[0]
            img = page.render(scale=4).to_pil()
        elif f.filename.endswith((".jpeg", ".jpg", ".png")):
            img = np.array(Image.open(f"{DIR}/{f.filename}"))
        text = pytesseract.image_to_string(img)
        parsed_text = llm.generate(text)
        print(parsed_text)
        return render_template("response.html", content=parsed_text)
    if request.method == "GET":
        return render_template("index.html")


app.run("0.0.0.0", 3002)
