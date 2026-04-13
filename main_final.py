import os
import hashlib
import numpy as np
import cv2
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
PASSWORD_ALLOWED = "242016"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



def password_to_seed(password: str):
    h = hashlib.sha256(password.encode('utf-8')).hexdigest()
    first = int(h[:16], 16)
    seed = (first % (10**8)) / float(10**8)
    if seed == 0:
        seed = 0.123456789
    last = int(h[16:32], 16)
    r = 3.99 + (last % 1000000) / 1e6 * 0.009999
    return seed, r

def logistic_map_sequence(x0: float, r: float, length: int, discard: int = 100):
    x = x0
    seq = []
    for _ in range(discard):
        x = r * x * (1 - x)
    for _ in range(length):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq)

BITS_TO_NUC = {'00':'A','01':'C','10':'G','11':'T'}
NUC_TO_BITS = {v:k for k,v in BITS_TO_NUC.items()}

def byte_to_nucleotides(b):
    bits = f"{b:08b}"
    return [BITS_TO_NUC[bits[i:i+2]] for i in range(0,8,2)]

def nucleotides_to_byte(nucs):
    bits = ''.join([NUC_TO_BITS[n] for n in nucs])
    return int(bits,2)

def dna_xor_byte(a,b):
    A = byte_to_nucleotides(a)
    B = byte_to_nucleotides(b)
    out = []
    for x,y in zip(A,B):
        r = int(NUC_TO_BITS[x],2) ^ int(NUC_TO_BITS[y],2)
        out.append(BITS_TO_NUC[f"{r:02b}"])
    return nucleotides_to_byte(out)

def dna_xor_array(arr, key):
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = dna_xor_byte(int(arr[i]), int(key[i]))
    return out

def generate_permutation_from_seq(seq):
    return np.argsort(seq)

def inverse_permutation(p):
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv

def encrypt_image(img_cv, password):
    if len(img_cv.shape) == 2:
        mode = "L"
    elif img_cv.shape[2] == 3:
        mode = "RGB"
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    else:
        mode = "RGBA"
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)

    orig_shape = img_cv.shape
    flat = img_cv.flatten().astype(np.uint8)
    n = flat.size

    seed, r = password_to_seed(password)

    seq_perm = logistic_map_sequence(seed,r,n,200)
    perm = generate_permutation_from_seq(seq_perm)
    permuted = flat[perm]

    seed2 = (seed + 0.314159265) % 1.0
    seq_key = logistic_map_sequence(seed2,r,n,300)
    key_bytes = np.floor(seq_key*256).astype(np.uint8)

    encrypted = dna_xor_array(permuted,key_bytes)
    encrypted_img = encrypted.reshape(orig_shape)

    if mode == "RGB":
        encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_RGB2BGR)
    elif mode == "RGBA":
        encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_RGBA2BGRA)

    metadata = {
        "orig_shape": orig_shape,
        "mode": mode,
        "seed": seed,
        "r": r,
        "n": n
    }
    return encrypted_img, metadata

def decrypt_image(img_cv, password, meta):
    mode = meta["mode"]
    if mode == "RGB":
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    elif mode == "RGBA":
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)

    flat = img_cv.flatten().astype(np.uint8)
    n = meta["n"]

    seed, r = password_to_seed(password)
    seed2 = (seed + 0.314159265) % 1.0

    seq_key = logistic_map_sequence(seed2,r,n,300)
    key_bytes = np.floor(seq_key*256).astype(np.uint8)

    after_sub = dna_xor_array(flat,key_bytes)

    seq_perm = logistic_map_sequence(seed,r,n,200)
    perm = generate_permutation_from_seq(seq_perm)
    inv_perm = inverse_permutation(perm)

    orig = after_sub[inv_perm]
    recovered = orig.reshape(meta["orig_shape"])

    if mode == "RGB":
        recovered = cv2.cvtColor(recovered, cv2.COLOR_RGB2BGR)
    elif mode == "RGBA":
        recovered = cv2.cvtColor(recovered, cv2.COLOR_RGBA2BGRA)

    return recovered



def allowed(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    name = secure_filename(file.filename)
    new = f"{uuid.uuid4().hex}_{name}"
    path = os.path.join(UPLOAD_FOLDER,new)
    file.save(path)
    return new, path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/encrypt", methods=["GET","POST"])
def encrypt_page():
    if request.method == "GET":
        return render_template("encrypt.html")

    file = request.files.get("image")
    username = request.form.get("username","")
    password = request.form.get("password","")

    if not file or file.filename == "":
        flash("No file selected")
        return redirect(request.url)

    if not allowed(file.filename):
        flash("File type not allowed")
        return redirect(request.url)

    saved_name, saved_path = save_file(file)

    img = cv2.imdecode(np.fromfile(saved_path, np.uint8), cv2.IMREAD_UNCHANGED)

    encrypted_img, meta = encrypt_image(img, password)

    enc_name = username+ "enc_" + saved_name
    enc_path = os.path.join(UPLOAD_FOLDER, enc_name)
    cv2.imencode(".png", encrypted_img)[1].tofile(enc_path)

    meta_path = enc_path + ".meta.npy"
    np.save(meta_path, meta)

    return render_template("result_encrypt.html",
                           input_image=saved_name,
                           encrypted_image=enc_name)

@app.route("/decrypt", methods=["GET","POST"])
def decrypt_page():
    if request.method == "GET":
        return render_template("decrypt.html")

    file = request.files.get("image")
    password = request.form.get("password","")
    if not file or file.filename == "":
        flash("No file selected")
        return redirect(request.url)

    saved_name, saved_path = save_file(file)

    img = cv2.imdecode(np.fromfile(saved_path, np.uint8), cv2.IMREAD_UNCHANGED)

    # Load matching metadata
    meta_file = None
    for f in os.listdir(UPLOAD_FOLDER):
        if f.endswith(".meta.npy"):
            meta_file = os.path.join(UPLOAD_FOLDER,f)
            break

    if meta_file is None:
        flash("Metadata file missing. Cannot decrypt.")
        return redirect(request.url)

    meta = np.load(meta_file, allow_pickle=True).item()
    decrypted = decrypt_image(img, password, meta)

    dec_name = "dec_" + saved_name
    dec_path = os.path.join(UPLOAD_FOLDER, dec_name)

    cv2.imencode(".png", decrypted)[1].tofile(dec_path)

    return render_template("result_decrypt.html", decrypted_image=dec_name)

@app.route("/uploads/<filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()
