from flask import Flask

app = Flask(__name__, template_folder='templates')
app.secret_key = "secretkey"

from app import routes