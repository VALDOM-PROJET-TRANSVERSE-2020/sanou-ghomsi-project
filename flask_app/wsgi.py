from waitress import serve

from api import app

if __name__ == "__main__":
    #serve(app, host="127.0.0.1", port=5005)

    app.run(host="0.0.0.0", debug=True, port=8000)