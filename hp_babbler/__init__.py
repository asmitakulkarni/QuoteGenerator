import os
from flask import render_template, request, Flask, session
from flask_pymongo import PyMongo
import random
import pickle
from bson.binary import Binary


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        MONGO_URI='mongodb://localhost:27017/character_models',
    )
    
    mongo = PyMongo(app)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/', methods = ['POST', 'GET'])
    def hello():
        character = None
        output = None
        if request.method == 'POST':
            character = request.form['character']
            # call model on character HERE
            output = generate_sentence(mongo, character)
        return render_template('index.html', character=character, output=output)

    return app


def generate_sentence(mongo, character):
    quote_gen = load_quotegen(mongo, character)
    text = [None, None]
    sentence_finished = False
    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in quote_gen[tuple(text[-2:])].keys():
            accumulator += quote_gen[tuple(text[-2:])][word]

            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True

    return ' '.join([t for t in text if t])

def load_quotegen(mongo, character):
    return pickle.loads(mongo.db.character.find_one({'character': character})['model'])
    # return pickle.load(open(character + '.pkl', 'rb') )
