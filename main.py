from flask import Flask
from flask_restful import Resource, Api, reqparse
from predictKeyPhrase import predictKey

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('document', type=str)

class HelloWorld(Resource):
    def post(self):
        args = parser.parse_args()
        result = predictKey(args['document'])
        task = {'result': result}
        return task, 201
    def get(self):
        return {'Hi':2}

api.add_resource(HelloWorld, '/api/predict')

if __name__ == '__main__':
    app.run(debug=True)