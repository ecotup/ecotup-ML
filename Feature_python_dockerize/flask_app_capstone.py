from flask import Flask
from finding_driver import finding_driver_bp
from clustering_and_sort import clustering_and_sort_bp

flask_feature = Flask(__name__)

flask_feature.register_blueprint(finding_driver_bp)
flask_feature.register_blueprint(clustering_and_sort_bp)

@flask_feature.route('/')
def Starting():
    return 'Welcome, flask app and port is online! please add /find_nearest_driver and /clustering_and_sorting to access the features!!'

if __name__ == '__main__':
    flask_feature.run(host='0.0.0.0', port=5000)