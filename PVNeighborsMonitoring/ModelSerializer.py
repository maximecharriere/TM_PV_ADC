import os
import pickle
import joblib


class ModelSerializer:
    def _save_model(self, model, serial_type, save_params):
        serial_type.dump(model, save_params)

    def _retrieve_model(self, serial_type, retrieve_params):
        return serial_type.load(retrieve_params)


class JoblibSerializer(ModelSerializer):
    def save_model(self, model, save_model_path, filename):
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        super()._save_model(model, joblib, os.path.join(save_model_path, filename + ".joblib"))

    def retrieve_model(self, save_model_path, filename):
        return super()._retrieve_model(joblib, os.path.join(save_model_path, filename + '.joblib'))


class PickleSerializer(ModelSerializer):
    def save_model(self, model, save_model_path, filename):
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        with open(os.path.join(save_model_path, filename + ".pkl"), 'wb') as f:
            super()._save_model(model, pickle, f)

    def retrieve_model(self, save_model_path, filename):
        with open(os.path.join(save_model_path, filename + ".pkl"), 'rb') as f:
            return super()._retrieve_model(pickle, f)