import joblib
# import onnx
# import skl2onnx

def export_to_onnx(model_path: str, output_path: str):
    """
    Loads a trained model and exports it to ONNX format.
    """
    print(f"Exporting {model_path} to ONNX format at {output_path}...")
    # model = joblib.load(model_path)
    # initial_type = [('float_input', FloatTensorType([None, 4]))]
    # onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
    # with open(output_path, "wb") as f:
    #     f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    export_to_onnx("model.pkl", "model.onnx")
