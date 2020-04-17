import os

import DeepSparseCoding.utils.file_utils as file_utils

def load_model(model_type, root_dir):
    if(model_type.lower() == "mlp"):
        module_name = "Mlp"
        file_name = os.path.join(root_dir, "models/mlp.py")

    elif(model_type.lower() == "lca"):
        module_name = "Lca"
        file_name = os.path.join(root_dir, "models/lca.py")

    elif(model_type.lower() == "ensemble"):
        module_name = "Ensemble"
        file_name = os.path.join(root_dir, "models/ensemble.py")

    else:
        accepted_names = ["mlp", "lca", "ensemble"]
        assert False, (
            "Acceptible model_types are %s', not %s"%(",".join(accepted_names), model_type))

    module = file_utils.module_from_file(module_name, file_name)
    model = getattr(module, module_name)
    return model()

if __name__ == "__main__":
    # print list of models
    print("TODO.")
