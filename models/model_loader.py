import PyTorchDisentanglement.utils.file_utils as file_utils

def load_model(model_type):
    if(model_type.lower() == "mlp"):
        module_name = "Mlp"
        file_name = "models/mlp.py"

    elif(model_type.lower() == "lca"):
        module_name = "Lca"
        file_name = "models/lca.py"

    elif(model_type.lower() == "ensemble"):
        module_name = "Ensemble"
        file_name = "models/ensemble.py"

    else:
        assert False, ("Acceptible model_types are 'mlp' and 'lca', not %s"%(model_type))

    module = file_utils.module_from_file(module_name, file_name)
    model = getattr(module, module_name)
    return model()

if __name__ == "__main__":
    # print list of models
    print("TODO.")
