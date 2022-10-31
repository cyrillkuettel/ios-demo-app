# After loading the model in swift, I got the runtime Error:
# "Lite Interpreter version number does not match.
# The model version must be between 3 and 7 but the model version is 8"
# That's why this snippet here is being added, to convert back to version 7, the last version that worked.

# However, I ended up not using this, as it crashed as well. Still keeping it here, may be useful sometimes.
from torch.jit.mobile import (
    _backport_for_mobile,
    _get_model_bytecode_version,
)

MODEL_INPUT_FILE = "ImageSegmentation/model/model.ptl"
MODEL_OUTPUT_FILE = "model_v5.ptl"

print("model version", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))
exit(0)
_backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=7)

print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE))
