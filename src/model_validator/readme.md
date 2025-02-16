validator = ModelValidator()

# Validate multiple models
valid_models = validator.validate_models([
    "sentence-transformers/all-MiniLM-L6-v2",
    "invalid/model-name"
])

# Validate single model
is_valid = validator.is_valid_model("sentence-transformers/all-MiniLM-L6-v2")