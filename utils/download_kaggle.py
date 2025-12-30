import kagglehub

# Download latest version
path = kagglehub.dataset_download("ucimachinelearning/deep-fake-detection-cropped-dataset")

print("Path to dataset files:", path)
