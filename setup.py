from setuptools import setup, find_packages

setup(
    name="model_fusion",
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "torch",
        "torchvision",
        "lightning",
        "pyhessian",
        "python-dotenv",
        "wandb"
    ],
    # author='Your Name',
    # author_email='your_email@example.com',
    # description='Description of your package',
    # url='https://github.com/your_username/your_repository',
)
